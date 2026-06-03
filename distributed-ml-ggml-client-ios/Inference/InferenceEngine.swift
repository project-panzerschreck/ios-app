// InferenceEngine.swift
//
// High-level Swift wrapper around LlamaBridge.
// Manages model lifecycle and exposes an async streaming generation API.
//
// Thread model
// ────────────
// All @Observable state mutations happen on MainActor.
// Blocking llama.cpp work (model load, inference) runs in Task.detached so it
// doesn't block the main thread.  `bridge` is marked nonisolated(unsafe) to
// allow capture from detached tasks; the caller guarantees single-threaded
// access to the bridge (only one operation at a time).

import Foundation
import Combine
import Network
import Darwin
#if canImport(UIKit)
import UIKit
#endif

// ── Chat message ──────────────────────────────────────────────────────────────

struct ChatMessage: Identifiable, Equatable {
    let id = UUID()
    let role: String    // "user" | "assistant"
    var content: String
}

// ── Generation result ─────────────────────────────────────────────────────────

struct GenerationResult: Sendable {
    let text: String
    let isDone: Bool
    let tokensPerSecond: Double
}

// ── RPC server state ──────────────────────────────────────────────────────────

enum RPCServerState: Equatable {
    case idle
    case starting
    case running(endpoint: String)
    case recovering(reason: String)
    case degraded(reason: String)
    case unavailable(String)
}

// ── Model state ───────────────────────────────────────────────────────────────

enum ModelState: Equatable {
    case unloaded
    case loading
    case ready(modelName: String, nLayers: Int)
    case generating
    case error(String)
}

// ── InferenceEngine ───────────────────────────────────────────────────────────

@MainActor
final class InferenceEngine: ObservableObject {
    // ── Published state ───────────────────────────────────────────────────────
    @Published var modelState: ModelState = .unloaded
    @Published var generatedText: String  = ""
    @Published var tokensPerSecond: Double = 0
    @Published var rpcServerState: RPCServerState = .idle
    @Published var chatMessages: [ChatMessage] = []

    // ── nonisolated(unsafe): accessed from detached tasks, single-threaded ────
    nonisolated(unsafe) private let bridge = LlamaBridge()
    private var generationTask: Task<Void, Never>?
    private var rpcServerTask:  Task<Void, Never>?
    private var supervisorTask: Task<Void, Never>?
    private var storageServer: StorageServer?

    private struct NodeRuntimeConfig: Equatable {
        let coordinatorHost: String
        let coordinatorPort: Int
        let nickname: String
        let threads: Int
        let deviceId: String
    }

    private let generalLogTag = "GENERAL"
    private let rpcServerLogTag = "RPC SERVER"
    private let storageLogTag = "STORAGE"
    private let healthCheckTimeout: TimeInterval = 1.5
    private let recoveryBackoffNs: UInt64 = 2_000_000_000
    private let startupGraceInterval: TimeInterval = 2.0
    private let defaultAnnounceInterval: Double = 10.0

    private var desiredNodeConfig: NodeRuntimeConfig?
    private var nodeShouldBeRunning = false
    private var appIsActive = true
    private var runtimeIsShuttingDown = false
    private var discoveryActive = false
    private var announceEligible = false
    private var rpcHealthy = false
    private var storageHealthy = false
    private var currentRPCEndpoint = ""
    private var currentStorageEndpoint = ""
    private var lastRuntimeError = ""
    private var lastRPCStartAt: Date?
    private var lastStorageStartAt: Date?

    static let shared = InferenceEngine()

    // ── Model loading ─────────────────────────────────────────────────────────

    func loadModel(from url: URL, contextLength: Int = 1024) async {
        modelState    = .loading
        generatedText = ""

        let path = url.path

        // Detach so the blocking file-read doesn't stall the main actor.
        // Await `.value` on its own line so Swift sees the async suspension.
        let loadOp = Task.detached(priority: .userInitiated) { [bridge] in
            // ObjC `- (BOOL)method:(T)a error:(NSError **)e` is bridged as
            // `func method(_ a: T) throws` in Swift — no `error:` argument.
            do {
                try bridge.loadModel(fromPath: path, nCtx: contextLength)
                let name    = bridge.modelInfo?.name    ?? "Unknown"
                let nLayers = bridge.modelInfo?.nLayers ?? 0
                return (name, nLayers, "" as String)
            } catch {
                return ("", 0, error.localizedDescription)
            }
        }
        let (name, nLayers, errMsg) = await loadOp.value

        if errMsg.isEmpty {
            modelState = .ready(modelName: name, nLayers: nLayers)
        } else {
            modelState = .error(errMsg)
        }
    }

    func unloadModel() {
        generationTask?.cancel()
        generationTask = nil
        bridge.unloadModel()
        modelState    = .unloaded
        generatedText = ""
        tokensPerSecond = 0
        chatMessages  = []
    }

    var modelInfo: LlamaModelInfo? { bridge.modelInfo }
    var eosTokenID: Int32 { bridge.modelInfo?.eosTokenID ?? 2 }

    // ── Single-device generation ──────────────────────────────────────────────

    func generate(
        prompt: String,
        config: LlamaGenerationConfig = .defaults()
    ) -> AsyncStream<GenerationResult> {
        // IMPORTANT: do NOT touch `generationTask` here.
        // `generateIntoState` already assigned `generationTask` to the Task
        // that is currently running this call.  If we cancel/replace it here
        // we would cancel our own caller, causing the `for await` loop to exit
        // immediately and the model state to revert to .ready without generating
        // a single token.
        AsyncStream { continuation in
            let innerTask = Task.detached(priority: .userInitiated) { [bridge] in
                var accumulated = ""
                var tokenCount  = 0
                let start       = Date()

                // ObjC callback: (NSString * _Nonnull, BOOL) → (String, Bool)
                // NSString * inside NS_ASSUME_NONNULL_BEGIN is non-optional.
                bridge.generate(fromPrompt: prompt, config: config) { piece, done in
                    accumulated += piece
                    tokenCount  += 1

                    let elapsed = Date().timeIntervalSince(start)
                    let tps     = elapsed > 0 ? Double(tokenCount) / elapsed : 0

                    continuation.yield(GenerationResult(
                        text: accumulated,
                        isDone: done,
                        tokensPerSecond: tps
                    ))
                    if done { continuation.finish() }
                }
            }
            // When the consumer (generateIntoState's for-await loop) is cancelled,
            // also cancel the inner detached task so it can exit cleanly.
            continuation.onTermination = { _ in innerTask.cancel() }
        }
    }

    func generateIntoState(prompt: String, config: LlamaGenerationConfig = .defaults()) {
        guard case .ready = modelState else { return }
        modelState    = .generating
        generatedText = ""
        tokensPerSecond = 0

        generationTask = Task { @MainActor in
            for await result in generate(prompt: prompt, config: config) {
                generatedText   = result.text
                tokensPerSecond = result.tokensPerSecond
            }
            if case .generating = modelState, let info = bridge.modelInfo {
                modelState = .ready(modelName: info.name, nLayers: info.nLayers)
            }
        }
    }

    func cancelGeneration() {
        generationTask?.cancel()
        generationTask = nil
        if let info = bridge.modelInfo {
            modelState = .ready(modelName: info.name, nLayers: info.nLayers)
        }
    }

    // ── Conversation mode ─────────────────────────────────────────────────────

    /// Send a user message and stream the assistant reply into `chatMessages`.
    func sendMessage(_ text: String, config: LlamaGenerationConfig = .defaults()) {
        guard case .ready = modelState else { return }

        // Build the message list for the template (all history + new user turn).
        let historyForTemplate = chatMessages + [ChatMessage(role: "user", content: text)]
        let nsMessages = historyForTemplate.map { ["role": $0.role, "content": $0.content] }

        // Apply the model's built-in chat template (from GGUF metadata).
        guard let formatted = bridge.applyChatTemplate(nsMessages, addAssistantPrefix: true) else {
            modelState = .error("Model has no chat template — cannot use conversation mode")
            return
        }

        chatMessages.append(ChatMessage(role: "user", content: text))
        chatMessages.append(ChatMessage(role: "assistant", content: ""))
        let assistantIndex = chatMessages.count - 1

        modelState      = .generating
        tokensPerSecond = 0

        generationTask = Task { @MainActor in
            for await result in generate(prompt: formatted, config: config) {
                chatMessages[assistantIndex].content = result.text
                tokensPerSecond = result.tokensPerSecond
            }
            if case .generating = modelState, let info = bridge.modelInfo {
                modelState = .ready(modelName: info.name, nLayers: info.nLayers)
            }
        }
    }

    func clearChat() {
        generationTask?.cancel()
        generationTask = nil
        chatMessages   = []
        tokensPerSecond = 0
        if let info = bridge.modelInfo {
            modelState = .ready(modelName: info.name, nLayers: info.nLayers)
        }
    }

    // ── GGML RPC worker server ────────────────────────────────────────────────

    /// Whether the GGML RPC backend was compiled in.
    /// Requires ggml-rpc.xcframework (rebuild with GGML_RPC=ON).
    var rpcAvailable: Bool { LlamaBridge.rpcAvailable() }

    /// Whether the device supports Metal acceleration for llama.cpp.
    var metalAvailable: Bool { LlamaBridge.metalAvailable() }

    /// Start the GGML RPC server so an external llama-cli can use this device
    /// as a Metal compute backend.  The phone is a leaf node only — it never
    /// coordinates inference itself.
    func startRPCServer(
        coordinatorHost: String,
        coordinatorPort: Int,
        nickname: String,
        threads: Int,
        deviceId: String
    ) {
        let config = NodeRuntimeConfig(
            coordinatorHost: coordinatorHost.trimmingCharacters(in: .whitespacesAndNewlines),
            coordinatorPort: coordinatorPort,
            nickname: nickname,
            threads: threads,
            deviceId: deviceId
        )

        guard !config.coordinatorHost.isEmpty else {
            AppLogger.log("WARN", tag: generalLogTag, "node.start.rejected reason=missing_coordinator_host")
            return
        }

        desiredNodeConfig = config
        nodeShouldBeRunning = true
        runtimeIsShuttingDown = false
        lastRuntimeError = ""
        rpcServerState = .starting
        AppLogger.log(tag: generalLogTag, "node.start.requested coordinator=\(config.coordinatorHost):\(config.coordinatorPort)")
        applyKeepAwakePolicy()
        publishRuntimeHealth(statusOverride: "starting")

        if appIsActive {
            startNodeSupervisorIfNeeded()
        } else {
            rpcServerState = .degraded(reason: "Waiting for app to become active")
            publishRuntimeHealth(statusOverride: "degraded")
        }
    }

    /// Returns (freeMB, totalMB) reflecting actual device memory at call time.
    /// Uses os_proc_available_memory() — the per-process jetsam headroom —
    /// so the coordinator never offloads more than the phone can hold.
    /// A 10% safety margin is subtracted for app/Metal overhead.
    private nonisolated static func deviceMemoryMB() -> (freeMB: UInt, totalMB: UInt) {
        let total = UInt(ProcessInfo.processInfo.physicalMemory / 1_048_576)
        let rawFree = UInt(LlamaBridge.processAvailableMemoryBytes() / 1_048_576)
        let free = UInt(Double(rawFree) * 0.9)   // 10% headroom for app/Metal overhead
        return (freeMB: free, totalMB: total)
    }

    /// Stop the RPC server.
    /// Note: this cancels the Swift Task; the underlying C server loop will be
    /// interrupted when the OS reclaims the socket on thread teardown.
    func stopRPCServer() {
        AppLogger.log(tag: generalLogTag, "node.stop.requested reason=user")
        stopNodeRuntime(preserveDesiredConfig: false, reason: "Stopped by user")
    }

    func handleAppDidBecomeActive() {
        appIsActive = true
        AppLogger.log(tag: generalLogTag, "app.active")
        applyKeepAwakePolicy()
        if nodeShouldBeRunning {
            if case .degraded(_) = rpcServerState {
                rpcServerState = .starting
            }
            startNodeSupervisorIfNeeded()
        } else {
            publishRuntimeHealth(statusOverride: "idle")
        }
    }

    func handleAppWillResignActive() {
        appIsActive = false
        AppLogger.log(tag: generalLogTag, "app.inactive stopping_runtime=true")
        if nodeShouldBeRunning {
            stopNodeRuntime(preserveDesiredConfig: true, reason: "App moved to background")
        } else {
            applyKeepAwakePolicy()
            publishRuntimeHealth(statusOverride: "idle")
        }
    }

    private func startNodeSupervisorIfNeeded() {
        guard supervisorTask == nil, nodeShouldBeRunning, appIsActive else { return }
        AppLogger.log(tag: generalLogTag, "supervisor.start")
        supervisorTask = Task.detached(priority: .utility) { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let interval = await self.runNodeSupervisorIteration()
                do {
                    try await Task.sleep(nanoseconds: UInt64(max(interval, 0.5) * 1_000_000_000))
                } catch {
                    break
                }
            }
        }
    }

    private func stopNodeRuntime(preserveDesiredConfig: Bool, reason: String) {
        runtimeIsShuttingDown = true
        stopDiscoveryLoop()
        stopStorageServer()
        stopRPCWorker(reason: reason)
        supervisorTask?.cancel()
        supervisorTask = nil
        rpcHealthy = false
        storageHealthy = false
        announceEligible = false
        discoveryActive = false

        if preserveDesiredConfig {
            rpcServerState = .degraded(reason: reason)
        } else {
            nodeShouldBeRunning = false
            desiredNodeConfig = nil
            lastRuntimeError = ""
            currentRPCEndpoint = ""
            rpcServerState = .idle
        }

        applyKeepAwakePolicy()
        publishRuntimeHealth(statusOverride: preserveDesiredConfig ? "degraded" : "idle")
    }

    private func startDiscoveryLoop() {
        discoveryActive = true
    }

    private func stopDiscoveryLoop() {
        discoveryActive = false
    }

    private func startStorageServer() -> Bool {
        guard storageServer == nil else { return true }
        let server = StorageServer(storageDir: RpcSettings.shared.storageDirectory)
        let started = server.start(port: RpcSettings.storagePort)
        if started {
            storageServer = server
            currentStorageEndpoint = "127.0.0.1:\(RpcSettings.storagePort)"
            lastStorageStartAt = Date()
            AppLogger.log(tag: storageLogTag, "storage.runtime.started endpoint=\(currentStorageEndpoint)")
        } else {
            lastRuntimeError = "Storage failed to bind port \(RpcSettings.storagePort)"
            AppLogger.log("ERROR", tag: generalLogTag, "storage.runtime.start_failed port=\(RpcSettings.storagePort)")
        }
        return started
    }

    private func stopStorageServer() {
        storageServer?.stop()
        storageServer = nil
        currentStorageEndpoint = ""
    }

    private func restartStorageServer() {
        AppLogger.log(tag: generalLogTag, "storage.runtime.restart_requested")
        stopStorageServer()
        _ = startStorageServer()
    }

    private func startRPCWorker(_ config: NodeRuntimeConfig) {
        guard rpcServerTask == nil else { return }
        guard LlamaBridge.rpcAvailable() else {
            let message = "ggml-rpc not compiled in. Run scripts/build-ggml-ios.sh then add ggml-rpc.xcframework to the Xcode target."
            rpcServerState = .unavailable(message)
            lastRuntimeError = message
            nodeShouldBeRunning = false
            desiredNodeConfig = nil
            AppLogger.log("ERROR", tag: rpcServerLogTag, "rpc.unavailable reason=not_compiled")
            publishRuntimeHealth(statusOverride: "unavailable")
            return
        }

        runtimeIsShuttingDown = false
        currentRPCEndpoint = "\(RpcSettings.listenHost):\(RpcSettings.listenPort)"
        lastRPCStartAt = Date()
        let endpoint = currentRPCEndpoint
        let (freeMB, totalMB) = Self.deviceMemoryMB()
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first?.path
        let threads = config.threads
        rpcServerState = .starting
        AppLogger.log(tag: rpcServerLogTag, "rpc.start.begin endpoint=\(endpoint) threads=\(threads) cache=\(cacheDir ?? "disabled") free_mb=\(freeMB) total_mb=\(totalMB)")
        if let cacheDir, !cacheDir.isEmpty {
            AppLogger.log(tag: rpcServerLogTag, "rpc.cache.enabled path=\(cacheDir)")
        } else {
            AppLogger.log("WARN", tag: rpcServerLogTag, "rpc.cache.disabled")
        }

        rpcServerTask = Task.detached(priority: .userInitiated) { [bridge] in
            bridge.startRPCServer(
                endpoint,
                cacheDir: cacheDir,
                freeMB: freeMB,
                totalMB: totalMB,
                threads: UInt(threads)
            )

            await MainActor.run {
                self.handleRPCWorkerExit(endpoint: endpoint)
            }
        }
    }

    private func stopRPCWorker(reason: String) {
        guard rpcServerTask != nil else { return }
        AppLogger.log(tag: rpcServerLogTag, "rpc.stop.requested reason=\(reason)")
        runtimeIsShuttingDown = true
        if !currentRPCEndpoint.isEmpty {
            bridge.stopRPCServer(currentRPCEndpoint)
        }
        rpcServerTask?.cancel()
    }

    private func restartRPCWorker(_ config: NodeRuntimeConfig, reason: String) {
        if rpcServerTask != nil {
            stopRPCWorker(reason: reason)
        } else {
            AppLogger.log(tag: rpcServerLogTag, "rpc.restart.requested")
            startRPCWorker(config)
        }
    }

    private func handleRPCWorkerExit(endpoint: String) {
        let expected = runtimeIsShuttingDown || !nodeShouldBeRunning || !appIsActive
        rpcServerTask = nil
        rpcHealthy = false
        announceEligible = false

        if expected {
            AppLogger.log(tag: rpcServerLogTag, "rpc.exit.expected endpoint=\(endpoint)")
            if !nodeShouldBeRunning {
                currentRPCEndpoint = ""
                rpcServerState = .idle
            }
        } else {
            lastRuntimeError = "RPC worker exited unexpectedly"
            rpcServerState = .recovering(reason: lastRuntimeError)
            AppLogger.log("ERROR", tag: rpcServerLogTag, "rpc.exit.unexpected endpoint=\(endpoint)")
        }
        publishRuntimeHealth(statusOverride: expected ? currentRuntimeStatusName() : "recovering")
    }

    private func isRPCHealthy() async -> Bool {
        guard rpcServerTask != nil else { return false }
        return await Self.probeTCP(host: "127.0.0.1", port: RpcSettings.listenPort, timeout: healthCheckTimeout)
    }

    private func isStorageHealthy() async -> Bool {
        guard storageServer != nil else { return false }
        guard let url = URL(string: "http://127.0.0.1:\(RpcSettings.storagePort)/storage_info") else { return false }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = healthCheckTimeout

        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else { return false }
            return httpResponse.statusCode == 200
        } catch {
            return false
        }
    }

    private func runNodeSupervisorIteration() async -> Double {
        guard nodeShouldBeRunning, appIsActive, let config = desiredNodeConfig else {
            publishRuntimeHealth(statusOverride: currentRuntimeStatusName())
            return defaultAnnounceInterval
        }

        applyKeepAwakePolicy()
        let _ = startStorageServer()
        if rpcServerTask == nil {
            startRPCWorker(config)
        }

        let rpcProbe = await isRPCHealthy()
        let storageProbe = await isStorageHealthy()
        let rpcWithinGrace = lastRPCStartAt.map { Date().timeIntervalSince($0) < startupGraceInterval } ?? false
        let storageWithinGrace = lastStorageStartAt.map { Date().timeIntervalSince($0) < startupGraceInterval } ?? false
        let effectiveRPCHealthy = rpcProbe || (rpcServerTask != nil && rpcWithinGrace)
        let effectiveStorageHealthy = storageProbe || (storageServer != nil && storageWithinGrace)

        rpcHealthy = effectiveRPCHealthy
        storageHealthy = effectiveStorageHealthy
        announceEligible = rpcHealthy && storageHealthy
        AppLogger.log("DEBUG", tag: generalLogTag, "health.check rpc_probe=\(rpcProbe) storage_probe=\(storageProbe) rpc_grace=\(rpcWithinGrace) storage_grace=\(storageWithinGrace) rpc_healthy=\(rpcHealthy) storage_healthy=\(storageHealthy) announce_eligible=\(announceEligible)")

        if !storageHealthy {
            stopDiscoveryLoop()
            lastRuntimeError = "Storage server unhealthy"
            rpcServerState = .recovering(reason: lastRuntimeError)
            publishRuntimeHealth(statusOverride: "recovering")
            if !storageWithinGrace {
                AppLogger.log("WARN", tag: generalLogTag, "health.storage_unhealthy action=restart_storage announce=skipped")
                restartStorageServer()
            }
            return Double(recoveryBackoffNs) / 1_000_000_000
        }

        if !rpcHealthy {
            stopDiscoveryLoop()
            lastRuntimeError = "RPC worker unhealthy"
            rpcServerState = .recovering(reason: lastRuntimeError)
            publishRuntimeHealth(statusOverride: "recovering")
            if !rpcWithinGrace {
                AppLogger.log("WARN", tag: generalLogTag, "health.rpc_unhealthy action=restart_rpc announce=skipped")
                restartRPCWorker(config, reason: lastRuntimeError)
            }
            return Double(recoveryBackoffNs) / 1_000_000_000
        }

        rpcServerState = .running(endpoint: currentRPCEndpoint)
        publishRuntimeHealth(statusOverride: "running")
        return await announceToCoordinator(config)
    }

    private func announceToCoordinator(_ config: NodeRuntimeConfig) async -> Double {
        startDiscoveryLoop()

        do {
            let hwModel = Self.hardwareModel()
            let maxBytes = LlamaBridge.availableProcessMemoryBytes()
            let localIP = Self.primaryIPv4Address()
            #if canImport(UIKit)
            UIDevice.current.isBatteryMonitoringEnabled = true
            defer { UIDevice.current.isBatteryMonitoringEnabled = false }
            let battery = UIDevice.current.batteryLevel
            #else
            let battery: Float = -1
            #endif
            let tempC = Self.thermalStateTemperature(ProcessInfo.processInfo.thermalState)

            var comps = URLComponents()
            comps.scheme = "http"
            comps.host = config.coordinatorHost
            comps.port = config.coordinatorPort
            comps.path = "/announce"
            var items: [URLQueryItem] = [
                .init(name: "id", value: config.deviceId),
                .init(name: "port", value: "\(RpcSettings.listenPort)"),
                .init(name: "storage_port", value: "\(RpcSettings.storagePort)"),
                .init(name: "ip", value: localIP),
                .init(name: "model", value: hwModel),
                .init(name: "max_size", value: "\(maxBytes)")
            ]
            if battery >= 0 {
                items.append(.init(name: "battery", value: String(format: "%.1f", battery * 100)))
            }
            if !tempC.isNaN {
                items.append(.init(name: "temperature", value: String(format: "%.1f", tempC)))
            }
            let trimmedNickname = config.nickname.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmedNickname.isEmpty {
                items.append(.init(name: "nickname", value: trimmedNickname))
            }
            comps.queryItems = items

            guard let url = comps.url else {
                lastRuntimeError = "Failed to build announce URL"
                AppLogger.log("ERROR", tag: generalLogTag, "announce.request.invalid_url")
                publishRuntimeHealth(statusOverride: "running")
                return 1
            }

            AppLogger.log("DEBUG", tag: generalLogTag, "announce.request url=\(url.absoluteString)")

            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            request.timeoutInterval = 5
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                lastRuntimeError = "Coordinator announce failed"
                AppLogger.log("WARN", tag: generalLogTag, "announce.failed status=\((response as? HTTPURLResponse)?.statusCode ?? -1)")
                publishRuntimeHealth(statusOverride: "running")
                return 1
            }

            lastRuntimeError = ""
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            let intervalSec = (json?["interval"] as? NSNumber)?.doubleValue ?? defaultAnnounceInterval
            AppLogger.log(tag: generalLogTag, "announce.ok interval_sec=\(intervalSec)")
            publishRuntimeHealth(statusOverride: "running")
            return intervalSec
        } catch is CancellationError {
            return defaultAnnounceInterval
        } catch {
            lastRuntimeError = "Coordinator announce error: \(error.localizedDescription)"
            AppLogger.log("WARN", tag: generalLogTag, "announce.error error=\(error.localizedDescription)")
            publishRuntimeHealth(statusOverride: "running")
            return 1
        }
    }

    private func applyKeepAwakePolicy() {
        #if canImport(UIKit)
        UIApplication.shared.isIdleTimerDisabled = nodeShouldBeRunning && appIsActive
        #endif
    }

    private func publishRuntimeHealth(statusOverride: String? = nil) {
        let coordinator = desiredNodeConfig.map { "\($0.coordinatorHost):\($0.coordinatorPort)" } ?? ""
        let status = statusOverride ?? currentRuntimeStatusName()
        AppLogger.log("DEBUG", tag: generalLogTag, "health.publish status=\(status) rpc=\(currentRPCEndpoint.isEmpty ? "none" : currentRPCEndpoint) storage=\(currentStorageEndpoint.isEmpty ? "none" : currentStorageEndpoint) coordinator=\(coordinator.isEmpty ? "none" : coordinator) rpc_healthy=\(rpcHealthy) storage_healthy=\(storageHealthy) announce_eligible=\(announceEligible) discovery_active=\(discoveryActive)")
        AppLogger.rpcHealth(status: status, details: [
            "endpoint": currentRPCEndpoint,
            "storage_endpoint": currentStorageEndpoint,
            "coordinator": coordinator,
            "last_error": lastRuntimeError,
            "rpc_available": rpcAvailable,
            "metal_available": metalAvailable,
            "discovery_active": discoveryActive,
            "rpc_healthy": rpcHealthy,
            "storage_healthy": storageHealthy,
            "announce_eligible": announceEligible
        ])
    }

    private func currentRuntimeStatusName() -> String {
        switch rpcServerState {
        case .idle:
            return "idle"
        case .starting:
            return "starting"
        case .running:
            return "running"
        case .recovering:
            return "recovering"
        case .degraded:
            return "degraded"
        case .unavailable:
            return "unavailable"
        }
    }

    // Returns the hardware model identifier, e.g. "iPhone16,1".
    private nonisolated static func hardwareModel() -> String {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var machine = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.machine", &machine, &size, nil, 0)
        return String(cString: machine)
    }

    private nonisolated static func primaryIPv4Address() -> String {
        let monitor = NWPathMonitor()
        let primaryInterfaceName = monitor.currentPath.availableInterfaces.first?.name

        var ifAddr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifAddr) == 0, let first = ifAddr else { return "0.0.0.0" }
        defer { freeifaddrs(first) }

        return sequence(first: first, next: { $0.pointee.ifa_next })
            .compactMap { node -> String? in
                let ifa = node.pointee
                guard ifa.ifa_addr.pointee.sa_family == UInt8(AF_INET) else { return nil }

                let name = String(cString: ifa.ifa_name)
                if name == primaryInterfaceName || (primaryInterfaceName == nil && name != "lo0") {
                    var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                    getnameinfo(ifa.ifa_addr, socklen_t(ifa.ifa_addr.pointee.sa_len),
                                &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST)
                    return String(cString: host)
                }
                return nil
            }
            .first ?? "0.0.0.0"
    }

    private nonisolated static func thermalStateTemperature(_ state: ProcessInfo.ThermalState) -> Double {
        switch state {
        case .nominal:  return 30.0
        case .fair:     return 38.0
        case .serious:  return 45.0
        case .critical: return 55.0
        @unknown default: return Double.nan
        }
    }

    private nonisolated static func probeTCP(host: String, port: Int, timeout: TimeInterval) async -> Bool {
        guard let nwPort = NWEndpoint.Port(rawValue: UInt16(port)) else { return false }

        return await withCheckedContinuation { continuation in
            let queue = DispatchQueue(label: "InferenceEngine.probeTCP")
            let connection = NWConnection(host: NWEndpoint.Host(host), port: nwPort, using: .tcp)
            var completed = false

            @Sendable func finish(_ result: Bool) {
                guard !completed else { return }
                completed = true
                connection.cancel()
                continuation.resume(returning: result)
            }

            connection.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    finish(true)
                case .failed(_), .cancelled:
                    finish(false)
                default:
                    break
                }
            }

            connection.start(queue: queue)
            queue.asyncAfter(deadline: .now() + timeout) {
                finish(false)
            }
        }
    }

    // ── Distributed shard helpers ─────────────────────────────────────────────

    func runFirstShard(
        tokens: [Int32],
        endLayer: Int
    ) async -> (hiddenState: Data, tokenCount: Int, nEmbd: Int) {
        // The ObjC callback is synchronous (fires before the method returns),
        // so we don't need withCheckedContinuation — just run inside a detached
        // task so we don't block the main actor.
        let op = Task.detached(priority: .userInitiated) { [bridge] in
            let nsTokens = tokens.map { NSNumber(value: $0) }
            var result: (Data, Int, Int) = (Data(), 0, 0)
            bridge.runFirstShard(withTokens: nsTokens, endLayer: endLayer) {
                state, count, embd, _ in
                result = (state, Int(count), Int(embd))
            }
            return result
        }
        return await op.value
    }

    func runShard(
        hiddenState: Data,
        tokenCount: Int,
        startLayer: Int,
        endLayer: Int
    ) async -> (hiddenState: Data, tokenCount: Int, nEmbd: Int) {
        let op = Task.detached(priority: .userInitiated) { [bridge] in
            var result: (Data, Int, Int) = (Data(), 0, 0)
            bridge.runShard(withHiddenState: hiddenState,
                            tokenCount: tokenCount,
                            startLayer: startLayer,
                            endLayer: endLayer) { state, count, embd, _ in
                result = (state, Int(count), Int(embd))
            }
            return result
        }
        return await op.value
    }

    // ── Tokenization helpers ──────────────────────────────────────────────────

    func tokenize(text: String, addBOS: Bool = true) -> [Int32] {
        bridge.tokenizeText(text, addBOS: addBOS).map { $0.int32Value }
    }

    func tokenToPiece(_ tokenID: Int32) -> String {
        // ObjC `tokenToPiece:` is renamed by Swift to `tokenPiece(_:)`
        // via NS_SWIFT_NAME(tokenPiece(_:)) in LlamaBridge.h
        bridge.tokenPiece(tokenID)
    }
}
