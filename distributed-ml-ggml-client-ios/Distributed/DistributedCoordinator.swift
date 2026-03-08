// DistributedCoordinator.swift
//
// Orchestrates pipeline-parallel inference across a cluster of devices.
//
// Roles
// ─────
//   Coordinator  – the device that drives the full generation loop.
//                  It tokenizes the prompt, triggers the first shard, waits
//                  for token results from the last shard, and decides when
//                  to stop (EOS / maxNewTokens).
//                  (Coordinator == first-shard device in the simplest topology.)
//
//   Shard worker – any device (including the coordinator) that runs a layer
//                  range and forwards activations to the next device.
//
// Session lifecycle
// ─────────────────
//   1. Coordinator discovers peers and assigns ShardPlan.
//   2. Each peer starts its ShardNetwork server and acknowledges readiness.
//   3. Coordinator tokenizes prompt → sends ActivationPacket to its own
//      first-shard InferenceEngine.
//   4. First shard produces hidden state → ShardNetwork POSTs to second device.
//   5. …
//   6. Last shard produces a TokenResult → POSTs to coordinator.
//   7. Coordinator checks EOS; if not done, feeds new token back as step N+1.
//   8. Repeat from step 3 for each autoregressive decode step.
//
// Current status: coordinator and single-device fallback are fully implemented.
// Multi-device path is structurally complete; actual hidden-state transfer
// requires the llama-partial-eval.patch (see LlamaBridge.mm).

import Combine
import Foundation
import Network
#if canImport(UIKit)
import UIKit
#endif

// ── Session state ─────────────────────────────────────────────────────────────

enum SessionState: Equatable {
    case idle
    case planningSession
    case waitingForPeers
    case running(step: Int)
    case finished(reason: FinishReason)
    case error(String)
}

enum FinishReason: String, Equatable {
    case eos          = "End of sequence"
    case maxTokens    = "Max tokens reached"
    case cancelled    = "Cancelled"
    case networkError = "Network error"
}

// ── Coordinator ───────────────────────────────────────────────────────────────

@MainActor
final class DistributedCoordinator: ObservableObject {

    // ── Published ─────────────────────────────────────────────────────────────
    @Published var sessionState: SessionState = .idle
    @Published var generatedTokens: [String]  = []
    @Published var currentStep: Int           = 0
    @Published var activePlan: ShardPlan?

    // ── Dependencies ──────────────────────────────────────────────────────────
    private let engine:  InferenceEngine
    private let network: ShardNetwork

    // ── Internal ──────────────────────────────────────────────────────────────
    private var generationTask: Task<Void, Never>?
    private var tokenResultContinuation: AsyncStream<TokenResult>.Continuation?
    /// Last sampled token ID — used as input for the next decode step.
    private var lastTokenID: Int32 = 0

    init(engine: InferenceEngine? = nil) {
        self.engine  = engine ?? InferenceEngine.shared
        self.network = ShardNetwork()
    }

    // ── Plan management ───────────────────────────────────────────────────────

    /// Build a plan for single-device inference (the "sole shard" case).
    func buildSolePlan(modelID: String) {
        guard let info = engine.modelInfo else { return }
        let localEndpoint = URL(string: "http://localhost:\(ShardNetwork.defaultPort)")!
        activePlan = ShardPlan.balanced(
            modelID: modelID,
            totalLayers: info.nLayers,
            nEmbd: info.nEmbd,
            eosTokenID: engine.eosTokenID,
            deviceDescriptors: [(
                label: UIDeviceLabel.current,
                deviceID: UIDeviceLabel.deviceID,
                endpoint: localEndpoint
            )]
        )
    }

    /// Build a pipeline plan across multiple peers.
    /// `peerDescriptors` should be ordered first→last (excluding self; self is always prepended as first).
    func buildDistributedPlan(
        modelID: String,
        peerDescriptors: [(label: String, deviceID: String, endpoint: URL)]
    ) {
        guard let info = engine.modelInfo else { return }
        let localEndpoint = URL(string: "http://localhost:\(ShardNetwork.defaultPort)")!
        var all = [(label: String, deviceID: String, endpoint: URL)]()
        all.append((UIDeviceLabel.current, UIDeviceLabel.deviceID, localEndpoint))
        all.append(contentsOf: peerDescriptors)
        activePlan = ShardPlan.balanced(
            modelID: modelID,
            totalLayers: info.nLayers,
            nEmbd: info.nEmbd,
            eosTokenID: engine.eosTokenID,
            deviceDescriptors: all
        )
    }

    // ── Generation ────────────────────────────────────────────────────────────

    /// Run a full generation session using the current `activePlan`.
    /// Yields one `String` token piece per autoregressive step.
    func generate(
        prompt: String,
        config: LlamaGenerationConfig = .defaults(),
        maxNewTokens: Int = 200
    ) -> AsyncStream<String> {
        AsyncStream { continuation in
            generationTask?.cancel()
            generationTask = Task { @MainActor in
                guard let plan = activePlan else {
                    continuation.finish()
                    return
                }

                // Determine local shard
                guard let myShard = plan.shards.first(where: {
                    $0.deviceID == UIDeviceLabel.deviceID
                }) else {
                    continuation.finish()
                    return
                }

                // ── Single-device fast path ───────────────────────────────────
                if myShard.role == .sole {
                    sessionState = .running(step: 0)
                    var step = 0
                    for await result in engine.generate(prompt: prompt, config: config) {
                        // Stream each token piece to the caller
                        let pieces = result.text.dropFirst(
                            generatedTokens.joined().count
                        )
                        if !pieces.isEmpty {
                            let s = String(pieces)
                            generatedTokens.append(s)
                            continuation.yield(s)
                        }
                        step += 1
                        sessionState = .running(step: step)
                        if result.isDone { break }
                    }
                    sessionState = .finished(reason: .eos)
                    continuation.finish()
                    return
                }

                // ── Distributed path ──────────────────────────────────────────
                sessionState = .planningSession

                // Start the local shard server so other devices can POST to us.
                await network.startServer()
                sessionState = .waitingForPeers

                // Distribute the ShardPlan to every peer device so they know
                // their layer range and can start their own inference loops.
                if myShard.role == .first || myShard.role == .sole {
                    for peerShard in plan.shards where peerShard.deviceID != UIDeviceLabel.deviceID {
                        try? await network.sendShardPlan(plan, to: peerShard.selfEndpoint)
                    }
                }

                // Tokenize the prompt (first shard only).
                var tokenIDs: [Int32] = []
                if myShard.role == .first {
                    tokenIDs = engine.tokenize(text: prompt, addBOS: true)
                }

                lastTokenID = 0
                var step = 0
                var done = false

                while !done && step < maxNewTokens {
                    sessionState = .running(step: step)

                    switch myShard.role {
                    case .first:
                        // Run the full forward pass locally (proxy for first-shard
                        // computation until llama_decode_partial() is available).
                        // Returns raw logits (n_vocab floats) as the "hidden state".
                        let inputTokens = step == 0 ? tokenIDs : [lastTokenID]
                        let (hiddenState, tokenCount, nEmbd) = await engine.runFirstShard(
                            tokens: inputTokens,
                            endLayer: myShard.endLayer
                        )
                        if let nextURL = myShard.nextDeviceEndpoint {
                            // Pipeline: forward logits to the next (last) shard device.
                            let packet = ActivationPacket(
                                sessionID: plan.sessionID,
                                step: step,
                                tokenPosition: step,
                                tokenCount: tokenCount,
                                nEmbd: nEmbd,
                                hiddenState: hiddenState.toFloatArray(),
                                isDone: false
                            )
                            try? await network.sendActivation(packet, to: nextURL)
                            // Wait for sampled token from the last shard.
                            if let result = await network.waitForTokenResult(sessionID: plan.sessionID) {
                                lastTokenID = result.tokenID
                                generatedTokens.append(result.tokenPiece)
                                currentStep = step
                                continuation.yield(result.tokenPiece)
                                done = result.isEOS
                            } else {
                                done = true
                            }
                        } else {
                            // No next device → this device is sole shard; sample locally.
                            // (Shouldn't reach here since .sole is handled above.)
                            done = true
                        }

                    case .middle:
                        // Receive activation from previous shard, pass through, forward.
                        if let packet = await network.waitForActivation(sessionID: plan.sessionID, step: step) {
                            let (hs, count, embd) = await engine.runShard(
                                hiddenState: packet.hiddenStateData,
                                tokenCount: packet.tokenCount,
                                startLayer: myShard.startLayer,
                                endLayer: myShard.endLayer
                            )
                            let outPacket = ActivationPacket(
                                sessionID: plan.sessionID,
                                step: step,
                                tokenPosition: packet.tokenPosition,
                                tokenCount: count,
                                nEmbd: embd,
                                hiddenState: hs.toFloatArray()
                            )
                            if let nextURL = myShard.nextDeviceEndpoint {
                                try? await network.sendActivation(outPacket, to: nextURL)
                            }
                        }
                        done = (step >= maxNewTokens - 1)

                    case .last:
                        // Receive logits from previous shard, sample a token, return it.
                        if let packet = await network.waitForActivation(sessionID: plan.sessionID, step: step) {
                            let (hs, _, _) = await engine.runShard(
                                hiddenState: packet.hiddenStateData,
                                tokenCount: packet.tokenCount,
                                startLayer: myShard.startLayer,
                                endLayer: myShard.endLayer   // == totalLayers
                            )
                            let tokenID = hs.toFloatArray().first.map { Int32($0) } ?? 0
                            let piece   = engine.tokenToPiece(tokenID)
                            let isEOS   = (tokenID == plan.eosTokenID)

                            let result = TokenResult(
                                sessionID: plan.sessionID,
                                step: step,
                                tokenID: tokenID,
                                tokenPiece: piece,
                                isEOS: isEOS
                            )
                            // Send result back to the coordinator (first shard / prev device).
                            let coordinatorURL = myShard.prevDeviceEndpoint
                                ?? plan.shards.first(where: { $0.role == .first })?.selfEndpoint
                            if let url = coordinatorURL {
                                try? await network.sendTokenResult(result, to: url)
                            }
                        }
                        done = (step >= maxNewTokens - 1)

                    case .sole:
                        break // handled in single-device fast path above
                    }

                    step += 1
                }

                sessionState = .finished(reason: done ? .eos : .maxTokens)
                await network.stopServer()
                continuation.finish()
            }
        }
    }

    func cancel() {
        generationTask?.cancel()
        generationTask = nil
        sessionState = .finished(reason: .cancelled)
        Task { await network.stopServer() }
    }

    // ── Worker mode ───────────────────────────────────────────────────────────

    /// Run this device as a non-coordinator shard worker.
    /// Starts the local HTTP server, waits for the coordinator to POST a
    /// ShardPlan, then runs the appropriate shard loop (middle or last).
    ///
    /// `statusCallback` receives human-readable status strings for the UI.
    func startAsWorker(statusCallback: @escaping (String) -> Void) async {
        sessionState = .planningSession
        await network.startServer()
        statusCallback("Server running – waiting for ShardPlan…")

        // Block until the coordinator sends us our assignment.
        guard let plan = await network.waitForShardPlan() else {
            statusCallback("Timed out waiting for plan.")
            sessionState = .error("No ShardPlan received")
            return
        }

        activePlan = plan

        guard let myShard = plan.shards.first(where: {
            $0.deviceID == UIDeviceLabel.deviceID
        }) else {
            statusCallback("This device is not in the received plan.")
            sessionState = .error("Device not in plan")
            return
        }

        statusCallback("Received plan – running as \(myShard.role.rawValue) shard (layers \(myShard.startLayer)–\(myShard.endLayer - 1))…")
        sessionState = .running(step: 0)

        // Run shard loop: process each decode step as activations arrive.
        var step = 0
        var shardDone = false
        while !Task.isCancelled && !shardDone {
            switch myShard.role {
            case .middle:
                if let packet = await network.waitForActivation(sessionID: plan.sessionID, step: step) {
                    let (hs, count, embd) = await engine.runShard(
                        hiddenState: packet.hiddenStateData,
                        tokenCount: packet.tokenCount,
                        startLayer: myShard.startLayer,
                        endLayer: myShard.endLayer
                    )
                    let outPacket = ActivationPacket(
                        sessionID: plan.sessionID,
                        step: step,
                        tokenPosition: packet.tokenPosition,
                        tokenCount: count,
                        nEmbd: embd,
                        hiddenState: hs.toFloatArray()
                    )
                    if let nextURL = myShard.nextDeviceEndpoint {
                        try? await network.sendActivation(outPacket, to: nextURL)
                    }
                } else {
                    shardDone = true
                }

            case .last:
                if let packet = await network.waitForActivation(sessionID: plan.sessionID, step: step) {
                    let (hs, _, _) = await engine.runShard(
                        hiddenState: packet.hiddenStateData,
                        tokenCount: packet.tokenCount,
                        startLayer: myShard.startLayer,
                        endLayer: myShard.endLayer
                    )
                    let tokenID = hs.toFloatArray().first.map { Int32($0) } ?? 0
                    let piece   = engine.tokenToPiece(tokenID)
                    let isEOS   = (tokenID == plan.eosTokenID)
                    let result  = TokenResult(
                        sessionID: plan.sessionID,
                        step: step,
                        tokenID: tokenID,
                        tokenPiece: piece,
                        isEOS: isEOS
                    )
                    let coordinatorURL = myShard.prevDeviceEndpoint
                        ?? plan.shards.first(where: { $0.role == .first })?.selfEndpoint
                    if let url = coordinatorURL {
                        try? await network.sendTokenResult(result, to: url)
                    }
                    statusCallback("Step \(step): \"\(piece)\"")
                    if isEOS { shardDone = true }
                } else {
                    shardDone = true
                }

            default:
                shardDone = true
            }
            step += 1
            sessionState = .running(step: step)
        }

        await network.stopServer()
        sessionState = .finished(reason: .eos)
        statusCallback("Session complete.")
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

private extension Data {
    func toFloatArray() -> [Float] {
        withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float.self))
        }
    }
}

/// Minimal device identification helpers.
enum UIDeviceLabel {
    static var current: String {
        #if os(iOS)
        return UIDevice.current.name
        #else
        return Host.current().localizedName ?? "Mac"
        #endif
    }
    static var deviceID: String {
        #if os(iOS)
        return UIDevice.current.identifierForVendor?.uuidString ?? UUID().uuidString
        #else
        return ProcessInfo.processInfo.hostName
        #endif
    }
}
