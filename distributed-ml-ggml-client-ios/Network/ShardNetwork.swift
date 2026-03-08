// ShardNetwork.swift
//
// Lightweight HTTP server + client used to exchange activation tensors and
// token results between shard devices in a pipeline-parallel session.
//
// Server endpoints (per-device):
//   POST /activation        – receive an ActivationPacket from the previous shard
//   POST /token-result      – receive a TokenResult from the last shard
//   POST /session           – receive a ShardPlan from the coordinator
//   GET  /health            – liveness probe for peer discovery
//
// The server runs on a configurable port (default 58080) using Network.framework.
// All bodies are JSON-encoded.  For production use, add TLS and authentication.

import Foundation
import Network
import Darwin   // getifaddrs / getnameinfo

@MainActor
final class ShardNetwork {

    static let defaultPort: UInt16 = 58080

    // ── Queues for received messages ──────────────────────────────────────────
    private var activationQueue: [ActivationPacket] = []
    private var tokenResultQueue: [TokenResult]     = []
    private var activationWaiters: [CheckedContinuation<ActivationPacket?, Never>] = []
    private var tokenResultWaiters: [CheckedContinuation<TokenResult?, Never>]     = []

    // ── ShardPlan distribution ────────────────────────────────────────────────
    /// Plan received from the coordinator via POST /session.
    private(set) var receivedPlan: ShardPlan?
    private var sessionPlanWaiters: [CheckedContinuation<ShardPlan?, Never>] = []

    // ── Network.framework listener ────────────────────────────────────────────
    private var listener: NWListener?
    private var connections: [NWConnection] = []

    // ── Server ────────────────────────────────────────────────────────────────

    func startServer() async {
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true

        guard let listener = try? NWListener(using: params,
                                             on: NWEndpoint.Port(rawValue: ShardNetwork.defaultPort)!) else {
            return
        }
        self.listener = listener

        listener.newConnectionHandler = { [weak self] connection in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.connections.append(connection)
                connection.start(queue: .main)
                self.handleConnection(connection)
            }
        }

        listener.start(queue: .main)
    }

    func stopServer() async {
        connections.forEach { $0.cancel() }
        connections.removeAll()
        listener?.cancel()
        listener = nil

        // Drain waiters with nil
        activationWaiters.forEach   { $0.resume(returning: nil) }
        activationWaiters.removeAll()
        tokenResultWaiters.forEach  { $0.resume(returning: nil) }
        tokenResultWaiters.removeAll()
        sessionPlanWaiters.forEach  { $0.resume(returning: nil) }
        sessionPlanWaiters.removeAll()
        receivedPlan = nil
    }

    // ── Receive side ──────────────────────────────────────────────────────────

    /// Block until an ActivationPacket arrives for the given session + step.
    func waitForActivation(sessionID: UUID, step: Int) async -> ActivationPacket? {
        // Check queue first
        if let idx = activationQueue.firstIndex(where: {
            $0.sessionID == sessionID && $0.step == step
        }) {
            return activationQueue.remove(at: idx)
        }
        // Otherwise suspend until network delivers it
        return await withCheckedContinuation { cont in
            activationWaiters.append(cont)
        }
    }

    /// Block until a TokenResult arrives for the given session.
    func waitForTokenResult(sessionID: UUID) async -> TokenResult? {
        if let idx = tokenResultQueue.firstIndex(where: { $0.sessionID == sessionID }) {
            return tokenResultQueue.remove(at: idx)
        }
        return await withCheckedContinuation { cont in
            tokenResultWaiters.append(cont)
        }
    }

    // ── Send side ─────────────────────────────────────────────────────────────

    func sendActivation(_ packet: ActivationPacket, to url: URL) async throws {
        let envelope = ActivationEnvelope(packet: packet,
                                          senderDeviceID: UIDeviceLabel.deviceID)
        try await post(envelope, to: url.appendingPathComponent("activation"))
    }

    func sendTokenResult(_ result: TokenResult, to url: URL) async throws {
        try await post(result, to: url.appendingPathComponent("token-result"))
    }

    /// Send the ShardPlan to a peer device so it knows its layer assignment.
    func sendShardPlan(_ plan: ShardPlan, to url: URL) async throws {
        try await post(plan, to: url.appendingPathComponent("session"))
    }

    /// Wait for a ShardPlan to arrive on this device via POST /session.
    /// Used by non-coordinator devices to receive their assignment.
    func waitForShardPlan() async -> ShardPlan? {
        if let plan = receivedPlan { return plan }
        return await withCheckedContinuation { cont in
            sessionPlanWaiters.append(cont)
        }
    }

    // ── Peer discovery ────────────────────────────────────────────────────────

    /// Check whether a peer at `endpoint` is reachable and ready.
    func probePeer(_ endpoint: URL) async -> Bool {
        let healthURL = endpoint.appendingPathComponent("health")
        var req = URLRequest(url: healthURL)
        req.timeoutInterval = 3
        do {
            let (_, resp) = try await URLSession.shared.data(for: req)
            return (resp as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private func post<T: Encodable>(_ body: T, to url: URL) async throws {
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONEncoder().encode(body)
        req.timeoutInterval = 10
        let (_, _) = try await URLSession.shared.data(for: req)
    }

    private func handleConnection(_ connection: NWConnection) {
        receiveHTTPRequest(from: connection)
    }

    /// Minimal HTTP/1.1 request parser – reads headers + body and routes to handler.
    private func receiveHTTPRequest(from connection: NWConnection) {
        connection.receive(minimumIncompleteLength: 1,
                           maximumLength: 4 * 1024 * 1024) { [weak self] data, _, isComplete, _ in
            guard let self, let data, !data.isEmpty else { return }

            // Parse the raw HTTP request (pure value work – safe in nonisolated context)
            guard let rawString = String(data: data, encoding: .utf8) else { return }
            let parts = rawString.components(separatedBy: "\r\n\r\n")
            let headerBlock = parts[0]
            let bodyString  = parts.count > 1 ? parts[1...].joined(separator: "\r\n\r\n") : ""
            let bodyData    = bodyString.data(using: .utf8) ?? Data()

            // Extract request line
            let firstLine = headerBlock.components(separatedBy: "\r\n").first ?? ""
            let requestParts = firstLine.components(separatedBy: " ")
            guard requestParts.count >= 2 else { return }
            let method = requestParts[0]
            let path   = requestParts[1]

            // Route and optionally re-arm on the MainActor (ShardNetwork is @MainActor).
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.routeRequest(method: method, path: path, body: bodyData,
                                  connection: connection)
                if !isComplete {
                    self.receiveHTTPRequest(from: connection)
                }
            }
        }
    }

    private func routeRequest(method: String, path: String,
                               body: Data, connection: NWConnection) {
        switch (method, path) {
        case ("POST", "/activation"):
            if let envelope = try? JSONDecoder().decode(ActivationEnvelope.self, from: body) {
                deliverActivation(envelope.packet)
            }
            sendHTTPResponse(status: 200, body: Data(), to: connection)

        case ("POST", "/token-result"):
            if let result = try? JSONDecoder().decode(TokenResult.self, from: body) {
                deliverTokenResult(result)
            }
            sendHTTPResponse(status: 200, body: Data(), to: connection)

        case ("POST", "/session"):
            if let plan = try? JSONDecoder().decode(ShardPlan.self, from: body) {
                deliverShardPlan(plan)
            }
            sendHTTPResponse(status: 200, body: Data(), to: connection)

        case ("GET", "/health"):
            let ok = "{\"status\":\"ok\"}".data(using: .utf8)!
            sendHTTPResponse(status: 200, body: ok, to: connection)

        default:
            sendHTTPResponse(status: 404, body: Data(), to: connection)
        }
    }

    private func deliverActivation(_ packet: ActivationPacket) {
        if let cont = activationWaiters.first {
            activationWaiters.removeFirst()
            cont.resume(returning: packet)
        } else {
            activationQueue.append(packet)
        }
    }

    private func deliverTokenResult(_ result: TokenResult) {
        if let cont = tokenResultWaiters.first {
            tokenResultWaiters.removeFirst()
            cont.resume(returning: result)
        } else {
            tokenResultQueue.append(result)
        }
    }

    private func deliverShardPlan(_ plan: ShardPlan) {
        receivedPlan = plan
        sessionPlanWaiters.forEach { $0.resume(returning: plan) }
        sessionPlanWaiters.removeAll()
    }

    private func sendHTTPResponse(status: Int, body: Data, to connection: NWConnection) {
        let header = "HTTP/1.1 \(status) \(status == 200 ? "OK" : "Error")\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n"
        var response = header.data(using: .utf8)!
        response.append(body)
        connection.send(content: response, completion: .contentProcessed { _ in })
    }
}

// ── Local network helpers ─────────────────────────────────────────────────────

/// A single network interface with a human-readable label and IPv4 address.
struct LocalInterface: Identifiable {
    let id: String      // interface name, e.g. "en0"
    let label: String   // e.g. "Wi-Fi", "Tailscale", "Cellular"
    let ip: String
}

extension ShardNetwork {

    /// All non-loopback IPv4 addresses on this device, sorted by priority.
    /// Includes Wi-Fi, Hotspot, Cellular, Tailscale/VPN, and any other active interfaces.
    static var allLocalIPv4s: [LocalInterface] {
        var ifAddr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifAddr) == 0, let first = ifAddr else { return [] }
        defer { freeifaddrs(first) }

        var results: [LocalInterface] = []
        var seen = Set<String>()

        var node = first
        while true {
            let ifa = node.pointee
            if ifa.ifa_addr.pointee.sa_family == UInt8(AF_INET) {
                let name = String(cString: ifa.ifa_name)
                // Skip loopback
                guard name != "lo0", !seen.contains(name) else {
                    if let next = ifa.ifa_next { node = next; continue } else { break }
                }
                var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                getnameinfo(ifa.ifa_addr,
                            socklen_t(ifa.ifa_addr.pointee.sa_len),
                            &host, socklen_t(host.count),
                            nil, 0, NI_NUMERICHOST)
                let ip = String(cString: host)
                seen.insert(name)
                results.append(LocalInterface(id: name, label: Self.label(for: name, ip: ip), ip: ip))
            }
            guard let next = ifa.ifa_next else { break }
            node = next
        }

        // Sort: Wi-Fi first, then hotspot, then Tailscale/VPN, then cellular, then rest
        let order = ["en0", "bridge100", "utun", "pdp_ip0"]
        return results.sorted { a, b in
            let ai = order.firstIndex(where: { a.id.hasPrefix($0) }) ?? order.count
            let bi = order.firstIndex(where: { b.id.hasPrefix($0) }) ?? order.count
            return ai == bi ? a.id < b.id : ai < bi
        }
    }

    /// The primary IPv4 (first in the sorted list), for backwards-compatible use.
    static var wifiIPv4: String? { allLocalIPv4s.first?.ip }

    private static func label(for interface: String, ip: String) -> String {
        switch true {
        case interface == "en0":              return "Wi-Fi"
        case interface == "bridge100":        return "Hotspot"
        case interface.hasPrefix("pdp_ip"):   return "Cellular"
        case interface.hasPrefix("utun"):
            // Tailscale uses the 100.64.0.0/10 range (100.64.x.x – 100.127.x.x)
            if isTailscaleIP(ip) { return "Tailscale" }
            return "VPN"
        case interface.hasPrefix("en"):       return "Ethernet"
        default:                              return interface
        }
    }

    private static func isTailscaleIP(_ ip: String) -> Bool {
        let parts = ip.split(separator: ".").compactMap { UInt8($0) }
        guard parts.count == 4, parts[0] == 100 else { return false }
        return parts[1] >= 64 && parts[1] <= 127
    }
}
