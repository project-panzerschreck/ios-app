// LayerShard.swift
//
// Describes how the transformer layers of a model are distributed across
// the cluster of devices participating in a pipeline-parallel session.
//
// Example – 12-layer GPT-2 split across 3 phones:
//   Shard 0  →  layers  0.. 3  (Device A, "first")
//   Shard 1  →  layers  4.. 7  (Device B, "middle")
//   Shard 2  →  layers  8..11  (Device C, "last")
//
// The ShardPlan is computed once by the session coordinator and distributed
// to every participating device so each device knows:
//   • which layer range to execute
//   • where to send its hidden-state output
//   • where to receive hidden-state input from

import Foundation

// ── Shard role ────────────────────────────────────────────────────────────────

enum ShardRole: String, Codable, Sendable {
    /// First device: embeds tokens and runs early layers.
    case first
    /// Middle device: consumes activations, runs middle layers, forwards result.
    case middle
    /// Last device: runs final layers, produces output tokens, returns them to first.
    case last
    /// Degenerate single-device case: first == last.
    case sole
}

// ── Per-device shard descriptor ───────────────────────────────────────────────

struct LayerShard: Codable, Identifiable, Sendable {
    /// Stable identifier for this shard assignment.
    let id: UUID
    /// Human-readable device label (e.g. "iPhone 12 Pro – Sunny").
    let deviceLabel: String
    /// Unique identifier for the device (e.g. UIDevice.identifierForVendor).
    let deviceID: String
    /// Which role this shard plays in the pipeline.
    let role: ShardRole
    /// First transformer block this shard executes (inclusive, 0-indexed).
    let startLayer: Int
    /// One past the last transformer block this shard executes (exclusive).
    let endLayer: Int
    /// This device's own shard-server endpoint (used by the coordinator to
    /// distribute the ShardPlan and by peers to send activations).
    let selfEndpoint: URL
    /// Network endpoint of the *next* device's shard server.
    /// Nil for the last shard (it returns results to the coordinator).
    let nextDeviceEndpoint: URL?
    /// Network endpoint of the *previous* device's shard server.
    /// Nil for the first shard.
    let prevDeviceEndpoint: URL?

    /// Number of layers handled by this shard.
    var layerCount: Int { endLayer - startLayer }
}

// ── Session-wide plan ─────────────────────────────────────────────────────────

struct ShardPlan: Codable, Sendable {
    /// Unique ID for this inference session.
    let sessionID: UUID
    /// Identifier of the model being run (e.g. "gpt-2-117M-Q4_K_M").
    let modelID: String
    /// Total number of transformer layers in the model.
    let totalLayers: Int
    /// Hidden-state (embedding) dimension of the model.
    let nEmbd: Int
    /// EOS token ID for the model (used for generation termination).
    let eosTokenID: Int32
    /// Ordered list of shards from first to last.
    let shards: [LayerShard]

    // ── Factory ───────────────────────────────────────────────────────────────

    /// Create a balanced shard plan for `deviceEndpoints` devices.
    /// Layers are distributed as evenly as possible; any remainder goes to
    /// the last shard.
    ///
    /// - Parameters:
    ///   - modelID:          identifier string for the model.
    ///   - totalLayers:      total transformer block count.
    ///   - nEmbd:            hidden-state dimension.
    ///   - eosTokenID:       model's end-of-sequence token ID.
    ///   - deviceDescriptors: array of (label, deviceID, endpoint) tuples,
    ///                        ordered first→last in the pipeline.
    static func balanced(
        modelID: String,
        totalLayers: Int,
        nEmbd: Int,
        eosTokenID: Int32 = 2,
        deviceDescriptors: [(label: String, deviceID: String, endpoint: URL)]
    ) -> ShardPlan {
        let n = deviceDescriptors.count
        precondition(n > 0, "Need at least one device")

        let baseCount  = totalLayers / n
        let remainder  = totalLayers % n
        var shards: [LayerShard] = []
        var cursor = 0

        for (i, desc) in deviceDescriptors.enumerated() {
            let count = baseCount + (i < remainder ? 1 : 0)
            let start = cursor
            let end   = cursor + count
            cursor    = end

            let role: ShardRole
            switch (i, n) {
            case (_, 1):                   role = .sole
            case (0, _):                   role = .first
            case (let x, let y) where x == y - 1: role = .last
            default:                       role = .middle
            }

            let nextEndpoint = (i + 1 < n) ? deviceDescriptors[i + 1].endpoint : nil
            let prevEndpoint = (i > 0)      ? deviceDescriptors[i - 1].endpoint : nil

            shards.append(LayerShard(
                id: UUID(),
                deviceLabel: desc.label,
                deviceID: desc.deviceID,
                role: role,
                startLayer: start,
                endLayer: end,
                selfEndpoint: desc.endpoint,
                nextDeviceEndpoint: nextEndpoint,
                prevDeviceEndpoint: prevEndpoint
            ))
        }

        return ShardPlan(
            sessionID: UUID(),
            modelID: modelID,
            totalLayers: totalLayers,
            nEmbd: nEmbd,
            eosTokenID: eosTokenID,
            shards: shards
        )
    }
}
