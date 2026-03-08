// ActivationPacket.swift
//
// Wire format for inter-device hidden-state transfer in pipeline-parallel inference.
//
// One packet is sent from shard N to shard N+1 for each decode step.
// The hidden-state payload is a flat array of IEEE-754 float32 values with
// shape [tokenCount × nEmbd], stored row-major.
//
// For GPT-2 117M (nEmbd = 768) at sequence length 1:
//   payload = 1 × 768 × 4 bytes = 3 072 bytes  ≈ 3 KB per step
//
// This is small enough for reliable transfer over local WiFi or even BLE.

import Foundation

// ── Packet ────────────────────────────────────────────────────────────────────

struct ActivationPacket: Codable, Sendable {
    /// Session this packet belongs to.
    let sessionID: UUID
    /// Monotonically increasing step counter (0 = first autoregressive step).
    let step: Int
    /// Token position in the sequence (0-indexed).
    let tokenPosition: Int
    /// Number of tokens encoded in `hiddenStateData` (almost always 1 during generation).
    let tokenCount: Int
    /// Embedding dimension of the model.
    let nEmbd: Int
    /// Raw float32 hidden-state tensor: `tokenCount × nEmbd` floats, row-major.
    let hiddenStateData: Data
    /// True if the previous shard has signalled end-of-sequence.
    let isDone: Bool
    /// When `isDone`, this carries the final sampled token ID (as Int32).
    let finalToken: Int32?

    // ── Convenience ───────────────────────────────────────────────────────────

    /// Create a packet from a raw float buffer.
    init(sessionID: UUID,
         step: Int,
         tokenPosition: Int,
         tokenCount: Int,
         nEmbd: Int,
         hiddenState: [Float],
         isDone: Bool = false,
         finalToken: Int32? = nil) {
        self.sessionID      = sessionID
        self.step           = step
        self.tokenPosition  = tokenPosition
        self.tokenCount     = tokenCount
        self.nEmbd          = nEmbd
        self.hiddenStateData = hiddenState.withUnsafeBytes { Data($0) }
        self.isDone         = isDone
        self.finalToken     = finalToken
    }

    /// Extract the hidden-state tensor as a Float array.
    func hiddenStateFloats() -> [Float] {
        hiddenStateData.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float.self))
        }
    }

    /// Validate that the payload size matches the declared dimensions.
    var isValid: Bool {
        hiddenStateData.count == tokenCount * nEmbd * MemoryLayout<Float>.size
    }
}

// ── Transport envelope ────────────────────────────────────────────────────────

/// HTTP/WebSocket envelope wrapping an ActivationPacket.
/// The shard server deserialises this from the request body.
struct ActivationEnvelope: Codable, Sendable {
    let packet: ActivationPacket
    /// Optional metadata for debugging / tracing.
    let senderDeviceID: String
    let timestamp: Date

    init(packet: ActivationPacket, senderDeviceID: String) {
        self.packet         = packet
        self.senderDeviceID = senderDeviceID
        self.timestamp      = Date()
    }
}

// ── Token result (last shard → coordinator) ───────────────────────────────────

/// Sent by the last shard back to the coordinator (first shard or orchestrator).
struct TokenResult: Codable, Sendable {
    let sessionID: UUID
    let step: Int
    /// Sampled token ID.
    let tokenID: Int32
    /// Decoded text piece (UTF-8 sub-word fragment).
    let tokenPiece: String
    /// True if the model emitted an EOS token.
    let isEOS: Bool
}
