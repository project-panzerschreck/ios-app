//
//  distributed_ml_ggml_client_iosTests.swift
//  distributed-ml-ggml-client-iosTests

import Testing
import Foundation
@testable import distributed_ml_ggml_client_ios

// MARK: - ShardPlan.balanced

struct ShardPlanTests {

    private func makeEndpoint(_ i: Int) -> URL {
        URL(string: "http://192.168.1.\(i):58080")!
    }

    @Test func soleDeviceGetsSoleRole() {
        let plan = ShardPlan.balanced(
            modelID: "gpt2",
            totalLayers: 12,
            nEmbd: 768,
            eosTokenID: 50256,
            deviceDescriptors: [("Phone A", "dev-a", makeEndpoint(1))]
        )
        #expect(plan.shards.count == 1)
        #expect(plan.shards[0].role == .sole)
        #expect(plan.shards[0].startLayer == 0)
        #expect(plan.shards[0].endLayer == 12)
    }

    @Test func twoDevicesGetFirstAndLastRoles() {
        let plan = ShardPlan.balanced(
            modelID: "gpt2",
            totalLayers: 12,
            nEmbd: 768,
            deviceDescriptors: [
                ("Phone A", "dev-a", makeEndpoint(1)),
                ("Phone B", "dev-b", makeEndpoint(2))
            ]
        )
        #expect(plan.shards.count == 2)
        #expect(plan.shards[0].role == .first)
        #expect(plan.shards[1].role == .last)
    }

    @Test func threeDevicesHaveMiddleShard() {
        let plan = ShardPlan.balanced(
            modelID: "gpt2",
            totalLayers: 12,
            nEmbd: 768,
            deviceDescriptors: [
                ("Phone A", "dev-a", makeEndpoint(1)),
                ("Phone B", "dev-b", makeEndpoint(2)),
                ("Phone C", "dev-c", makeEndpoint(3))
            ]
        )
        #expect(plan.shards[0].role == .first)
        #expect(plan.shards[1].role == .middle)
        #expect(plan.shards[2].role == .last)
    }

    @Test func layersAreBalancedEvenly() {
        let plan = ShardPlan.balanced(
            modelID: "gpt2",
            totalLayers: 12,
            nEmbd: 768,
            deviceDescriptors: [
                ("A", "a", makeEndpoint(1)),
                ("B", "b", makeEndpoint(2)),
                ("C", "c", makeEndpoint(3))
            ]
        )
        #expect(plan.shards[0].layerCount == 4)
        #expect(plan.shards[1].layerCount == 4)
        #expect(plan.shards[2].layerCount == 4)
    }

    @Test func remainderGoesToEarlyShards() {
        // 13 layers / 3 devices → 4, 4, 5
        let plan = ShardPlan.balanced(
            modelID: "gpt2",
            totalLayers: 13,
            nEmbd: 768,
            deviceDescriptors: [
                ("A", "a", makeEndpoint(1)),
                ("B", "b", makeEndpoint(2)),
                ("C", "c", makeEndpoint(3))
            ]
        )
        #expect(plan.shards[0].layerCount == 5)
        #expect(plan.shards[1].layerCount == 4)
        #expect(plan.shards[2].layerCount == 4)
    }

    @Test func layerRangesAreContinuousAndCoverAll() {
        let total = 20
        let plan = ShardPlan.balanced(
            modelID: "llama",
            totalLayers: total,
            nEmbd: 4096,
            deviceDescriptors: [
                ("A", "a", makeEndpoint(1)),
                ("B", "b", makeEndpoint(2)),
                ("C", "c", makeEndpoint(3)),
                ("D", "d", makeEndpoint(4))
            ]
        )
        var cursor = 0
        for shard in plan.shards {
            #expect(shard.startLayer == cursor)
            cursor = shard.endLayer
        }
        #expect(cursor == total)
    }

    @Test func nextAndPrevEndpointsAreWiredCorrectly() {
        let ep1 = makeEndpoint(1)
        let ep2 = makeEndpoint(2)
        let ep3 = makeEndpoint(3)
        let plan = ShardPlan.balanced(
            modelID: "gpt2",
            totalLayers: 9,
            nEmbd: 768,
            deviceDescriptors: [("A", "a", ep1), ("B", "b", ep2), ("C", "c", ep3)]
        )
        // First shard points forward, not backward.
        #expect(plan.shards[0].nextDeviceEndpoint == ep2)
        #expect(plan.shards[0].prevDeviceEndpoint == nil)
        // Middle shard points in both directions.
        #expect(plan.shards[1].nextDeviceEndpoint == ep3)
        #expect(plan.shards[1].prevDeviceEndpoint == ep1)
        // Last shard has no next pointer.
        #expect(plan.shards[2].nextDeviceEndpoint == nil)
        #expect(plan.shards[2].prevDeviceEndpoint == ep2)
    }

    @Test func planCarriesCorrectModelMetadata() {
        let plan = ShardPlan.balanced(
            modelID: "tinyllama",
            totalLayers: 22,
            nEmbd: 2048,
            eosTokenID: 1,
            deviceDescriptors: [("A", "a", makeEndpoint(1))]
        )
        #expect(plan.modelID == "tinyllama")
        #expect(plan.totalLayers == 22)
        #expect(plan.nEmbd == 2048)
        #expect(plan.eosTokenID == 1)
    }

    @Test func soleShardHasNilNeighbors() {
        let plan = ShardPlan.balanced(
            modelID: "gpt2",
            totalLayers: 12,
            nEmbd: 768,
            deviceDescriptors: [("A", "a", makeEndpoint(1))]
        )
        #expect(plan.shards[0].nextDeviceEndpoint == nil)
        #expect(plan.shards[0].prevDeviceEndpoint == nil)
    }
}

// MARK: - ActivationPacket

struct ActivationPacketTests {

    private let sessionID = UUID()

    private func makePacket(tokenCount: Int, nEmbd: Int) -> ActivationPacket {
        let floats = (0..<(tokenCount * nEmbd)).map { Float($0) }
        return ActivationPacket(
            sessionID: sessionID,
            step: 0,
            tokenPosition: 0,
            tokenCount: tokenCount,
            nEmbd: nEmbd,
            hiddenState: floats
        )
    }

    @Test func validPacketPassesValidation() {
        let pkt = makePacket(tokenCount: 1, nEmbd: 768)
        #expect(pkt.isValid)
    }

    @Test func hiddenStateRoundtrips() {
        let floats: [Float] = [1.0, -2.5, 0.0, 3.14]
        let pkt = ActivationPacket(
            sessionID: UUID(),
            step: 0,
            tokenPosition: 0,
            tokenCount: 1,
            nEmbd: 4,
            hiddenState: floats
        )
        let recovered = pkt.hiddenStateFloats()
        #expect(recovered.count == floats.count)
        for (a, b) in zip(recovered, floats) {
            #expect(a == b)
        }
    }

    @Test func largerPayloadIsValid() {
        let pkt = makePacket(tokenCount: 4, nEmbd: 256)
        #expect(pkt.isValid)
        #expect(pkt.hiddenStateData.count == 4 * 256 * 4)
    }

    @Test func isDoneDefaultsFalse() {
        let pkt = makePacket(tokenCount: 1, nEmbd: 8)
        #expect(pkt.isDone == false)
        #expect(pkt.finalToken == nil)
    }

    @Test func donePacketCarriesFinalToken() {
        let floats = [Float](repeating: 0, count: 8)
        let pkt = ActivationPacket(
            sessionID: UUID(),
            step: 5,
            tokenPosition: 5,
            tokenCount: 1,
            nEmbd: 8,
            hiddenState: floats,
            isDone: true,
            finalToken: 50256
        )
        #expect(pkt.isDone == true)
        #expect(pkt.finalToken == 50256)
    }

    @Test func packetIsJsonCodable() throws {
        let pkt = makePacket(tokenCount: 1, nEmbd: 4)
        let encoder = JSONEncoder()
        let data = try encoder.encode(pkt)
        let decoded = try JSONDecoder().decode(ActivationPacket.self, from: data)
        #expect(decoded.sessionID == pkt.sessionID)
        #expect(decoded.step == pkt.step)
        #expect(decoded.nEmbd == pkt.nEmbd)
        #expect(decoded.isValid)
    }
}

// MARK: - LayerShard

struct LayerShardTests {

    @Test func layerCountComputed() {
        let shard = LayerShard(
            id: UUID(),
            deviceLabel: "iPhone",
            deviceID: "dev-1",
            role: .sole,
            startLayer: 3,
            endLayer: 7,
            selfEndpoint: URL(string: "http://localhost:58080")!,
            nextDeviceEndpoint: nil,
            prevDeviceEndpoint: nil
        )
        #expect(shard.layerCount == 4)
    }

    @Test func shardIsCodable() throws {
        let shard = LayerShard(
            id: UUID(),
            deviceLabel: "iPad",
            deviceID: "dev-2",
            role: .middle,
            startLayer: 4,
            endLayer: 8,
            selfEndpoint: URL(string: "http://192.168.1.2:58080")!,
            nextDeviceEndpoint: URL(string: "http://192.168.1.3:58080"),
            prevDeviceEndpoint: URL(string: "http://192.168.1.1:58080")
        )
        let data    = try JSONEncoder().encode(shard)
        let decoded = try JSONDecoder().decode(LayerShard.self, from: data)
        #expect(decoded.id == shard.id)
        #expect(decoded.role == shard.role)
        #expect(decoded.startLayer == shard.startLayer)
        #expect(decoded.endLayer == shard.endLayer)
    }
}

// MARK: - TokenResult

struct TokenResultTests {

    @Test func tokenResultIsCodable() throws {
        let result = TokenResult(
            sessionID: UUID(),
            step: 3,
            tokenID: 1234,
            tokenPiece: " hello",
            isEOS: false
        )
        let data    = try JSONEncoder().encode(result)
        let decoded = try JSONDecoder().decode(TokenResult.self, from: data)
        #expect(decoded.tokenID == result.tokenID)
        #expect(decoded.tokenPiece == result.tokenPiece)
        #expect(decoded.isEOS == false)
    }

    @Test func eosTokenResultFlagsCorrectly() throws {
        let result = TokenResult(
            sessionID: UUID(),
            step: 10,
            tokenID: 50256,
            tokenPiece: "<|endoftext|>",
            isEOS: true
        )
        #expect(result.isEOS == true)
    }
}

// MARK: - SessionState & FinishReason

struct SessionStateTests {

    @Test func idleStateEquality() {
        #expect(SessionState.idle == SessionState.idle)
    }

    @Test func runningStateEquality() {
        #expect(SessionState.running(step: 5) == SessionState.running(step: 5))
        #expect(SessionState.running(step: 5) != SessionState.running(step: 6))
    }

    @Test func finishedStateEquality() {
        #expect(SessionState.finished(reason: .eos) == SessionState.finished(reason: .eos))
        #expect(SessionState.finished(reason: .eos) != SessionState.finished(reason: .maxTokens))
    }

    @Test func errorStateEquality() {
        #expect(SessionState.error("oops") == SessionState.error("oops"))
        #expect(SessionState.error("a") != SessionState.error("b"))
    }

    @Test func finishReasonRawValues() {
        #expect(FinishReason.eos.rawValue == "End of sequence")
        #expect(FinishReason.maxTokens.rawValue == "Max tokens reached")
        #expect(FinishReason.cancelled.rawValue == "Cancelled")
        #expect(FinishReason.networkError.rawValue == "Network error")
    }
}

// MARK: - ActivationEnvelope

struct ActivationEnvelopeTests {

    @Test func envelopeWrapsPacket() throws {
        let pkt = ActivationPacket(
            sessionID: UUID(),
            step: 0,
            tokenPosition: 0,
            tokenCount: 1,
            nEmbd: 4,
            hiddenState: [1.0, 2.0, 3.0, 4.0]
        )
        let envelope = ActivationEnvelope(packet: pkt, senderDeviceID: "dev-x")
        #expect(envelope.senderDeviceID == "dev-x")
        #expect(envelope.packet.sessionID == pkt.sessionID)

        let data    = try JSONEncoder().encode(envelope)
        let decoded = try JSONDecoder().decode(ActivationEnvelope.self, from: data)
        #expect(decoded.senderDeviceID == "dev-x")
        #expect(decoded.packet.nEmbd == 4)
    }
}
