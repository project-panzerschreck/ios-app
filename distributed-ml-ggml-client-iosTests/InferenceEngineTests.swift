import Testing
import Foundation
@testable import distributed_ml_ggml_client_ios

// MARK: - ChatMessage

struct ChatMessageTests {

    @Test func fieldsAreStoredCorrectly() {
        let msg = ChatMessage(role: "user", content: "hello")
        #expect(msg.role == "user")
        #expect(msg.content == "hello")
    }

    @Test func idIsUnique() {
        let a = ChatMessage(role: "user", content: "x")
        let b = ChatMessage(role: "user", content: "x")
        #expect(a.id != b.id)
    }

    @Test func equalityChecksAllFields() {
        var a = ChatMessage(role: "assistant", content: "hi")
        var b = a
        #expect(a == b)
        b.content = "bye"
        #expect(a != b)
    }

    @Test func contentIsMutable() {
        var msg = ChatMessage(role: "assistant", content: "")
        msg.content += "streamed token"
        #expect(msg.content == "streamed token")
    }
}

// MARK: - GenerationResult

struct GenerationResultTests {

    @Test func fieldsRoundtrip() {
        let r = GenerationResult(text: "hello", isDone: false, tokensPerSecond: 12.5)
        #expect(r.text == "hello")
        #expect(r.isDone == false)
        #expect(r.tokensPerSecond == 12.5)
    }

    @Test func doneResult() {
        let r = GenerationResult(text: "done", isDone: true, tokensPerSecond: 0)
        #expect(r.isDone == true)
    }
}

// MARK: - RPCServerState

struct RPCServerStateTests {

    @Test func idleEquality() {
        #expect(RPCServerState.idle == RPCServerState.idle)
    }

    @Test func startingEquality() {
        #expect(RPCServerState.starting == RPCServerState.starting)
    }

    @Test func runningEquality() {
        #expect(RPCServerState.running(endpoint: "host:50") == RPCServerState.running(endpoint: "host:50"))
        #expect(RPCServerState.running(endpoint: "a") != RPCServerState.running(endpoint: "b"))
    }

    @Test func unavailableEquality() {
        #expect(RPCServerState.unavailable("err") == RPCServerState.unavailable("err"))
        #expect(RPCServerState.unavailable("a") != RPCServerState.unavailable("b"))
    }

    @Test func differentCasesAreNotEqual() {
        #expect(RPCServerState.idle != RPCServerState.starting)
        #expect(RPCServerState.idle != RPCServerState.running(endpoint: "x"))
    }
}

// MARK: - ModelState

struct ModelStateTests {

    @Test func unloadedEquality() {
        #expect(ModelState.unloaded == ModelState.unloaded)
    }

    @Test func loadingEquality() {
        #expect(ModelState.loading == ModelState.loading)
    }

    @Test func readyEquality() {
        #expect(ModelState.ready(modelName: "llama", nLayers: 32) == ModelState.ready(modelName: "llama", nLayers: 32))
        #expect(ModelState.ready(modelName: "a", nLayers: 1) != ModelState.ready(modelName: "b", nLayers: 1))
        #expect(ModelState.ready(modelName: "a", nLayers: 1) != ModelState.ready(modelName: "a", nLayers: 2))
    }

    @Test func generatingEquality() {
        #expect(ModelState.generating == ModelState.generating)
    }

    @Test func errorEquality() {
        #expect(ModelState.error("oops") == ModelState.error("oops"))
        #expect(ModelState.error("a") != ModelState.error("b"))
    }

    @Test func differentCasesAreNotEqual() {
        #expect(ModelState.unloaded != ModelState.loading)
        #expect(ModelState.loading != ModelState.generating)
    }
}

// MARK: - InferenceEngine state machine (no model required)

@MainActor
struct InferenceEngineTests {

    @Test func initialStateIsUnloaded() {
        let engine = InferenceEngine()
        #expect(engine.modelState == .unloaded)
    }

    @Test func initialRPCStateIsIdle() {
        let engine = InferenceEngine()
        #expect(engine.rpcServerState == .idle)
    }

    @Test func initialGeneratedTextIsEmpty() {
        let engine = InferenceEngine()
        #expect(engine.generatedText == "")
    }

    @Test func initialTokensPerSecondIsZero() {
        let engine = InferenceEngine()
        #expect(engine.tokensPerSecond == 0)
    }

    @Test func initialChatMessagesIsEmpty() {
        let engine = InferenceEngine()
        #expect(engine.chatMessages.isEmpty)
    }

    @Test func modelInfoIsNilWithNoModel() {
        let engine = InferenceEngine()
        #expect(engine.modelInfo == nil)
    }

    @Test func eosTokenIDDefaultsToTwoWithNoModel() {
        let engine = InferenceEngine()
        #expect(engine.eosTokenID == 2)
    }

    @Test func rpcAvailableReturnsBool() {
        let engine = InferenceEngine()
        // Just verify it returns without crashing; value depends on build.
        _ = engine.rpcAvailable
    }

    @Test func metalAvailableReturnsBool() {
        let engine = InferenceEngine()
        _ = engine.metalAvailable
    }

    @Test func unloadModelResetsState() {
        let engine = InferenceEngine()
        engine.generatedText = "something"
        engine.tokensPerSecond = 42
        engine.chatMessages = [ChatMessage(role: "user", content: "hi")]
        engine.unloadModel()
        #expect(engine.modelState == .unloaded)
        #expect(engine.generatedText == "")
        #expect(engine.tokensPerSecond == 0)
        #expect(engine.chatMessages.isEmpty)
    }

    @Test func cancelGenerationWhenNotGeneratingIsNoop() {
        let engine = InferenceEngine()
        engine.cancelGeneration()
        #expect(engine.modelState == .unloaded)
    }

    @Test func generateIntoStateGuardsWhenNotReady() {
        let engine = InferenceEngine()
        #expect(engine.modelState == .unloaded)
        engine.generateIntoState(prompt: "test")
        // State should be unchanged — guard exits when not .ready
        #expect(engine.modelState == .unloaded)
    }

    @Test func sendMessageGuardsWhenNotReady() {
        let engine = InferenceEngine()
        engine.sendMessage("hello")
        #expect(engine.modelState == .unloaded)
        #expect(engine.chatMessages.isEmpty)
    }

    @Test func clearChatEmptiesMessages() {
        let engine = InferenceEngine()
        engine.chatMessages = [
            ChatMessage(role: "user", content: "hi"),
            ChatMessage(role: "assistant", content: "hey"),
        ]
        engine.tokensPerSecond = 10
        engine.clearChat()
        #expect(engine.chatMessages.isEmpty)
        #expect(engine.tokensPerSecond == 0)
    }

    @Test func clearChatWhenNoModelDoesNotCrash() {
        let engine = InferenceEngine()
        engine.clearChat()
        #expect(engine.modelState == .unloaded)
    }

    @Test func stopRPCServerSetsIdle() {
        let engine = InferenceEngine()
        engine.stopRPCServer()
        #expect(engine.rpcServerState == .idle)
    }

    @Test func startRPCServerWhenNotIdleIsNoop() {
        let engine = InferenceEngine()
        // Force a non-idle state so the guard triggers.
        engine.rpcServerState = .starting
        engine.startRPCServer(
            coordinatorHost: "192.168.1.1",
            coordinatorPort: 8080,
            nickname: "test",
            threads: 4,
            deviceId: "test-id"
        )
        // Should still be .starting — guard returned early without side effects.
        #expect(engine.rpcServerState == .starting)
        // Clean up
        engine.rpcServerState = .idle
    }

    @Test func tokenizeReturnsEmptyArrayWithNoModel() {
        let engine = InferenceEngine()
        let tokens = engine.tokenize(text: "hello")
        // Bridge returns empty when no model loaded.
        #expect(tokens.isEmpty)
    }

    @Test func tokenToPieceReturnsStringWithNoModel() {
        let engine = InferenceEngine()
        let piece = engine.tokenToPiece(0)
        // Just verify it returns without crashing.
        _ = piece
    }
}
