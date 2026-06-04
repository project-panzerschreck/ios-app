import Testing
import Foundation
@testable import distributed_ml_ggml_client_ios

// MARK: - DistributedCoordinator tests

@MainActor
struct DistributedCoordinatorTests {

    // MARK: Initial state

    @Test func initialSessionStateIsIdle() {
        let coord = DistributedCoordinator()
        #expect(coord.sessionState == .idle)
    }

    @Test func initialGeneratedTokensIsEmpty() {
        let coord = DistributedCoordinator()
        #expect(coord.generatedTokens.isEmpty)
    }

    @Test func initialCurrentStepIsZero() {
        let coord = DistributedCoordinator()
        #expect(coord.currentStep == 0)
    }

    @Test func initialActivePlanIsNil() {
        let coord = DistributedCoordinator()
        #expect(coord.activePlan == nil)
    }

    // MARK: cancel()

    @Test func cancelSetsFinishedCancelled() {
        let coord = DistributedCoordinator()
        coord.cancel()
        #expect(coord.sessionState == .finished(reason: .cancelled))
    }

    @Test func cancelIsIdempotent() {
        let coord = DistributedCoordinator()
        coord.cancel()
        coord.cancel()
        #expect(coord.sessionState == .finished(reason: .cancelled))
    }

    // MARK: buildSolePlan / buildDistributedPlan (guard: no model)

    @Test func buildSolePlanDoesNothingWithNoModel() {
        let coord = DistributedCoordinator()
        coord.buildSolePlan(modelID: "test-model")
        // engine.modelInfo is nil → guard exits, activePlan stays nil
        #expect(coord.activePlan == nil)
    }

    @Test func buildDistributedPlanDoesNothingWithNoModel() {
        let coord = DistributedCoordinator()
        let ep = URL(string: "http://192.168.1.2:58080")!
        coord.buildDistributedPlan(
            modelID: "test-model",
            peerDescriptors: [("Peer", "peer-id", ep)]
        )
        #expect(coord.activePlan == nil)
    }

    // MARK: generate() with no plan

    @Test func generateWithNoPlanFinishesImmediately() async {
        let coord = DistributedCoordinator()
        var collected: [String] = []
        for await token in coord.generate(prompt: "hello") {
            collected.append(token)
        }
        #expect(collected.isEmpty)
    }

    @Test func generateDoesNotMutateStateWhenNoPlan() async {
        let coord = DistributedCoordinator()
        for await _ in coord.generate(prompt: "hello") {}
        // sessionState unchanged since the guard exits without setting it
        // (generationTask is cancelled/finished before it touches sessionState)
        #expect(coord.generatedTokens.isEmpty)
    }

    // MARK: UIDeviceLabel helpers

    @Test func deviceIDIsNonEmpty() {
        #expect(!UIDeviceLabel.deviceID.isEmpty)
    }

    @Test func deviceLabelIsNonEmpty() {
        #expect(!UIDeviceLabel.current.isEmpty)
    }

    @Test func deviceIDIsStable() {
        // Should return the same value on repeated calls within a session.
        #expect(UIDeviceLabel.deviceID == UIDeviceLabel.deviceID)
    }

    // MARK: SessionState / FinishReason (verify via coordinator)

    @Test func planningSessionState() {
        let s = SessionState.planningSession
        #expect(s == .planningSession)
        #expect(s != .idle)
    }

    @Test func waitingForPeersState() {
        let s = SessionState.waitingForPeers
        #expect(s == .waitingForPeers)
        #expect(s != .planningSession)
    }

    @Test func finishReasonNetworkError() {
        #expect(FinishReason.networkError.rawValue == "Network error")
    }

    @Test func allFinishReasonsHaveRawValues() {
        let reasons: [FinishReason] = [.eos, .maxTokens, .cancelled, .networkError]
        for reason in reasons {
            #expect(!reason.rawValue.isEmpty)
        }
    }

    @Test func sessionStateRunningAssociatedValue() {
        let s = SessionState.running(step: 42)
        if case .running(let step) = s {
            #expect(step == 42)
        } else {
            Issue.record("Expected .running")
        }
    }

    @Test func sessionStateErrorAssociatedValue() {
        let s = SessionState.error("boom")
        if case .error(let msg) = s {
            #expect(msg == "boom")
        } else {
            Issue.record("Expected .error")
        }
    }

    @Test func sessionStateFinishedAssociatedValue() {
        let s = SessionState.finished(reason: .maxTokens)
        if case .finished(let reason) = s {
            #expect(reason == .maxTokens)
        } else {
            Issue.record("Expected .finished")
        }
    }
}
