import Foundation
import Combine

struct RPCHealthSnapshot: Equatable {
    var status: String = "idle"
    var endpoint: String = ""
    var storageEndpoint: String = ""
    var coordinator: String = ""
    var lastError: String = ""
    var lastTransitionAt: String = ""
    var rpcAvailable: Bool = false
    var metalAvailable: Bool = false
    var discoveryActive: Bool = false
    var rpcHealthy: Bool = false
    var storageHealthy: Bool = false
    var announceEligible: Bool = false
}

@MainActor
final class AppDiagnosticsModel: ObservableObject {
    static let shared = AppDiagnosticsModel()

    @Published var logsText: String = AppDiagnostics.logsSnapshot()
    @Published var rpcHealth = RPCHealthSnapshot()

    private var observer: NSObjectProtocol?

    private init() {
        refresh()
        observer = NotificationCenter.default.addObserver(
            forName: .AppDiagnosticsDidUpdate,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor [self] in
                self.refresh()
            }
        }
    }

    deinit {
        if let observer {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    func refresh() {
        logsText = AppDiagnostics.logsSnapshot()
        let raw = AppDiagnostics.rpcHealthSnapshot()
        rpcHealth = RPCHealthSnapshot(
            status: raw["status"] as? String ?? "idle",
            endpoint: raw["endpoint"] as? String ?? "",
            storageEndpoint: raw["storage_endpoint"] as? String ?? "",
            coordinator: raw["coordinator"] as? String ?? "",
            lastError: raw["last_error"] as? String ?? "",
            lastTransitionAt: raw["last_transition_at"] as? String ?? "",
            rpcAvailable: raw["rpc_available"] as? Bool ?? false,
            metalAvailable: raw["metal_available"] as? Bool ?? false,
            discoveryActive: raw["discovery_active"] as? Bool ?? false,
            rpcHealthy: raw["rpc_healthy"] as? Bool ?? false,
            storageHealthy: raw["storage_healthy"] as? Bool ?? false,
            announceEligible: raw["announce_eligible"] as? Bool ?? false
        )
    }
}
