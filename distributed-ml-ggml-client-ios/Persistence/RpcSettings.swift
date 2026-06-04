import Foundation
import Combine

/// Centralized persistence for all app settings.
final class RpcSettings: ObservableObject {
    static let shared = RpcSettings()
    static let listenHost = "0.0.0.0"
    static let listenPort = 47651
    static let storagePort = 47672

    enum Keys {
        static let nickname = "rpcNickname"
        static let threads = "rpcThreads"
        static let deviceId = "rpcDeviceId"
        static let clusterServerHost = "clusterServerHost"
        static let clusterServerPort = "clusterServerPort"
        static let clusterToken = "clusterToken"
        static let verboseRPCLogging = "verboseRPCLogging"
    }

    static let defaultClusterServerPort = 4917

    static func loadClusterServerHost() -> String {
        UserDefaults.standard.string(forKey: Keys.clusterServerHost) ?? ""
    }

    static func loadClusterServerPort() -> Int {
        let stored = UserDefaults.standard.integer(forKey: Keys.clusterServerPort)
        return stored == 0 ? defaultClusterServerPort : stored
    }

    static func loadClusterToken() -> String {
        UserDefaults.standard.string(forKey: Keys.clusterToken) ?? ""
    }

    static func saveClusterConnection(host: String, port: Int, token: String) {
        UserDefaults.standard.set(host, forKey: Keys.clusterServerHost)
        UserDefaults.standard.set(port, forKey: Keys.clusterServerPort)
        UserDefaults.standard.set(token, forKey: Keys.clusterToken)
    }

    // ── Persistence ──────────────────────────────────────────────────────────

    @Published var nickname: String {
        didSet { UserDefaults.standard.set(nickname, forKey: Keys.nickname) }
    }
    @Published var threads: Int {
        didSet { UserDefaults.standard.set(threads, forKey: Keys.threads) }
    }
    @Published var deviceId: String {
        didSet { UserDefaults.standard.set(deviceId, forKey: Keys.deviceId) }
    }
    @Published var verboseRPCLogging: Bool {
        didSet {
            UserDefaults.standard.set(verboseRPCLogging, forKey: Keys.verboseRPCLogging)
            LlamaBridge.configureRPCLoggingVerbose(verboseRPCLogging)
        }
    }

    private init() {
        self.nickname = UserDefaults.standard.string(forKey: Keys.nickname) ?? ""

        let t = UserDefaults.standard.integer(forKey: Keys.threads)
        self.threads = (t == 0) ? 4 : t

        if let existing = UserDefaults.standard.string(forKey: Keys.deviceId), !existing.isEmpty {
            self.deviceId = existing
        } else {
            let newId = UUID().uuidString
            UserDefaults.standard.set(newId, forKey: Keys.deviceId)
            self.deviceId = newId
        }

        #if VERBOSE_RPC_DEFAULT
        if UserDefaults.standard.object(forKey: Keys.verboseRPCLogging) == nil {
            self.verboseRPCLogging = true
        } else {
            self.verboseRPCLogging = UserDefaults.standard.bool(forKey: Keys.verboseRPCLogging)
        }
        #elseif DEBUG
        if UserDefaults.standard.object(forKey: Keys.verboseRPCLogging) == nil {
            self.verboseRPCLogging = true
        } else {
            self.verboseRPCLogging = UserDefaults.standard.bool(forKey: Keys.verboseRPCLogging)
        }
        #else
        self.verboseRPCLogging = UserDefaults.standard.bool(forKey: Keys.verboseRPCLogging)
        #endif
        LlamaBridge.configureRPCLoggingVerbose(verboseRPCLogging)
    }

    // ── Storage ──────────────────────────────────────────────────────────────
    var storageDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("StorageApp", isDirectory: true)
    }
}
