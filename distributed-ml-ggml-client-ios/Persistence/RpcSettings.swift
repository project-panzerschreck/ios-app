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
    }

    // ── Storage ──────────────────────────────────────────────────────────────
    var storageDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("StorageApp", isDirectory: true)
    }
}
