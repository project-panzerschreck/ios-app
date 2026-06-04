import Foundation

enum AppLogger {
    static func log(_ level: String = "INFO", tag: String, _ message: String) {
        AppDiagnostics.log(withLevel: level, tag: tag, message: message)
        NSLog("[%@] %@: %@", level, tag, message)
    }

    static func rpcHealth(status: String, details: [String: Any]) {
        AppDiagnostics.setRPCHealthStatus(status, details: details)
        NSLog("[RPC_HEALTH] status=%@ details=%@", status, String(describing: details))
    }
}
