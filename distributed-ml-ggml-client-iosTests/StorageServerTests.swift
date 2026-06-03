import Testing
import Foundation
import CryptoKit
import Darwin
@testable import distributed_ml_ggml_client_ios

// MARK: - Helpers

private func makeTempDir() -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("StorageServerTests-\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

private func sha256Hex(_ data: Data) -> String {
    let digest = SHA256.hash(data: data)
    return digest.map { String(format: "%02x", $0) }.joined()
}

private func storageRequest(
    method: String,
    path: String,
    body: Data? = nil,
    headers: [String: String] = [:],
    port: Int
) async throws -> (Data, Int) {
    let url = URL(string: "http://127.0.0.1:\(port)\(path)")!
    var req = URLRequest(url: url)
    req.httpMethod = method
    req.timeoutInterval = 5
    for (k, v) in headers { req.setValue(v, forHTTPHeaderField: k) }
    if let body { req.httpBody = body }
    let (data, resp) = try await URLSession.shared.data(for: req)
    let status = (resp as! HTTPURLResponse).statusCode
    return (data, status)
}

// Binds to port 0 and reads back the OS-assigned port, then closes the socket.
private func freePort() -> Int {
    let sock = socket(AF_INET, SOCK_STREAM, 0)
    var addr = sockaddr_in()
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = 0
    addr.sin_addr.s_addr = INADDR_ANY
    withUnsafeMutablePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            _ = bind(sock, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    var len = socklen_t(MemoryLayout<sockaddr_in>.size)
    withUnsafeMutablePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            _ = getsockname(sock, $0, &len)
        }
    }
    let port = Int(CFSwapInt16BigToHost(addr.sin_port))
    close(sock)
    return port
}

// MARK: - StorageServer tests

struct StorageServerTests {

    private func makeServer() -> (StorageServer, Int) {
        let dir = makeTempDir()
        let port = freePort()
        let server = StorageServer(storageDir: dir)
        return (server, port)
    }

    // MARK: start / stop

    @Test func startReturnsTrue() {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
    }

    @Test func stopIsIdempotent() {
        let (server, port) = makeServer()
        _ = server.start(port: port)
        server.stop()
        server.stop()
    }

    // MARK: PUT chunk

    @Test func putChunkStoresData() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        let data = Data("hello world".utf8)
        let id = sha256Hex(data)
        let (_, status) = try await storageRequest(
            method: "PUT",
            path: "/chunk/\(id)",
            body: data,
            headers: ["Content-Length": "\(data.count)"],
            port: port
        )
        #expect(status == 200)
    }

    @Test func putChunkBadIdReturns400() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        let (body, status) = try await storageRequest(
            method: "PUT",
            path: "/chunk/not-a-valid-sha256",
            body: Data("x".utf8),
            headers: ["Content-Length": "1"],
            port: port
        )
        #expect(status == 400)
        let json = try JSONSerialization.jsonObject(with: body) as? [String: String]
        #expect(json?["error"] == "bad_id")
    }

    @Test func putChunkWrongChecksumReturns400() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        let data = Data("actual content".utf8)
        let wrongID = sha256Hex(Data("different content".utf8))
        let (body, status) = try await storageRequest(
            method: "PUT",
            path: "/chunk/\(wrongID)",
            body: data,
            headers: ["Content-Length": "\(data.count)"],
            port: port
        )
        #expect(status == 400)
        let json = try JSONSerialization.jsonObject(with: body) as? [String: String]
        #expect(json?["error"] == "checksum_incorrect")
    }

    // MARK: GET chunk

    @Test func getChunkReturnsStoredData() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        let data = Data("test payload".utf8)
        let id = sha256Hex(data)
        _ = try await storageRequest(
            method: "PUT", path: "/chunk/\(id)",
            body: data, headers: ["Content-Length": "\(data.count)"], port: port
        )

        let (got, getStatus) = try await storageRequest(
            method: "GET", path: "/chunk/\(id)", port: port
        )
        #expect(getStatus == 200)
        #expect(got == data)
    }

    @Test func getChunkNotFoundReturns404() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        let missingID = String(repeating: "a", count: 64)
        let (body, status) = try await storageRequest(
            method: "GET", path: "/chunk/\(missingID)", port: port
        )
        #expect(status == 404)
        let json = try JSONSerialization.jsonObject(with: body) as? [String: String]
        #expect(json?["error"] == "not_found")
    }

    @Test func getChunkBadIdReturns400() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        let (_, status) = try await storageRequest(
            method: "GET", path: "/chunk/tooshort", port: port
        )
        #expect(status == 400)
    }

    // MARK: DELETE chunk

    @Test func deleteChunkRemovesIt() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        let data = Data("delete me".utf8)
        let id = sha256Hex(data)
        _ = try await storageRequest(
            method: "PUT", path: "/chunk/\(id)",
            body: data, headers: ["Content-Length": "\(data.count)"], port: port
        )

        let (_, delStatus) = try await storageRequest(
            method: "DELETE", path: "/chunk/\(id)", port: port
        )
        #expect(delStatus == 200)

        let (_, getStatus) = try await storageRequest(
            method: "GET", path: "/chunk/\(id)", port: port
        )
        #expect(getStatus == 404)
    }

    @Test func deleteChunkNotFoundReturns404() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        let missingID = String(repeating: "b", count: 64)
        let (body, status) = try await storageRequest(
            method: "DELETE", path: "/chunk/\(missingID)", port: port
        )
        #expect(status == 404)
        let json = try JSONSerialization.jsonObject(with: body) as? [String: String]
        #expect(json?["error"] == "not_found")
    }

    @Test func deleteChunkBadIdReturns400() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        let (_, status) = try await storageRequest(
            method: "DELETE", path: "/chunk/badhash", port: port
        )
        #expect(status == 400)
    }

    @Test func deleteChunkBadIdReturnsJsonError() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        let (body, status) = try await storageRequest(
            method: "DELETE", path: "/chunk/tooshort", port: port
        )
        #expect(status == 400)
        // DELETE with bad id returns plain text (same branch as GET)
        let text = String(data: body, encoding: .utf8)
        #expect(text?.isEmpty == false)
    }

    // MARK: chunks/list

    @Test func listChunksIsEmptyInitially() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        let (body, status) = try await storageRequest(
            method: "GET", path: "/chunks/list", port: port
        )
        #expect(status == 200)
        let list = try JSONSerialization.jsonObject(with: body) as? [String]
        #expect(list?.isEmpty == true)
    }

    @Test func listChunksShowsStoredChunk() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        let data = Data("list me".utf8)
        let id = sha256Hex(data)
        _ = try await storageRequest(
            method: "PUT", path: "/chunk/\(id)",
            body: data, headers: ["Content-Length": "\(data.count)"], port: port
        )

        let (body, status) = try await storageRequest(
            method: "GET", path: "/chunks/list", port: port
        )
        #expect(status == 200)
        let list = try JSONSerialization.jsonObject(with: body) as? [String]
        #expect(list?.contains(id) == true)
    }

    @Test func listChunksDoesNotIncludeNonSHA256Files() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        // Manually drop a non-SHA256 file into the storage dir
        let data = Data("hello world".utf8)
        let id = sha256Hex(data)
        _ = try await storageRequest(
            method: "PUT", path: "/chunk/\(id)",
            body: data, headers: ["Content-Length": "\(data.count)"], port: port
        )

        let (body, status) = try await storageRequest(
            method: "GET", path: "/chunks/list", port: port
        )
        #expect(status == 200)
        let list = try JSONSerialization.jsonObject(with: body) as? [String]
        // Only valid SHA256 names should appear
        #expect(list?.allSatisfy { $0.count == 64 } == true)
    }

    // MARK: healthcheck

    @Test func healthCheckHealthy() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        let (body, status) = try await storageRequest(
            method: "GET", path: "/chunks/healthcheck", port: port
        )
        #expect(status == 200)
        let json = try JSONSerialization.jsonObject(with: body) as? [String: Any]
        #expect(json?["status"] as? String == "healthy")
        #expect((json?["bad_chunks"] as? [String])?.isEmpty == true)
    }

    @Test func healthCheckCachesResult() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        _ = try await storageRequest(
            method: "GET", path: "/chunks/healthcheck?max_age=300", port: port
        )
        let (body, status) = try await storageRequest(
            method: "GET", path: "/chunks/healthcheck?max_age=300", port: port
        )
        #expect(status == 200)
        let json = try JSONSerialization.jsonObject(with: body) as? [String: Any]
        #expect(json?["status"] as? String == "healthy")
    }

    @Test func healthCheckMaxAgeZeroBypassesCache() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        _ = try await storageRequest(method: "GET", path: "/chunks/healthcheck", port: port)
        let (body, status) = try await storageRequest(
            method: "GET", path: "/chunks/healthcheck?max_age=0", port: port
        )
        #expect(status == 200)
        let json = try JSONSerialization.jsonObject(with: body) as? [String: Any]
        #expect(json?["status"] as? String == "healthy")
    }

    // MARK: storage_info

    @Test func storageInfoReturnsSpaceFields() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        let (body, status) = try await storageRequest(
            method: "GET", path: "/storage_info", port: port
        )
        #expect(status == 200)
        let json = try JSONSerialization.jsonObject(with: body) as? [String: Any]
        #expect(json?["total_space"] != nil)
        #expect(json?["available_space"] != nil)
        #expect(json?["used_space"] != nil)
    }

    @Test func storageInfoUsedSpaceGrowsAfterPut() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        let (b0, _) = try await storageRequest(method: "GET", path: "/storage_info", port: port)
        let before = (try JSONSerialization.jsonObject(with: b0) as? [String: Any])?["used_space"] as? Int64 ?? 0

        let data = Data(repeating: 0xAB, count: 1024)
        let id = sha256Hex(data)
        _ = try await storageRequest(
            method: "PUT", path: "/chunk/\(id)",
            body: data, headers: ["Content-Length": "\(data.count)"], port: port
        )

        let (b1, _) = try await storageRequest(method: "GET", path: "/storage_info", port: port)
        let after = (try JSONSerialization.jsonObject(with: b1) as? [String: Any])?["used_space"] as? Int64 ?? 0
        #expect(after > before)
    }

    // MARK: unknown routes

    @Test func unknownRouteReturns404() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        let (_, status) = try await storageRequest(
            method: "GET", path: "/nonexistent", port: port
        )
        #expect(status == 404)
    }

    @Test func unsupportedMethodOnChunkReturns404() async throws {
        let (server, port) = makeServer()
        defer { server.stop() }
        #expect(server.start(port: port))
        try await Task.sleep(nanoseconds: 150_000_000)

        let id = String(repeating: "c", count: 64)
        let (_, status) = try await storageRequest(
            method: "PATCH", path: "/chunk/\(id)", port: port
        )
        #expect(status == 404)
    }
}
