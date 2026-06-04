// InferenceView.swift
//
// Primary UI for single-device GGML inference and GGML RPC worker mode.
//
// Sections:
//   1. Model loader – pick a .gguf file from the Documents directory.
//   2. Model status card – name, layers, embedding dimension, etc.
//   3. Prompt + generation configuration controls (single-device).
//   4. Streaming output area.
//   5. RPC Worker panel – expose this device as a Metal compute backend.
//      The phone is a *leaf node* only; a llama-cli process on a laptop or
//      server acts as the coordinator via the GGML RPC protocol.

import SwiftUI
import UIKit
import AVFoundation
import UniformTypeIdentifiers

struct InferenceView: View {

    @EnvironmentObject private var engine: InferenceEngine
    @EnvironmentObject private var settings: RpcSettings

    // ── UI state ──────────────────────────────────────────────────────────────
    @State private var chatInput     = ""
    @State private var contextSize   = 1024
    @State private var maxTokens     = 200
    @State private var temperature   = 0.8
    @State private var showDocPicker = false
    @State private var localModels: [URL] = []

    // ── RPC worker state (loaded once per launch; persisted when connecting) ───
    @State private var clusterServerHost: String = ""
    @State private var clusterServerPort: Int = RpcSettings.defaultClusterServerPort
    @State private var clusterToken: String = ""
    private static var didLoadCoordinatorSettings = false

    @State private var connectionString: String = ""
    @State private var endpointsExpanded: Bool = false
    @State private var selectedTab: Int  = 1
    @State private var showQRScanner: Bool = false
    @State private var importStatus: String = ""

    // ── Body ──────────────────────────────────────────────────────────────────

    var body: some View {
        TabView(selection: $selectedTab) {
            NavigationView {
                List {
                    modelSection
                    if case .ready = engine.modelState {
                        chatSection
                    }
                    if case .generating = engine.modelState {
                        chatSection
                    }
                }
                .navigationTitle("Inference")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar { toolbarContent }
                .sheet(isPresented: $showDocPicker) { documentPicker }
            }
            .navigationViewStyle(.stack)
            .tabItem { Label("Inference", systemImage: "cpu") }
            .tag(0)

            NavigationView {
                List {
                    rpcWorkerSection
                }
                .navigationTitle("RMCluster Node")
                .navigationTitle("RMCluster Node")
                .navigationBarTitleDisplayMode(.inline)
                .modifier(CompactSectionSpacing())
            }
            .navigationViewStyle(.stack)
            .tabItem { Label("RMCluster Node", systemImage: "network") }
            .tabItem { Label("RMCluster Node", systemImage: "network") }
            .tag(1)

            LogsView()
                .tabItem { Label("Logs", systemImage: "terminal") }
                .tag(2)
        }
        .sheet(isPresented: $showQRScanner) {
            QRScannerSheet(
                onCodeScanned: { code in
                    showQRScanner = false
                    AppLogger.log(tag: "InferenceView", "QR code scanned successfully")
                    applyConnectionConfig(from: code)
                },
                onFailure: { message in
                    showQRScanner = false
                    importStatus = message
                    AppLogger.log("WARN", tag: "InferenceView", "QR scanner failed: \(message)")
                }
            )
        }
        .onOpenURL { incomingURL in
            AppLogger.log(tag: "InferenceView", "Received deep link: \(incomingURL.absoluteString)")
            applyConnectionConfig(from: incomingURL.absoluteString)
        }
        .onAppear {
            loadCoordinatorSettingsIfNeeded()
            refreshLocalModels()
        }
    }

    // ── Model section ─────────────────────────────────────────────────────────

    @ViewBuilder
    private var modelSection: some View {
        Section(header: Text("Model")) {
            switch engine.modelState {
            case .unloaded:
                if localModels.isEmpty {
                    Button { showDocPicker = true } label: {
                        Label("Load .gguf model…", systemImage: "doc.badge.plus")
                    }
                } else {
                    ForEach(localModels, id: \.self) { url in
                        Button {
                            Task { await engine.loadModel(from: url, contextLength: contextSize) }
                        } label: {
                            Label(url.lastPathComponent, systemImage: "cpu")
                        }
                    }
                    Button { showDocPicker = true } label: {
                        Label("Load other…", systemImage: "doc.badge.plus")
                    }
                    .foregroundColor(.secondary)
                }

            case .loading:
                HStack {
                    ProgressView()
                    Text("Loading model…").foregroundColor(.secondary)
                }

            case .ready(let name, let nLayers):
                VStack(alignment: .leading, spacing: 4) {
                    Label(name, systemImage: "cpu").font(.headline)
                    if let info = engine.modelInfo {
                        HStack(spacing: 16) {
                            StatChip(label: "\(nLayers) layers")
                            StatChip(label: "embd \(info.nEmbd)")
                            StatChip(label: "ctx \(info.nCtx)")
                            StatChip(label: sizeString(info.fileSizeBytes))
                        }
                    }
                }
                .padding(.vertical, 2)
                Button { engine.unloadModel() } label: {
                    Label("Unload model", systemImage: "eject")
                }
                .foregroundColor(.red)

            case .generating:
                HStack {
                    ProgressView()
                    Text("Generating…").foregroundColor(.secondary)
                    Spacer()
                    if engine.tokensPerSecond > 0 {
                        Text(String(format: "%.1f tok/s", engine.tokensPerSecond))
                            .font(.caption.monospacedDigit())
                            .foregroundColor(.secondary)
                    }
                    Button("Stop") { engine.cancelGeneration() }
                        .foregroundColor(.red)
                }

            case .error(let msg):
                Label(msg, systemImage: "exclamationmark.triangle").foregroundColor(.red)
                Button("Try again") { showDocPicker = true }
            }
        }
    }

    // ── Conversation ──────────────────────────────────────────────────────────

    private var isGenerating: Bool {
        if case .generating = engine.modelState { return true }
        return false
    }

    @ViewBuilder
    private var chatSection: some View {
        // Message history
        if !engine.chatMessages.isEmpty {
            Section(header: Text("Conversation")) {
                ForEach(engine.chatMessages) { msg in
                    VStack(alignment: msg.role == "user" ? .trailing : .leading, spacing: 2) {
                        Text(msg.role == "user" ? "You" : "Assistant")
                            .font(.caption2.weight(.semibold))
                            .foregroundColor(.secondary)
                        Text(msg.content.isEmpty ? "…" : msg.content)
                            .font(.body)
                            .frame(maxWidth: .infinity,
                                   alignment: msg.role == "user" ? .trailing : .leading)
                    }
                    .padding(.vertical, 2)
                }
                if engine.tokensPerSecond > 0 {
                    HStack {
                        Spacer()
                        Text(String(format: "%.1f tok/s", engine.tokensPerSecond))
                            .font(.caption.monospacedDigit())
                            .foregroundColor(.secondary)
                    }
                }
            }
        }

        // Input bar
        Section {
            HStack(alignment: .bottom, spacing: 10) {
                Group {
                    if #available(iOS 16.0, *) {
                        TextField("Message…", text: $chatInput, axis: .vertical)
                            .lineLimit(1...5)
                    } else {
                        TextField("Message…", text: $chatInput)
                            .lineLimit(5)
                    }
                }
                .disabled(isGenerating)
                Button {
                    if isGenerating {
                        engine.cancelGeneration()
                    } else {
                        let text = chatInput.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !text.isEmpty else { return }
                        chatInput = ""
                        let config = LlamaGenerationConfig.defaults()
                        config.maxNewTokens = maxTokens
                        config.temperature  = Float(temperature)
                        engine.sendMessage(text, config: config)
                    }
                } label: {
                    Image(systemName: isGenerating ? "stop.circle.fill" : "arrow.up.circle.fill")
                        .font(.title2)
                        .foregroundColor(
                            chatInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isGenerating
                                ? Color.secondary : Color.accentColor
                        )
                }
                .buttonStyle(.plain)
            }
        }

        // Parameters
        Section(header: Text("Parameters")) {
            HStack {
                Text("Max tokens")
                Spacer()
                Stepper("\(maxTokens)", value: $maxTokens, in: 1...2048, step: 50)
            }
            HStack {
                Text(String(format: "Temperature  %.2f", temperature))
                Spacer()
                Slider(value: $temperature, in: 0...2, step: 0.05)
            }
        }

        // Clear
        if !engine.chatMessages.isEmpty {
            Section {
                Button {
                    engine.clearChat()
                } label: {
                    Label("Clear conversation", systemImage: "trash")
                        .frame(maxWidth: .infinity)
                }
                .foregroundColor(.red)
            }
        }
    }

    // ── RPC Worker panel ──────────────────────────────────────────────────────

    @ViewBuilder
    private var rpcWorkerSection: some View {
        let isRunning = rpcIsRunning
        let interfaces = ShardNetwork.allLocalIPv4s

        Section(header: Text("Node Name (Optional)")) {
            TextField("e.g. John's iPhone", text: $settings.nickname)
                .disabled(isRunning)
        }

        Section {
            if endpointsExpanded {
                if interfaces.isEmpty {
                    Label("No network interfaces found", systemImage: "wifi.slash")
                        .font(.caption)
                        .foregroundColor(.secondary)
                } else {
                    ForEach(interfaces) { iface in
                        HStack(spacing: 12) {
                            Circle()
                                .fill(isRunning ? Color.green : Color.secondary.opacity(0.35))
                                .frame(width: 9, height: 9)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(verbatim: "RPC \(iface.ip)")
                                    .font(.system(.body, design: .monospaced).bold())
                                Text(verbatim: "Storage \(iface.ip)")
                                    .font(.system(.caption2, design: .monospaced))
                                    .foregroundColor(.secondary)
                                Text(iface.label)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            Spacer()
                            Button {
                                UIPasteboard.general.string = "\(iface.ip):\(RpcSettings.listenPort)"
                            } label: {
                                Image(systemName: "doc.on.doc")
                            }
                            .buttonStyle(.plain)
                            .foregroundColor(Color.accentColor)
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
        } header: {
            Button {
                withAnimation { endpointsExpanded.toggle() }
            } label: {
                HStack {
                    Text("Endpoints")
                        .font(.headline)
                        .foregroundColor(.primary)
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.subheadline)
                        .foregroundColor(.primary)
                        .rotationEffect(.degrees(endpointsExpanded ? 90 : 0))
                }
            }
            .buttonStyle(.plain)
        }

        Section {
            Button {
                showQRScanner = true
            } label: {
                Label("Scan QR code", systemImage: "qrcode.viewfinder")
            }

            TextField("Paste rmcluster:// connection URL ", text: $connectionString)
                .autocapitalization(.none)
                .disableAutocorrection(true)
                .keyboardType(.URL)

            if !importStatus.isEmpty {
                Text(importStatus)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        } header: {
            Text("Connection")
        } footer: {
            Text("Paste a rmcluster://connect URL, scan a QR code, or type the coordinator server IP, port, and token below.")
        }

        Section(header: Text("Coordinator")) {
            TextField("Server IP or host", text: $clusterServerHost)
                .autocapitalization(.none)
                .disableAutocorrection(true)
                .keyboardType(.URL)

            IntStepperField("Server port", value: $clusterServerPort, in: 1...65535, disabled: false)
            IntStepperField("Thread count", value: $settings.threads, in: 1...64, disabled: isRunning)
        }

        Section {
            rpcActionContent(isRunning: isRunning)
        }
    }

    @ViewBuilder
    private func rpcActionContent(isRunning: Bool) -> some View {
        switch engine.rpcServerState {
        case .unavailable(let msg):
            Label(msg.isEmpty ? "Unavailable" : msg, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundColor(.red)
                .padding(.vertical, 4)
        case .starting:
            Button {
                AppLogger.log(tag: "InferenceView", "Cancel connection tapped")
                engine.stopRPCServer()
            } label: {
                Label("Cancel connection", systemImage: "xmark.circle")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity, minHeight: 44)
                    .background(Color.orange)
            }
            .buttonStyle(.plain)
            .listRowInsets(EdgeInsets())
        default:
            if isRunning {
                Button {
                    AppLogger.log(tag: "InferenceView", "Disconnect tapped")
                    engine.stopRPCServer()
                } label: {
                    Label("Disconnect from cluster", systemImage: "stop.circle")
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity, minHeight: 44)
                        .background(Color.red)
                }
                .buttonStyle(.plain)
                .listRowInsets(EdgeInsets())
            } else {
                Button {
                    guard prepareCoordinatorSettingsForStart() else { return }
                    AppLogger.log(tag: "InferenceView", "Start RPC server tapped")
                    engine.startRPCServer(
                        coordinatorHost: clusterServerHost,
                        coordinatorPort: clusterServerPort,
                        nickname: settings.nickname,
                        threads: settings.threads,
                        deviceId: settings.deviceId
                    )
                } label: {
                    Text("Connect to cluster")
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundColor(canStartRPCServer ? .white : .secondary)
                        .frame(maxWidth: .infinity, minHeight: 44)
                        .background(canStartRPCServer ? Color.accentColor : Color.gray.opacity(0.3))
                }
                .buttonStyle(.plain)
                .disabled(!canStartRPCServer)
                .listRowInsets(EdgeInsets())
            }
        }
    }

    // ── RPC helpers ───────────────────────────────────────────────────────────

    private var rpcIsRunning: Bool {
        if case .running = engine.rpcServerState { return true }
        return false
    }

    private var rpcStatusText: String {
        switch engine.rpcServerState {
        case .idle:
            return "Idle"
        case .starting:
            return "Starting"
        case .running:
            return "Running"
        case .recovering:
            return "Recovering"
        case .degraded:
            return "Paused"
        case .unavailable:
            return "Unavailable"
        }
    }

    private var rpcStatusDetail: String {
        switch engine.rpcServerState {
        case .running(let endpoint):
            return endpoint
        case .recovering(let reason), .degraded(let reason), .unavailable(let reason):
            return reason
        case .idle, .starting:
            return ""
        }
    }

    private var rpcStatusIcon: String {
        switch engine.rpcServerState {
        case .idle:
            return "circle.dashed"
        case .starting:
            return "hourglass"
        case .running:
            return "checkmark.circle.fill"
        case .recovering:
            return "arrow.triangle.2.circlepath.circle.fill"
        case .degraded:
            return "pause.circle.fill"
        case .unavailable:
            return "exclamationmark.triangle.fill"
        }
    }

    private var rpcStatusColor: Color {
        switch engine.rpcServerState {
        case .running:
            return .green
        case .recovering, .starting:
            return .orange
        case .degraded, .idle:
            return .secondary
        case .unavailable:
            return .red
        }
    }

    private var canStartRPCServer: Bool {
        let pendingConnection = !connectionString.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let hasCoordinatorHost = !clusterServerHost.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        return pendingConnection || hasCoordinatorHost
    }

    private func loadCoordinatorSettingsIfNeeded() {
        guard !Self.didLoadCoordinatorSettings else { return }
        Self.didLoadCoordinatorSettings = true
        clusterServerHost = RpcSettings.loadClusterServerHost()
        clusterServerPort = RpcSettings.loadClusterServerPort()
        clusterToken = RpcSettings.loadClusterToken()
    }

    private func persistCoordinatorSettings() {
        RpcSettings.saveClusterConnection(
            host: clusterServerHost,
            port: clusterServerPort,
            token: clusterToken
        )
    }

    private func prepareCoordinatorSettingsForStart() -> Bool {
        let trimmedHost = clusterServerHost.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedHost.isEmpty {
            importStatus = ""
            persistCoordinatorSettings()
            return true
        }

        let pendingConnection = connectionString.trimmingCharacters(in: .whitespacesAndNewlines)
        if !pendingConnection.isEmpty {
            guard applyConnectionConfig(from: pendingConnection) else { return false }
            persistCoordinatorSettings()
            return true
        }

        importStatus = "Enter a coordinator server IP or paste a connection string."
        AppLogger.log("WARN", tag: "InferenceView", importStatus)
        return false
    }

    @discardableResult
    private func applyConnectionConfig(from rawValue: String) -> Bool {
        guard let parsed = ConnectionBootstrapPayload.parse(rawValue) else {
            importStatus = "Could not parse connection data."
            AppLogger.log("WARN", tag: "InferenceView", "Could not parse connection data: \(rawValue)")
            return false
        }

        connectionString = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        clusterServerHost = parsed.host
        if let port = parsed.port {
            clusterServerPort = port
        }
        if let token = parsed.token, !token.isEmpty {
            clusterToken = token
        }
        selectedTab = 1
        importStatus = ""
        AppLogger.log(tag: "InferenceView", "Applied connection config for \(parsed.host):\(parsed.port ?? clusterServerPort)")
        return true
    }

    // ── Toolbar ───────────────────────────────────────────────────────────────

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            if case .unloaded = engine.modelState {
                Button { showDocPicker = true } label: {
                    Image(systemName: "folder")
                }
            }
        }
    }

    // ── Document picker ───────────────────────────────────────────────────────

    private var documentPicker: some View {
        DocumentPicker(contentTypes: [.init(filenameExtension: "gguf") ?? .data]) { url in
            Task { await engine.loadModel(from: url, contextLength: contextSize) }
        }
    }

    // ── Actions ───────────────────────────────────────────────────────────────

    private func refreshLocalModels() {
        let docs = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
        localModels = (try? FileManager.default.contentsOfDirectory(
            at: docs, includingPropertiesForKeys: nil))?.filter {
            $0.pathExtension.lowercased() == "gguf"
        }.sorted { $0.lastPathComponent < $1.lastPathComponent } ?? []
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private func sizeString(_ bytes: UInt) -> String {
        let mb = Double(bytes) / 1_048_576
        if mb >= 1000 { return String(format: "%.1f GB", mb / 1024) }
        return String(format: "%.0f MB", mb)
    }
}

// ── Sub-views ─────────────────────────────────────────────────────────────────

private struct StatChip: View {
    let label: String
    var body: some View {
        Text(label)
            .font(.system(.caption2, design: .monospaced))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Capsule().fill(Color(UIColor.quaternarySystemFill)))
    }
}

// ── UIDocumentPickerViewController wrapper ────────────────────────────────────

private struct DocumentPicker: UIViewControllerRepresentable {
    let contentTypes: [UTType]
    let onPick: (URL) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onPick: onPick) }

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let vc = UIDocumentPickerViewController(forOpeningContentTypes: contentTypes)
        vc.delegate = context.coordinator
        vc.allowsMultipleSelection = false
        return vc
    }

    func updateUIViewController(_ vc: UIDocumentPickerViewController, context: Context) {}

    final class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: (URL) -> Void
        init(onPick: @escaping (URL) -> Void) { self.onPick = onPick }

        func documentPicker(_ controller: UIDocumentPickerViewController,
                            didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else { return }
            guard url.startAccessingSecurityScopedResource() else { return }
            defer { url.stopAccessingSecurityScopedResource() }

            // Copy to app's Documents directory so we keep access after the picker closes
            let dest = FileManager.default
                .urls(for: .documentDirectory, in: .userDomainMask)[0]
                .appendingPathComponent(url.lastPathComponent)
            try? FileManager.default.copyItem(at: url, to: dest)
            onPick(dest)
        }
    }
}

// ── Compact section spacing (iOS 17+) ────────────────────────────────────────
private struct CompactSectionSpacing: ViewModifier {
    func body(content: Content) -> some View {
        if #available(iOS 17, *) {
            content.listSectionSpacing(8)
        } else {
            content
        }
    }
}

// ── Int field with inline text entry + stepper ────────────────────────────────

private struct IntStepperField: View {
    let label: String
    @Binding var value: Int
    let range: ClosedRange<Int>
    let disabled: Bool

    @State private var text: String = ""

    init(_ label: String, value: Binding<Int>, in range: ClosedRange<Int>, disabled: Bool) {
        self.label    = label
        self._value   = value
        self.range    = range
        self.disabled = disabled
        self._text    = State(initialValue: String(value.wrappedValue))
    }

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            TextField("", text: $text)
                .keyboardType(.numberPad)
                .multilineTextAlignment(.trailing)
                .frame(width: 64)
                .disabled(disabled)
                .onReceive(NotificationCenter.default.publisher(for: UITextField.textDidEndEditingNotification)) { _ in
                    commit()
                }
                .onChange(of: value) { newVal in
                    text = String(newVal)
                }
            Stepper("", value: $value, in: range, step: 1)
                .labelsHidden()
                .disabled(disabled)
                .onChange(of: value) { newVal in
                    text = String(newVal)
                }
        }
    }

    private func commit() {
        if let parsed = Int(text), range.contains(parsed) {
            value = parsed
        } else {
            text = String(value)
        }
    }
}

// ── Preview ───────────────────────────────────────────────────────────────────

struct InferenceView_Previews: PreviewProvider {
    static var previews: some View {
        InferenceView()
            .environmentObject(InferenceEngine.shared)
            .environmentObject(RpcSettings.shared)
            .environmentObject(AppDiagnosticsModel.shared)
    }
}

private struct ConnectionBootstrapPayload {
    let host: String
    let port: Int?
    let token: String?
    let device: String?

    static func parse(_ raw: String) -> ConnectionBootstrapPayload? {
        let input = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty else { return nil }

        if let deepLink = URL(string: input),
           deepLink.scheme?.lowercased() == "rmcluster",
           let comps = URLComponents(url: deepLink, resolvingAgainstBaseURL: false) {
            let query = queryMap(from: comps)
            guard let serverValue = query["url"] ?? query["host"],
                  let (host, embeddedPort) = parseHostAndPort(serverValue) else {
                return nil
            }

            let explicitPort = query["port"].flatMap(Int.init)
            return ConnectionBootstrapPayload(
                host: host,
                port: explicitPort ?? embeddedPort,
                token: query["token"],
                device: query["device"] ?? query["label"] ?? query["name"]
            )
        }

        if let comps = URLComponents(string: input), let host = comps.host {
            let query = queryMap(from: comps)
            return ConnectionBootstrapPayload(
                host: host,
                port: comps.port ?? query["port"].flatMap(Int.init),
                token: query["token"],
                device: query["device"] ?? query["label"] ?? query["name"]
            )
        }

        if let (host, port) = parseHostAndPort(input) {
            return ConnectionBootstrapPayload(host: host, port: port, token: nil, device: nil)
        }

        return nil
    }

    private static func queryMap(from comps: URLComponents) -> [String: String] {
        Dictionary(uniqueKeysWithValues: (comps.queryItems ?? []).map { ($0.name.lowercased(), $0.value ?? "") })
    }

    private static func parseHostAndPort(_ value: String) -> (String, Int?)? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if let withScheme = URLComponents(string: trimmed), let host = withScheme.host {
            return (host, withScheme.port)
        }

        if let fallback = URLComponents(string: "http://\(trimmed)"), let host = fallback.host {
            return (host, fallback.port)
        }

        return nil
    }
}

private struct QRScannerSheet: UIViewControllerRepresentable {
    let onCodeScanned: (String) -> Void
    let onFailure: (String) -> Void

    func makeUIViewController(context: Context) -> QRScannerViewController {
        let controller = QRScannerViewController()
        controller.onCodeScanned = onCodeScanned
        controller.onFailure = onFailure
        return controller
    }

    func updateUIViewController(_ uiViewController: QRScannerViewController, context: Context) {}
}

private final class QRScannerViewController: UIViewController, AVCaptureMetadataOutputObjectsDelegate {
    var onCodeScanned: ((String) -> Void)?
    var onFailure: ((String) -> Void)?

    private let captureSession = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer?

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        configureSession()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        startIfAuthorized()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }

    private func configureSession() {
        guard let camera = AVCaptureDevice.default(for: .video) else {
            onFailure?("Camera unavailable on this device.")
            return
        }

        guard let input = try? AVCaptureDeviceInput(device: camera), captureSession.canAddInput(input) else {
            onFailure?("Failed to open camera input.")
            return
        }
        captureSession.addInput(input)

        let output = AVCaptureMetadataOutput()
        guard captureSession.canAddOutput(output) else {
            onFailure?("Failed to start QR scanner.")
            return
        }
        captureSession.addOutput(output)
        output.setMetadataObjectsDelegate(self, queue: .main)
        output.metadataObjectTypes = [.qr]

        let preview = AVCaptureVideoPreviewLayer(session: captureSession)
        preview.videoGravity = .resizeAspectFill
        preview.frame = view.layer.bounds
        view.layer.addSublayer(preview)
        previewLayer = preview
    }

    private func startIfAuthorized() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            if !captureSession.isRunning {
                captureSession.startRunning()
            }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    if granted {
                        self?.captureSession.startRunning()
                    } else {
                        self?.onFailure?("Camera permission denied.")
                    }
                }
            }
        default:
            onFailure?("Camera permission denied.")
        }
    }

    func metadataOutput(
        _ output: AVCaptureMetadataOutput,
        didOutput metadataObjects: [AVMetadataObject],
        from connection: AVCaptureConnection
    ) {
        guard let object = metadataObjects.first as? AVMetadataMachineReadableCodeObject,
              let code = object.stringValue else {
            return
        }

        if captureSession.isRunning {
            captureSession.stopRunning()
        }
        onCodeScanned?(code)
    }
}
