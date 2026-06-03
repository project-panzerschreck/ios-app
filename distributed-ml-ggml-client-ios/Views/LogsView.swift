import SwiftUI
import UIKit

struct LogsView: View {
    @EnvironmentObject private var diagnostics: AppDiagnosticsModel
    @Environment(\.colorScheme) private var colorScheme
    @State private var activeFilters = LogCategory.allCases

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                healthSummary
                categoryFilters
                Divider()
                LogTextView(text: filteredLogsText)
                    .background(Color.black)
            }
            .navigationTitle("Logs")
            .navigationBarTitleDisplayMode(.inline)
        }
        .navigationViewStyle(.stack)
    }

    private var healthSummary: some View {
        let health = diagnostics.rpcHealth

        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label(health.status.capitalized, systemImage: statusIcon(for: health.status))
                    .font(.headline)
                Spacer()
            }
            if !health.lastError.isEmpty {
                Text(health.lastError)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(uiColor: .systemBackground))
    }

    private var categoryFilters: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(LogCategory.allCases) { category in
                    let isActive = activeFilters.contains(category)
                    let healthColor = healthColor(for: category)

                    Button {
                        toggle(category)
                    } label: {
                        Text(category.title)
                            .font(.caption.weight(.semibold))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 7)
                            .background(
                                isActive ? selectedFilterBackground : Color.secondary.opacity(0.12),
                                in: Capsule()
                            )
                            .foregroundColor(isActive ? selectedFilterForeground : Color.secondary)
                            .overlay(
                                Capsule()
                                    .stroke(healthColor, lineWidth: 1.5)
                            )
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 10)
        }
        .background(Color(uiColor: .systemBackground))
    }

    private var filteredLogsText: String {
        diagnostics.logsText
            .split(whereSeparator: \.isNewline)
            .map(String.init)
            .filter { line in
                guard let category = LogCategory.category(for: line) else { return false }
                return activeFilters.contains(category)
            }
            .joined(separator: "\n")
    }

    private func toggle(_ category: LogCategory) {
        if let index = activeFilters.firstIndex(of: category) {
            activeFilters.remove(at: index)
        } else {
            activeFilters.append(category)
            activeFilters.sort { $0.sortOrder < $1.sortOrder }
        }
    }

    private func statusIcon(for status: String) -> String {
        switch status {
        case "running":
            return "checkmark.circle.fill"
        case "recovering":
            return "arrow.triangle.2.circlepath.circle.fill"
        case "degraded", "unavailable":
            return "exclamationmark.triangle.fill"
        default:
            return "circle.dashed"
        }
    }

    private var selectedFilterBackground: Color {
        colorScheme == .dark ? .white : .black
    }

    private var selectedFilterForeground: Color {
        colorScheme == .dark ? .black : .white
    }

    private func healthColor(for category: LogCategory) -> Color {
        let health = diagnostics.rpcHealth
        guard showsHealthOutline(for: health) else {
            return .clear
        }

        let isHealthy: Bool

        switch category {
        case .rpc:
            isHealthy = health.rpcHealthy
        case .storage:
            isHealthy = health.storageHealthy
        case .general:
            isHealthy = health.announceEligible
        }

        return isHealthy ? .green : .red
    }

    private func showsHealthOutline(for health: RPCHealthSnapshot) -> Bool {
        switch health.status {
        case "starting", "running", "recovering", "degraded", "unavailable":
            return true
        default:
            return false
        }
    }
}

private enum LogCategory: String, CaseIterable, Identifiable {
    case rpc
    case storage
    case general

    var id: String { rawValue }

    var title: String {
        switch self {
        case .rpc:
            return "RPC"
        case .storage:
            return "Storage"
        case .general:
            return "General"
        }
    }
    var sortOrder: Int {
        switch self {
        case .rpc:
            return 0
        case .storage:
            return 1
        case .general:
            return 2
        }
    }

    static func category(for line: String) -> LogCategory? {
        if line.contains("[STORAGE]") {
            return .storage
        }
        if line.contains("[RPC SERVER]") {
            return .rpc
        }

        // Everything else, including [GENERAL] and legacy untagged lines, belongs in General.
        return .general
    }
}

private struct LogTextView: UIViewRepresentable {
    let text: String

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> UITextView {
        let view = UITextView()
        view.isEditable = false
        view.isSelectable = true
        view.backgroundColor = .black
        view.textColor = .systemGreen
        view.font = UIFont.monospacedSystemFont(ofSize: 12, weight: .regular)
        view.alwaysBounceVertical = true
        view.delegate = context.coordinator
        view.textContainerInset = UIEdgeInsets(top: 12, left: 12, bottom: 12, right: 12)
        return view
    }

    func updateUIView(_ uiView: UITextView, context: Context) {
        let shouldScroll = context.coordinator.shouldAutoScroll(for: uiView)
        if uiView.text != text {
            uiView.text = text
        }
        if shouldScroll {
            let offset = CGPoint(x: 0, y: max(-uiView.adjustedContentInset.top, uiView.contentSize.height - uiView.bounds.height + uiView.adjustedContentInset.bottom))
            uiView.setContentOffset(offset, animated: false)
        }
        context.coordinator.didInitialScroll = true
    }

    final class Coordinator: NSObject, UITextViewDelegate {
        var didInitialScroll = false
        private var isNearBottom = true

        func shouldAutoScroll(for textView: UITextView) -> Bool {
            return !didInitialScroll || isNearBottom || textView.text.isEmpty
        }

        func scrollViewDidScroll(_ scrollView: UIScrollView) {
            let bottomOffset = scrollView.contentSize.height - (scrollView.contentOffset.y + scrollView.bounds.height - scrollView.adjustedContentInset.bottom)
            isNearBottom = bottomOffset < 80
        }
    }
}
