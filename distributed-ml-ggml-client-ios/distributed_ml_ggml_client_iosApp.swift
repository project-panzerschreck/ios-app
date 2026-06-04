//
//  distributed_ml_ggml_client_iosApp.swift
//  distributed-ml-ggml-client-ios
//
//  Created by Sandeep Reehal on 2/23/26.
//

import SwiftUI

@main
struct distributed_ml_ggml_client_iosApp: App {
    @Environment(\.scenePhase) private var scenePhase

    init() {
        // Install GGML log forwarding before any framework static initializers need GGML_RPC_DEBUG.
        _ = RpcSettings.shared
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(InferenceEngine.shared)
                .environmentObject(RpcSettings.shared)
                .environmentObject(AppDiagnosticsModel.shared)
                .onChange(of: scenePhase) { newPhase in
                    switch newPhase {
                    case .active:
                        InferenceEngine.shared.handleAppDidBecomeActive()
                    case .inactive, .background:
                        InferenceEngine.shared.handleAppWillResignActive()
                    @unknown default:
                        break
                    }
                }
        }
    }
}
