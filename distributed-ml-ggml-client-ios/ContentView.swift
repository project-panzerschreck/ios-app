//
//  ContentView.swift
//  distributed-ml-ggml-client-ios
//
//  Created by Sandeep Reehal on 2/23/26.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        InferenceView()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .environmentObject(InferenceEngine.shared)
            .environmentObject(RpcSettings.shared)
            .environmentObject(AppDiagnosticsModel.shared)
    }
}
