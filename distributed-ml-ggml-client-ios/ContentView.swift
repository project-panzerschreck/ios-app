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

#Preview {
    ContentView()
        .environmentObject(InferenceEngine.shared)
}
