//
//  ContentView.swift
//  textmods
//
//  Created by origin echo on 2026/1/5.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            EmbeddingTestView()
                .tabItem {
                    Label("Embedding", systemImage: "text.magnifyingglass")
                }
            
            SettingsView()
                .tabItem {
                    Label("设置", systemImage: "gear")
                }
        }
    }
}

struct SettingsView: View {
    var body: some View {
        NavigationStack {
            Form {
                Section("关于") {
                    LabeledContent("版本", value: "1.0.0")
                    LabeledContent("模型", value: "Qwen3-Embedding-0.6B-4bit-DWQ")
                }
                
                Section("说明") {
                    Text("基于 MLX-Swift 框架的端侧 Embedding 模型测试应用。")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("设置")
        }
    }
}

#Preview {
    ContentView()
}
