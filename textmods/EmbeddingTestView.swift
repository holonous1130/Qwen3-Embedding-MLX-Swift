//
//  EmbeddingTestView.swift
//  textmods
//
//  Qwen3-Embedding 测试界面
//

import SwiftUI

struct EmbeddingTestView: View {
    @StateObject private var embeddingManager = Qwen3EmbeddingEngine()
    
    // 输入状态
    @State private var text1 = "今天天气很好"
    @State private var text2 = "今天阳光明媚"
    @State private var selectedDimension = 1024
    @State private var useWhitening = false
    
    // 结果状态
    @State private var similarity: Float?
    @State private var elapsedTime: Double?
    @State private var vector1Preview: [Float]?
    @State private var vector2Preview: [Float]?
    @State private var isCalculating = false
    @State private var calculationError: String?
    
    // 可选维度
    private let dimensions = [32, 64, 128, 256, 512, 768, 1024]
    
    var body: some View {
        NavigationStack {
            Form {
                // 模型控制区
                modelControlSection
                
                // 输入区
                inputSection
                
                // 设置区
                settingsSection
                
                // 操作区
                actionSection
                
                // 结果区
                if similarity != nil {
                    resultSection
                }
            }
            .navigationTitle("Qwen3 Engine 测试")
        }
    }
    
    // MARK: - Sections
    
    private var modelControlSection: some View {
        Section("模型状态") {
            HStack {
                Circle()
                    .fill(embeddingManager.isModelLoaded ? .green : .gray)
                    .frame(width: 10, height: 10)
                
                Text(embeddingManager.isModelLoaded ? "已加载" : "未加载")
                
                Spacer()
                
                if embeddingManager.isModelLoaded {
                    Text("\(Int(embeddingManager.memoryUsageMB))MB")
                        .foregroundStyle(.secondary)
                }
            }
            
            if let error = embeddingManager.errorMessage ?? calculationError {
                Text(error)
                    .foregroundStyle(.red)
                    .font(.caption)
            }
            
            HStack {
                Button("加载模型") {
                    Task {
                        await embeddingManager.loadModel()
                    }
                }
                .disabled(embeddingManager.isLoading || embeddingManager.isModelLoaded)
                
                Button("卸载模型") {
                    embeddingManager.unloadModel()
                    clearResults()
                }
                .disabled(!embeddingManager.isModelLoaded)
                .foregroundStyle(.red)
            }
            
            if embeddingManager.isLoading {
                ProgressView("加载中...")
            }
        }
    }
    
    private var inputSection: some View {
        Section("输入文本") {
            VStack(alignment: .leading, spacing: 8) {
                Text("文本 1")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextEditor(text: $text1)
                    .frame(minHeight: 60)
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("文本 2")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextEditor(text: $text2)
                    .frame(minHeight: 60)
            }
        }
    }
    
    private var settingsSection: some View {
        Section("设置") {
            Picker("向量维度", selection: $selectedDimension) {
                ForEach(dimensions, id: \.self) { dim in
                    Text("\(dim)").tag(dim)
                }
            }
            
            Toggle("BERT-Whitening", isOn: $useWhitening)
            
            if useWhitening {
                Text("白化处理可对抗各向异性问题")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
    
    private var actionSection: some View {
        Section {
            Button(action: calculateSimilarity) {
                HStack {
                    Spacer()
                    if isCalculating {
                        ProgressView()
                            .padding(.trailing, 8)
                    }
                    Text(isCalculating ? "计算中..." : "计算相似度")
                        .fontWeight(.semibold)
                    Spacer()
                }
            }
            .disabled(!embeddingManager.isModelLoaded || isCalculating || text1.isEmpty || text2.isEmpty)
        }
    }
    
    private var resultSection: some View {
        Section("计算结果") {
            if let sim = similarity {
                HStack {
                    Text("相似度")
                    Spacer()
                    Text(String(format: "%.4f", sim))
                        .fontWeight(.bold)
                        .foregroundStyle(similarityColor(sim))
                }
                
                // 相似度可视化
                ProgressView(value: Double(max(0, min(1, sim))))
                    .tint(similarityColor(sim))
            }
            
            if let time = elapsedTime {
                HStack {
                    Text("耗时")
                    Spacer()
                    Text(String(format: "%.2f ms", time * 1000))
                        .foregroundStyle(.secondary)
                }
            }
            
            // 向量预览
            if let v1 = vector1Preview {
                VStack(alignment: .leading, spacing: 4) {
                    Text("向量 1 预览")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(formatVector(v1))
                        .font(.system(.caption, design: .monospaced))
                }
            }
            
            if let v2 = vector2Preview {
                VStack(alignment: .leading, spacing: 4) {
                    Text("向量 2 预览")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(formatVector(v2))
                        .font(.system(.caption, design: .monospaced))
                }
            }
        }
    }
    
    // MARK: - Actions
    
    private func calculateSimilarity() {
        guard !text1.isEmpty, !text2.isEmpty else { return }
        
        isCalculating = true
        clearResults()
        
        Task {
            let startTime = Date()
            
            do {
                // 现在支持维度参数了
                var v1 = try await embeddingManager.embed(text1, dimension: selectedDimension)
                var v2 = try await embeddingManager.embed(text2, dimension: selectedDimension)
                
                // 可选白化处理
                if useWhitening {
                    let whitened = SimilarityCalculator.whiten([v1, v2])
                    v1 = whitened[0]
                    v2 = whitened[1]
                }
                
                let sim = SimilarityCalculator.cosine(v1, v2)
                let elapsed = Date().timeIntervalSince(startTime)
                
                await MainActor.run {
                    similarity = sim
                    elapsedTime = elapsed
                    vector1Preview = Array(v1.prefix(8))
                    vector2Preview = Array(v2.prefix(8))
                    isCalculating = false
                }
            } catch {
                await MainActor.run {
                    calculationError = error.localizedDescription
                    isCalculating = false
                }
            }
        }
    }
    
    private func clearResults() {
        similarity = nil
        elapsedTime = nil
        vector1Preview = nil
        vector2Preview = nil
        calculationError = nil
    }
    
    // MARK: - Helpers
    
    private func similarityColor(_ value: Float) -> Color {
        if value > 0.7 {
            return .green
        } else if value > 0.3 {
            return .orange
        } else {
            return .red
        }
    }
    
    private func formatVector(_ vector: [Float]) -> String {
        let formatted = vector.map { String(format: "%.3f", $0) }.joined(separator: ", ")
        return "[\(formatted), ...]"
    }
}

#Preview {
    EmbeddingTestView()
}
