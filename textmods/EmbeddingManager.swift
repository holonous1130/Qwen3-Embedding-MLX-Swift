//
//  EmbeddingManager.swift
//  textmods
//
//  Qwen3-Embedding 完整模型管理器
//  基于 mlx-swift + swift-transformers 手动实现完整 Transformer 架构
//

import Foundation
import Combine
import MLX
import MLXNN
import MLXFast
import Tokenizers
import Hub

// MARK: - EmbeddingManager

public class Qwen3EmbeddingEngine: ObservableObject {
    
    // --- 库级公共属性 ---
    public let modelId = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
    
    /// 默认指令（如：'Given a web search query, retrieve relevant passages that answer the query'）
    public var defaultInstruction: String?
    
    /// 目标向量维度（支持 Matryoshka 截断，如 512, 768, 1024）
    public var targetDimension: Int = 1024
    
    @Published public var isModelLoaded = false
    @Published public var isLoading = false
    @Published public var loadingProgress = ""
    @Published public var errorMessage: String?
    @Published public var memoryUsageMB: Int = 0
    
    private var model: Qwen3Model?
    private var tokenizer: (any Tokenizer)?
    
    public init() {}
    
    // MARK: - 公共方法
    
    public func loadModel() async {
        guard !isLoading && !isModelLoaded else { return }
        
        await MainActor.run {
            self.isLoading = true
            self.errorMessage = nil
            self.loadingProgress = "正在初始化..."
        }
        
        do {
            let hub = HubApi()
            let repo = Hub.Repo(id: modelId)
            
            // --- 离线加载逻辑：优先检查 Documents 目录 ---
            let fileManager = FileManager.default
            let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
            // 将 "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ" 转换为文件夹名，或者直接探测
            let localModelFolder = documentsURL.appendingPathComponent("Qwen3-Embedding-0.6B-4bit-DWQ")
            
            var modelFolder: URL
            if fileManager.fileExists(atPath: localModelFolder.path) {
                print("发现本地模型目录，切换至离线模式: \(localModelFolder.path)")
                modelFolder = localModelFolder
                await MainActor.run { self.loadingProgress = "正在从本地 Documents 加载..." }
            } else {
                await MainActor.run { self.loadingProgress = "正在读取配置 (HuggingFace)..." }
                modelFolder = try await hub.snapshot(from: repo, matching: ["config.json", "tokenizer.json", "tokenizer_config.json"])
            }
            // ----------------------------------------
            
            let configURL = modelFolder.appendingPathComponent("config.json")
            if !fileManager.fileExists(atPath: configURL.path) {
                throw NSError(domain: "Qwen3EmbeddingEngine", code: 404, userInfo: [NSLocalizedDescriptionKey: "未找到 config.json，请检查模型文件夹结构"])
            }
            
            let configData = try Data(contentsOf: configURL)
            let config = try JSONDecoder().decode(Qwen3Configuration.self, from: configData)
            
            await MainActor.run { self.loadingProgress = "正在加载 Tokenizer..." }
            tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
            
            await MainActor.run { self.loadingProgress = "正在构建模型结构..." }
            let qwenModel = Qwen3Model(config)
            
            await MainActor.run { self.loadingProgress = "正在查找模型权重..." }
            
            // 搜索 safetensors 文件
            var safetensorsFiles: [URL] = []
            let enumerator = fileManager.enumerator(at: modelFolder, includingPropertiesForKeys: nil)
            while let fileURL = enumerator?.nextObject() as? URL {
                if fileURL.pathExtension == "safetensors" {
                    safetensorsFiles.append(fileURL)
                }
            }
            
            if safetensorsFiles.isEmpty && !fileManager.fileExists(atPath: localModelFolder.path) {
                 // 如果本地没搜到，且还没试过 Hub 搜，则试一下 Hub
                 await MainActor.run { self.loadingProgress = "本地无权重，尝试从云端下载..." }
                 let hubWeights = try await hub.snapshot(from: repo, matching: ["*.safetensors"])
                 let hubEnumerator = fileManager.enumerator(at: hubWeights, includingPropertiesForKeys: nil)
                 while let fileURL = hubEnumerator?.nextObject() as? URL {
                     if fileURL.pathExtension == "safetensors" {
                         safetensorsFiles.append(fileURL)
                     }
                 }
            }

            var weights: [String: MLXArray] = [:]
            for fileURL in safetensorsFiles {
                let fileWeights = try MLX.loadArrays(url: fileURL)
                for (key, value) in fileWeights {
                    var newKey = key
                    if newKey.hasPrefix("model.") {
                        newKey = String(newKey.dropFirst(6))
                    }
                    weights[newKey] = value
                }
            }
            
            if weights.isEmpty {
                throw NSError(domain: "Qwen3EmbeddingEngine", code: 404, userInfo: [NSLocalizedDescriptionKey: "未找到权重文件 (.safetensors)"])
            }
            
            await MainActor.run { self.loadingProgress = "正在应用权重..." }
            
            // 检测是否为量化模型 (检查键中是否包含 scales)
            let isQuantized = weights.keys.contains { $0.contains(".scales") }
            if isQuantized {
                print("检测到量化模型，正在转换层结构...")
                // 根据 config.json, Qwen3-Embedding-0.6B 使用 groupSize 64
                QuantizedLinear.quantize(model: qwenModel, groupSize: 64, bits: 4)
            }
            
            qwenModel.update(parameters: ModuleParameters.unflattened(weights))
            self.model = qwenModel
            
            await MainActor.run {
                self.isModelLoaded = true
                self.memoryUsageMB = 440
                self.loadingProgress = "加载完成"
                self.isLoading = false
            }
            
        } catch {
            print("模型加载失败详情: \(error)")
            // 尝试获取更具体的解码错误信息
            let detailedError: String
            if let decodingError = error as? DecodingError {
                switch decodingError {
                case .keyNotFound(let key, _): detailedError = "缺少字段: \(key.stringValue)"
                case .typeMismatch(let type, let context): detailedError = "类型不匹配: \(type), 路径: \(context.codingPath.map { $0.stringValue }.joined(separator: "."))"
                default: detailedError = error.localizedDescription
                }
            } else {
                detailedError = error.localizedDescription
            }
            
            await MainActor.run {
                self.errorMessage = "模型加载失败: \(detailedError)"
                self.loadingProgress = ""
                self.isLoading = false
            }
        }
    }
    
    public func unloadModel() {
        model = nil
        tokenizer = nil
        isModelLoaded = false
        memoryUsageMB = 0
        loadingProgress = ""
    }
    
    /// 生成词嵌入向量
    /// - Parameters:
    ///   - text: 输入文本
    ///   - overrideInstruction: 可选的指令覆盖（若为 nil 则使用 defaultInstruction）
    ///   - overrideDimension: 可选的维度覆盖（若为 nil 则使用 targetDimension）
    public func embed(_ text: String, instruction: String? = nil, dimension: Int? = nil) async throws -> [Float] {
        guard let model = model, let tokenizer = tokenizer else {
            throw NSError(domain: "Qwen3EmbeddingEngine", code: 401, userInfo: [NSLocalizedDescriptionKey: "模型未加载"])
        }
        
        let activeInstruction = instruction ?? defaultInstruction
        let activeDimension = dimension ?? targetDimension
        
        // 遵循官方格式: Instruct: {task}\nQuery:{query}
        let inputText = activeInstruction != nil ? "Instruct: \(activeInstruction!)\nQuery:\(text)" : text
        let tokens = tokenizer.encode(text: inputText)
        guard !tokens.isEmpty else { return [] }
        
        let inputIds = MLXArray(tokens).reshaped(1, -1)
        
        // 推理并应用可能的维度截断
        let lastHiddenStates = model(inputIds, targetDimension: activeDimension)
        
        // 池化策略: Qwen3-Embedding 使用 Last Token Pool
        let embedding = lastHiddenStates[0, -1]
        
        // 执行 eval 确保计算完成并同步回 CPU
        MLX.eval(embedding)
        
        return embedding.asType(DType.float32).asArray(Float.self)
    }
    
    private func updateMemoryUsage() {
        memoryUsageMB = 420
    }
}

// MARK: - Error Types

public enum EmbeddingError: LocalizedError {
    case modelNotLoaded
    case embeddingFailed
    case nanDetected
    
    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "模型未加载"
        case .embeddingFailed: return "向量生成失败"
        case .nanDetected: return "检测到 NaN 值"
        }
    }
}
