import Foundation
import MLX
import Qwen3Embedding

/// 情绪词分条目
public struct EmotionEntry: Codable, Identifiable {
    public var id: String { label }
    public let label: String
    public let pleasure: Float
    public let arousal: Float
    public let dominance: Float
    public var embedding: [Float]?
    
    public init(label: String, pleasure: Float, arousal: Float, dominance: Float, embedding: [Float]? = nil) {
        self.label = label
        self.pleasure = pleasure
        self.arousal = arousal
        self.dominance = dominance
        self.embedding = embedding
    }
}

/// 情绪向量数据库
@MainActor
public class EmotionVectorStore: ObservableObject {
    
    @Published public var entries: [EmotionEntry] = []
    @Published public var isIndexing = false
    @Published public var indexProgress: Double = 0
    
    private let engine: Qwen3EmbeddingEngine
    private let fileManager = FileManager.default
    
    public init(engine: Qwen3EmbeddingEngine) {
        self.engine = engine
    }
    
    /// 从 CSV 加载原始数据
    public func loadFromCSV(url: URL) throws {
        let content = try String(contentsOf: url)
        let lines = content.components(separatedBy: .newlines)
        
        var newEntries: [EmotionEntry] = []
        
        // 跳过表头 Label,Pleasure,Arousal,Dominance
        for i in 1..<lines.count {
            let line = lines[i].trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty { continue }
            
            let parts = line.components(separatedBy: ",")
            if parts.count >= 4 {
                let label = parts[0]
                let p = Float(parts[1]) ?? 0
                let a = Float(parts[2]) ?? 0
                let d = Float(parts[3]) ?? 0
                
                newEntries.append(EmotionEntry(label: label, pleasure: p, arousal: a, dominance: d))
            }
        }
        
        self.entries = newEntries
        print("CSV 加载完成，共 \(entries.count) 条记录")
    }
    
    /// 构建向量索引（注意：14k 条数据建议分批处理或加载预热向量）
    public func buildIndex() async {
        guard !isIndexing else { return }
        isIndexing = true
        
        let total = entries.count
        for i in 0..<total {
            if entries[i].embedding != nil { continue }
            
            do {
                let vector = try await engine.embed(entries[i].label)
                entries[i].embedding = vector
            } catch {
                print("词条 [\(entries[i].label)] 嵌入失败: \(error)")
            }
            
            if i % 10 == 0 {
                indexProgress = Double(i) / Double(total)
            }
        }
        
        isIndexing = false
        indexProgress = 1.0
        saveToCache()
    }
    
    /// 模糊搜索最匹配的情绪词（基于向量余弦相似度）
    public func search(query: String, topK: Int = 5) async throws -> [(EmotionEntry, Float)] {
        let queryVector = try await engine.embed(query)
        let qArray = MLXArray(queryVector)
        
        var results: [(EmotionEntry, Float)] = []
        
        // 提取已有向量的词条
        let validEntries = entries.filter { $0.embedding != nil }
        guard !validEntries.isEmpty else { return [] }
        
        // 构造矩阵进行批量运算 [N, Dim]
        let allEmbeds = validEntries.compactMap { $0.embedding }
        let matrix = MLXArray(allEmbeds.flatMap { $0 }).reshaped(allEmbeds.count, -1)
        
        // 计算余弦相似度: (A · B) / (|A| * |B|)
        // 假设模型输出已归一化，则只需计算点积
        let scores = (matrix * qArray).sum(axis: -1)
        MLX.eval(scores)
        
        let scoreArray = scores.asArray(Float.self)
        
        for (idx, score) in scoreArray.enumerated() {
            results.append((validEntries[idx], score))
        }
        
        // 按分数排序并返回 TopK
        return results.sorted { $0.1 > $1.1 }.prefix(topK).map { $0 }
    }
    
    // MARK: - 持久化逻辑
    
    private var cacheURL: URL {
        let docs = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("emotion_vectors.bin")
    }
    
    public func saveToCache() {
        do {
            let data = try JSONEncoder().encode(entries)
            try data.write(to: cacheURL)
            print("向量索引已保存至: \(cacheURL.path)")
        } catch {
            print("保存缓存失败: \(error)")
        }
    }
    
    public func loadFromCache() -> Bool {
        guard fileManager.fileExists(atPath: cacheURL.path) else { return false }
        do {
            let data = try Data(contentsOf: cacheURL)
            let decoded = try JSONDecoder().decode([EmotionEntry].self, from: data)
            self.entries = decoded
            print("从缓存加载了 \(entries.count) 条已索引记录")
            return true
        } catch {
            print("加载缓存失败: \(error)")
            return false
        }
    }
}
