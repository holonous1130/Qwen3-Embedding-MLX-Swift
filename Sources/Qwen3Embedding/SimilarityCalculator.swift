//
//  SimilarityCalculator.swift
//  textmods
//
//  相似度计算工具
//

import Foundation

/// 相似度计算工具
public struct SimilarityCalculator {
    
    /// 计算余弦相似度
    /// - Parameters:
    ///   - a: 向量 A
    ///   - b: 向量 B
    /// - Returns: 相似度 [-1, 1]
    public static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        
        var dotProduct: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        
        for i in 0..<a.count {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        
        let denominator = sqrt(normA) * sqrt(normB)
        guard denominator > 0 else { return 0 }
        
        return dotProduct / denominator
    }
    
    /// 计算欧几里得距离
    /// - Note: 用于处理各向异性问题（Gemini 报告建议）
    public static func euclidean(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return Float.infinity }
        
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        
        return sqrt(sum)
    }
    
    /// BERT-Whitening 白化处理（简化版）
    /// 用于对抗嵌入向量的各向异性问题
    public static func whiten(_ vectors: [[Float]]) -> [[Float]] {
        guard !vectors.isEmpty, !vectors[0].isEmpty else { return vectors }
        
        let dim = vectors[0].count
        let n = Float(vectors.count)
        
        // 计算均值
        var mean = [Float](repeating: 0, count: dim)
        for vector in vectors {
            for i in 0..<dim {
                mean[i] += vector[i]
            }
        }
        mean = mean.map { $0 / n }
        
        // 均值中心化
        var centered = vectors.map { vector in
            zip(vector, mean).map { $0 - $1 }
        }
        
        // 计算方差
        var variance = [Float](repeating: 0, count: dim)
        for vector in centered {
            for i in 0..<dim {
                variance[i] += vector[i] * vector[i]
            }
        }
        variance = variance.map { sqrt($0 / n + 1e-8) }
        
        // 方差归一化
        centered = centered.map { vector in
            zip(vector, variance).map { $0 / $1 }
        }
        
        // L2 归一化
        return centered.map { vector in
            let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
            return norm > 0 ? vector.map { $0 / norm } : vector
        }
    }
}
