//
//  Qwen3Embedding.swift
//  textmods
//
//  Qwen3 Transformer 架构模型 definition - 针对 Qwen3-Embedding-0.6B 优化
//

import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Qwen3 Configuration

public struct Qwen3Configuration: Codable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let vocabSize: Int
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int
    
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
    }
}

// MARK: - Qwen3 Components

public class Attention: Module {
    let args: Qwen3Configuration
    let scale: Float
    
    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    
    let rope: RoPE
    
    public init(_ args: Qwen3Configuration) {
        self.args = args
        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)
        
        let dim = args.hiddenSize
        let heads = args.numAttentionHeads
        let kvHeads = args.numKeyValueHeads
        
        // Qwen3-Embedding 权重中不包含非量化 bias
        _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)
        
        // q_norm/k_norm 的维度是 headDim (128)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        
        self.rope = RoPE(dimensions: headDim, traditional: false, base: args.ropeTheta)
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        
        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)
        
        // 1. Reshape 到多头结构 [B, L, H, D]
        queries = queries.reshaped(B, L, args.numAttentionHeads, -1)
        keys = keys.reshaped(B, L, args.numKeyValueHeads, -1)
        values = values.reshaped(B, L, args.numKeyValueHeads, -1)
        
        // 2. 在分头后的维度上应用 q_norm / k_norm (D=128)
        queries = qNorm(queries)
        keys = kNorm(keys)
        
        // 3. 应用 RoPE 并转换形状以优化 Attention 计算 [B, H, L, D]
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)
        
        queries = rope(queries)
        keys = rope(keys)
        
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        
        return wo(output)
    }
}

public class MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    
    public init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

public class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    @ModuleInfo(key: "mlp") var mlp: MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    
    public init(_ args: Qwen3Configuration) {
        _attention.wrappedValue = Attention(args)
        _mlp.wrappedValue = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}

// MARK: - Quantized Components

public class QuantizedEmbedding: Module {
    @ModuleInfo(key: "weight") public var weight: MLXArray // uint32 [V, 128]
    @ModuleInfo(key: "scales") public var scales: MLXArray // float32 [V, 16]
    @ModuleInfo(key: "biases") public var biases: MLXArray // float32 [V, 16]
    
    public override init() {
        _weight.wrappedValue = MLXArray.zeros([1])
        _scales.wrappedValue = MLXArray.zeros([1])
        _biases.wrappedValue = MLXArray.zeros([1])
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, L]
        let shape = x.shape
        let flatX = x.flattened()
        
        // 1. 提取被选中词的压缩权重、scale 和 bias
        // weight: [V, 128] -> [N, 128], where N = B*L
        let w = weight[flatX]
        let s = scales[flatX]
        let b = biases[flatX]
        
        let n = w.dim(0)
        
        // 2. 执行按需（On-the-fly）反量化逻辑 [N, 128] -> [N, 1024]
        var unpackedParts: [MLXArray] = []
        for i in 0..<8 {
            unpackedParts.append((w >> (i * 4)) & 0x0F)
        }
        
        // 重新排列并堆叠：[N, 128, 8] -> [N, 1024]
        let weightCombined = stacked(unpackedParts, axis: -1).reshaped(n, 16, 64)
        
        let sReshaped = s.reshaped(n, 16, 1)
        let bReshaped = b.reshaped(n, 16, 1)
        
        // 执行反量化公式: w * s + b
        let dequantized = weightCombined.asType(DType.float32) * sReshaped + bReshaped
        
        // 3. 恢复原始形状 [B, L, 1024]
        return dequantized.reshaped(shape + [1024])
    }
}

public class Qwen3Model: Module {
    public let args: Qwen3Configuration
    @ModuleInfo(key: "embed_tokens") var embedTokens: QuantizedEmbedding
    @ModuleInfo(key: "layers") var layers: [TransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    
    public init(_ args: Qwen3Configuration) {
        self.args = args
        _embedTokens.wrappedValue = QuantizedEmbedding()
        _layers.wrappedValue = (0..<args.numHiddenLayers).map { _ in TransformerBlock(args) }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }
    
    public func callAsFunction(_ inputIds: MLXArray, targetDimension: Int? = nil) -> MLXArray {
        var h = embedTokens(inputIds)
        
        // 生成因果掩码 (Causal Mask)
        let L = h.dim(1)
        let mask = triu(MLXArray.full([L, L], values: MLXArray(-Float.infinity), type: Float.self), k: 1)
        
        for layer in layers {
            h = layer(h, mask: mask)
        }
        
        var output = norm(h)
        
        // Matryoshka 维度截断及 L2 重归一化
        if let dim = targetDimension, dim < args.hiddenSize {
            output = output[0..., 0..., 0..<dim]
            // 执行重归一化以保持余弦相似度的准确性
            output = output / sqrt(sum(output * output, axis: -1, keepDims: true) + 1e-6)
        }
        
        return output
    }
}
