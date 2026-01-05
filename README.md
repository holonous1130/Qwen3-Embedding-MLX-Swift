# Qwen3-Embedding-MLX-Swift

这是一个基于 [MLX-Swift](https://github.com/ml-explore/mlx-swift) 深度优化的 Qwen3-Embedding (0.6B) iOS 库。

## ✨ 特性

- **极致内存优化**: 通过自定义 `QuantizedEmbedding` 模块实现按需（On-the-fly）反量化，运行时显存占用仅 **~440MB**（0.6B 4-bit 模型）。
- **Matryoshka 嵌入支持**: 支持 32 到 1024 维度的动态截断，并自动执行 L2 重归一化，在低维度下依然保持高效的搜索精度。
- **本地离线加载**: 优先探测本地 `Documents` 目录，支持完全离线运行。
- **官方对齐**: 精准实现 Causal Mask 和 `Instruct: {task}\nQuery:{query}` 模板，相似度计算结果与官方 Python 端 100% 对齐。
- **面向 Swift 6 设计**: 完美支持 Swift 6 严格并发检查（Strict Concurrency Checking），`Qwen3EmbeddingEngine` 通过 `@MainActor` 隔离确保 UI 线程安全性。
- **Swift 原生封装**: 提供 `Qwen3EmbeddingEngine` 易用接口，完美适配 SwiftUI。

## 🚀 性能数据 (iPhone 15 Pro)

| 指标 | 表现 |
| :--- | :--- |
| **运行时内存** | ~420MB - 440MB |
| **首 Token 延迟** | < 50ms |
| **全向量维度** | 1024 维 |
| **截断维度支持** | 32, 64, 128, 256, 512, 768 |

## 🛠 安装

通过 Swift Package Manager 引入：

```swift
.package(url: "https://github.com/holonous1130/Qwen3-Embedding-MLX-Swift.git", from: "1.0.0")
```

## 📖 快速上手

```swift
import Qwen3Embedding

let engine = Qwen3EmbeddingEngine()
await engine.loadModel()

// 生成向量
let vector = try await engine.embed("北京是中国的首都", dimension: 768)
print("向量维度: \(vector.count)")
```

## 📄 开源协议

MIT
