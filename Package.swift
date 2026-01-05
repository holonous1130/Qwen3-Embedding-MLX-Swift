// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Qwen3Embedding",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(
            name: "Qwen3Embedding",
            targets: ["Qwen3Embedding"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.29.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.0"),
    ],
    targets: [
        .target(
            name: "Qwen3Embedding",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
    ]
)
