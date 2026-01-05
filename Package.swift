// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "textmods",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(
            name: "textmods",
            targets: ["textmods"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/mzbac/mlx.embeddings", branch: "main"),
    ],
    targets: [
        .target(
            name: "textmods",
            dependencies: [
                .product(name: "mlx_embeddings", package: "mlx.embeddings"),
            ]
        ),
    ]
)
