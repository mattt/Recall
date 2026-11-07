// swift-tools-version: 6.0

import PackageDescription

let package = Package(
  name: "Recall",
  platforms: [
    .macOS(.v14),
    .iOS(.v17),
  ],
  products: [
    .library(name: "Recall", targets: ["Recall"])
  ],
  targets: [
    // Apple platforms provide the `SQLite3` module.
    // On Linux, this system library target provides the same C API.
    .systemLibrary(
      name: "CSQLite",
      pkgConfig: "sqlite3",
      providers: [
        .apt(["libsqlite3-dev"]),
        .yum(["sqlite-devel"]),
      ]
    ),
    .target(
      name: "Recall",
      dependencies: [
        .target(name: "CSQLite", condition: .when(platforms: [.linux]))
      ]
    ),
    .testTarget(
      name: "RecallTests",
      dependencies: [
        "Recall",
        .target(name: "CSQLite", condition: .when(platforms: [.linux])),
      ]
    ),
  ],
  swiftLanguageModes: [.v6]
)
