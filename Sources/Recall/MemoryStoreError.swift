import Foundation

/// An error that a memory store throws.
///
/// Errors from an ``Embedder`` pass through unchanged.
public enum MemoryStoreError: Error, Hashable, Sendable {
  /// SQLite returned an error.
  ///
  /// The code is an extended SQLite result code.
  case sqlite(code: Int32, message: String)

  /// The database has a schema version that is newer than this version of Recall supports.
  case unsupportedSchemaVersion(Int)

  /// No memory has the identifier.
  case notFound(UUID)

  /// The text of a memory is empty or contains only whitespace.
  case emptyText

  /// An embedder returned a vector that is empty, has zero length,
  /// contains a value that isn't finite,
  /// or has a different dimension than the embedder's other vectors.
  case invalidEmbedding(embedder: String)

  /// A ``RecallConfiguration`` has a value that can't produce meaningful scores.
  case invalidConfiguration(String)

  /// The stored metadata of a memory isn't a JSON object of strings.
  case invalidMetadata(UUID)
}

extension MemoryStoreError: LocalizedError {
  public var errorDescription: String? {
    switch self {
    case .sqlite(let code, let message):
      "SQLite error \(code): \(message)"
    case .unsupportedSchemaVersion(let version):
      "The database has schema version \(version), which this version of Recall doesn't support."
    case .notFound(let id):
      "No memory has the identifier \(id)."
    case .emptyText:
      "A memory must contain text."
    case .invalidEmbedding(let embedder):
      "The embedder \(embedder) returned a vector that the store can't use."
    case .invalidConfiguration(let reason):
      "The recall configuration isn't valid. \(reason)"
    case .invalidMetadata(let id):
      "The metadata of memory \(id) isn't valid."
    }
  }
}
