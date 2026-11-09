import Foundation

/// An error that a memory store throws.
///
/// Errors that an ``Embedder`` throws pass through unchanged.
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
  /// or contains a value that isn't finite.
  case invalidEmbedding(embedder: String)

  /// An embedder returned a vector with a different number of elements
  /// than the vectors that the store saved from it.
  ///
  /// The store throws this error when it saves a memory's vector
  /// and when it compares a query's vector with the saved vectors.
  case dimensionMismatch(embedder: String, expected: Int, actual: Int)

  /// A ``RecallConfiguration`` has a value that can't produce meaningful scores.
  case invalidConfiguration(RecallConfiguration.Problem)

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
    case .dimensionMismatch(let embedder, let expected, let actual):
      "The embedder \(embedder) returned a vector with \(actual) elements instead of \(expected)."
    case .invalidConfiguration(let problem):
      "The recall configuration isn't valid. \(problem.description)"
    case .invalidMetadata(let id):
      "The metadata of memory \(id) isn't valid."
    }
  }
}
