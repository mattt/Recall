import Foundation

/// An error that a memory store throws.
///
/// Errors that an ``Embedder`` throws pass through unchanged,
/// except that ``MemoryStore/reembed()`` collects the errors for single memories
/// in a ``ReembeddingError``.
public enum MemoryStoreError: Error, Hashable, Sendable {
  /// SQLite returned an error.
  ///
  /// The code is an extended SQLite result code.
  case sqlite(code: Int32, message: String)

  /// The database has a schema version that this version of Recall doesn't support,
  /// such as a newer version or an invalid negative one.
  case unsupportedSchemaVersion(Int)

  /// No memory has the identifier.
  case notFound(UUID)

  /// The text of a memory is empty or contains only whitespace.
  case emptyText

  /// An embedder returned a vector that the store can't normalize.
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

/// An error that ``MemoryStore/reembed()`` throws when it can't embed some memories.
///
/// The store throws this error after it embeds every memory that it can,
/// and it keeps the vectors that it saved.
public struct ReembeddingError: Error, Sendable {
  /// A memory that the store couldn't embed.
  public struct Failure: Sendable {
    /// The identifier of the memory.
    public let id: UUID

    /// The error from the embedder.
    public let error: any Error
  }

  /// The number of vectors that the store saved before it threw this error.
  public let embeddedCount: Int

  /// The memories that the store couldn't embed, in the order that the store tried them.
  public let failures: [Failure]

  /// The number of memories that the store couldn't embed.
  public var count: Int { failures.count }
}

extension ReembeddingError: LocalizedError {
  public var errorDescription: String? {
    let listed = failures.prefix(10).map(\.id.uuidString).joined(separator: ", ")
    let remaining = failures.count - min(failures.count, 10)
    let noun = failures.count == 1 ? "memory" : "memories"
    let suffix = remaining > 0 ? ", and \(remaining) more" : ""
    return "The embedder couldn't embed \(failures.count) \(noun): \(listed)\(suffix)."
  }
}
