/// The use of a text that an ``Embedder`` embeds.
///
/// Many retrieval models embed queries and documents differently,
/// usually by adding a prefix such as `"query: "` or `"passage: "` to the text.
/// An embedder for such a model uses the purpose to choose the prefix.
public enum EmbeddingPurpose: Sendable, Hashable {
  /// Text to search with, such as the query of a recall.
  case query

  /// Text to search for, such as the text of a memory.
  case document
}

/// A type that turns text into a vector for semantic search.
///
/// A memory store calls ``embed(_:for:)`` with ``EmbeddingPurpose/document``
/// for each memory that it saves or embeds again,
/// and with ``EmbeddingPurpose/query`` for each query that it recalls,
/// then compares vectors by cosine similarity.
/// The store keeps the vectors of documents and never stores the vectors of queries.
///
/// The store saves each vector with the embedder's ``identifier``.
/// When you open a store with an embedder that has a different identifier,
/// the store embeds each memory that has no vector from that embedder.
/// It keeps the vectors from other embedders,
/// so stores with different embedders can share one database.
/// Change the identifier whenever a change to the model or its settings
/// would change the vectors of documents,
/// including a change to the prefix for documents.
/// A change that affects only query vectors doesn't need a new identifier,
/// because the store makes a new query vector for each recall.
///
/// Every vector from one embedder must have the same number of elements,
/// for both purposes.
public protocol Embedder: Sendable {
  /// A string that identifies the model and its version,
  /// such as `"example-model-v2"`.
  var identifier: String { get }

  /// Returns a vector that represents the meaning of `text`.
  ///
  /// - Parameters:
  ///   - text: The text to embed.
  ///   - purpose: Whether `text` is a query or a document to search.
  func embed(_ text: String, for purpose: EmbeddingPurpose) async throws -> [Float]
}
