/// A type that turns text into a vector for semantic search.
///
/// A memory store calls ``embed(_:)`` for each memory that it saves
/// and for each query that it recalls,
/// then compares vectors by cosine similarity.
///
/// The store saves each vector with the embedder's ``identifier``.
/// When you open a store with an embedder that has a different identifier,
/// the store embeds each memory that has no vector from that embedder.
/// It keeps the vectors from other embedders,
/// so stores with different embedders can share one database.
/// Change the identifier whenever a change to the model or its settings
/// would change the vectors it produces.
///
/// Every vector from one embedder must have the same number of elements.
public protocol Embedder: Sendable {
  /// A string that identifies the model and its version,
  /// such as `"example-model-v2"`.
  var identifier: String { get }

  /// Returns a vector that represents the meaning of `text`.
  func embed(_ text: String) async throws -> [Float]
}
