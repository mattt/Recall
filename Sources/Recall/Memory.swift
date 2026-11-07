import Foundation

/// A short text record that a user or an agent asked the store to keep.
public struct Memory: Identifiable, Hashable, Sendable, Codable {
  /// The identifier that the store assigned to the memory.
  public let id: UUID

  /// The text of the memory.
  public var text: String

  /// The date when the memory was first remembered.
  ///
  /// Updates don't change this date.
  public let createdAt: Date

  /// The name of the client or tool that wrote the memory, if known.
  public var source: String?

  /// Arbitrary string key-value pairs that a client attaches to the memory.
  ///
  /// The store saves and returns this dictionary
  /// but doesn't search it.
  public var metadata: [String: String]

  /// Creates a memory value.
  ///
  /// You don't usually create memories directly.
  /// Call ``MemoryStore/remember(_:source:metadata:createdAt:)`` instead,
  /// which assigns the identifier and saves the memory.
  public init(
    id: UUID = UUID(),
    text: String,
    createdAt: Date = Date(),
    source: String? = nil,
    metadata: [String: String] = [:]
  ) {
    self.id = id
    self.text = text
    self.createdAt = createdAt
    self.source = source
    self.metadata = metadata
  }
}
