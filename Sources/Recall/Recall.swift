import Foundation

/// Settings that control how ``MemoryStore/recall(_:limit:configuration:)`` ranks memories.
///
/// Recall builds up to three ranked lists of memories
/// and merges them with weighted reciprocal rank fusion.
/// A memory at rank `r` in a list with weight `w`
/// gets `w / (k + r)` from that list,
/// where ranks start at 1.
/// The fused score of a memory is the sum over the lists that contain it.
///
/// The lists are:
///
/// - term Keyword: Memories that contain words from the query, ordered by BM25.
/// - term Vector: Memories ordered by the cosine similarity of their vectors to the query's vector.
///   The store makes this list only if it has an ``Embedder``.
/// - term Recency: The memories from the other two lists, newest first.
///   This list only reorders the memories that the query found;
///   it doesn't add memories.
///
/// A list with a weight of zero isn't made,
/// and its rank in each result is `nil`.
public struct RecallConfiguration: Hashable, Sendable {
  /// The constant that controls how fast a list's contribution falls with rank.
  ///
  /// A larger value makes the difference between adjacent ranks smaller.
  /// The default is 60.
  public var k: Double

  /// The weight of the keyword list.
  public var keywordWeight: Double

  /// The weight of the vector list.
  public var vectorWeight: Double

  /// The weight of the recency list.
  public var recencyWeight: Double

  /// The maximum number of memories that the keyword and vector lists each contain.
  public var candidateLimit: Int

  /// The cosine similarity that a memory needs to be in the vector list,
  /// or `nil` to allow every memory with a vector.
  public var minimumSimilarity: Float?

  /// Creates a configuration.
  ///
  /// With the default weights,
  /// the keyword and vector lists count equally,
  /// and recency counts half as much as either.
  /// Set `recencyWeight` to zero to ignore creation dates.
  public init(
    k: Double = 60,
    keywordWeight: Double = 1,
    vectorWeight: Double = 1,
    recencyWeight: Double = 0.5,
    candidateLimit: Int = 50,
    minimumSimilarity: Float? = nil
  ) {
    self.k = k
    self.keywordWeight = keywordWeight
    self.vectorWeight = vectorWeight
    self.recencyWeight = recencyWeight
    self.candidateLimit = candidateLimit
    self.minimumSimilarity = minimumSimilarity
  }

  /// Throws if a value can't produce meaningful scores.
  func validate() throws(MemoryStoreError) {
    guard k.isFinite, k >= 0 else {
      throw .invalidConfiguration("k must be a finite number that isn't negative.")
    }
    for weight in [keywordWeight, vectorWeight, recencyWeight] {
      guard weight.isFinite, weight >= 0 else {
        throw .invalidConfiguration("Each weight must be a finite number that isn't negative.")
      }
    }
    guard candidateLimit > 0 else {
      throw .invalidConfiguration("candidateLimit must be greater than zero.")
    }
  }
}

/// A memory that recall found, with the information that explains its position.
public struct RecallResult: Identifiable, Hashable, Sendable {
  /// The memory.
  public var memory: Memory

  /// The fused score: the sum of `weight / (k + rank)` over the lists that contain the memory.
  ///
  /// Results are sorted by this score, highest first.
  public var score: Double

  /// The rank of the memory in the keyword list, starting at 1,
  /// or `nil` if the list doesn't contain it.
  public var keywordRank: Int?

  /// The rank of the memory in the vector list, starting at 1,
  /// or `nil` if the list doesn't contain it.
  public var vectorRank: Int?

  /// The rank of the memory in the recency list, starting at 1,
  /// or `nil` if the list doesn't contain it.
  public var recencyRank: Int?

  /// The cosine similarity of the memory's vector to the query's vector,
  /// or `nil` if the vector list doesn't contain the memory.
  public var similarity: Float?

  /// The identifier of the memory.
  public var id: UUID { memory.id }

  /// Creates a result.
  public init(
    memory: Memory,
    score: Double,
    keywordRank: Int? = nil,
    vectorRank: Int? = nil,
    recencyRank: Int? = nil,
    similarity: Float? = nil
  ) {
    self.memory = memory
    self.score = score
    self.keywordRank = keywordRank
    self.vectorRank = vectorRank
    self.recencyRank = recencyRank
    self.similarity = similarity
  }
}

/// Weighted reciprocal rank fusion.
enum Fusion {
  /// Returns the fused score of each identifier in `lists`.
  ///
  /// The identifier at index `i` of a list has rank `i + 1`
  /// and gets `weight / (k + i + 1)` from that list.
  static func scores<ID: Hashable>(
    _ lists: [(weight: Double, ids: [ID])],
    k: Double
  ) -> [ID: Double] {
    var scores: [ID: Double] = [:]
    for list in lists {
      for (index, id) in list.ids.enumerated() {
        scores[id, default: 0] += list.weight / (k + Double(index + 1))
      }
    }
    return scores
  }

  /// Returns the rank, starting at 1, of each identifier in `ids`.
  static func ranks<ID: Hashable>(_ ids: [ID]) -> [ID: Int] {
    var ranks: [ID: Int] = [:]
    for (index, id) in ids.enumerated() where ranks[id] == nil {
      ranks[id] = index + 1
    }
    return ranks
  }
}

extension MemoryStore {
  /// Finds the memories that best match a query.
  ///
  /// The store ranks memories by keywords,
  /// by vector similarity if it has an embedder,
  /// and by recency,
  /// then merges the lists as ``RecallConfiguration`` describes.
  /// Before a vector search,
  /// the store embeds any memories that have no vector from the current embedder.
  ///
  /// - Parameters:
  ///   - query: Text that describes what to recall.
  ///   - limit: The maximum number of results.
  ///   - configuration: The settings for ranking.
  /// - Returns: Results with the highest score first.
  ///   Equal scores put newer memories first.
  public func recall(
    _ query: String,
    limit: Int = 10,
    configuration: RecallConfiguration = RecallConfiguration()
  ) async throws -> [RecallResult] {
    try configuration.validate()
    guard limit > 0 else { return [] }

    let keyword =
      configuration.keywordWeight > 0
      ? try keywordSearch(query, limit: configuration.candidateLimit) : []
    let vector =
      configuration.vectorWeight > 0
      ? try await vectorSearch(
        query,
        limit: configuration.candidateLimit,
        minimumSimilarity: configuration.minimumSimilarity
      ) : []

    // Load each candidate once.
    // A memory that was forgotten during the vector search is skipped.
    var memories: [UUID: Memory] = [:]
    for id in keyword + vector.map(\.id) where memories[id] == nil {
      memories[id] = try memory(id)
    }

    let recency =
      configuration.recencyWeight > 0
      ? memories.values.sorted(by: Self.newerFirst).map(\.id) : []

    let keywordRanks = Fusion.ranks(keyword)
    let vectorRanks = Fusion.ranks(vector.map(\.id))
    let recencyRanks = Fusion.ranks(recency)
    let similarities = Dictionary(vector.map { ($0.id, $0.similarity) }) { first, _ in first }
    let scores = Fusion.scores(
      [
        (configuration.keywordWeight, keyword),
        (configuration.vectorWeight, vector.map(\.id)),
        (configuration.recencyWeight, recency),
      ],
      k: configuration.k
    )

    let results = memories.values.map { memory in
      RecallResult(
        memory: memory,
        score: scores[memory.id] ?? 0,
        keywordRank: keywordRanks[memory.id],
        vectorRank: vectorRanks[memory.id],
        recencyRank: recencyRanks[memory.id],
        similarity: similarities[memory.id]
      )
    }
    return Array(
      results.sorted { left, right in
        left.score != right.score
          ? left.score > right.score : Self.newerFirst(left.memory, right.memory)
      }
      .prefix(limit)
    )
  }

  /// Orders memories newest first,
  /// and by identifier when the dates are equal,
  /// so that the order is stable.
  static func newerFirst(_ left: Memory, _ right: Memory) -> Bool {
    left.createdAt != right.createdAt
      ? left.createdAt > right.createdAt : left.id.uuidString < right.id.uuidString
  }
}
