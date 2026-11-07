import Foundation

/// A store of memories in a SQLite database.
///
/// A store is an actor,
/// so you can share one instance between tasks.
/// Each store has its own connection to the database.
/// Two stores can open the same file;
/// SQLite serializes their writes.
///
/// If you supply an ``Embedder``,
/// the store also keeps a vector for each memory
/// and uses it for semantic search.
public actor MemoryStore {
  let database: Database

  /// The embedder that the store uses for vector search, if any.
  public nonisolated let embedder: (any Embedder)?

  /// The task that is embedding stale memories, if one is running.
  private var reembedding: Task<Int, any Error>?

  /// Opens or creates a store in the SQLite file at `url`.
  ///
  /// If the file has an older schema,
  /// the store migrates it to the current schema.
  ///
  /// - Parameters:
  ///   - url: A file URL for the database.
  ///   - embedder: The embedder for vector search,
  ///     or `nil` to use keyword search only.
  /// - Throws: ``MemoryStoreError/unsupportedSchemaVersion(_:)``
  ///   if a newer version of Recall created the file,
  ///   or ``MemoryStoreError/sqlite(code:message:)``
  ///   if SQLite can't open it.
  public init(url: URL, embedder: (any Embedder)? = nil) throws {
    try self.init(database: Database(url: url), isFile: true, embedder: embedder)
  }

  /// Creates a store in a private in-memory database.
  ///
  /// The memories exist only as long as the store does.
  ///
  /// - Parameter embedder: The embedder for vector search,
  ///   or `nil` to use keyword search only.
  public static func inMemory(embedder: (any Embedder)? = nil) throws -> MemoryStore {
    try MemoryStore(database: Database(url: nil), isFile: false, embedder: embedder)
  }

  init(
    database: Database,
    isFile: Bool,
    embedder: (any Embedder)?
  ) throws(MemoryStoreError) {
    self.embedder = embedder
    if isFile {
      try database.execute("PRAGMA journal_mode = WAL")
    }
    try database.execute("PRAGMA foreign_keys = ON")
    try Schema.migrate(database)
    self.database = database
  }

  // MARK: - Writing

  /// Saves a new memory.
  ///
  /// - Parameters:
  ///   - text: The text to remember.
  ///   - source: The name of the client or tool that writes the memory.
  ///   - metadata: String key-value pairs to keep with the memory.
  ///   - createdAt: The creation date of the memory.
  ///     Supply a date to import memories from elsewhere.
  /// - Returns: The saved memory.
  /// - Throws: ``MemoryStoreError/emptyText`` if `text` contains only whitespace,
  ///   or any error from the embedder.
  ///   If the embedder fails, the store doesn't save the memory.
  @discardableResult
  public func remember(
    _ text: String,
    source: String? = nil,
    metadata: [String: String] = [:],
    createdAt: Date = Date()
  ) async throws -> Memory {
    try Self.validate(text)
    let memory = Memory(text: text, createdAt: createdAt, source: source, metadata: metadata)
    let vector = try await embedding(for: text)
    try database.transaction { () throws(MemoryStoreError) in
      let rowid = try insert(memory)
      if let vector { try saveEmbedding(vector, rowid: rowid) }
    }
    return memory
  }

  /// Saves changes to the text, source, or metadata of a memory.
  ///
  /// The identifier and creation date don't change.
  ///
  /// - Parameter memory: The memory with its new values.
  /// - Returns: The saved memory.
  /// - Throws: ``MemoryStoreError/notFound(_:)`` if the store has no memory with that identifier,
  ///   or any error from the embedder.
  ///   If the embedder fails, the store doesn't save the change.
  @discardableResult
  public func update(_ memory: Memory) async throws -> Memory {
    try Self.validate(memory.text)
    guard let existing = try self.memory(memory.id) else {
      throw MemoryStoreError.notFound(memory.id)
    }
    let vector = existing.text == memory.text ? nil : try await embedding(for: memory.text)
    try database.transaction { () throws(MemoryStoreError) in
      let rowid = try save(memory)
      if let vector { try saveEmbedding(vector, rowid: rowid) }
    }
    return try self.memory(memory.id) ?? memory
  }

  /// Replaces the text of a memory.
  ///
  /// - Returns: The saved memory.
  /// - Throws: ``MemoryStoreError/notFound(_:)`` if the store has no memory with that identifier.
  @discardableResult
  public func update(_ id: UUID, text: String) async throws -> Memory {
    guard var memory = try memory(id) else { throw MemoryStoreError.notFound(id) }
    memory.text = text
    return try await update(memory)
  }

  /// Deletes a memory.
  ///
  /// - Returns: `true` if the store had the memory, or `false` if it didn't.
  @discardableResult
  public func forget(_ id: UUID) throws -> Bool {
    try database.prepare("DELETE FROM memories WHERE id = ?")
      .bind(.text(id.uuidString))
      .step()
    return database.changes > 0
  }

  // MARK: - Reading

  /// Returns the memory with an identifier, or `nil` if there isn't one.
  public func memory(_ id: UUID) throws -> Memory? {
    let statement = try database.prepare(
      "SELECT \(Self.columns) FROM memories m WHERE m.id = ?"
    )
    try statement.bind(.text(id.uuidString))
    guard try statement.step() else { return nil }
    return try Self.memory(from: statement)
  }

  /// Returns memories, newest first.
  ///
  /// - Parameters:
  ///   - source: If not `nil`, only memories with this source.
  ///   - limit: The maximum number of memories to return, or `nil` for all.
  ///   - offset: The number of memories to skip.
  public func list(source: String? = nil, limit: Int? = nil, offset: Int = 0) throws -> [Memory] {
    let statement = try database.prepare(
      """
      SELECT \(Self.columns) FROM memories m
      WHERE ?1 IS NULL OR m.source = ?1
      ORDER BY m.created_at DESC, m.rowid DESC
      LIMIT ?2 OFFSET ?3
      """
    )
    try statement.bind(.optionalText(source), .integer(limit ?? -1), .integer(max(offset, 0)))
    var memories: [Memory] = []
    while try statement.step() {
      memories.append(try Self.memory(from: statement))
    }
    return memories
  }

  /// The number of memories in the store.
  public func count() throws -> Int {
    let statement = try database.prepare("SELECT count(*) FROM memories")
    try statement.step()
    return statement.int(at: 0)
  }

  // MARK: - Keyword search

  /// Returns the identifiers of memories that match the words in `query`,
  /// best match first by BM25.
  func keywordSearch(_ query: String, limit: Int) throws -> [UUID] {
    guard limit > 0, let match = KeywordQuery.match(for: query) else { return [] }
    let statement = try database.prepare(
      """
      SELECT m.id FROM memories_fts
      JOIN memories m ON m.rowid = memories_fts.rowid
      WHERE memories_fts MATCH ?
      ORDER BY bm25(memories_fts), m.created_at DESC, m.rowid DESC
      LIMIT ?
      """
    )
    try statement.bind(.text(match), .integer(limit))
    var ids: [UUID] = []
    while try statement.step() {
      if let id = statement.text(at: 0).flatMap(UUID.init(uuidString:)) {
        ids.append(id)
      }
    }
    return ids
  }

  // MARK: - Vectors

  /// Embeds each memory that has no vector from the current embedder,
  /// and returns the number of memories that it embedded.
  ///
  /// Recall calls this method before a vector search,
  /// so you don't need to call it.
  /// Call it after you open a store with a new embedder
  /// to do the work before the first recall.
  ///
  /// If another call is already embedding memories,
  /// this call waits for it and returns the same count.
  @discardableResult
  public func reembed() async throws -> Int {
    guard let embedder else { return 0 }
    if let reembedding { return try await reembedding.value }
    let task = Task { try await embedStaleMemories(with: embedder) }
    reembedding = task
    defer { reembedding = nil }
    return try await task.value
  }

  /// The number of memories that have no vector from the current embedder.
  public func staleEmbeddingCount() throws -> Int {
    guard let embedder else { return 0 }
    let statement = try database.prepare(
      """
      SELECT count(*) FROM memories m
      LEFT JOIN embeddings e ON e.memory_rowid = m.rowid
      WHERE e.memory_rowid IS NULL OR e.embedder IS NOT ?
      """
    )
    try statement.bind(.text(embedder.identifier))
    try statement.step()
    return statement.int(at: 0)
  }

  private func embedStaleMemories(with embedder: any Embedder) async throws -> Int {
    let batchSize = 64
    var embedded = 0
    // Rows that fail are skipped so that the loop ends;
    // a later call tries them again.
    var lastRowid = 0
    while true {
      let statement = try database.prepare(
        """
        SELECT m.rowid, m.text FROM memories m
        LEFT JOIN embeddings e ON e.memory_rowid = m.rowid
        WHERE (e.memory_rowid IS NULL OR e.embedder IS NOT ?) AND m.rowid > ?
        ORDER BY m.rowid
        LIMIT ?
        """
      )
      try statement.bind(.text(embedder.identifier), .integer(lastRowid), .integer(batchSize))
      var batch: [(rowid: Int, text: String)] = []
      while try statement.step() {
        batch.append((statement.int(at: 0), statement.text(at: 1) ?? ""))
      }
      guard !batch.isEmpty else { return embedded }
      for row in batch {
        lastRowid = row.rowid
        let vector = try await normalizedEmbedding(for: row.text, with: embedder)
        // The text can change while the embedder runs.
        // Save the vector only if it still describes the memory.
        let current = try database.prepare("SELECT text FROM memories WHERE rowid = ?")
        try current.bind(.integer(row.rowid))
        guard try current.step(), current.text(at: 0) == row.text else { continue }
        try saveEmbedding(vector, rowid: row.rowid)
        embedded += 1
      }
    }
  }

  /// Returns the identifiers of the memories whose vectors are most similar to the vector of `query`,
  /// most similar first, with their cosine similarities.
  func vectorSearch(
    _ query: String,
    limit: Int,
    minimumSimilarity: Float? = nil
  ) async throws -> [(id: UUID, similarity: Float)] {
    guard limit > 0, let embedder else { return [] }
    try await reembed()
    guard let vector = try await embedding(for: query) else { return [] }
    let results = try nearest(to: vector, embedder: embedder.identifier, limit: limit)
    guard let minimumSimilarity else { return results }
    return results.filter { $0.similarity >= minimumSimilarity }
  }

  /// Returns the memories nearest to a unit vector by exact cosine similarity.
  private func nearest(
    to query: [Float],
    embedder: String,
    limit: Int
  ) throws(MemoryStoreError) -> [(id: UUID, similarity: Float)] {
    let statement = try database.prepare(
      """
      SELECT m.id, e.vector FROM embeddings e
      JOIN memories m ON m.rowid = e.memory_rowid
      WHERE e.embedder = ? AND e.dimension = ?
      """
    )
    try statement.bind(.text(embedder), .integer(query.count))
    var scored: [(id: UUID, similarity: Float)] = []
    while try statement.step() {
      guard let id = statement.text(at: 0).flatMap(UUID.init(uuidString:)),
        let vector = Vector.decode(statement.data(at: 1), dimension: query.count)
      else { continue }
      scored.append((id, Vector.dot(query, vector)))
    }
    scored.sort { $0.similarity > $1.similarity }
    return Array(scored.prefix(limit))
  }

  /// Returns the unit vector for `text` from the store's embedder,
  /// or `nil` if the store has no embedder.
  private func embedding(for text: String) async throws -> [Float]? {
    guard let embedder else { return nil }
    return try await normalizedEmbedding(for: text, with: embedder)
  }

  private func normalizedEmbedding(
    for text: String,
    with embedder: any Embedder
  ) async throws -> [Float] {
    guard let vector = Vector.normalized(try await embedder.embed(text)) else {
      throw MemoryStoreError.invalidEmbedding(embedder: embedder.identifier)
    }
    return vector
  }

  /// Saves the vector of a memory from the current embedder.
  private func saveEmbedding(_ vector: [Float], rowid: Int) throws(MemoryStoreError) {
    guard let embedder else { return }
    let identifier = embedder.identifier
    // Vectors of different dimensions can't be compared,
    // so all vectors from one embedder must agree.
    let existing = try database.prepare(
      "SELECT dimension FROM embeddings WHERE embedder = ? AND memory_rowid != ? LIMIT 1"
    )
    try existing.bind(.text(identifier), .integer(rowid))
    if try existing.step(), existing.int(at: 0) != vector.count {
      throw .invalidEmbedding(embedder: identifier)
    }
    try database.prepare(
      """
      INSERT OR REPLACE INTO embeddings (memory_rowid, embedder, dimension, vector)
      VALUES (?, ?, ?, ?)
      """
    )
    .bind(.integer(rowid), .text(identifier), .integer(vector.count), .blob(Vector.encode(vector)))
    .step()
  }

  // MARK: - Rows

  /// The memory columns, in the order that ``memory(from:)`` reads them.
  static let columns = "m.id, m.text, m.created_at, m.source, m.metadata"

  static func memory(from statement: Statement) throws(MemoryStoreError) -> Memory {
    guard let id = statement.text(at: 0).flatMap(UUID.init(uuidString:)) else {
      throw .sqlite(code: 0, message: "A memory has an identifier that isn't a UUID.")
    }
    var metadata: [String: String] = [:]
    if let json = statement.text(at: 4) {
      do {
        metadata = try JSONDecoder().decode([String: String].self, from: Data(json.utf8))
      } catch {
        throw .invalidMetadata(id)
      }
    }
    return Memory(
      id: id,
      text: statement.text(at: 1) ?? "",
      createdAt: Date(timeIntervalSinceReferenceDate: statement.double(at: 2)),
      source: statement.text(at: 3),
      metadata: metadata
    )
  }

  static func encode(_ metadata: [String: String]) -> SQLValue {
    guard !metadata.isEmpty else { return .null }
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    // Encoding a dictionary of strings can't fail.
    let data = (try? encoder.encode(metadata)) ?? Data("{}".utf8)
    return .text(String(decoding: data, as: UTF8.self))
  }

  static func validate(_ text: String) throws(MemoryStoreError) {
    guard text.contains(where: { !$0.isWhitespace }) else { throw .emptyText }
  }

  /// Inserts a new memory row and returns its row identifier.
  @discardableResult
  func insert(_ memory: Memory) throws(MemoryStoreError) -> Int {
    try database.prepare(
      """
      INSERT INTO memories (id, text, created_at, source, metadata)
      VALUES (?, ?, ?, ?, ?)
      """
    )
    .bind(
      .text(memory.id.uuidString),
      .text(memory.text),
      .real(memory.createdAt.timeIntervalSinceReferenceDate),
      .optionalText(memory.source),
      Self.encode(memory.metadata)
    )
    .step()
    return try rowid(of: memory.id)
  }

  /// Saves the mutable fields of an existing memory and returns its row identifier.
  @discardableResult
  func save(_ memory: Memory) throws(MemoryStoreError) -> Int {
    try database.prepare(
      "UPDATE memories SET text = ?, source = ?, metadata = ? WHERE id = ?"
    )
    .bind(
      .text(memory.text),
      .optionalText(memory.source),
      Self.encode(memory.metadata),
      .text(memory.id.uuidString)
    )
    .step()
    guard database.changes > 0 else { throw .notFound(memory.id) }
    return try rowid(of: memory.id)
  }

  func rowid(of id: UUID) throws(MemoryStoreError) -> Int {
    let statement = try database.prepare("SELECT rowid FROM memories WHERE id = ?")
    try statement.bind(.text(id.uuidString))
    guard try statement.step() else { throw .notFound(id) }
    return statement.int(at: 0)
  }
}
