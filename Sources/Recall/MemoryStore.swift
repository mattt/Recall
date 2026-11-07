import Foundation

/// A store of memories in a SQLite database.
///
/// A store is an actor,
/// so you can share one instance between tasks.
/// Each store has its own connection to the database.
/// Two stores can open the same file;
/// SQLite serializes their writes.
public actor MemoryStore {
  let database: Database

  /// Opens or creates a store in the SQLite file at `url`.
  ///
  /// If the file has an older schema,
  /// the store migrates it to the current schema.
  ///
  /// - Parameter url: A file URL for the database.
  /// - Throws: ``MemoryStoreError/unsupportedSchemaVersion(_:)``
  ///   if a newer version of Recall created the file,
  ///   or ``MemoryStoreError/sqlite(code:message:)``
  ///   if SQLite can't open it.
  public init(url: URL) throws {
    try self.init(database: Database(url: url), isFile: true)
  }

  /// Creates a store in a private in-memory database.
  ///
  /// The memories exist only as long as the store does.
  public static func inMemory() throws -> MemoryStore {
    try MemoryStore(database: Database(url: nil), isFile: false)
  }

  init(database: Database, isFile: Bool) throws(MemoryStoreError) {
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
  /// - Throws: ``MemoryStoreError/emptyText`` if `text` contains only whitespace.
  @discardableResult
  public func remember(
    _ text: String,
    source: String? = nil,
    metadata: [String: String] = [:],
    createdAt: Date = Date()
  ) async throws -> Memory {
    try Self.validate(text)
    let memory = Memory(text: text, createdAt: createdAt, source: source, metadata: metadata)
    try insert(memory)
    return memory
  }

  /// Saves changes to the text, source, or metadata of a memory.
  ///
  /// The identifier and creation date don't change.
  ///
  /// - Parameter memory: The memory with its new values.
  /// - Returns: The saved memory.
  /// - Throws: ``MemoryStoreError/notFound(_:)`` if the store has no memory with that identifier.
  @discardableResult
  public func update(_ memory: Memory) async throws -> Memory {
    try Self.validate(memory.text)
    try save(memory)
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
