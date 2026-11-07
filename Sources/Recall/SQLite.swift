import Foundation

#if canImport(SQLite3)
  import SQLite3
#else
  import CSQLite
#endif

/// The destructor value that tells SQLite to copy bound text and blobs.
private let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

/// A thin wrapper around one SQLite connection.
///
/// This type isn't thread-safe.
/// `MemoryStore` owns one instance
/// and uses it only from the actor's isolation domain.
final class Database {
  private let handle: OpaquePointer

  /// Opens the database at a file URL,
  /// or a private in-memory database if the URL is `nil`.
  init(url: URL?) throws(MemoryStoreError) {
    let path = url?.path ?? ":memory:"
    var handle: OpaquePointer?
    let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX
    let code = sqlite3_open_v2(path, &handle, flags, nil)
    guard let handle else {
      throw .sqlite(code: code, message: "Cannot open the database at \(path).")
    }
    self.handle = handle
    guard code == SQLITE_OK else {
      let message = String(cString: sqlite3_errmsg(handle))
      sqlite3_close_v2(handle)
      throw .sqlite(code: code, message: message)
    }
    sqlite3_busy_timeout(handle, 5_000)
    sqlite3_extended_result_codes(handle, 1)
  }

  deinit {
    sqlite3_close_v2(handle)
  }

  /// The message for the most recent error on this connection.
  var errorMessage: String {
    String(cString: sqlite3_errmsg(handle))
  }

  /// Runs one or more SQL statements that take no parameters.
  func execute(_ sql: String) throws(MemoryStoreError) {
    var message: UnsafeMutablePointer<CChar>?
    let code = sqlite3_exec(handle, sql, nil, nil, &message)
    guard code == SQLITE_OK else {
      let text = message.map { String(cString: $0) } ?? errorMessage
      sqlite3_free(message)
      throw .sqlite(code: code, message: text)
    }
  }

  /// Puts a file database in write-ahead logging mode,
  /// so that readers don't block the writer.
  ///
  /// SQLite can return `SQLITE_BUSY` from this change
  /// without calling the busy handler
  /// when another connection opens the file at the same time,
  /// so this method retries for up to five seconds.
  func enableWriteAheadLog() throws(MemoryStoreError) {
    let deadline = Date().addingTimeInterval(5)
    while true {
      do {
        let statement = try prepare("PRAGMA journal_mode = WAL")
        try statement.step()
        return
      } catch .sqlite(let code, _) where code & 0xFF == SQLITE_BUSY && Date() < deadline {
        Thread.sleep(forTimeInterval: 0.01)
      }
    }
  }

  /// Compiles one SQL statement.
  func prepare(_ sql: String) throws(MemoryStoreError) -> Statement {
    var statement: OpaquePointer?
    let code = sqlite3_prepare_v2(handle, sql, -1, &statement, nil)
    guard code == SQLITE_OK, let statement else {
      throw .sqlite(code: code, message: errorMessage)
    }
    return Statement(database: self, handle: statement)
  }

  /// Runs `body` in a transaction that takes the write lock at its start,
  /// and commits if `body` returns or rolls back if it throws.
  func transaction<T>(_ body: () throws(MemoryStoreError) -> T) throws(MemoryStoreError) -> T {
    try execute("BEGIN IMMEDIATE")
    do {
      let value = try body()
      try execute("COMMIT")
      return value
    } catch {
      try? execute("ROLLBACK")
      throw error
    }
  }

  /// The number of rows that the most recent statement changed.
  var changes: Int {
    Int(sqlite3_changes(handle))
  }

  /// The version number in the database header.
  var userVersion: Int {
    get throws(MemoryStoreError) {
      let statement = try prepare("PRAGMA user_version")
      guard try statement.step() else { return 0 }
      return statement.int(at: 0)
    }
  }
}

/// One compiled SQL statement.
final class Statement {
  private let database: Database
  private let handle: OpaquePointer

  fileprivate init(database: Database, handle: OpaquePointer) {
    self.database = database
    self.handle = handle
  }

  deinit {
    sqlite3_finalize(handle)
  }

  /// Binds values to the statement's parameters, in order.
  @discardableResult
  func bind(_ values: SQLValue...) throws(MemoryStoreError) -> Statement {
    for (offset, value) in values.enumerated() {
      let index = Int32(offset + 1)
      let code: Int32
      switch value {
      case .null:
        code = sqlite3_bind_null(handle, index)
      case .integer(let integer):
        code = sqlite3_bind_int64(handle, index, Int64(integer))
      case .real(let real):
        code = sqlite3_bind_double(handle, index, real)
      case .text(let text):
        code = sqlite3_bind_text(handle, index, text, -1, transient)
      case .blob(let data):
        code = data.withUnsafeBytes { buffer in
          sqlite3_bind_blob(handle, index, buffer.baseAddress, Int32(buffer.count), transient)
        }
      }
      guard code == SQLITE_OK else {
        throw .sqlite(code: code, message: database.errorMessage)
      }
    }
    return self
  }

  /// Advances to the next row.
  ///
  /// - Returns: `true` if a row is available, or `false` when the statement is done.
  @discardableResult
  func step() throws(MemoryStoreError) -> Bool {
    let code = sqlite3_step(handle)
    switch code {
    case SQLITE_ROW:
      return true
    case SQLITE_DONE:
      return false
    default:
      throw .sqlite(code: code, message: database.errorMessage)
    }
  }

  func isNull(at column: Int32) -> Bool {
    sqlite3_column_type(handle, column) == SQLITE_NULL
  }

  func int(at column: Int32) -> Int {
    Int(sqlite3_column_int64(handle, column))
  }

  func double(at column: Int32) -> Double {
    sqlite3_column_double(handle, column)
  }

  func text(at column: Int32) -> String? {
    guard let pointer = sqlite3_column_text(handle, column) else { return nil }
    return String(cString: pointer)
  }

  func data(at column: Int32) -> Data {
    let count = Int(sqlite3_column_bytes(handle, column))
    guard count > 0, let pointer = sqlite3_column_blob(handle, column) else { return Data() }
    return Data(bytes: pointer, count: count)
  }
}

/// A value to bind to a statement parameter.
enum SQLValue {
  case null
  case integer(Int)
  case real(Double)
  case text(String)
  case blob(Data)
}

extension SQLValue {
  static func optionalText(_ text: String?) -> SQLValue {
    text.map(SQLValue.text) ?? .null
  }
}
