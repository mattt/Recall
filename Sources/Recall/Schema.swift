import Foundation

/// The database schema and its migrations.
///
/// The `user_version` field in the database header stores the schema version.
/// Migration `n` (counting from 1) changes a database from version `n - 1` to version `n`.
/// Never edit a migration after it ships;
/// append a new one instead.
enum Schema {
  static let migrations: [String] = [
    // Version 1: memories and their full-text index.
    // `created_at` holds seconds since 2001-01-01 00:00:00 UTC,
    // which is the value that `Date` stores,
    // so dates read back unchanged.
    """
    CREATE TABLE memories (
      rowid INTEGER PRIMARY KEY,
      id TEXT NOT NULL UNIQUE,
      text TEXT NOT NULL,
      created_at REAL NOT NULL,
      source TEXT,
      metadata TEXT
    );
    CREATE INDEX memories_created_at ON memories (created_at);
    CREATE VIRTUAL TABLE memories_fts USING fts5(
      text,
      content = 'memories',
      content_rowid = 'rowid',
      tokenize = 'porter unicode61 remove_diacritics 2'
    );
    CREATE TRIGGER memories_after_insert AFTER INSERT ON memories BEGIN
      INSERT INTO memories_fts (rowid, text) VALUES (new.rowid, new.text);
    END;
    CREATE TRIGGER memories_after_delete AFTER DELETE ON memories BEGIN
      INSERT INTO memories_fts (memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    END;
    CREATE TRIGGER memories_after_update_text AFTER UPDATE OF text ON memories BEGIN
      INSERT INTO memories_fts (memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
      INSERT INTO memories_fts (rowid, text) VALUES (new.rowid, new.text);
    END;
    """,
    // Version 2: one embedding vector for each memory.
    // `vector` holds `dimension` 32-bit floats of unit length, in host byte order.
    // A change to the text of a memory deletes its vector.
    """
    CREATE TABLE embeddings (
      memory_rowid INTEGER PRIMARY KEY REFERENCES memories (rowid) ON DELETE CASCADE,
      embedder TEXT NOT NULL,
      dimension INTEGER NOT NULL,
      vector BLOB NOT NULL
    );
    CREATE INDEX embeddings_embedder ON embeddings (embedder, dimension);
    CREATE TRIGGER memories_after_update_text_embedding AFTER UPDATE OF text ON memories
    WHEN old.text IS NOT new.text BEGIN
      DELETE FROM embeddings WHERE memory_rowid = new.rowid;
    END;
    """,
  ]

  /// The schema version that this version of Recall creates.
  static var currentVersion: Int { migrations.count }

  /// Brings the database up to `version`, which defaults to the current version.
  ///
  /// The migration runs in one write transaction,
  /// so two connections that open the same file at the same time
  /// don't both apply it.
  static func migrate(
    _ database: Database,
    to version: Int = currentVersion
  ) throws(MemoryStoreError) {
    try database.transaction { () throws(MemoryStoreError) in
      let existing = try database.userVersion
      guard existing <= currentVersion else {
        throw .unsupportedSchemaVersion(existing)
      }
      guard existing < version else { return }
      for migration in migrations[existing..<version] {
        try database.execute(migration)
      }
      try database.execute("PRAGMA user_version = \(version)")
    }
  }
}
