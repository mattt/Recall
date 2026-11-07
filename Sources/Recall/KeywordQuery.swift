/// Converts user text into an FTS5 query that can't fail to parse.
///
/// FTS5 gives meaning to quotation marks, parentheses, `*`, `^`, `:`, `-`, `+`,
/// and the words `AND`, `OR`, `NOT`, and `NEAR`.
/// To keep user text from using that syntax,
/// this type splits the text into runs of letters and numbers,
/// puts each run in double quotation marks,
/// and joins the runs with `OR`.
/// A quoted run is an FTS5 string, so FTS5 reads it only as a term to match.
enum KeywordQuery {
  /// Returns an FTS5 `MATCH` expression for `text`,
  /// or `nil` if the text contains no letters or numbers.
  static func match(for text: String) -> String? {
    var seen: Set<String> = []
    var terms: [String] = []
    for run in text.split(whereSeparator: { !($0.isLetter || $0.isNumber) }) {
      let term = String(run)
      guard seen.insert(term.lowercased()).inserted else { continue }
      terms.append(quoted(term))
    }
    guard !terms.isEmpty else { return nil }
    return terms.joined(separator: " OR ")
  }

  /// Returns `term` as an FTS5 string,
  /// with each double quotation mark in it doubled.
  static func quoted(_ term: String) -> String {
    "\"" + term.replacing("\"", with: "\"\"") + "\""
  }
}
