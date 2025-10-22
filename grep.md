GLOB_SEARCH_TOOL_DESCRIPTION = """Fast file pattern matching tool.
Supports glob patterns like **/*.js or src/**/*.ts.
Returns matching file paths sorted by modification time (most recent first).
Use this tool when you need to find files by name patterns.
Usage:
- The pattern parameter is a glob pattern to match files against.
- You can optionally provide a path parameter to search in a specific directory.
- Returns a newline-separated list of matching file paths.
- Results are sorted by modification time (most recently modified first).
"""


GREP_SEARCH_TOOL_DESCRIPTION = """Fast content search tool.
Searches file contents using regular expressions. Supports full regex
syntax and filters files by pattern with the include parameter.
Usage:
- The pattern parameter is a regular expression to search for in file contents.
- You can optionally provide a path parameter to search in a specific directory.
- The include parameter filters files by glob pattern (e.g., "*.js", "*.{ts,tsx}").
- The output_mode parameter controls the output format:
  - "files_with_matches": Only file paths containing matches (default)
  - "content": Matching lines with file:line:content format
  - "count": Count of matches per file
"""