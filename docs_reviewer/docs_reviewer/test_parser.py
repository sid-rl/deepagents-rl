"""Simple test to verify the markdown parser works."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from docs_reviewer.markdown_parser import extract_code_snippets, filter_executable_snippets, categorize_snippets


def test_parser():
    """Test the markdown parser with the example file."""
    example_file = Path(__file__).parent / "example_docs.md"

    print("Testing markdown parser...")
    print(f"File: {example_file}\n")

    # Extract all snippets
    snippets = extract_code_snippets(example_file)
    print(f"Total snippets found: {len(snippets)}\n")

    for i, snippet in enumerate(snippets, 1):
        print(f"Snippet {i}:")
        print(f"  Language: {snippet['language']}")
        print(f"  Lines: {snippet['start_line']}-{snippet['end_line']}")
        print(f"  Code length: {len(snippet['code'])} chars")
        print(f"  Metadata: {snippet['metadata']}")
        print()

    # Filter executable snippets
    executable = filter_executable_snippets(snippets)
    print(f"Executable snippets: {len(executable)}\n")

    for snippet in executable:
        print(f"  - {snippet['language']} (lines {snippet['start_line']}-{snippet['end_line']})")

    # Categorize snippets
    categories = categorize_snippets(snippets)
    print(f"\nSnippets by language:")
    for lang, lang_snippets in categories.items():
        print(f"  {lang}: {len(lang_snippets)}")


if __name__ == "__main__":
    test_parser()
