"""Markdown writing utilities for creating corrected documentation."""

from pathlib import Path
from typing import Any
from datetime import datetime

from docs_reviewer.markdown_parser import CodeSnippet


def write_corrected_markdown(
    original_file: Path,
    output_file: Path,
    snippets: list[CodeSnippet],
    results: list[dict[str, Any]],
) -> None:
    """
    Write a corrected markdown file with updated code snippets.

    Args:
        original_file: Path to the original markdown file
        output_file: Path to write the corrected markdown
        snippets: List of original code snippets
        results: List of review results for each snippet
    """
    # Read the original file
    with open(original_file, encoding="utf-8") as f:
        lines = f.readlines()

    # Create a mapping of snippet indices to results
    result_map = {r["snippet_index"]: r for r in results}

    # Track which lines need to be replaced
    replacements: dict[int, list[str]] = {}

    for i, snippet in enumerate(snippets):
        if i not in result_map:
            continue

        result = result_map[i]

        # Only replace if there were corrections
        if result["corrected_code"] != result["original_code"] or not result["success"]:
            start_line = snippet["start_line"] - 1  # Convert to 0-indexed
            end_line = snippet["end_line"]

            # Build the replacement content
            replacement_lines = []

            # Add a comment about the correction
            if not result["success"]:
                replacement_lines.append(f"<!-- REVIEW NOTE: This snippet had issues -->\n")
                replacement_lines.append(f"<!-- {_escape_html_comment(result['analysis'][:200])} -->\n")
                replacement_lines.append("\n")

            # Add the code fence
            replacement_lines.append(f"```{snippet['language']}\n")
            replacement_lines.append(result["corrected_code"])
            if not result["corrected_code"].endswith("\n"):
                replacement_lines.append("\n")
            replacement_lines.append("```\n")

            replacements[start_line] = replacement_lines

    # Write the corrected file
    with open(output_file, "w", encoding="utf-8") as f:
        # Add header comment
        f.write(f"<!-- Corrected by Docs Reviewer on {datetime.now().isoformat()} -->\n")
        f.write(f"<!-- Original file: {original_file} -->\n\n")

        i = 0
        while i < len(lines):
            if i in replacements:
                # Write the replacement
                for line in replacements[i]:
                    f.write(line)

                # Skip the original code block
                # Find the closing ```
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    i += 1
                i += 1  # Skip the closing ```
            else:
                f.write(lines[i])
                i += 1


def write_review_report(
    output_file: Path,
    markdown_file: Path,
    snippets: list[CodeSnippet],
    results: list[dict[str, Any]],
) -> None:
    """
    Write a detailed review report.

    Args:
        output_file: Path to write the report
        markdown_file: Path to the reviewed markdown file
        snippets: List of code snippets
        results: List of review results
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Docs Review Report\n\n")
        f.write(f"**File:** {markdown_file}\n")
        f.write(f"**Date:** {datetime.now().isoformat()}\n\n")

        # Summary statistics
        total = len(results)
        successful = sum(1 for r in results if r["success"])
        failed = total - successful

        f.write(f"## Summary\n\n")
        f.write(f"- Total snippets reviewed: {total}\n")
        f.write(f"- Successful: {successful}\n")
        f.write(f"- Failed: {failed}\n")
        f.write(f"- Success rate: {(successful / total * 100):.1f}%\n\n")

        # Detailed results
        f.write(f"## Detailed Results\n\n")

        for i, result in enumerate(results, 1):
            snippet = snippets[result["snippet_index"]]

            status = "✅ Success" if result["success"] else "❌ Failed"
            f.write(f"### Snippet {i}: {status}\n\n")
            f.write(f"**Language:** {result['language']}\n")
            f.write(f"**Lines:** {result['start_line']}-{result['end_line']}\n\n")

            f.write(f"**Analysis:**\n")
            f.write(f"{result['analysis']}\n\n")

            if result["corrected_code"] != result["original_code"]:
                f.write(f"**Original Code:**\n")
                f.write(f"```{result['language']}\n")
                f.write(result["original_code"])
                f.write(f"\n```\n\n")

                f.write(f"**Corrected Code:**\n")
                f.write(f"```{result['language']}\n")
                f.write(result["corrected_code"])
                f.write(f"\n```\n\n")

            f.write("---\n\n")


def _escape_html_comment(text: str) -> str:
    """Escape text for safe inclusion in HTML comments."""
    return text.replace("--", "- -").replace("<!--", "< !--").replace("-->", "-- >")
