#!/usr/bin/env python3
"""Verify the streaming code structure is correct."""

import sys
from pathlib import Path

# Read the cli_agent.py file
cli_agent_path = Path(__file__).parent / "docs_reviewer" / "cli_agent.py"
code = cli_agent_path.read_text()

print("="*60)
print("Streaming Implementation Verification")
print("="*60)
print()

# Check 1: Using multiple stream modes
if 'stream_mode=["messages", "updates"]' in code:
    print("✓ Using dual stream modes: ['messages', 'updates']")
else:
    print("✗ Not using dual stream modes correctly")
    sys.exit(1)

# Check 2: Proper tuple unpacking
if "stream_mode_name, content = stream_item" in code:
    print("✓ Proper tuple unpacking: (stream_mode_name, content)")
else:
    print("✗ Tuple unpacking not correct")
    sys.exit(1)

# Check 3: Handling messages mode
if 'if stream_mode_name == "messages"' in code:
    print("✓ Handling 'messages' stream mode")
else:
    print("✗ Not handling 'messages' stream mode")
    sys.exit(1)

# Check 4: Handling updates mode
if 'stream_mode_name == "updates"' in code:
    print("✓ Handling 'updates' stream mode")
else:
    print("✗ Not handling 'updates' stream mode")
    sys.exit(1)

# Check 5: Streaming LLM content
if "console.print(content.content, end=\"\")" in code:
    print("✓ Streaming LLM tokens with end=''")
else:
    print("✗ Not streaming LLM tokens correctly")
    sys.exit(1)

# Check 6: Tool call detection
if "hasattr(content, 'tool_calls') and content.tool_calls" in code:
    print("✓ Detecting tool calls from message chunks")
else:
    print("✗ Tool call detection not found")
    sys.exit(1)

# Check 7: Tool call formatting
if "→ {tool_name}" in code and "tool_args" in code:
    print("✓ Formatting tool calls with arguments")
else:
    print("✗ Tool call formatting incomplete")
    sys.exit(1)

# Check 8: Tool response handling
if "hasattr(content, 'name') and content.name" in code:
    print("✓ Detecting tool responses")
else:
    print("✗ Tool response detection not found")
    sys.exit(1)

# Check 9: Custom formatting for list_snippets
if '"list_snippets"' in code and "Found {total} snippet" in code:
    print("✓ Custom formatting for list_snippets tool")
else:
    print("✗ Missing custom formatting for list_snippets")
    sys.exit(1)

# Check 10: Custom formatting for review_markdown_file
if '"review_markdown_file"' in code and "Review complete" in code:
    print("✓ Custom formatting for review_markdown_file tool")
else:
    print("✗ Missing custom formatting for review_markdown_file")
    sys.exit(1)

# Check 11: Todo tracking from updates
if "todos" in code and '"todos" in node_data' in code:
    print("✓ Todo tracking from updates stream")
else:
    print("✗ Todo tracking not found")
    sys.exit(1)

print()
print("="*60)
print("✓ All streaming implementation checks passed!")
print("="*60)
print()
print("The streaming implementation should correctly handle:")
print("  • LLM token streaming (messages mode)")
print("  • Tool calls with formatted arguments")
print("  • Tool results with custom formatting")
print("  • Todo updates (updates mode)")
print()
print("To test with real API calls:")
print("  1. Set ANTHROPIC_API_KEY environment variable")
print("  2. Run: docs-reviewer chat -m 'list snippets in example_docs.md'")
print()
