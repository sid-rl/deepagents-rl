# Claude Code-Like Streaming Implementation

## Overview

The Docs Reviewer CLI has been updated to provide a Claude Code-like streaming experience with real-time feedback on LLM generation, tool calls, and tool results.

## Key Implementation Details

### 1. Dual Stream Modes

```python
stream_mode=["messages", "updates"]
```

Using multiple stream modes returns tuples: `(stream_mode_name, content)`

- **`"messages"`** - Streams message chunks as they're generated
  - LLM token streaming (AIMessageChunk with content)
  - Tool calls (message with tool_calls attribute)
  - Tool responses (ToolMessage with name attribute)

- **`"updates"`** - Streams node-level state updates
  - Todo list updates from TodoListMiddleware
  - Other state changes from custom middleware

### 2. Stream Item Structure

```python
for stream_item in self.agent.stream(..., stream_mode=["messages", "updates"]):
    stream_mode_name, content = stream_item  # Tuple unpacking

    if stream_mode_name == "messages":
        # Handle LLM tokens, tool calls, tool responses

    elif stream_mode_name == "updates":
        # Handle state updates (todos, etc.)
```

### 3. Real-time LLM Token Streaming

```python
if hasattr(content, 'content') and content.content:
    console.print(content.content, end="")  # No newline - stream continuously
    final_response += content.content
```

**Result:** Tokens appear as the LLM generates them, character by character or in small chunks.

### 4. Tool Call Display

When the LLM decides to use a tool:

```python
if hasattr(content, 'tool_calls') and content.tool_calls:
    for tool_call in content.tool_calls:
        tool_name = tool_call.get('name', 'unknown')
        tool_args = tool_call.get('args', {})

        console.print(f"\n[yellow]→ {tool_name}[/yellow]")

        # Show formatted arguments
        for key, value in tool_args.items():
            if isinstance(value, str) and len(value) > 60:
                console.print(f"  [dim]{key}:[/dim] {value[:60]}...")
            else:
                console.print(f"  [dim]{key}:[/dim] {value}")
```

**Output:**
```
→ list_snippets
  markdown_file: example_docs.md
```

### 5. Tool Result Display

Custom formatting for each tool:

#### list_snippets
```python
if tool_name == "list_snippets":
    total = result.get("total_snippets", 0)
    console.print(f"[green]  ✓ Found {total} snippet{'s' if total != 1 else ''}[/green]")

    snippets = result.get("snippets", [])
    for i, snip in enumerate(snippets, 1):
        lang = snip.get("language", "unknown")
        lines = snip.get("lines", "?")
        console.print(f"    {i}. [cyan]{lang}[/cyan] (lines {lines})")
```

**Output:**
```
  ✓ Found 6 snippets
    1. python (lines 9-13)
    2. python (lines 20-32)
    3. javascript (lines 52-58)
    ...
```

#### review_markdown_file
```python
if tool_name == "review_markdown_file":
    console.print(f"[green]  ✓ Review complete[/green]")
    console.print(f"    {successful}/{total} passed")
    if failed > 0:
        console.print(f"    [yellow]{failed} failed[/yellow]")
    console.print(f"    [dim]→ {corrected_file}[/dim]")
```

**Output:**
```
  ✓ Review complete
    5/6 passed
    1 failed
    → example_docs_corrected.md
```

#### find_markdown_files
```python
if tool_name == "find_markdown_files":
    count = result.get("count", 0)
    console.print(f"[green]  ✓ Found {count} markdown file{'s' if count != 1 else ''}[/green]")
    files = result.get("files", [])
    for f in files[:10]:
        console.print(f"    • {f}")
    if len(files) > 10:
        console.print(f"    [dim]... and {len(files) - 10} more[/dim]")
```

**Output:**
```
  ✓ Found 8 markdown files
    • README.md
    • example_docs.md
    • SIMPLE_USAGE.md
    ...
```

### 6. Todo Tracking

From the `updates` stream mode:

```python
elif stream_mode_name == "updates":
    for node_name, node_data in content.items():
        if "todos" in node_data:
            todos = node_data["todos"]
            for todo in todos:
                status = todo.get("status", "pending")
                status_emoji = {
                    "pending": "○",
                    "in_progress": "◐",
                    "completed": "●"
                }.get(status, "•")

                if status == "in_progress":
                    console.print(f"  {status_emoji} {todo.get('activeForm')}")
                else:
                    console.print(f"  [dim]{status_emoji} {todo.get('content')}[/dim]")
```

**Output:**
```
Tasks:
  ○ Extract code snippets
  ◐ Validating snippets
  ● Parse markdown file
```

### 7. Deduplication

To prevent showing duplicate tool calls or todos:

```python
seen_tool_calls = set()
shown_todos = set()

# For tool calls
tool_id = tool_call.get('id', '') or str(tool_call)
if tool_id not in seen_tool_calls:
    seen_tool_calls.add(tool_id)
    # Display tool call

# For todos
todos_hash = str([(t.get("content"), t.get("status")) for t in todos])
if todos_hash not in shown_todos:
    shown_todos.add(todos_hash)
    # Display todos
```

## User Experience

### Example Session

```
Docs Reviewer ready. Type your request or 'exit' to quit.

You
list snippets in example_docs.md

Docs Reviewer

→ list_snippets
  markdown_file: example_docs.md

  ✓ Found 6 snippets
    1. python (lines 9-13)
    2. python (lines 20-32)
    3. python (lines 39-45)
    4. javascript (lines 52-58)
    5. bash (lines 63-65)
    6. text (lines 72-75)

I found 6 code snippets in example_docs.md: 3 Python snippets, 1 JavaScript snippet, 1 bash snippet, and 1 text block. Would you like me to review these for correctness?

You
yes please

Docs Reviewer

→ review_markdown_file
  markdown_file: example_docs.md

  ✓ Review complete
    5/6 passed
    → example_docs_corrected.md

The review is complete! 5 out of 6 snippets passed validation. I've created a corrected version at example_docs_corrected.md with fixes applied.
```

## Features Demonstrated

✅ **Real-time streaming** - Tokens appear as generated
✅ **Tool visibility** - See exactly what tools are being called
✅ **Formatted arguments** - Clear display of tool inputs
✅ **Custom results** - Tool-specific formatting for better readability
✅ **Progress tracking** - Todo updates show what's happening
✅ **Clean interface** - Minimal, focused output
✅ **Clear turns** - "You" and "Docs Reviewer" labels

## Testing

### Code Verification
```bash
python verify_streaming_code.py
```

Verifies the implementation has all required components without API calls.

### End-to-End Test
```bash
# With .env file containing ANTHROPIC_API_KEY
python test_e2e_streaming.py

# Or with environment variable
export ANTHROPIC_API_KEY='your-key'
python test_e2e_streaming.py
```

Tests actual streaming with API calls.

### Manual Testing
```bash
# Interactive mode
docs-reviewer chat

# Single message mode
docs-reviewer chat -m "list snippets in example_docs.md"
```

## Files Modified

1. **`docs_reviewer/cli.py`** - Added "You" / "Docs Reviewer" labels back
2. **`docs_reviewer/cli_agent.py`** - Complete streaming rewrite:
   - Dual stream modes: `["messages", "updates"]`
   - Proper tuple unpacking: `(stream_mode_name, content)`
   - Real-time LLM token streaming
   - Tool call display with formatted arguments
   - Custom tool result formatting
   - Todo tracking from updates stream

## Architecture Benefits

1. **Separation of Concerns**
   - `messages` mode handles all LLM/tool communication
   - `updates` mode handles state changes (todos, etc.)

2. **Extensibility**
   - Easy to add new tool result formatters
   - Can add more stream modes if needed (e.g., `custom`)

3. **Performance**
   - Streaming provides immediate feedback
   - No waiting for full completion

4. **User Experience**
   - Clear visibility into agent's thought process
   - Professional, polished interface
   - Matches Claude Code expectations
