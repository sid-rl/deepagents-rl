"""CLI agent that handles natural language commands for docs reviewing."""

import os
from pathlib import Path
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

from deepagents import create_deep_agent
from docs_reviewer.markdown_parser import extract_code_snippets, filter_executable_snippets, categorize_snippets
from docs_reviewer.markdown_writer import write_corrected_markdown, write_review_report
from docs_reviewer.agent import create_code_execution_tool


class DocsReviewerState(TypedDict):
    """State schema for the docs reviewer agent."""
    messages: Annotated[list, add_messages]
    file_snippets_cache: dict[str, dict]  # Maps file paths to snippet data


class DocsReviewerCLIAgent:
    """Simple agent for natural language docs reviewing."""

    def __init__(self):
        """Initialize the CLI agent."""
        import uuid

        self.current_working_directory = Path.cwd()
        self.thread_id = f"docs-reviewer-{uuid.uuid4().hex[:8]}"  # Unique thread per session
        self.last_message_count = 0  # Track how many messages we've displayed

        # Get API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        # Initialize LLM - simple, no config needed
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0.1,
            api_key=api_key,
        )

        # Create tools and agent
        self.tools = self._create_cli_tools()

        # Create code execution tool for subagents
        execute_code_snippet = create_code_execution_tool()

        # Define snippet fixer subagent - handles ALL testing and fixing
        snippet_fixer_subagent = {
            "name": "snippet_fixer",
            "description": "Test and fix a code snippet, returning the working version",
            "system_prompt": """You are a code testing and fixing specialist.

Your job:
1. Test the code snippet by calling execute_code_snippet(code, language, dependencies)
2. If it passes → return the original code with success=true
3. If it fails → analyze the error, fix the code, test again (max 3 attempts)
4. Return the final result

You MUST return a JSON response:
{
  "success": true/false,
  "original_code": "...",
  "fixed_code": "...",  // same as original if it worked first try
  "error": "..."  // empty if success
}

Important:
- Analyze imports to detect dependencies (pandas, numpy, axios, etc)
- Be surgical with fixes - only change what's broken
- Preserve original intent and structure
- Give up after 3 fix attempts
""",
            "tools": [execute_code_snippet],
            "model": self.llm,
        }

        # Create agent with checkpointer for conversation memory
        from langgraph.checkpoint.memory import MemorySaver

        self.agent = create_deep_agent(
            model=self.llm,
            tools=self.tools,  # Main agent only has file/directory tools
            system_prompt=self._get_system_prompt(),
            checkpointer=MemorySaver(),  # In-memory checkpointer for conversation state
            context_schema=DocsReviewerState,  # Custom state schema with file cache
            subagents=[snippet_fixer_subagent],  # Subagent has execute_code_snippet
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the CLI agent."""
        return f"""You are a documentation reviewer that validates and fixes code snippets in markdown files.

Working directory: {self.current_working_directory}

## Workflow:

1. Call `review_markdown_file(file)` - extracts all snippets
2. For EACH snippet, spawn a `snippet_fixer` subagent using the `task` tool (spawn all in parallel)
3. Each subagent will test and fix its snippet, returning a JSON result
4. Collect all subagent results and call `finalize_review([results])` to update the markdown

## Spawning snippet_fixer subagents:

For each snippet, call task with:
```
task(
  subagent="snippet_fixer",
  prompt="Test and fix this code snippet.

Code:
```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
result = factorial('not a number')
print(f'Result: {{result}}')
```

Language: python
Dependencies: [] (detect from imports if any)
"
)
```

The subagent returns JSON like:
{{"success": true, "original_code": "...", "fixed_code": "...", "error": ""}}

Call finalize_review with ALL these results.
"""

    def _create_cli_tools(self) -> list:
        """Create tools for the CLI agent."""
        tools = []

        @tool
        def review_markdown_file(markdown_file: str) -> str:
            """
            Extract code snippets from a markdown file.

            IMPORTANT: After calling this tool, spawn a snippet_fixer subagent for EACH
            snippet using the task tool. Do this IN PARALLEL for speed.

            Args:
                markdown_file: Path to the markdown file to review

            Returns:
                List of snippets with their code and metadata
            """
            try:
                file_path = Path(markdown_file)
                if not file_path.is_absolute():
                    file_path = self.current_working_directory / file_path

                if not file_path.exists():
                    return f"Error: File not found: {file_path}"

                # Extract snippets
                snippets = extract_code_snippets(file_path)
                if not snippets:
                    return "No code snippets found in file"

                # Get executable snippets
                executable = filter_executable_snippets(snippets)
                if not executable:
                    return "No executable code snippets found in this file"

                # Store context for finalize_review
                self._review_context = {
                    "file_path": file_path,
                    "snippets": snippets,
                    "executable": executable,
                }

                # Create line number mapping for display
                self._code_line_map = {
                    s['code']: f"lines {s['start_line']}-{s['end_line']}"
                    for s in executable
                }

                # Return snippet details with code preview
                snippet_details = []
                for i, s in enumerate(executable):
                    # Get first 10 lines of code
                    code_lines = s['code'].split('\n')[:10]
                    code_preview = '\n'.join(code_lines)
                    if len(s['code'].split('\n')) > 10:
                        code_preview += '\n...'

                    snippet_details.append(
                        f"Snippet {i+1} ({s['language']}, lines {s['start_line']}-{s['end_line']}):\n{code_preview}"
                    )

                return f"""Found {len(executable)} executable snippets in {file_path.name}:

{chr(10).join(snippet_details)}

NEXT: Spawn a snippet_fixer subagent for EACH snippet (all in parallel).
After all complete, call finalize_review with ALL the results."""
            except Exception as e:
                return f"Error: {str(e)}"

        @tool
        def finalize_review(snippet_results: list[dict]) -> str:
            """
            Apply fixes to the markdown file and show diff.

            Call this AFTER testing/fixing all snippets.

            Args:
                snippet_results: List of results for each snippet, in order.
                                Each dict should have:
                                - "original_code": str - original snippet code
                                - "fixed_code": str - fixed code (or same as original if it worked)
                                - "success": bool - whether the final version works
                                - "error": str - error message if failed

            Returns:
                JSON string with diff and summary statistics
            """
            import json

            try:
                if not hasattr(self, '_review_context'):
                    return json.dumps({"error": "No active review context. Call review_markdown_file first."})

                context = self._review_context
                file_path = context["file_path"]
                snippets = context["snippets"]
                executable = context["executable"]

                # Build results from snippet_results
                results = []
                for i, (snippet, result_dict) in enumerate(zip(executable, snippet_results)):
                    # Extract data from result dict
                    fixed_code = result_dict.get("fixed_code", snippet["code"])
                    success = result_dict.get("success", False)
                    error = result_dict.get("error", "")

                    # Build analysis
                    if success:
                        analysis = f"✓ Success"
                        if fixed_code != snippet["code"]:
                            analysis += " (code was fixed)"
                    else:
                        analysis = f"✗ Failed\nError: {error}"

                    results.append({
                        "snippet_index": i,
                        "success": success,
                        "analysis": analysis,
                        "original_code": snippet["code"],
                        "corrected_code": fixed_code,  # Use the fixed code
                        "language": snippet["language"],
                        "start_line": snippet["start_line"],
                        "end_line": snippet["end_line"],
                    })

                # Write corrected markdown with inline editing
                diff_info = write_corrected_markdown(
                    file_path,
                    file_path,  # Output path (ignored in inline mode)
                    snippets,
                    results,
                    inline=True  # Enable inline editing
                )

                # Calculate stats
                total = len(results)
                successful = sum(1 for r in results if r["success"])
                failed = total - successful

                # Clear context
                del self._review_context

                return json.dumps({
                    "success": True,
                    "file": str(file_path),
                    "total_snippets": total,
                    "successful": successful,
                    "failed": failed,
                    "success_rate": f"{(successful / total * 100):.1f}%" if total > 0 else "0%",
                    "has_changes": diff_info["has_changes"],
                    "diff": diff_info["diff"],
                })
            except Exception as e:
                return json.dumps({"error": str(e)})

        @tool
        def change_directory(directory: str) -> dict[str, Any]:
            """
            Change the current working directory.

            Args:
                directory: Path to the new working directory

            Returns:
                Dictionary with the new working directory
            """
            try:
                new_dir = Path(directory)
                if not new_dir.is_absolute():
                    new_dir = self.current_working_directory / new_dir

                if not new_dir.exists():
                    return {"error": f"Directory not found: {new_dir}"}

                if not new_dir.is_dir():
                    return {"error": f"Not a directory: {new_dir}"}

                self.current_working_directory = new_dir.resolve()
                return {
                    "success": True,
                    "working_directory": str(self.current_working_directory),
                }
            except Exception as e:
                return {"error": str(e)}

        @tool
        def get_working_directory() -> str:
            """
            Get the current working directory.

            Returns:
                The current working directory path
            """
            return str(self.current_working_directory)

        @tool
        def find_markdown_files(directory: Optional[str] = None) -> dict[str, Any]:
            """
            Find all markdown files in a directory.

            Args:
                directory: Directory to search (default: current working directory)

            Returns:
                Dictionary with list of markdown files found
            """
            try:
                search_dir = Path(directory) if directory else self.current_working_directory
                if not search_dir.is_absolute():
                    search_dir = self.current_working_directory / search_dir

                if not search_dir.exists():
                    return {"error": f"Directory not found: {search_dir}"}

                md_files = list(search_dir.rglob("*.md"))
                return {
                    "directory": str(search_dir),
                    "count": len(md_files),
                    "files": [str(f.relative_to(search_dir)) for f in md_files],
                }
            except Exception as e:
                return {"error": str(e)}

        tools.append(review_markdown_file)
        tools.append(finalize_review)
        tools.append(change_directory)
        tools.append(get_working_directory)
        tools.append(find_markdown_files)

        return tools

    def process_message(self, message: str, console) -> str:
        """
        Process a user message with streaming output.

        Args:
            message: User's message
            console: Rich console for output

        Returns:
            Agent's response as a string
        """
        import json
        from rich.syntax import Syntax

        # Stream the agent's response
        try:
            final_response = ""
            shown_todos = set()
            last_ai_message = None
            # Track tool call details for displaying with results
            tool_call_details = {}  # Maps tool_call_id -> (tool_name, args)

            # Use "values" stream mode to get the full state including all messages
            # We'll only display messages after self.last_message_count
            for state in self.agent.stream(
                {"messages": [HumanMessage(content=message)]},
                config={"configurable": {"thread_id": self.thread_id}},
                stream_mode="values"
            ):
                # state contains the full graph state including all messages
                if not isinstance(state, dict) or "messages" not in state:
                    continue

                messages = state["messages"]

                # If message count decreased, summarization happened - reset counter
                if len(messages) < self.last_message_count:
                    self.last_message_count = 0

                # Only process messages we haven't seen yet
                new_messages = messages[self.last_message_count:]

                for msg in new_messages:
                    # Skip human messages (never display them)
                    if hasattr(msg, 'type') and msg.type == 'human':
                        self.last_message_count += 1
                        continue

                    # Handle AI messages
                    if hasattr(msg, 'type') and msg.type == 'ai':
                        last_ai_message = msg
                        self.last_message_count += 1

                        # Check if this AI message has text content (not just tool calls)
                        has_content = False
                        if hasattr(msg, 'content'):
                            content = msg.content
                            if isinstance(content, str) and content.strip():
                                has_content = True
                                # Print the AI's text content
                                console.print(content, end="")
                                final_response += content
                            elif isinstance(content, list):
                                # Handle Anthropic content blocks format
                                for block in content:
                                    if isinstance(block, dict) and block.get('type') == 'text':
                                        text = block.get('text', '')
                                        if text:
                                            has_content = True
                                            console.print(text, end="")
                                            final_response += text

                        # Check for tool calls
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                tool_id = tool_call.get('id', '') or str(tool_call)
                                tool_name = tool_call.get('name', 'unknown')
                                tool_args = tool_call.get('args', {})

                                # Store tool call details for result display
                                tool_call_details[tool_id] = (tool_name, tool_args)

                                # For execute_code_snippet, DON'T print anything yet - wait for result
                                # This prevents the display from showing tool call then result separately
                                if tool_name == "execute_code_snippet":
                                    continue

                                # Add newline if we just printed content
                                if has_content and final_response and not final_response.endswith('\n'):
                                    console.print()

                                # Special formatting for task (subagent spawning)
                                if tool_name == "task":
                                    subagent = tool_args.get('subagent', 'unknown')
                                    prompt_preview = tool_args.get('prompt', '')[:80]
                                    console.print(f"\n[yellow]🤖 Spawning subagent:[/yellow] [cyan]{subagent}[/cyan]")
                                    if prompt_preview:
                                        console.print(f"  [dim]{prompt_preview}...[/dim]")
                                    continue

                                # For other tools, show them immediately
                                console.print(f"\n[#beb4fd]🔧 {tool_name}[/#beb4fd]")
                                # Show args if they're concise
                                if tool_args:
                                    for key, value in tool_args.items():
                                        if isinstance(value, str) and len(value) > 60:
                                            console.print(f"  [dim]{key}:[/dim] {value[:60]}...")
                                        elif isinstance(value, (list, dict)):
                                            # Skip verbose data structures
                                            continue
                                        else:
                                            console.print(f"  [dim]{key}:[/dim] {value}")

                    # Handle tool response messages
                    elif hasattr(msg, 'type') and msg.type == 'tool':
                        self.last_message_count += 1
                        tool_call_id = getattr(msg, 'tool_call_id', None)

                        # Get tool name and args from stored details
                        tool_name = 'unknown'
                        tool_args = {}
                        if tool_call_id in tool_call_details:
                            tool_name, tool_args = tool_call_details[tool_call_id]

                        tool_content = getattr(msg, 'content', '')

                        # Parse and display tool results
                        try:
                            result = json.loads(tool_content) if isinstance(tool_content, str) else tool_content

                            # Special formatting for execute_code_snippet (handle first, before generic error check)
                            if tool_name == "execute_code_snippet":
                                # Result is a JSON string
                                try:
                                    exec_result = json.loads(result) if isinstance(result, str) else result

                                    # Get language and line info from stored tool args
                                    language = tool_args.get('language', 'unknown')
                                    code = tool_args.get('code', '')
                                    line_info = None
                                    if hasattr(self, '_code_line_map') and code in self._code_line_map:
                                        line_info = self._code_line_map[code]

                                    # Show as a combined tool call + result (inline style)
                                    if exec_result.get("success"):
                                        output = exec_result.get("output", "")
                                        preview = output[:100] + "..." if len(output) > 100 else output
                                        # Show on single line: "🔧 execute_code_snippet → ✓ python (lines X-Y) → 'output'"
                                        console.print(f"\n[#beb4fd]🔧 execute_code_snippet[/#beb4fd] [dim]→[/dim] [#2f6868]✓ {language}[/#2f6868]", end="")
                                        if line_info:
                                            console.print(f" [dim]{line_info}[/dim]", end="")
                                        if preview.strip():
                                            console.print(f" [dim]→[/dim] {repr(preview)}")
                                        else:
                                            console.print()
                                    else:
                                        error = exec_result.get("error", "Unknown error")
                                        # Show last line of error (usually most useful)
                                        error_lines = error.split('\n')
                                        error_preview = error_lines[-1].strip() if error_lines else error
                                        if len(error_preview) > 80:
                                            error_preview = error_preview[:80] + "..."
                                        console.print(f"\n[#beb4fd]🔧 execute_code_snippet[/#beb4fd] [dim]→[/dim] [red]✗ {language}[/red]", end="")
                                        if line_info:
                                            console.print(f" [dim]{line_info}[/dim]", end="")
                                        console.print(f" [dim]→[/dim] {error_preview}")
                                except (json.JSONDecodeError, TypeError):
                                    console.print(f"\n[#beb4fd]🔧 execute_code_snippet[/#beb4fd] [dim]→[/dim] [#2f6868]✓[/#2f6868]")
                                continue

                            # Check for errors in other tools
                            if isinstance(result, dict) and "error" in result:
                                console.print(f"[red]  ✗ Error: {result['error']}[/red]\n")
                                continue

                            # Special formatting for task (subagent) results
                            if tool_name == "task" and isinstance(result, str):
                                # Show subagent completion
                                subagent_name = "unknown"
                                if tool_call_id in tool_call_details:
                                    _, args = tool_call_details[tool_call_id]
                                    subagent_name = args.get('subagent', 'unknown')

                                console.print(f"\n[yellow]✅ Subagent complete:[/yellow] [cyan]{subagent_name}[/cyan]")
                                # Show first 200 chars of result
                                result_preview = result[:200] + "..." if len(result) > 200 else result
                                console.print(f"  [dim]{result_preview}[/dim]\n")
                                continue

                            # Special formatting for review_markdown_file - show snippets
                            if tool_name == "review_markdown_file" and isinstance(result, str):
                                # Parse snippet info and display nicely
                                if "Found" in result and "snippets" in result:
                                    console.print(f"[#2f6868]  ✅ {result.split(chr(10))[0]}[/#2f6868]")
                                    # Display code snippets with syntax highlighting
                                    lines = result.split('\n')
                                    for line in lines[2:]:  # Skip first two lines
                                        if line.startswith('Snippet'):
                                            console.print(f"\n  [cyan]{line}[/cyan]")
                                        elif line.strip() and not line.startswith('NEXT'):
                                            console.print(f"    [dim]{line}[/dim]")
                                    console.print()
                                continue

                            # Special formatting for finalize_review - show diff
                            if tool_name == "finalize_review":
                                try:
                                    review_result = json.loads(result) if isinstance(result, str) else result

                                    if review_result.get("success"):
                                        total = review_result.get("total_snippets", 0)
                                        successful = review_result.get("successful", 0)
                                        failed = review_result.get("failed", 0)

                                        console.print(f"[#2f6868]  ✅ Review finalized[/#2f6868]")
                                        console.print(f"    [dim]{successful}/{total} passed[/dim]", end="")
                                        if failed > 0:
                                            console.print(f", [yellow]{failed} failed[/yellow]")
                                        else:
                                            console.print()

                                        # Show diff if there are changes
                                        if review_result.get("has_changes") and review_result.get("diff"):
                                            console.print(f"\n[dim]Changes:[/dim]")
                                            from rich.syntax import Syntax
                                            diff_syntax = Syntax(review_result["diff"], "diff", theme="monokai", line_numbers=False)
                                            console.print(diff_syntax)
                                        else:
                                            console.print(f"    [dim]No changes needed[/dim]")
                                        console.print()
                                except (json.JSONDecodeError, TypeError):
                                    console.print(f"[#2f6868]  ✅[/#2f6868]\n")
                                continue

                            # Special formatting for find_markdown_files
                            if tool_name == "find_markdown_files" and isinstance(result, dict):
                                count = result.get("count", 0)
                                console.print(f"[#2f6868]  ✅ Found {count} markdown file{'s' if count != 1 else ''}[/#2f6868]")
                                files = result.get("files", [])
                                if files:
                                    for f in files[:10]:  # Show first 10
                                        console.print(f"    [dim]•[/dim] {f}")
                                    if len(files) > 10:
                                        console.print(f"    [dim]... and {len(files) - 10} more[/dim]")
                                console.print()
                                continue

                            # Generic success for other tools
                            if isinstance(result, dict) and result.get("success"):
                                console.print(f"[#2f6868]  ✅ {tool_name}[/#2f6868]\n")
                            # For tools that return simple values (strings, etc), don't show anything
                            # They already showed the tool call, that's enough

                        except (json.JSONDecodeError, TypeError):
                            # Non-JSON response, don't show anything - tool call was already shown
                            pass

                # Handle todos from state
                if "todos" in state:
                    todos = state["todos"]
                    if todos:
                        todos_hash = str([(t.get("content"), t.get("status")) for t in todos])
                        if todos_hash not in shown_todos:
                            shown_todos.add(todos_hash)

                            if final_response and not final_response.endswith('\n'):
                                console.print()

                            console.print("\n[dim]Tasks:[/dim]")
                            for todo in todos:
                                status = todo.get("status", "pending")
                                status_emoji = {"pending": "○", "in_progress": "◐", "completed": "●"}.get(status, "•")
                                todo_content = todo.get("content", "Unknown task")

                                if status == "completed":
                                    console.print(f"  [dim]{status_emoji} {todo_content}[/dim]")
                                elif status == "in_progress":
                                    active_form = todo.get("activeForm", todo_content)
                                    console.print(f"  {status_emoji} {active_form}")
                                else:
                                    console.print(f"  [dim]{status_emoji} {todo_content}[/dim]")
                            console.print()

            # Ensure final newline
            if final_response and not final_response.endswith('\n'):
                console.print()

            # Conversation history is automatically managed by LangGraph checkpointer
            return final_response

        except Exception as e:
            error_message = f"Error: {str(e)}"
            console.print(f"\n[red]{error_message}[/red]")
            return error_message

    def reset_conversation(self) -> None:
        """Reset the conversation history by creating a new thread."""
        import uuid
        self.thread_id = f"docs-reviewer-{uuid.uuid4().hex[:8]}"
