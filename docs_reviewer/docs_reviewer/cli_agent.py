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

        # Add code execution tool
        self.execute_code_snippet = create_code_execution_tool()
        all_tools = self.tools + [self.execute_code_snippet]

        # Create agent with checkpointer for conversation memory
        from langgraph.checkpoint.memory import MemorySaver

        self.agent = create_deep_agent(
            model=self.llm,
            tools=all_tools,
            system_prompt=self._get_system_prompt(),
            checkpointer=MemorySaver(),  # In-memory checkpointer for conversation state
            context_schema=DocsReviewerState,  # Custom state schema with file cache
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the CLI agent."""
        return f"""You are a documentation reviewer that validates code snippets in markdown files.

Working directory: {self.current_working_directory}

## Workflow for reviewing files:

1. Use `list_snippets` to show what code snippets exist in a file
2. Use `review_markdown_file` to start the review process - this extracts executable snippets
3. Call `execute_code_snippet` for EACH snippet IN PARALLEL to validate them
   - Pass: code, language, dependencies (list), python_version (if applicable)
   - Example: execute_code_snippet(code="import pandas as pd...", language="python", dependencies=["pandas"])
   - You can make multiple execute_code_snippet calls at the same time
   - This is the key to fast reviews - parallel execution!
4. Use `finalize_review` to apply inline edits and show the diff

## execute_code_snippet Tool Details:

For **Python** snippets:
- If code imports packages (pandas, numpy, requests, etc.), include them in dependencies
- Example: dependencies=["pandas", "numpy", "matplotlib"]
- Optionally specify python_version="3.11" if needed
- Uses uv/uvx for isolated execution with dependency installation

For **JavaScript** snippets:
- If code requires npm packages, include them in dependencies
- Example: dependencies=["axios", "lodash"]
- Uses node with automatic npm install

For **Bash** snippets:
- No dependencies needed, executes directly

## Important Notes:

- Files are edited inline (no separate _corrected.md files)
- LangGraph will automatically execute parallel tool calls concurrently
- Always call execute_code_snippet IN PARALLEL for all snippets (not one at a time)
- Analyze code to detect required dependencies from import statements
- Be direct and concise in your responses
- Use relative paths from the working directory
"""

    def _create_cli_tools(self) -> list:
        """Create tools for the CLI agent."""
        tools = []

        @tool
        def list_snippets(markdown_file: str) -> dict[str, Any]:
            """
            List all code snippets found in a markdown file without executing them.

            This tool caches the snippet data so it can be reused by review_markdown_file.

            Args:
                markdown_file: Path to the markdown file (relative or absolute)

            Returns:
                Dictionary with snippet information
            """
            try:
                file_path = Path(markdown_file)
                if not file_path.is_absolute():
                    file_path = self.current_working_directory / file_path

                if not file_path.exists():
                    return {"error": f"File not found: {file_path}"}

                # Extract snippets
                snippets = extract_code_snippets(file_path)
                executable = filter_executable_snippets(snippets)
                categories = categorize_snippets(snippets)

                # Store in cache for later use by review_markdown_file
                file_key = str(file_path)
                cache_data = {
                    "snippets": snippets,
                    "executable": executable,
                    "file_path": file_path,
                }

                # Store in instance variable for immediate access
                if not hasattr(self, '_file_cache'):
                    self._file_cache = {}
                self._file_cache[file_key] = cache_data

                return {
                    "file": str(file_path),
                    "total_snippets": len(snippets),
                    "executable_snippets": len(executable),
                    "categories": {lang: len(snips) for lang, snips in categories.items()},
                    "snippets": [
                        {
                            "language": s["language"],
                            "lines": f"{s['start_line']}-{s['end_line']}",
                            "length": len(s["code"]),
                            "preview": s["code"][:100] + "..." if len(s["code"]) > 100 else s["code"],
                        }
                        for s in snippets
                    ],
                }
            except Exception as e:
                return {"error": str(e)}

        @tool
        def review_markdown_file(markdown_file: str, output_file: Optional[str] = None) -> str:
            """
            Review a markdown file and return the list of code snippets to execute.

            If you previously called list_snippets on this file, I'll reuse that data
            to avoid redundant work!

            IMPORTANT: After calling this tool, you MUST call execute_code_snippet for EACH
            snippet returned. You can call execute_code_snippet multiple times IN PARALLEL
            to speed up the review process.

            Args:
                markdown_file: Path to the markdown file to review
                output_file: Optional path for the corrected markdown (default: <input>_corrected.md)

            Returns:
                Instructions on what snippets to execute next
            """
            try:
                file_path = Path(markdown_file)
                if not file_path.is_absolute():
                    file_path = self.current_working_directory / file_path

                if not file_path.exists():
                    return f"Error: File not found: {file_path}"

                file_key = str(file_path)

                # Check if we have cached data from list_snippets
                if hasattr(self, '_file_cache') and file_key in self._file_cache:
                    cache_data = self._file_cache[file_key]
                    snippets = cache_data["snippets"]
                    executable = cache_data["executable"]
                    used_cache = True
                else:
                    # Extract snippets fresh
                    snippets = extract_code_snippets(file_path)
                    if not snippets:
                        return "Error: No code snippets found in file"

                    # Get executable snippets
                    executable = filter_executable_snippets(snippets)
                    used_cache = False

                if not executable:
                    return "No executable code snippets found in this file."

                # Store the context for later use
                self._review_context = {
                    "file_path": file_path,
                    "output_file": output_file,
                    "snippets": snippets,
                    "executable": executable,
                }

                # Return detailed snippet info
                snippet_details = []
                for i, s in enumerate(executable):
                    code_preview = s['code'][:100] + "..." if len(s['code']) > 100 else s['code']
                    snippet_details.append(
                        f"  {i+1}. {s['language']} (lines {s['start_line']}-{s['end_line']})\n"
                        f"     Code: {repr(code_preview)}"
                    )

                snippet_list = "\n".join(snippet_details)

                # Indicate if we reused cached data
                cache_note = " (reusing previously scanned snippets)" if used_cache else ""

                return f"""Found {len(executable)} executable code snippets in {file_path.name}{cache_note}:

{snippet_list}

NEXT STEP: Call execute_code_snippet for each snippet IN PARALLEL.
Analyze each snippet for import statements to detect dependencies.

For each snippet:
- Look for imports (Python: import/from, JS: require/import)
- Call execute_code_snippet with: code, language, dependencies (if any)
- Make ALL calls in parallel (not one at a time)

After all snippets execute, call finalize_review with the results."""
            except Exception as e:
                return f"Error: {str(e)}"

        @tool
        def finalize_review(execution_results: list[str]) -> str:
            """
            Finalize the review by applying inline edits to the markdown file.

            Call this AFTER you have executed all code snippets using execute_code_snippet.

            Args:
                execution_results: List of JSON strings from execute_code_snippet calls
                                 Each should be: {"success": bool, "output": str, "error": str}

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

                # Parse execution results
                results = []
                for i, (snippet, exec_result_str) in enumerate(zip(executable, execution_results)):
                    try:
                        exec_result = json.loads(exec_result_str) if isinstance(exec_result_str, str) else exec_result_str
                    except json.JSONDecodeError:
                        exec_result = {"success": False, "output": "", "error": "Invalid result format"}

                    # Build analysis from execution result
                    if exec_result.get("success"):
                        analysis = f"✓ Success\nOutput: {exec_result.get('output', '')}"
                    else:
                        analysis = f"✗ Failed\nError: {exec_result.get('error', '')}"

                    results.append({
                        "snippet_index": i,
                        "success": exec_result.get("success", False),
                        "analysis": analysis,
                        "original_code": snippet["code"],
                        "corrected_code": snippet["code"],  # For now, no corrections - just validation
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

        tools.append(list_snippets)
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
            seen_tool_calls = set()
            shown_todos = set()
            last_ai_message = None

            # Use only "updates" stream mode for smoother, less jumpy output
            # Pass only the new message - LangGraph will handle conversation history via checkpointer
            for stream_item in self.agent.stream(
                {"messages": [HumanMessage(content=message)]},
                config={"configurable": {"thread_id": self.thread_id}},
                stream_mode="updates"
            ):
                # stream_item is a dict mapping node_name -> node_output
                if not isinstance(stream_item, dict):
                    continue

                # Process each node's update
                for node_name, node_data in stream_item.items():
                    if not isinstance(node_data, dict):
                        continue

                    # Extract messages from the update
                    if "messages" in node_data:
                        messages = node_data["messages"]
                        if not isinstance(messages, list):
                            messages = [messages]

                        for msg in messages:
                            # Handle AI messages
                            if hasattr(msg, 'type') and msg.type == 'ai':
                                last_ai_message = msg

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

                                        if tool_id not in seen_tool_calls:
                                            seen_tool_calls.add(tool_id)

                                            # Add newline if we just printed content
                                            if has_content and final_response and not final_response.endswith('\n'):
                                                console.print()

                                            tool_name = tool_call.get('name', 'unknown')
                                            tool_args = tool_call.get('args', {})

                                            # Format tool call nicely - but only show args if they're concise
                                            args_str = json.dumps(tool_args)
                                            if len(args_str) > 150:
                                                # Too verbose, just show the tool name with emoji
                                                console.print(f"\n[#beb4fd]🔧 {tool_name}[/#beb4fd]")
                                            else:
                                                # Show args nicely if they're reasonable
                                                console.print(f"\n[#beb4fd]🔧 {tool_name}[/#beb4fd]")
                                                if tool_args:
                                                    for key, value in tool_args.items():
                                                        if isinstance(value, str) and len(value) > 60:
                                                            console.print(f"  [dim]{key}:[/dim] {value[:60]}...")
                                                        else:
                                                            console.print(f"  [dim]{key}:[/dim] {value}")

                            # Handle tool response messages
                            elif hasattr(msg, 'type') and msg.type == 'tool':
                                # Get tool name from tool_call_id by looking up in recent AI message
                                tool_name = 'unknown'
                                if last_ai_message and hasattr(last_ai_message, 'tool_calls'):
                                    tool_call_id = getattr(msg, 'tool_call_id', None)
                                    for tc in last_ai_message.tool_calls:
                                        if tc.get('id') == tool_call_id:
                                            tool_name = tc.get('name', 'unknown')
                                            break

                                tool_content = getattr(msg, 'content', '')

                                # Parse and display tool results
                                try:
                                    result = json.loads(tool_content) if isinstance(tool_content, str) else tool_content

                                    # Special formatting for execute_code_snippet (handle first, before generic error check)
                                    if tool_name == "execute_code_snippet":
                                        # Result is a JSON string
                                        try:
                                            exec_result = json.loads(result) if isinstance(result, str) else result

                                            # Get the language from tool args
                                            language = "unknown"
                                            if last_ai_message and hasattr(last_ai_message, 'tool_calls'):
                                                tool_call_id = getattr(msg, 'tool_call_id', None)
                                                for tc in last_ai_message.tool_calls:
                                                    if tc.get('id') == tool_call_id:
                                                        language = tc.get('args', {}).get('language', 'unknown')
                                                        break

                                            if exec_result.get("success"):
                                                output = exec_result.get("output", "")
                                                preview = output[:100] + "..." if len(output) > 100 else output
                                                console.print(f"[#2f6868]  ✓ {language}[/#2f6868]", end="")
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
                                                console.print(f"[red]  ✗ {language}[/red] [dim]→[/dim] {error_preview}")
                                        except (json.JSONDecodeError, TypeError):
                                            console.print(f"[#2f6868]  ✓[/#2f6868]")
                                        continue

                                    # Check for errors in other tools
                                    if isinstance(result, dict) and "error" in result:
                                        console.print(f"[red]  ✗ Error: {result['error']}[/red]\n")
                                        continue

                                    # Special formatting for list_snippets
                                    if tool_name == "list_snippets" and isinstance(result, dict):
                                        total = result.get("total_snippets", 0)
                                        console.print(f"[#2f6868]  ✅ Found {total} snippet{'s' if total != 1 else ''}[/#2f6868]")

                                        snippets = result.get("snippets", [])
                                        if snippets:
                                            for i, snip in enumerate(snippets, 1):
                                                lang = snip.get("language", "unknown")
                                                lines = snip.get("lines", "?")
                                                console.print(f"    [dim]{i}.[/dim] [cyan]{lang}[/cyan] [dim](lines {lines})[/dim]")
                                        console.print()

                                    # Special formatting for finalize_review - show diff
                                    elif tool_name == "finalize_review":
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
                                    elif tool_name == "find_markdown_files" and isinstance(result, dict):
                                        count = result.get("count", 0)
                                        console.print(f"[#2f6868]  ✅ Found {count} markdown file{'s' if count != 1 else ''}[/#2f6868]")
                                        files = result.get("files", [])
                                        if files:
                                            for f in files[:10]:  # Show first 10
                                                console.print(f"    [dim]•[/dim] {f}")
                                            if len(files) > 10:
                                                console.print(f"    [dim]... and {len(files) - 10} more[/dim]")
                                        console.print()

                                    # Generic success for other tools
                                    elif isinstance(result, dict) and result.get("success"):
                                        console.print(f"[#2f6868]  ✅ {tool_name}[/#2f6868]\n")

                                    else:
                                        # Default: just show completion
                                        console.print(f"[#2f6868]  ✅[/#2f6868]\n")

                                except (json.JSONDecodeError, TypeError):
                                    # Non-JSON response, just show completion
                                    console.print(f"[dim]  ✓[/dim]\n")

                    # Handle todos from state
                    if "todos" in node_data:
                        todos = node_data["todos"]
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
