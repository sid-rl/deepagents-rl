# DeepAgents Project Guide

## Project Overview

**DeepAgents** is a Python package that implements "deep agents" - LLM-based agents that can handle complex, multi-step tasks through:
1. **Planning tools** (todo list management)
2. **Sub-agent spawning** (context isolation and specialization)
3. **File system access** (short-term and long-term memory)
4. **Detailed prompting**

Inspired by Claude Code, Manus, and Deep Research architectures.

**Current Version:** 0.1.3
**License:** MIT
**Python:** >=3.11,<4.0

---

## Project Structure

```
deepagents/
├── src/deepagents/
│   ├── __init__.py              # Main exports
│   ├── graph.py                 # Core create_deep_agent() factory
│   ├── middleware/
│   │   ├── filesystem.py        # FilesystemMiddleware (ls, read_file, write_file, edit_file)
│   │   ├── subagents.py         # SubAgentMiddleware (task spawning)
│   │   └── patch_tool_calls.py  # PatchToolCallsMiddleware
│   └── memory/
│       └── backends/            # Memory backend implementations
├── examples/
│   └── research/
│       └── research_agent.py    # Example research agent
├── tests/
│   ├── integration_tests/       # Integration tests
│   └── test_middleware.py       # Middleware unit tests
├── docs_reviewer/               # Untracked docs review tool
└── pyproject.toml               # Package configuration
```

---

## Key Components

### 1. Core Factory: `create_deep_agent()`
**Location:** `src/deepagents/graph.py:39`

Main entry point for creating deep agents. Returns a `CompiledStateGraph` (LangGraph graph).

**Key Parameters:**
- `model`: LLM to use (defaults to `claude-sonnet-4-5-20250929`)
- `tools`: List of tools for the agent
- `system_prompt`: Custom instructions (appended to base prompt)
- `middleware`: Additional middleware beyond defaults
- `subagents`: List of specialized sub-agents
- `use_longterm_memory`: Enable persistent memory (requires `store`)
- `interrupt_on`: Human-in-the-loop tool approval configs
- `checkpointer`: State persistence between runs
- `store`: Long-term memory storage backend

**Default Middleware Stack** (applied automatically):
1. `TodoListMiddleware` - Task planning and tracking
2. `FilesystemMiddleware` - File system tools
3. `SubAgentMiddleware` - Sub-agent spawning
4. `SummarizationMiddleware` - Context management (170k token threshold)
5. `AnthropicPromptCachingMiddleware` - Prompt caching
6. `PatchToolCallsMiddleware` - Tool call patching
7. `HumanInTheLoopMiddleware` - Optional HITL (if `interrupt_on` set)

### 2. TodoListMiddleware
Provides `write_todos` tool for planning and tracking multi-step tasks. Enables agents to break down complex problems and adapt plans dynamically.

### 3. FilesystemMiddleware
**Tools provided:**
- `ls`: List files in filesystem
- `read_file`: Read entire file or specific lines
- `write_file`: Create new files
- `edit_file`: Modify existing files

**Supports:**
- Short-term memory (in-state, per-thread)
- Long-term memory (via `BaseStore`, cross-thread)

### 4. SubAgentMiddleware
Enables spawning specialized sub-agents via `task` tool.

**SubAgent Definition:**
```python
{
    "name": str,                    # Unique identifier
    "description": str,             # What it does (for main agent)
    "prompt": str,                  # System prompt for subagent
    "tools": list,                  # Tools available to subagent
    "model": str | BaseChatModel,   # Optional custom model
    "middleware": list,             # Optional additional middleware
}
```

**CompiledSubAgent:** For pre-built LangGraph graphs
```python
{
    "name": str,
    "description": str,
    "runnable": Runnable,  # Pre-compiled graph
}
```

---

## Development Workflow

### Installation
```bash
# Standard install
pip install deepagents

# Development install (from repo root)
pip install -e ".[dev]"
```

### Dependencies
**Core:**
- `langchain>=1.0.0,<2.0.0`
- `langchain-core>=1.0.0,<2.0.0`
- `langchain-anthropic>=1.0.0,<2.0.0`

**Dev:**
- `pytest`, `pytest-cov`
- `langchain-openai`
- `ruff` (linting)
- `mypy` (type checking)

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run integration tests only
pytest tests/integration_tests/
```

### Code Quality
```bash
# Lint with ruff
ruff check .

# Format with ruff
ruff format .

# Type check with mypy
mypy src/deepagents
```

**Linting Config:**
- Line length: 150
- Convention: Google-style docstrings
- Strict mode: Enabled for mypy

---

## Common Patterns

### Basic Agent Creation
```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    tools=[my_tool1, my_tool2],
    system_prompt="You are an expert at X. Do Y."
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Do something"}]
})
```

### With Subagents
```python
subagents = [{
    "name": "researcher",
    "description": "Deep research specialist",
    "prompt": "You are a thorough researcher.",
    "tools": [search_tool],
    "model": "gpt-4o"
}]

agent = create_deep_agent(
    subagents=subagents,
    system_prompt="Coordinate research tasks."
)
```

### With Long-term Memory
```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
agent = create_deep_agent(
    store=store,
    use_longterm_memory=True
)
```

### With Human-in-the-Loop
```python
agent = create_deep_agent(
    tools=[sensitive_tool],
    interrupt_on={
        "sensitive_tool": {
            "allowed_decisions": ["approve", "edit", "reject"]
        }
    }
)
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/deepagents/graph.py` | Core agent factory |
| `src/deepagents/middleware/filesystem.py` | File system tools |
| `src/deepagents/middleware/subagents.py` | Sub-agent spawning |
| `examples/research/research_agent.py` | Example implementation |
| `tests/integration_tests/test_deepagents.py` | Integration test examples |
| `pyproject.toml` | Package config & dependencies |

---

## Testing Strategy

**Unit Tests:** `tests/test_middleware.py`
- Test individual middleware components
- Mock dependencies

**Integration Tests:** `tests/integration_tests/`
- `test_deepagents.py` - End-to-end agent tests
- `test_filesystem_middleware.py` - File system operations
- `test_subagent_middleware.py` - Sub-agent spawning
- `test_hitl.py` - Human-in-the-loop workflows

---

## Common Tasks

### Adding a New Middleware
1. Create middleware class in `src/deepagents/middleware/`
2. Inherit from `AgentMiddleware`
3. Implement required methods
4. Add tests in `tests/test_middleware.py`
5. Update exports in `src/deepagents/middleware/__init__.py`

### Adding a New Tool
1. Define tool function with proper type hints
2. Add docstring (used by LLM for tool selection)
3. Pass to `create_deep_agent()` via `tools` parameter
4. Add integration test demonstrating usage

### Modifying Core Agent
1. Edit `src/deepagents/graph.py`
2. Update middleware stack if needed
3. Run full test suite
4. Update README.md if user-facing changes

### Releasing a New Version
1. Update version in `pyproject.toml`
2. Run `python -m build`
3. Run `twine upload dist/*`

---

## Branch Information

**Current Branch:** `sr/docs-editor`
**Main Branch:** `master`

**Recent Changes:**
- Latest: Release patch with unpinned versions (#187)
- Bumped langchain dependency versions
- Fixed middleware accumulation bug

**Untracked Files (on current branch):**
- `docs_reviewer/` - Documentation review tool
- Various markdown guides (DEEPAGENT_INTEGRATION_SUMMARY.md, etc.)

---

## Important Notes

### Sync vs Async
- Use `create_deep_agent` for both sync and async
- `async_create_deep_agent` has been deprecated and folded into main factory
- MCP tools require async usage (see README section)

### MCP Integration
- Compatible via `langchain-mcp-adapters`
- Use async agent creation
- Pass MCP tools to `tools` parameter

### Context Management
- Default summarization at 170k tokens
- Keeps last 6 messages
- Use filesystem to offload large results

### Recursion Limit
- Default: 1000 steps
- Set in `graph.py:146`

---

## Useful Commands

```bash
# Run tests
pytest

# Run specific test file
pytest tests/integration_tests/test_deepagents.py

# Lint code
ruff check src/

# Format code
ruff format src/

# Type check
mypy src/deepagents

# Build package
python -m build

# Install in editable mode
pip install -e .
```

---

## Tips for Working with DeepAgents

1. **Prompting is Critical:** The system prompt heavily influences agent behavior
2. **Use Subagents for Isolation:** Keep main agent context clean
3. **Leverage the Filesystem:** Offload variable-length results to files
4. **Plan with Todos:** Complex tasks benefit from explicit task tracking
5. **Test with Integration Tests:** Unit tests alone miss interaction patterns
6. **Monitor Context Usage:** Watch for summarization triggers at 170k tokens

---

## Questions or Issues?

- GitHub Issues: https://github.com/langchain-ai/deepagents/issues
- LangChain Docs: https://python.langchain.com/
- LangGraph Docs: https://langchain-ai.github.io/langgraph/
