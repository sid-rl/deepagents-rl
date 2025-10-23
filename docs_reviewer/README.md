# Docs Reviewer

AI-powered documentation code reviewer built with [DeepAgents](https://github.com/langchain-ai/deepagents).

Reviews markdown files, tests code snippets, and fixes broken examples automatically.

## Quick Start

```bash
# Install
pip install -e .

# Set API key
export ANTHROPIC_API_KEY='your-key'

# Run
docs-reviewer chat
```

## Usage

```bash
# Interactive mode
docs-reviewer chat

# Single message
docs-reviewer chat -m "Review README.md"

# Disable MCP (LangChain docs access)
docs-reviewer chat --no-mcp
```

## How it Works

1. **Extract** - Parses code snippets from markdown
2. **Test** - Runs each snippet in isolated environments
3. **Fix** - Uses AI to fix broken code (with access to LangChain docs via MCP)
4. **Report** - Shows diffs and applies fixes

## Features

- ✅ Tests Python, JavaScript, and Bash code snippets
- ✅ Parallel testing for speed
- ✅ Conservative fixes (only changes what's broken)
- ✅ Access to LangChain docs via MCP for fixing LangChain code
- ✅ Inline editing (modifies files directly)
- ✅ Clear diffs showing what changed

## Architecture

Built on [DeepAgents](https://github.com/langchain-ai/deepagents):
- **Main Agent**: Coordinates review, spawns subagents
- **Subagents**: Test and fix individual snippets in parallel
- **MCP Integration**: Subagents can fetch LangChain docs when fixing LangChain code

## Requirements

- Python >=3.11
- Anthropic API key
- `uvx` for code execution (via `uv`: `pip install uv`)
