# Docs Reviewer CLI

A conversational AI-powered CLI tool built on DeepAgents for reviewing and validating documentation code snippets. Just tell it what you want to do in natural language!

## Features

- **🤖 Conversational Interface**: Talk to the AI naturally - no complex commands to remember
- **📝 Automatic Code Extraction**: Parses markdown files and extracts all code snippets
- **🔒 Sandbox Execution**: Safely executes code snippets in an isolated environment
- **🧠 Smart Validation**: Uses AI-powered agents to validate syntax and semantics
- **🔗 LangChain Integration**: Connect to LangChain docs via MCP server for context-aware validation
- **✨ Automated Corrections**: Generates corrected markdown files with fixed code snippets
- **📊 Detailed Reports**: Provides comprehensive analysis of each code snippet

## Installation

### Prerequisites
- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- Anthropic API key

### Quick Install with uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to docs_reviewer directory
cd /Users/sydney_runkle/oss/deepagents/docs_reviewer

# Install in editable mode (includes deepagents as local dependency)
uv pip install -e .

# Set your API key
export ANTHROPIC_API_KEY="your-key-here"

# Make it permanent (optional)
echo 'export ANTHROPIC_API_KEY="your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### Alternative: Install with pip

```bash
# Navigate to deepagents root
cd /Users/sydney_runkle/oss/deepagents

# Install deepagents
pip install -e .

# Navigate to docs_reviewer
cd docs_reviewer

# Install docs_reviewer
pip install -e .

# Set API key
export ANTHROPIC_API_KEY="your-key-here"
```

### Verify Installation

```bash
docs-reviewer --version
```

You should see: `docs-reviewer version 0.1.0`

## Quick Start

### 1. Start the CLI

```bash
docs-reviewer
```

You'll see a welcome message and can start chatting with the AI!

### 2. Tell it what you want

Just describe what you want to do in natural language:

```
You: Review the file docs/tutorial.md

You: Show me all the code snippets in README.md

You: Check the Python code in my documentation folder

You: Find all markdown files in the current directory
```

### 3. The AI does the work!

The agent will:
1. Understand your request
2. Extract and analyze code snippets
3. Execute them safely in a sandbox
4. Generate corrected markdown files
5. Provide detailed feedback

## Usage Examples

### Interactive Mode (Default)

```bash
docs-reviewer
```

**Example conversation:**
```
You: Review the file example_docs.md

Agent: I'll review that file for you. Let me first see what code snippets are in there...

[The agent uses tools to extract, analyze, and review the code]

Agent: I found 6 code snippets in example_docs.md. Here's what I found:
- 3 Python snippets
- 1 JavaScript snippet
- 1 Bash snippet
- 1 text snippet (non-executable)

I've completed the review and generated:
- Corrected file: example_docs_corrected.md
- Detailed report: example_docs_review_report.md

2 out of 5 executable snippets had issues that I've fixed.
```

### Single Command Mode

Process one request and exit:

```bash
docs-reviewer chat --message "Review docs/tutorial.md"
```

### Initialize Configuration (Optional)

```bash
docs-reviewer init
```

Creates a `docs_reviewer_config.yaml` file you can customize.

## Configuration

The configuration file (`docs_reviewer_config.yaml`) supports the following options:

```yaml
# API Keys
anthropic_api_key: null  # Or use ANTHROPIC_API_KEY env var
openai_api_key: null     # Optional

# Agent Configuration
agent:
  model: "claude-sonnet-4-5-20250929"
  temperature: 0.0
  max_iterations: 50
  enable_subagents: true
  enable_todos: true
  sandbox_mode: true

# Tools Configuration
tools:
  enable_bash: true
  enable_python: true
  enable_filesystem: true
  custom_tools: []

# MCP Servers
mcp_servers:
  - name: langchain-docs
    command: npx
    args:
      - "-y"
      - "@modelcontextprotocol/server-langchain-docs"
    env: {}

# Output Settings
verbose: false
save_intermediate_results: false
```

## Architecture

The docs reviewer consists of several components:

- **CLI (`main.py`)**: Typer-based command-line interface
- **Agent (`agent.py`)**: DeepAgent integration for code review
- **Parser (`markdown_parser.py`)**: Extracts code snippets from markdown
- **Writer (`markdown_writer.py`)**: Generates corrected markdown files
- **Config (`config.py`)**: Configuration management
- **MCP Integration (`mcp_integration.py`)**: Model Context Protocol server integration

## How It Works

1. **Extraction**: The parser identifies all code fences in your markdown file
2. **Filtering**: Only executable code snippets are selected (Python, JavaScript, etc.)
3. **Analysis**: The DeepAgent analyzes each snippet for:
   - Syntax validity
   - Runtime errors
   - Missing imports or dependencies
   - LangChain/LangGraph specific issues
4. **Execution**: Safe execution in a sandboxed environment
5. **Correction**: AI suggests fixes for failing snippets
6. **Output**: Generates corrected markdown and detailed reports

## Custom Tools

You can extend the agent with custom tools:

```python
from langchain_core.tools import tool

@tool
def my_custom_validator(code: str) -> dict:
    """Custom validation logic."""
    # Your validation code here
    return {"valid": True, "message": "OK"}
```

## LangChain Docs Integration

The tool can connect to the LangChain documentation MCP server to:
- Look up API references
- Find working examples
- Validate against current best practices
- Check for deprecated APIs

Enable this by ensuring the `langchain-docs` MCP server is configured in your config file.

## Examples

### Example 1: Review a Tutorial

```bash
python -m docs_reviewer.main review docs/getting-started.md
```

### Example 2: Batch Processing

```bash
for file in docs/*.md; do
    python -m docs_reviewer.main review "$file"
done
```

### Example 3: CI/CD Integration

```yaml
# .github/workflows/docs-review.yml
name: Docs Review
on: [push]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -e ".[dev]"
      - run: python -m docs_reviewer.main review docs/
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Troubleshooting

### "Command not found: docs-reviewer"
Make sure the package is installed:
```bash
uv pip list | grep docs-reviewer
```
If not found, run `uv pip install -e .` from the docs_reviewer directory.

### "Config file not found"
Run `docs-reviewer init` to create a configuration file.

### "Anthropic API key not found"
Set your API key as an environment variable (recommended):
```bash
export ANTHROPIC_API_KEY=your-api-key-here
```
Or add it to your config file via `docs-reviewer init`.

Verify it's set:
```bash
echo $ANTHROPIC_API_KEY
```

### "No code snippets found"
Make sure your markdown file uses standard code fences:
````markdown
```python
print("Hello, world!")
```
````

### "Module not found: deepagents"
Install deepagents from the parent directory:
```bash
cd /Users/sydney_runkle/oss/deepagents
uv pip install -e .
cd docs_reviewer
uv pip install -e .
```

### Import errors
Ensure you're using the same Python environment:
```bash
which python
uv python list
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
ruff check docs_reviewer/
ruff format docs_reviewer/
```

## License

MIT License - See LICENSE file for details
