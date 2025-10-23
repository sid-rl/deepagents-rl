"""Configuration management for the docs reviewer."""

import os
from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel, Field


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""

    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class ToolConfig(BaseModel):
    """Configuration for default tools."""

    enable_bash: bool = True
    enable_python: bool = True
    enable_filesystem: bool = True
    custom_tools: list[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    """Configuration for the agent behavior."""

    model: str = "claude-sonnet-4-5-20250929"
    temperature: float = 0.0
    max_iterations: int = 50
    enable_subagents: bool = True
    enable_todos: bool = True
    sandbox_mode: bool = True


class DocsReviewerConfig(BaseModel):
    """Main configuration for the docs reviewer."""

    # API Keys
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key. Can also be set via ANTHROPIC_API_KEY environment variable.",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (optional). Can also be set via OPENAI_API_KEY environment variable.",
    )

    # Agent configuration
    agent: AgentConfig = Field(default_factory=AgentConfig)

    # Tools configuration
    tools: ToolConfig = Field(default_factory=ToolConfig)

    # MCP Servers
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)

    # Output settings
    verbose: bool = False
    save_intermediate_results: bool = False

    def get_anthropic_api_key(self) -> str:
        """Get Anthropic API key from config or environment."""
        return self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from config or environment."""
        return self.openai_api_key or os.environ.get("OPENAI_API_KEY")


def init_config(config_path: Path) -> None:
    """Create a default configuration file."""
    default_config = {
        "anthropic_api_key": None,  # Set to your API key or use ANTHROPIC_API_KEY env var
        "openai_api_key": None,  # Optional: set to your API key or use OPENAI_API_KEY env var
        "agent": {
            "model": "claude-sonnet-4-5-20250929",
            "temperature": 0.0,
            "max_iterations": 50,
            "enable_subagents": True,
            "enable_todos": True,
            "sandbox_mode": True,
        },
        "tools": {
            "enable_bash": True,
            "enable_python": True,
            "enable_filesystem": True,
            "custom_tools": [],
        },
        "mcp_servers": [
            {
                "name": "langchain-docs",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-langchain-docs"],
                "env": {},
            }
        ],
        "verbose": False,
        "save_intermediate_results": False,
    }

    with open(config_path, "w") as f:
        yaml.safe_dump(default_config, f, default_flow_style=False, sort_keys=False)


def load_config(config_path: Path) -> DocsReviewerConfig:
    """Load configuration from a YAML file."""
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    return DocsReviewerConfig(**config_data)


def save_config(config: DocsReviewerConfig, config_path: Path) -> None:
    """Save configuration to a YAML file."""
    with open(config_path, "w") as f:
        yaml.safe_dump(config.model_dump(exclude_none=True), f, default_flow_style=False, sort_keys=False)
