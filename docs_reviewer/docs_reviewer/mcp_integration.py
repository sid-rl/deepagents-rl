"""MCP (Model Context Protocol) server integration using langchain-mcp-adapters."""

from langchain_mcp_adapters.client import MultiServerMCPClient
from docs_reviewer.config import MCPServerConfig


async def get_langchain_docs_tools(server_configs: list[MCPServerConfig] | None = None) -> list:
    """
    Get web fetch MCP tools for accessing LangChain docs.

    Uses langchain-mcp-adapters' MultiServerMCPClient directly to connect to MCP servers.

    Default configuration uses mcp-server-fetch (Python-based) which provides a 'fetch' tool that can:
    - Fetch web pages and convert HTML to markdown
    - Access LangChain documentation at docs.langchain.com
    - Extract content in chunks if needed

    Args:
        server_configs: Optional list of MCP server configurations.
                       If None, uses default fetch server config.

    Returns:
        List of LangChain tools from MCP servers

    Example:
        ```python
        # Get tools
        mcp_tools = await get_langchain_docs_tools()

        # Use with deep agent
        agent = create_deep_agent(
            tools=base_tools + mcp_tools,
            system_prompt="You have access to web fetch tools to access LangChain docs."
        )
        ```

    Note:
        Requires mcp-server-fetch to be installed: `pip install mcp-server-fetch` or `uvx mcp-server-fetch`
    """
    if server_configs is None:
        # Default configuration using mcp-server-fetch (Python-based, no Node.js required)
        server_configs = [
            MCPServerConfig(
                name="fetch",
                command="uvx",
                args=["mcp-server-fetch"],
                env={},
            )
        ]

    # Convert server configs to the format expected by MultiServerMCPClient
    servers_dict = {}
    for config in server_configs:
        servers_dict[config.name] = {
            "command": config.command,
            "args": config.args,
            "env": config.env or {},
            "transport": "stdio",  # Required in langchain-mcp-adapters 0.1.0+
        }

    # Initialize the MCP client and get tools
    client = MultiServerMCPClient(servers_dict)
    tools = await client.get_tools()

    return tools
