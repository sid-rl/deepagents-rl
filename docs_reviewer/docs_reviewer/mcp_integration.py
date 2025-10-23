"""MCP (Model Context Protocol) server integration."""

import subprocess
from typing import Any
from docs_reviewer.config import MCPServerConfig


class MCPServerManager:
    """Manages MCP server connections for the docs reviewer."""

    def __init__(self, server_configs: list[MCPServerConfig]):
        """
        Initialize MCP server manager.

        Args:
            server_configs: List of MCP server configurations
        """
        self.server_configs = server_configs
        self.active_servers: dict[str, subprocess.Popen] = {}

    def start_server(self, server_name: str) -> bool:
        """
        Start an MCP server.

        Args:
            server_name: Name of the server to start

        Returns:
            True if server started successfully
        """
        config = next((s for s in self.server_configs if s.name == server_name), None)
        if not config:
            raise ValueError(f"Server {server_name} not found in configuration")

        try:
            # Start the MCP server process
            process = subprocess.Popen(
                [config.command] + config.args,
                env=config.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.active_servers[server_name] = process
            return True
        except Exception as e:
            print(f"Failed to start MCP server {server_name}: {e}")
            return False

    def stop_server(self, server_name: str) -> bool:
        """
        Stop an MCP server.

        Args:
            server_name: Name of the server to stop

        Returns:
            True if server stopped successfully
        """
        if server_name not in self.active_servers:
            return False

        process = self.active_servers[server_name]
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

        del self.active_servers[server_name]
        return True

    def stop_all_servers(self) -> None:
        """Stop all active MCP servers."""
        for server_name in list(self.active_servers.keys()):
            self.stop_server(server_name)

    def query_langchain_docs(self, query: str) -> dict[str, Any]:
        """
        Query the LangChain documentation MCP server.

        Args:
            query: The search query

        Returns:
            Dictionary with search results
        """
        # This is a placeholder - actual implementation would use the MCP protocol
        # to communicate with the LangChain docs server
        #
        # The MCP protocol typically involves:
        # 1. Sending a JSON-RPC request to the server
        # 2. Receiving structured responses
        # 3. Parsing the documentation content
        #
        # For now, we'll return a placeholder structure
        return {
            "query": query,
            "results": [],
            "status": "MCP integration pending - requires JSON-RPC implementation",
        }

    def __enter__(self) -> "MCPServerManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - clean up servers."""
        self.stop_all_servers()


def create_langchain_docs_tool(mcp_manager: MCPServerManager):
    """
    Create a LangChain tool that queries the LangChain docs MCP server.

    Args:
        mcp_manager: MCP server manager instance

    Returns:
        A LangChain tool for querying documentation
    """
    from langchain_core.tools import tool

    @tool
    def search_langchain_docs(query: str) -> str:
        """
        Search the LangChain documentation for relevant information.

        This tool connects to the LangChain docs MCP server to retrieve
        up-to-date documentation, examples, and API references.

        Args:
            query: What to search for in the LangChain documentation

        Returns:
            Relevant documentation content and examples
        """
        results = mcp_manager.query_langchain_docs(query)

        # Format the results for the agent
        if results["status"] != "success":
            return f"Documentation search status: {results['status']}"

        if not results["results"]:
            return f"No documentation found for query: {query}"

        # Format multiple results
        formatted = f"LangChain Documentation Results for '{query}':\n\n"
        for i, result in enumerate(results["results"][:3], 1):  # Limit to top 3
            formatted += f"{i}. {result.get('title', 'Untitled')}\n"
            formatted += f"   {result.get('summary', '')}\n"
            if "url" in result:
                formatted += f"   URL: {result['url']}\n"
            formatted += "\n"

        return formatted

    return search_langchain_docs
