"""CompositeBackend: Route operations to different backends based on path prefix."""

from typing import Any, Optional

from deepagents.memory.protocol import MemoryBackend
from langgraph.types import Command


class CompositeBackend:
    """Backend that routes operations to different backends based on path prefix.
    
    This backend enables hybrid storage strategies, such as:
    - Short-term files (/*) → StateBackend (ephemeral)
    - Long-term files (/memories/*) → StoreBackend or FilesystemBackend (persistent)
    
    The routing is transparent to tools - they just call backend.get(path) and
    CompositeBackend handles the routing internally.
    
    Example:
        ```python
        backend = CompositeBackend(
            default=StateBackend(runtime),
            routes={"/memories/": FilesystemBackend("/data/memories")}
        )
        
        # Routes to StateBackend
        backend.get("/temp.txt")
        
        # Routes to FilesystemBackend, strips prefix
        backend.get("/memories/notes.txt")  # → FilesystemBackend.get("/notes.txt")
        ```
    """
    
    def __init__(
        self,
        default: MemoryBackend,
        routes: dict[str, MemoryBackend],
    ) -> None:
        """Initialize composite backend with routing rules.
        
        Args:
            default: Default backend for paths that don't match any route.
            routes: Dict mapping path prefixes to backends. Keys should include
                   trailing slash (e.g., "/memories/").
        """
        self.default = default
        self.routes = routes
        
        # Sort routes by length (longest first) for correct prefix matching
        self.sorted_routes = sorted(routes.items(), key=lambda x: len(x[0]), reverse=True)

    def _get_backend_and_key(self, key: str) -> tuple[MemoryBackend, str]:
        """Determine which backend handles this key and strip prefix.
        
        Args:
            key: Original file path
        
        Returns:
            Tuple of (backend, stripped_key) where stripped_key has the route
            prefix removed (but keeps leading slash).
        """
        # Check routes in order of length (longest first)
        for prefix, backend in self.sorted_routes:
            if key.startswith(prefix):
                # Strip prefix but keep leading slash
                # e.g., "/memories/notes.txt" → "/notes.txt"
                stripped_key = key[len(prefix) - 1:] if key[len(prefix) - 1:] else "/"
                return backend, stripped_key
        
        return self.default, key
    
    def ls(self, prefix: Optional[str] = None) -> list[str]:
        """List files from all backends, with appropriate prefixes.
        
        Args:
            prefix: Optional path prefix to filter results.
        
        Returns:
            List of file paths with route prefixes added.
        """
        result: list[str] = []
        
        # If prefix matches a route, only query that backend
        if prefix is not None:
            for route_prefix, backend in self.sorted_routes:
                if prefix.startswith(route_prefix):
                    # Strip route prefix from search prefix
                    search_prefix = prefix[len(route_prefix) - 1:]
                    keys = backend.ls(search_prefix if search_prefix != "/" else None)
                    # Add route prefix back to results
                    result.extend(f"{route_prefix[:-1]}{key}" for key in keys)
                    return result
        
        # Query default backend
        default_keys = self.default.ls(prefix)
        result.extend(default_keys)
        
        # Query each routed backend
        for route_prefix, backend in self.routes.items():
            # Skip if prefix filter doesn't match this route
            if prefix is not None and not route_prefix.startswith(prefix):
                continue
            
            # Get keys from backend (without prefix in search)
            keys = backend.ls(None)
            # Add route prefix to each key
            result.extend(f"{route_prefix[:-1]}{key}" for key in keys)
        
        return result
    
    def read(
        self, 
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content, routing to appropriate backend.
        
        Args:
            file_path: Absolute file path
            offset: Line offset to start reading from (0-indexed)
            limit: Maximum number of lines to read
        
        Returns:
            Formatted file content with line numbers, or error message.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.read(stripped_key, offset=offset, limit=limit)
    
    def write(
        self, 
        file_path: str,
        content: str,
    ) -> Command | str:
        """Create a new file, routing to appropriate backend.
        
        Args:
            file_path: Absolute file path
            content: File content as a string
        
        Returns:
            Success message or Command object, or error if file already exists.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.write(stripped_key, content)
    
    def edit(
        self, 
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> Command | str:
        """Edit a file, routing to appropriate backend.
        
        Args:
            file_path: Absolute file path
            old_string: String to find and replace
            new_string: Replacement string
            replace_all: If True, replace all occurrences
        
        Returns:
            Success message or Command object, or error message on failure.
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.edit(stripped_key, old_string, new_string, replace_all=replace_all)
    
    def delete(self, file_path: str) -> Command | None:
        """Delete file, routing to appropriate backend.
        
        Args:
            file_path: File path to delete
        
        Returns:
            Return value from backend (None or Command).
        """
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.delete(stripped_key)