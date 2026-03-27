"""Agent state management.

This module provides the AgentState class for managing
dynamic agent state with snapshot capabilities.
"""

from typing import Any


class AgentState:
    """A container for dynamic agent state with snapshot support.

    AgentState allows setting arbitrary attributes that are stored
    internally and can be snapshotted for inspection or persistence.

    All attributes set on the instance (except _data) are stored in
    an internal dictionary, allowing for dynamic state management.

    Example:
        state = AgentState()
        state.current_task = "processing"
        state.confidence = 0.95
        snapshot = state.snapshot()  # {"current_task": "processing", "confidence": 0.95}

    Attributes:
        _data: Internal dictionary storing all state attributes
    """

    def __init__(self) -> None:
        """Initialize an empty agent state."""
        self._data: dict[str, Any] = {}

    def __setattr__(self, key: str, value: Any) -> None:
        """Set an attribute, storing it in the internal data dict.

        Args:
            key: The attribute name
            value: The attribute value
        """
        if key == "_data":
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __getattr__(self, key: str) -> Any:
        """Get an attribute from the internal data dict.

        Args:
            key: The attribute name

        Returns:
            The attribute value

        Raises:
            AttributeError: If the attribute doesn't exist
        """
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{key}'")

    def snapshot(self) -> dict[str, Any]:
        """Create a snapshot of the current state.

        Returns:
            A copy of all state attributes as a dictionary
        """
        return dict(self._data)

    def clear(self) -> None:
        """Clear all state attributes."""
        self._data.clear()

    def update(self, data: dict[str, Any]) -> None:
        """Update state with multiple attributes at once.

        Args:
            data: Dictionary of attributes to set
        """
        self._data.update(data)
