"""
Formatter module for the demo application.
Provides output presentation utilities such as currency, percentages, and JSON summaries.
"""

import json
from typing import Any, Dict, List


def format_currency(value: float, currency: str = "$") -> str:
    """Format a numeric value as a currency string with two decimal places."""
    return f"{currency}{value:,.2f}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format a floating point ratio (e.g. 0.15) as a percentage string (e.g. 15.00%)."""
    percent = value * 100.0
    return f"{percent:.{decimal_places}f}%"


class ResultFormatter:
    """Encapsulates structured formatting for calculator results."""

    def __init__(self, prefix: str = "Result:") -> None:
        """Initialize the formatter with a prefix label."""
        self.prefix = prefix

    def format_line(self, operation: str, result: float) -> str:
        """Return a formatted single-line representation of an operation result."""
        return f"{self.prefix} {operation} = {result}"

    def to_json(self, operation: str, operands: List[float], result: float) -> str:
        """Convert calculation metadata into a structured JSON string."""
        payload: Dict[str, Any] = {
            "operation": operation,
            "operands": operands,
            "result": result,
            "status": "success",
        }
        return json.dumps(payload, indent=2)
