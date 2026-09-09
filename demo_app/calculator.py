"""
Calculator module for the demo application.
Provides arithmetic operations with configurable precision.
"""

from typing import Union


class Calculator:
    """A simple arithmetic calculator with configurable floating point precision."""

    def __init__(self, precision: int = 2) -> None:
        """Initialize the calculator with a rounding precision."""
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        """Return the sum of two numbers rounded to precision."""
        return round(a + b, self.precision)

    def subtract(self, a: float, b: float) -> float:
        """Return the difference between two numbers rounded to precision."""
        return round(a - b, self.precision)

    def multiply(self, a: float, b: float) -> float:
        """Return the product of two numbers rounded to precision."""
        return round(a * b, self.precision)

    def divide(self, a: float, b: float) -> float:
        """Return the division of a by b. Raises ValueError if b is zero."""
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        return round(a / b, self.precision)

    def power(self, base: float, exp: float) -> float:
        """Return base raised to the power of exp."""
        return round(base ** exp, self.precision)

