from math import sqrt


class Calculator:
    def __init__(self, precision: int = 2) -> None:
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        return round(a + b, self.precision)

    def subtract(self, a: float, b: float) -> float:
        return round(a - b, self.precision)

    def multiply(self, a: float, b: float) -> float:
        return round(a * b, self.precision)

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError('Division by zero is not allowed.')
        return round(a / b, self.precision)

    def power(self, base: float, exp: float) -> float:
        return round(base ** exp, self.precision)

    def sqrt(self, value: float) -> float:
        return round(sqrt(value), self.precision)