"""
Main entry point for the demo application.
Demonstrates arithmetic operations and formatting.
"""

import sys

try:
    from demo_app.calculator import Calculator
    from demo_app.formatter import ResultFormatter, format_currency, format_percentage
except ImportError:
    from calculator import Calculator
    from formatter import ResultFormatter, format_currency, format_percentage


def calculate_tax(amount: float, tax_rate: float) -> float:
    """Calculate the tax amount for a given base amount and rate."""
    calc = Calculator(precision=2)
    return calc.multiply(amount, tax_rate)


def run_demo() -> int:
    """Run calculation demonstrations and print formatted outputs."""
    calc = Calculator(precision=2)
    formatter = ResultFormatter(prefix="[DemoApp]")

    sum_res = calc.add(150.50, 49.50)
    print(formatter.format_line("Addition", sum_res))

    tax_rate = 0.20
    tax_amount = calculate_tax(sum_res, tax_rate)
    print(f"Tax rate: {format_percentage(tax_rate)} -> Tax: {format_currency(tax_amount)}")

    json_output = formatter.to_json("tax_calculation", [sum_res, tax_rate], tax_amount)
    print("JSON summary:")
    print(json_output)

    return 0


def main() -> int:
    """CLI entrypoint with exception handling."""
    try:
        return run_demo()
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected failure: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[*] Interrupted.", file=sys.stderr)
        sys.exit(130)
