"""Price parsing for the sample shop."""


def parse_amount(value: str) -> int:
    """Convert a storefront price string to cents. Prices are whole dollars, such as "12"."""
    return int(value) * 100
