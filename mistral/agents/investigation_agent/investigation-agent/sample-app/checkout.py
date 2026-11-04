"""Sends one checkout error to Sentry so the investigation agent has something to investigate."""

import logging
import os
import subprocess

import sentry_sdk

from pricing import parse_amount

CART = [{"sku": "mug", "price": "12"}, {"sku": "poster", "price": "12.50"}]


def order_total(items: list[dict[str, str]]) -> int:
    """Return the cart total in cents."""
    return sum(parse_amount(item["price"]) for item in items)


def git_sha() -> str | None:
    """Tag the event with the current commit so the agent can find the exact code on GitHub."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        raise SystemExit("SENTRY_DSN is not set. Add your Sentry project's DSN to ../.env.")
    sentry_sdk.init(dsn=dsn, release=git_sha(), environment="demo")
    try:
        order_total(CART)
    except ValueError:
        event_id = sentry_sdk.capture_exception()
        sentry_sdk.flush()
        logging.info("Sent a checkout error to Sentry (event %s).", event_id)


if __name__ == "__main__":
    main()
