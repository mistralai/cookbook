#!/usr/bin/env python3
"""Mistral OCR 4 (Azure Foundry) — file in, markdown out.

Thin CLI over ocr_client. Reads a PDF or image, calls the deployed
`mistral-ocr-4` endpoint, and writes the extracted markdown next to the input.

Config comes from the environment (see .env.example).

Usage:
    python ocr.py mydoc.pdf                 # -> mydoc.md (all pages)
    python ocr.py mydoc.pdf --pages 0-4     # first 5 pages only (zero-indexed)
    python ocr.py mydoc.pdf --tables html   # tables returned separately as HTML
"""
import argparse
import sys
from pathlib import Path

from ocr_client import ocr_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Mistral OCR 4 (Azure Foundry): file in, markdown out.")
    parser.add_argument("file", help="PDF or image to OCR")
    parser.add_argument("--pages", help="Pages to process, zero-indexed (e.g. '0,2-4'). Default: all.")
    parser.add_argument("--tables", choices=["markdown", "html"],
                        help="Return tables separately in this format instead of inline in the markdown.")
    args = parser.parse_args()

    try:
        markdown = ocr_file(args.file, pages=args.pages, table_format=args.tables)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.exit(str(exc))

    out = Path(args.file).with_suffix(".md")
    out.write_text(markdown, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
