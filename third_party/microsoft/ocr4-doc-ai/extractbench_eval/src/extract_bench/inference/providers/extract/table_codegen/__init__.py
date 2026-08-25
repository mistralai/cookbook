"""Internals for the table→schema code-generation extractor.

A model authors a Python extractor script against a parsed document's tables and
the (resolved) output schema; the script is gated for validity and run over the
full document in a sandboxed subprocess (``sandbox.py`` — scrubbed env, hard
timeout). See ``table_codegen_extract.py`` for the Provider that runs LlamaParse
Agentic first and drives this.
"""

from .document import Document, Page, load_document, render_document
from .harness import generate_extractor
from .items_document import Item, ItemsDocument, ItemsPage, render_items_document

__all__ = [
    "Document",
    "Item",
    "ItemsDocument",
    "ItemsPage",
    "Page",
    "generate_extractor",
    "load_document",
    "render_document",
    "render_items_document",
]
