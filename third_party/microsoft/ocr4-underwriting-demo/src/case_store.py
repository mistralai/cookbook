"""Case store: a small interface plus a SQLite implementation for the demo.

The interface is the seam that lets the demo run on SQLite and production run on
Cosmos DB without touching the agents or the UI. Writes use optimistic concurrency
so the customer chat and a reviewer cannot clobber each other.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional, Protocol

from underwriting_schema import Stage, UnderwritingCase


class ConcurrencyError(RuntimeError):
    """Raised when a put's expected etag does not match the stored etag."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CaseStore(Protocol):
    def create(self, case: UnderwritingCase) -> UnderwritingCase: ...
    def get(self, case_id: str) -> Optional[UnderwritingCase]: ...
    def put(self, case: UnderwritingCase, expected_etag: Optional[str] = None) -> UnderwritingCase: ...
    def list_pending(self) -> list[UnderwritingCase]: ...
    def list_finalized(self) -> list[UnderwritingCase]: ...
    def list_by_owner(self, owner: str) -> list[UnderwritingCase]: ...


class SqliteCaseStore:
    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or os.environ.get("CASE_DB_PATH", "cases.db")
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self._path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS cases (
                   case_id TEXT PRIMARY KEY, owner TEXT, stage TEXT,
                   etag TEXT, updated_at TEXT, data TEXT)"""
        )
        self._db.commit()

    def _row_to_case(self, row) -> UnderwritingCase:
        case = UnderwritingCase.model_validate_json(row[0])
        case.etag = row[1]
        return case

    def create(self, case: UnderwritingCase) -> UnderwritingCase:
        with self._lock:
            case.etag = uuid.uuid4().hex
            case.created_at = case.created_at or _now()
            case.updated_at = _now()
            self._db.execute(
                "INSERT INTO cases(case_id, owner, stage, etag, updated_at, data) VALUES (?,?,?,?,?,?)",
                (case.case_id, case.owner, case.stage.value, case.etag, case.updated_at, case.model_dump_json()),
            )
            self._db.commit()
            return case

    def get(self, case_id: str) -> Optional[UnderwritingCase]:
        cur = self._db.execute("SELECT data, etag FROM cases WHERE case_id=?", (case_id,))
        row = cur.fetchone()
        return self._row_to_case(row) if row else None

    def put(self, case: UnderwritingCase, expected_etag: Optional[str] = None) -> UnderwritingCase:
        with self._lock:
            cur = self._db.execute("SELECT etag FROM cases WHERE case_id=?", (case.case_id,))
            row = cur.fetchone()
            if row is None:
                raise ConcurrencyError(f"case {case.case_id} does not exist")
            current = row[0]
            expected = expected_etag if expected_etag is not None else case.etag
            if expected is not None and expected != current:
                raise ConcurrencyError(f"etag mismatch for {case.case_id}: expected {expected}, found {current}")
            case.etag = uuid.uuid4().hex
            case.updated_at = _now()
            self._db.execute(
                "UPDATE cases SET owner=?, stage=?, etag=?, updated_at=?, data=? WHERE case_id=?",
                (case.owner, case.stage.value, case.etag, case.updated_at, case.model_dump_json(), case.case_id),
            )
            self._db.commit()
            return case

    def list_pending(self) -> list[UnderwritingCase]:
        cur = self._db.execute(
            "SELECT data, etag FROM cases WHERE stage IN (?,?) ORDER BY updated_at",
            (Stage.pending_review.value, Stage.in_review.value),
        )
        return [self._row_to_case(r) for r in cur.fetchall()]

    def list_finalized(self, limit: int = 200) -> list[UnderwritingCase]:
        cur = self._db.execute(
            "SELECT data, etag FROM cases WHERE stage=? ORDER BY updated_at DESC LIMIT ?",
            (Stage.finalized.value, limit),
        )
        return [self._row_to_case(r) for r in cur.fetchall()]

    def list_by_owner(self, owner: str) -> list[UnderwritingCase]:
        cur = self._db.execute(
            "SELECT data, etag FROM cases WHERE owner=? ORDER BY updated_at DESC", (owner,)
        )
        return [self._row_to_case(r) for r in cur.fetchall()]


_STORE: Optional[SqliteCaseStore] = None


def get_store() -> SqliteCaseStore:
    global _STORE
    if _STORE is None:
        _STORE = SqliteCaseStore()
    return _STORE


def new_case_id() -> str:
    return "case-" + uuid.uuid4().hex[:12]
