"""
db/manager.py — SQLite database helper for rules

Purpose
-------
Provide a thin, reliable wrapper around SQLite for storing and retrieving
validation rules. Handles table creation, optional CSV bootstrap, safe
upserts, and robust reads with lock-aware retries.

Key Responsibilities
--------------------
- Initialize the rules table (and useful indexes) if absent.
- Optionally ingest a CSV into SQLite on first run.
- Expose simple CRUD helpers (read all, upsert).
- Apply pragmatic SQLite PRAGMAs for app usage (WAL, busy_timeout, etc.).
- Log clearly at INFO level; include defensive error handling & retries.

Public API
----------
- DatabaseManager(db_path, table_name)
- init_db(csv_path: Optional[str] = None) -> None
- load_rules_from_csv(csv_path: str) -> None
- get_rules() -> list[tuple]
- upsert_rule(rule: dict) -> None

Notes
-----
- Column order in SELECT is canonical and matches the loader expected by
  `RuleDataLoader`. If you add/remove columns, update BOTH places.
"""

from __future__ import annotations

import os
import json
import time
import sqlite3
import logging
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


# Canonical schema/columns (keep in sync with RuleDataLoader)
COLUMNS: list[str] = [
    "rule_id",
    "rule_name",
    "rule_description",
    "bansta_error_code",
    "iso_error_code",
    "description_en",
    "description_de",
    "rule_code",
    "llm_description",
    "keywords",
    "rule_type",
    "country",
    "business_type",
    "party_agent",
    "embedding",
    "relevance",
    "version_major",
    "version_minor",
    "created_at",
    "updated_at",
]

# Tags represented as comma-separated strings in DB, expanded to lists in loader
TAG_FIELDS = ["rule_type", "country", "business_type", "party_agent"]


class DatabaseManager:
    """Lightweight, logging-rich wrapper for a single SQLite rules table."""

    def __init__(self, db_path: str, table_name: str):
        """
        Args:
            db_path: Filesystem path to the SQLite database file.
            table_name: Name of the table to operate on.
        """
        self.db_path = db_path
        self.table_name = table_name

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        logger.info("DatabaseManager init: db=%r, table=%r", db_path, table_name)

    # ---------------------------------------------------------------------
    # Connection helpers
    # ---------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """
        Open a SQLite connection with pragmatic defaults for a web app.
        - WAL journal for better concurrency
        - Busy timeout to mitigate `database is locked`
        - Foreign keys enabled (future-proofing)
        """
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")  # 5s
        # Row remains tuple-like; suitable for zipping with COLUMNS
        return conn

    def _execute(
        self,
        conn: sqlite3.Connection,
        sql: str,
        params: Optional[Iterable[Any]] = None,
        *,
        commit: bool = False,
        retries: int = 3,
        retry_backoff: float = 0.25,
    ) -> sqlite3.Cursor:
        """
        Execute a single statement with simple lock-aware retry logic.
        """
        attempt = 0
        while True:
            try:
                cur = conn.execute(sql, tuple(params or []))
                if commit:
                    conn.commit()
                return cur
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                if "locked" in msg or "busy" in msg:
                    if attempt < retries:
                        delay = retry_backoff * (2**attempt)
                        logger.warning(
                            "SQLite is locked/busy; retrying in %.2fs (attempt %d/%d): %s",
                            delay, attempt + 1, retries, e,
                        )
                        time.sleep(delay)
                        attempt += 1
                        continue
                logger.exception("SQLite execution failed (no retry): %s", e)
                raise

    # ---------------------------------------------------------------------
    # Schema management
    # ---------------------------------------------------------------------
    def init_db(self, csv_path: Optional[str] = None) -> None:
        """
        Create table and indexes if missing, then (optionally) load from CSV.
        """
        logger.info("Initializing table %r in %r", self.table_name, self.db_path)
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS "{self.table_name}" (
            rule_id TEXT PRIMARY KEY,
            rule_name TEXT,
            rule_description TEXT,
            bansta_error_code TEXT,
            iso_error_code TEXT,
            description_en TEXT,
            description_de TEXT,
            rule_code TEXT,
            llm_description TEXT,
            keywords TEXT,
            rule_type TEXT,
            country TEXT,
            business_type TEXT,
            party_agent TEXT,
            embedding TEXT,    -- JSON (list[float]) as string
            relevance REAL,
            version_major INTEGER,
            version_minor INTEGER,
            created_at TEXT,
            updated_at TEXT
        );
        """

        # Helpful indexes for common filters
        index_sql = [
            f'CREATE INDEX IF NOT EXISTS idx_{self.table_name}_name ON "{self.table_name}"(rule_name);',
            f'CREATE INDEX IF NOT EXISTS idx_{self.table_name}_country ON "{self.table_name}"(country);',
            f'CREATE INDEX IF NOT EXISTS idx_{self.table_name}_rule_type ON "{self.table_name}"(rule_type);',
            f'CREATE INDEX IF NOT EXISTS idx_{self.table_name}_business_type ON "{self.table_name}"(business_type);',
            f'CREATE INDEX IF NOT EXISTS idx_{self.table_name}_party_agent ON "{self.table_name}"(party_agent);',
        ]

        with self._connect() as conn:
            self._execute(conn, create_sql, commit=True)
            for stmt in index_sql:
                self._execute(conn, stmt, commit=True)

        logger.info("Table %r ready.", self.table_name)

        if csv_path:
            self.load_rules_from_csv(csv_path)

    # ---------------------------------------------------------------------
    # CSV bootstrap
    # ---------------------------------------------------------------------
    def load_rules_from_csv(self, csv_path: str) -> None:
        """
        Load rules from CSV into the table **only if the table is empty**.

        The CSV does not need to include every column. Missing columns are
        filled with NULL/defaults. `rule_id` is generated if absent.
        """
        import uuid
        import pandas as pd
        from datetime import datetime, timezone

        logger.info("CSV bootstrap requested: %s", csv_path)
        if not os.path.isfile(csv_path):
            logger.warning("CSV not found: %s (skipping)", csv_path)
            return

        with self._connect() as conn:
            # Short-circuit if data already exist
            cur = self._execute(conn, f'SELECT COUNT(*) FROM "{self.table_name}";')
            count = int(cur.fetchone()[0])
            if count > 0:
                logger.info(
                    "Table %r already contains %d rows; skipping CSV load.",
                    self.table_name, count,
                )
                return

        # Read CSV
        df = pd.read_csv(csv_path, dtype=str).fillna("")  # keep as strings; loader normalizes further
        logger.info("Loaded CSV with %d rows and %d columns.", len(df), len(df.columns))
        if df.empty:
            logger.warning("CSV is empty; nothing to load.")
            return

        # Prepare INSERT with all canonical columns
        cols_sql = ",".join(f'"{c}"' for c in COLUMNS)
        placeholders = ",".join("?" for _ in COLUMNS)
        insert_sql = f'INSERT INTO "{self.table_name}" ({cols_sql}) VALUES ({placeholders});'

        def _row_to_values(row: dict) -> list[Any]:
            # Start with defaults
            now = datetime.now(timezone.utc).isoformat()
            values: dict[str, Any] = {c: None for c in COLUMNS}
            values["rule_id"] = row.get("rule_id") or str(uuid.uuid4())
            values["created_at"] = row.get("created_at") or now
            values["updated_at"] = row.get("updated_at") or now

            # Copy pass-through columns if present in CSV
            for c in COLUMNS:
                if c in ("rule_id", "created_at", "updated_at"):
                    continue
                if c in row and row[c] != "":
                    values[c] = row[c]

            # Normalize tag fields → comma-separated strings
            for t in TAG_FIELDS:
                v = values.get(t)
                if isinstance(v, list):
                    values[t] = ",".join(map(str, v))
                elif v is None:
                    values[t] = ""

            # Embedding → JSON string if it looks like a list
            emb = values.get("embedding")
            if isinstance(emb, (list, tuple)):
                values["embedding"] = json.dumps([float(x) for x in emb])
            # if string, keep as-is (assume already JSON or empty)

            # Relevance / versions fallbacks
            if values.get("relevance") in (None, ""):
                values["relevance"] = 1.0
            if values.get("version_major") in (None, ""):
                values["version_major"] = 1
            if values.get("version_minor") in (None, ""):
                values["version_minor"] = 0

            # Return in canonical order
            return [values[c] for c in COLUMNS]

        rows_to_insert = ( _row_to_values(rec) for rec in df.to_dict(orient="records") )

        # Bulk insert with chunking to keep memory stable
        inserted = 0
        batch: list[Sequence[Any]] = []
        batch_size = 1000

        with self._connect() as conn:
            for rec in rows_to_insert:
                batch.append(rec)
                if len(batch) >= batch_size:
                    conn.executemany(insert_sql, batch)
                    conn.commit()
                    inserted += len(batch)
                    logger.info("Inserted %d rows (running total).", inserted)
                    batch.clear()

            if batch:
                conn.executemany(insert_sql, batch)
                conn.commit()
                inserted += len(batch)

        logger.info("CSV bootstrap complete. Inserted %d row(s).", inserted)

    # ---------------------------------------------------------------------
    # Reads
    # ---------------------------------------------------------------------
    def get_rules(self) -> list[tuple]:
        """
        Return **all rules** as a list of tuples **in canonical column order**.
        """
        select_sql = f'SELECT {",".join(COLUMNS)} FROM "{self.table_name}";'
        with self._connect() as conn:
            cur = self._execute(conn, select_sql)
            rows = cur.fetchall()
        logger.info("Fetched %d rule(s).", len(rows))
        return rows

    # ---------------------------------------------------------------------
    # Writes
    # ---------------------------------------------------------------------
    def upsert_rule(self, rule: dict) -> None:
        """
        Insert or update a single rule using ON CONFLICT(rule_id) DO UPDATE.

        Only columns present in `rule` and recognized by the schema are used.
        """
        if "rule_id" not in rule or not rule["rule_id"]:
            raise ValueError("upsert_rule requires a non-empty 'rule_id'")

        # Filter unknown columns & coerce types where sensible
        payload: dict[str, Any] = {k: rule[k] for k in rule.keys() if k in COLUMNS}

        # Normalize tag fields to comma-separated strings
        for t in TAG_FIELDS:
            if t in payload and isinstance(payload[t], list):
                payload[t] = ",".join(map(str, payload[t]))

        # Embedding: list -> JSON
        if "embedding" in payload and isinstance(payload["embedding"], (list, tuple)):
            payload["embedding"] = json.dumps([float(x) for x in payload["embedding"]])

        # Timestamps
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        payload.setdefault("updated_at", now)
        payload.setdefault("created_at", now)

        # Build UPSERT
        cols = [c for c in COLUMNS if c in payload]
        cols_sql = ",".join(f'"{c}"' for c in cols)
        placeholders = ",".join("?" for _ in cols)

        # Update set excludes PK
        set_sql = ",".join(f'"{c}"=excluded."{c}"' for c in cols if c != "rule_id")

        sql = (
            f'INSERT INTO "{self.table_name}" ({cols_sql}) VALUES ({placeholders}) '
            f'ON CONFLICT(rule_id) DO UPDATE SET {set_sql};'
        )

        with self._connect() as conn:
            self._execute(conn, sql, [payload[c] for c in cols], commit=True)

        logger.info("Upserted rule_id=%s", payload["rule_id"])
