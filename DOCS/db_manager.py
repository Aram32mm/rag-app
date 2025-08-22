"""
db_manager.py — Database Manager

Purpose
-------
Provides SQLite database utilities for validation rules.  
Handles schema creation, rule ingestion (from CSV), fetching, and upserting.

Key Responsibilities
--------------------
- Initialize database and create `rules` table if missing.
- Load rules from CSV into SQLite (skipping if table already populated).
- Fetch all rules as row lists.
- Insert or update rules (upsert) by primary key.

Dependencies
------------
- sqlite3
- pandas (for CSV ingestion)
- logging
"""

import sqlite3
from typing import List, Any
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite persistence for rules with schema-aware utilities."""

    def __init__(self, db_path: str, table_name: str):
        """
        Initialize the DatabaseManager.

        Args
        ----
        db_path : str
            Path to SQLite database file.
        table_name : str
            Name of the table to manage.
        """
        self.db_path = db_path
        self.table_name = table_name
        logger.info("DatabaseManager initialized (db=%s, table=%s)", db_path, table_name)

    def get_conn(self):
        """Establish and return a connection to the SQLite database."""
        return sqlite3.connect(self.db_path)

    def init_db(self, csv_path: str = None):
        """
        Create the rules table if it does not exist and optionally load from CSV.

        Args
        ----
        csv_path : str, optional
            Path to the CSV file to load rules from.
        """
        logger.info("Initializing table '%s'...", self.table_name)

        schema = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
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
            embedding TEXT, -- JSON string
            relevance REAL,
            version_major INTEGER,
            version_minor INTEGER,
            created_at TEXT,
            updated_at TEXT
        );
        """
        conn = self.get_conn()
        conn.execute(schema)
        conn.commit()
        conn.close()
        logger.info("Table '%s' ready.", self.table_name)

        if csv_path:
            self.load_rules_from_csv(csv_path)

    def load_rules_from_csv(self, csv_path: str):
        """
        Load rules from a CSV file into the database.

        Args
        ----
        csv_path : str
            Path to the CSV file containing rules.
        """
        import pandas as pd
        import uuid

        logger.info("Attempting to load rules from %s...", csv_path)
        df = pd.read_csv(csv_path)
        conn = self.get_conn()
        cursor = conn.cursor()

        # Skip if table already populated
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        count = cursor.fetchone()[0]
        if count > 0:
            logger.info("Table '%s' already contains %d rules. Skipping CSV load.",
                        self.table_name, count)
            conn.close()
            return

        logger.info("Loading %d new rules from CSV...", len(df))
        for _, row in df.iterrows():
            rule_id = str(uuid.uuid4())
            csv_values = [row.get(col, None) for col in df.columns]
            values = [rule_id] + csv_values + [
                1.0,  # relevance
                1,    # version_major
                0,
