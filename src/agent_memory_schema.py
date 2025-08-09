"""
Agent Memory Schema and Helper Utilities

This module defines a dedicated SQLite schema for agent memory as requested:

Tables:
1. attack_history (id, timestamp, attack_type, success, response, metadata)
2. learned_patterns (id, pattern_type, pattern_data, confidence_score, created_at)
3. adaptations_made (id, adaptation_type, rule_changes, reason, timestamp)
4. threat_database (id, threat_signature, severity, countermeasure, discovered_at)

Additional mapping tables to provide proper foreign keys:
- attack_pattern_map (attack_id -> learned_patterns.id)
- attack_adaptation_map (attack_id -> adaptations_made.id)
- pattern_threat_map (pattern_id -> threat_database.id)

Helper functions provided for:
- Storing attack results
- Querying historical patterns
- Updating adaptation strategies
- Calculating success rate trends
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    # Enforce foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


class AgentMemoryDB:
    def __init__(self, db_path: str = "memory/agent_memory.db") -> None:
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def migrate(self) -> None:
        with _connect(self.db_path) as conn:
            cur = conn.cursor()

            # Core tables
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS attack_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    attack_type TEXT NOT NULL,
                    success INTEGER NOT NULL CHECK (success IN (0, 1)),
                    response TEXT,
                    metadata TEXT
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS adaptations_made (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    adaptation_type TEXT NOT NULL,
                    rule_changes TEXT NOT NULL,
                    reason TEXT,
                    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS threat_database (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    threat_signature TEXT NOT NULL UNIQUE,
                    severity TEXT NOT NULL,
                    countermeasure TEXT NOT NULL,
                    discovered_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            # Mapping tables to provide FKs while keeping core table schemas minimal
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS attack_pattern_map (
                    attack_id INTEGER NOT NULL,
                    pattern_id INTEGER NOT NULL,
                    PRIMARY KEY (attack_id, pattern_id),
                    FOREIGN KEY (attack_id) REFERENCES attack_history(id) ON DELETE CASCADE,
                    FOREIGN KEY (pattern_id) REFERENCES learned_patterns(id) ON DELETE CASCADE
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS attack_adaptation_map (
                    attack_id INTEGER NOT NULL,
                    adaptation_id INTEGER NOT NULL,
                    PRIMARY KEY (attack_id, adaptation_id),
                    FOREIGN KEY (attack_id) REFERENCES attack_history(id) ON DELETE CASCADE,
                    FOREIGN KEY (adaptation_id) REFERENCES adaptations_made(id) ON DELETE CASCADE
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS pattern_threat_map (
                    pattern_id INTEGER NOT NULL,
                    threat_id INTEGER NOT NULL,
                    PRIMARY KEY (pattern_id, threat_id),
                    FOREIGN KEY (pattern_id) REFERENCES learned_patterns(id) ON DELETE CASCADE,
                    FOREIGN KEY (threat_id) REFERENCES threat_database(id) ON DELETE CASCADE
                );
                """
            )

            # Indexes for performance
            cur.execute("CREATE INDEX IF NOT EXISTS idx_attack_history_timestamp ON attack_history(timestamp);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_attack_history_type ON attack_history(attack_type);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_attack_history_success_time ON attack_history(success, timestamp);")

            cur.execute("CREATE INDEX IF NOT EXISTS idx_learned_patterns_type ON learned_patterns(pattern_type);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_learned_patterns_conf ON learned_patterns(confidence_score);")

            cur.execute("CREATE INDEX IF NOT EXISTS idx_adaptations_type ON adaptations_made(adaptation_type);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_adaptations_time ON adaptations_made(timestamp);")

            cur.execute("CREATE INDEX IF NOT EXISTS idx_threat_severity ON threat_database(severity);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_threat_discovered ON threat_database(discovered_at);")

            conn.commit()

    # -----------------------------
    # Helper functions
    # -----------------------------
    def store_attack_result(
        self,
        attack_type: str,
        success: bool,
        response: Optional[str] = None,
        metadata_json: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        related_pattern_ids: Optional[Iterable[int]] = None,
        related_adaptation_ids: Optional[Iterable[int]] = None,
    ) -> int:
        """Insert a row into attack_history and establish optional relationships."""
        with _connect(self.db_path) as conn:
            cur = conn.cursor()
            ts = (timestamp or datetime.now()).isoformat()
            cur.execute(
                """
                INSERT INTO attack_history (timestamp, attack_type, success, response, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ts, attack_type, 1 if success else 0, response, metadata_json),
            )
            attack_id = cur.lastrowid

            if related_pattern_ids:
                cur.executemany(
                    "INSERT OR IGNORE INTO attack_pattern_map (attack_id, pattern_id) VALUES (?, ?)",
                    [(attack_id, pid) for pid in related_pattern_ids],
                )
            if related_adaptation_ids:
                cur.executemany(
                    "INSERT OR IGNORE INTO attack_adaptation_map (attack_id, adaptation_id) VALUES (?, ?)",
                    [(attack_id, aid) for aid in related_adaptation_ids],
                )

            conn.commit()
            return attack_id

    def query_historical_patterns(
        self,
        pattern_type: Optional[str] = None,
        since: Optional[datetime] = None,
        min_confidence: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Query patterns with optional filters and usage counts from attack mapping."""
        with _connect(self.db_path) as conn:
            cur = conn.cursor()
            clauses: List[str] = []
            params: List[Any] = []

            if pattern_type:
                clauses.append("lp.pattern_type = ?")
                params.append(pattern_type)
            if min_confidence is not None:
                clauses.append("lp.confidence_score >= ?")
                params.append(min_confidence)

            where_patterns = (" WHERE " + " AND ".join(clauses)) if clauses else ""

            # Filter attack timeframe if provided
            attack_time_clause = ""
            if since:
                attack_time_clause = " AND ah.timestamp >= ?"
                params.append(since.isoformat())

            sql = f"""
                SELECT
                    lp.id AS pattern_id,
                    lp.pattern_type,
                    lp.pattern_data,
                    lp.confidence_score,
                    lp.created_at,
                    COUNT(apm.attack_id) AS usage_count
                FROM learned_patterns lp
                LEFT JOIN attack_pattern_map apm ON apm.pattern_id = lp.id
                LEFT JOIN attack_history ah ON ah.id = apm.attack_id
                {where_patterns}
                {attack_time_clause}
                GROUP BY lp.id
                ORDER BY lp.confidence_score DESC, usage_count DESC
            """

            rows = cur.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

    def update_adaptation_strategy(
        self,
        adaptation_id: int,
        *,
        rule_changes: Optional[str] = None,
        reason: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update an adaptation strategy row in adaptations_made."""
        sets: List[str] = []
        params: List[Any] = []

        if rule_changes is not None:
            sets.append("rule_changes = ?")
            params.append(rule_changes)
        if reason is not None:
            sets.append("reason = ?")
            params.append(reason)
        if timestamp is not None:
            sets.append("timestamp = ?")
            params.append(timestamp.isoformat())

        if not sets:
            return

        params.append(adaptation_id)

        with _connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(f"UPDATE adaptations_made SET {', '.join(sets)} WHERE id = ?", params)
            conn.commit()

    def calculate_success_rate_trends(
        self,
        *,
        window_days: int = 7,
        lookback_windows: int = 4,
        group_by: str = "attack_type",
    ) -> List[Dict[str, Any]]:
        """
        Calculate success rate trends across rolling windows for the given grouping.

        Returns a list of dicts like:
        [{
           'group': 'prompt_injection',
           'windows': [{'start': '...', 'end': '...', 'success_rate': 0.12, 'total': 25}],
           'trend': 'increasing' | 'decreasing' | 'stable' | 'insufficient_data'
        }, ...]
        """
        assert group_by in {"attack_type"}, "Only grouping by attack_type is supported currently."

        end = datetime.now()
        windows: List[Tuple[datetime, datetime]] = []
        for i in range(lookback_windows, 0, -1):
            w_end = end - timedelta(days=(i - 0) * window_days)
            w_start = w_end - timedelta(days=window_days)
            windows.append((w_start, w_end))

        with _connect(self.db_path) as conn:
            cur = conn.cursor()

            # Get all distinct groups
            groups = [r[0] for r in cur.execute("SELECT DISTINCT attack_type FROM attack_history").fetchall()]
            results: List[Dict[str, Any]] = []

            for grp in groups:
                windows_data: List[Dict[str, Any]] = []
                for w_start, w_end in windows:
                    rows = cur.execute(
                        """
                        SELECT COUNT(*) AS total,
                               SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS succ
                        FROM attack_history
                        WHERE attack_type = ?
                          AND timestamp >= ? AND timestamp < ?
                        """,
                        (grp, w_start.isoformat(), w_end.isoformat()),
                    ).fetchone()
                    total = rows[0] or 0
                    succ = rows[1] or 0
                    rate = (succ / total) if total else 0.0
                    windows_data.append({
                        "start": w_start.isoformat(),
                        "end": w_end.isoformat(),
                        "success_rate": rate,
                        "total": total,
                    })

                trend = "insufficient_data"
                if len(windows_data) >= 2:
                    first = windows_data[0]["success_rate"]
                    last = windows_data[-1]["success_rate"]
                    if last > first:
                        trend = "increasing"
                    elif last < first:
                        trend = "decreasing"
                    else:
                        trend = "stable"

                results.append({
                    "group": grp,
                    "windows": windows_data,
                    "trend": trend,
                })

            return results

