"""Persistent stage state for restartable FFOpt pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable


VALID_STATUSES = {
    "pending",
    "running",
    "waiting",
    "completed",
    "failed",
    "skipped",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class StageRecord:
    name: str
    signature: str
    status: str
    attempt: int
    command: list[str]
    output_dir: str
    artifacts: list[str]
    job_id: str | None
    message: str
    started_at: str | None
    finished_at: str | None
    updated_at: str


class WorkflowState:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS stages (
                name TEXT PRIMARY KEY,
                signature TEXT NOT NULL,
                status TEXT NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 0,
                command_json TEXT NOT NULL DEFAULT '[]',
                output_dir TEXT NOT NULL DEFAULT '',
                artifacts_json TEXT NOT NULL DEFAULT '[]',
                job_id TEXT,
                message TEXT NOT NULL DEFAULT '',
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage TEXT,
                status TEXT NOT NULL,
                message TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def initialize(self, metadata: dict[str, Any]) -> None:
        with self.connection:
            for key, value in metadata.items():
                self.connection.execute(
                    "INSERT INTO metadata(key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(key), json.dumps(value, default=str)),
                )

    def metadata(self) -> dict[str, Any]:
        rows = self.connection.execute("SELECT key, value FROM metadata")
        return {row["key"]: json.loads(row["value"]) for row in rows}

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> StageRecord | None:
        if row is None:
            return None
        return StageRecord(
            name=row["name"],
            signature=row["signature"],
            status=row["status"],
            attempt=int(row["attempt"]),
            command=json.loads(row["command_json"]),
            output_dir=row["output_dir"],
            artifacts=json.loads(row["artifacts_json"]),
            job_id=row["job_id"],
            message=row["message"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            updated_at=row["updated_at"],
        )

    def get(self, name: str) -> StageRecord | None:
        row = self.connection.execute(
            "SELECT * FROM stages WHERE name=?", (name,)
        ).fetchone()
        return self._decode(row)

    def list(self) -> list[StageRecord]:
        rows = self.connection.execute(
            "SELECT * FROM stages ORDER BY rowid"
        ).fetchall()
        return [self._decode(row) for row in rows if row is not None]

    def prepare(
        self,
        name: str,
        signature: str,
        command: Iterable[str],
        output_dir: str | Path,
        artifacts: Iterable[str | Path],
    ) -> StageRecord:
        existing = self.get(name)
        status = existing.status if existing and existing.signature == signature else "pending"
        attempt = existing.attempt if existing and existing.signature == signature else 0
        now = utc_now()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO stages(
                    name, signature, status, attempt, command_json, output_dir,
                    artifacts_json, job_id, message, started_at, finished_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, '', NULL, NULL, ?)
                ON CONFLICT(name) DO UPDATE SET
                    signature=excluded.signature,
                    status=excluded.status,
                    attempt=excluded.attempt,
                    command_json=CASE
                        WHEN stages.signature=excluded.signature
                         AND stages.status IN ('waiting', 'running')
                        THEN stages.command_json ELSE excluded.command_json END,
                    output_dir=excluded.output_dir,
                    artifacts_json=excluded.artifacts_json,
                    job_id=CASE WHEN stages.signature=excluded.signature
                                THEN stages.job_id ELSE NULL END,
                    message=CASE WHEN stages.signature=excluded.signature
                                 THEN stages.message ELSE '' END,
                    started_at=CASE WHEN stages.signature=excluded.signature
                                    THEN stages.started_at ELSE NULL END,
                    finished_at=CASE WHEN stages.signature=excluded.signature
                                     THEN stages.finished_at ELSE NULL END,
                    updated_at=excluded.updated_at
                """,
                (
                    name,
                    signature,
                    status,
                    attempt,
                    json.dumps([str(item) for item in command]),
                    str(Path(output_dir).resolve()),
                    json.dumps([str(Path(item).resolve()) for item in artifacts]),
                    now,
                ),
            )
        return self.get(name)  # type: ignore[return-value]

    def transition(
        self,
        name: str,
        status: str,
        *,
        message: str = "",
        job_id: str | None = None,
        increment_attempt: bool = False,
    ) -> StageRecord:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid workflow stage status: {status}")
        current = self.get(name)
        if current is None:
            raise KeyError(f"Stage is not prepared: {name}")
        now = utc_now()
        started_at = current.started_at
        finished_at = current.finished_at
        if status == "running":
            started_at = now
            finished_at = None
        elif status in {"completed", "failed", "skipped"}:
            finished_at = now
        with self.connection:
            self.connection.execute(
                """
                UPDATE stages SET status=?, attempt=attempt+?, job_id=?,
                    message=?, started_at=?, finished_at=?, updated_at=?
                WHERE name=?
                """,
                (
                    status,
                    1 if increment_attempt else 0,
                    job_id if job_id is not None else current.job_id,
                    message,
                    started_at,
                    finished_at,
                    now,
                    name,
                ),
            )
            self.connection.execute(
                "INSERT INTO events(stage, status, message, created_at) "
                "VALUES (?, ?, ?, ?)",
                (name, status, message, now),
            )
        return self.get(name)  # type: ignore[return-value]

    @staticmethod
    def artifacts_exist(record: StageRecord) -> bool:
        return bool(record.artifacts) and all(Path(path).exists() for path in record.artifacts)

    def is_complete(self, name: str, signature: str) -> bool:
        record = self.get(name)
        return bool(
            record
            and record.signature == signature
            and record.status in {"completed", "skipped"}
            and self.artifacts_exist(record)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "metadata": self.metadata(),
            "stages": [asdict(record) for record in self.list()],
        }
