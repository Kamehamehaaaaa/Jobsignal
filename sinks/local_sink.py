"""
JobSignal — local file sink (backup + development).

Writes enriched events to:
  - output/rejections.jsonl    (one JSON object per line)
  - output/applications.jsonl

JSONL is chosen over CSV because:
  - Each line is valid JSON — easy to inspect with jq or pandas
  - No quoting/escaping issues with free-text fields (email subjects/bodies)
  - Append-only writes are safe across restarts (no header row concerns)
  - Spark can read JSONL directly for downstream analysis

The sink also maintains a rolling CSV summary (latest 1000 rows) alongside
the full JSONL log — useful for quick Excel/Sheets import without the API.
"""

from __future__ import annotations
import csv
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "output"

REJECTION_CSV_HEADERS = [
    "date_received", "company", "role", "rejection_type",
    "source", "job_board", "confidence", "subject", "event_id",
]
APPLICATION_CSV_HEADERS = [
    "date_applied", "company", "role", "status",
    "source", "job_board", "subject", "event_id",
]


class LocalSink:
    """
    Writes enriched job events to local JSONL files + CSV summaries.

    Usage:
        sink = LocalSink(output_dir="output")
        sink.write_rejection(event_dict, enriched_fields)
        sink.write_application(event_dict, enriched_fields)
    """

    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._rejection_jsonl    = self.output_dir / "rejections.jsonl"
        self._application_jsonl  = self.output_dir / "applications.jsonl"
        self._rejection_csv      = self.output_dir / "rejections.csv"
        self._application_csv    = self.output_dir / "applications.csv"

        self._init_csv(self._rejection_csv,   REJECTION_CSV_HEADERS)
        self._init_csv(self._application_csv, APPLICATION_CSV_HEADERS)

        logger.info("LocalSink ready | output_dir=%s", self.output_dir.resolve())

    # ── CSV init ──────────────────────────────────────────────────────────────

    def _init_csv(self, path: Path, headers: list[str]) -> None:
        """Write headers only if the file doesn't exist yet."""
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(headers)

    # ── Write helpers ─────────────────────────────────────────────────────────

    def _append_jsonl(self, path: Path, record: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _append_csv(self, path: Path, row: list) -> None:
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _fmt_date(self, iso: str) -> str:
        try:
            return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            return iso

    # ── Public write methods ──────────────────────────────────────────────────

    def write_rejection(self, event: dict, enriched) -> None:
        """Write a rejection event to JSONL + CSV."""

        # Full JSONL record — keep everything for later analysis
        record = {
            **event,
            "company_name":    enriched.company_name,
            "role_title":      enriched.role_title,
            "rejection_type":  enriched.rejection_type,
            "job_board":       enriched.job_board,
            "nlp_confidence":  enriched.confidence,
            "processed_at":    self._now(),
        }
        self._append_jsonl(self._rejection_jsonl, record)

        # CSV summary row (human-readable)
        self._append_csv(
            self._rejection_csv,
            [
                self._fmt_date(event.get("received_at", "")),
                enriched.company_name,
                enriched.role_title,
                enriched.rejection_type,
                event.get("source", ""),
                enriched.job_board,
                f"{enriched.confidence:.0%}",
                event.get("raw_subject", "")[:120],
                event.get("event_id", ""),
            ],
        )
        logger.info(
            "Local ← rejection | company=%r role=%r type=%s",
            enriched.company_name, enriched.role_title, enriched.rejection_type,
        )

    def write_application(self, event: dict, enriched) -> None:
        """Write an application confirmation event to JSONL + CSV."""

        record = {
            **event,
            "company_name":   enriched.company_name,
            "role_title":     enriched.role_title,
            "job_board":      enriched.job_board,
            "nlp_confidence": enriched.confidence,
            "processed_at":   self._now(),
        }
        self._append_jsonl(self._application_jsonl, record)

        self._append_csv(
            self._application_csv,
            [
                self._fmt_date(event.get("received_at", "")),
                enriched.company_name,
                enriched.role_title,
                event.get("application_status", "applied"),
                event.get("source", ""),
                enriched.job_board,
                event.get("raw_subject", "")[:120],
                event.get("event_id", ""),
            ],
        )
        logger.info(
            "Local ← application | company=%r role=%r board=%s",
            enriched.company_name, enriched.role_title, enriched.job_board,
        )

    # ── Utility ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return counts of processed events — useful for health checks."""
        def count_lines(path: Path) -> int:
            if not path.exists():
                return 0
            with open(path, encoding="utf-8") as f:
                return sum(1 for _ in f)

        return {
            "rejections":   count_lines(self._rejection_jsonl),
            "applications": count_lines(self._application_jsonl),
            "output_dir":   str(self.output_dir.resolve()),
        }