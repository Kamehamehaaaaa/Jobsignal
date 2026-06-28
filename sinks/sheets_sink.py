"""
JobSignal — Google Sheets sink.

Appends enriched job events to a Google Sheet with two tabs:
  - "Rejections"   — one row per rejection event
  - "Applications" — one row per application confirmation

Auth:
    Reuses the same Gmail OAuth2 credentials (same Google account).
    The credentials need the 'spreadsheets' scope added — see .env.example.

Sheet structure (Rejections tab):
    A: Date Received  B: Company  C: Role  D: Rejection Type
    E: Source  F: Job Board  G: Confidence  H: Subject  I: Event ID

Sheet structure (Applications tab):
    A: Date Applied  B: Company  C: Role  D: Status
    E: Source  F: Job Board  G: Subject  H: Event ID
"""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

# Scopes needed — Gmail read + Sheets write
SCOPES = [
    "https://mail.google.com/",
    "https://www.googleapis.com/auth/spreadsheets",
]

# Tab names inside the spreadsheet
TAB_REJECTIONS   = "Rejections"
TAB_APPLICATIONS = "Applications"

# Column headers
REJECTION_HEADERS = [
    "Date Received", "Company", "Role", "Rejection Type",
    "Source", "Job Board", "Confidence", "Subject", "Event ID",
]
APPLICATION_HEADERS = [
    "Date Applied", "Company", "Role", "Status",
    "Source", "Job Board", "Subject", "Event ID",
]


class SheetsSink:
    """
    Writes enriched job events to Google Sheets.

    Usage:
        sink = SheetsSink(
            spreadsheet_id="your-sheet-id",
            credentials_path="secrets/gmail_credentials.json",
            token_path="secrets/gmail_token.json",
        )
        sink.write_rejection(event_dict, enriched_fields)
        sink.write_application(event_dict, enriched_fields)
    """

    def __init__(
        self,
        spreadsheet_id: str,
        credentials_path: str,
        token_path: str,
    ):
        self.spreadsheet_id = spreadsheet_id
        self._service = self._build_service(credentials_path, token_path)
        self._ensure_tabs()

    # ── Auth ──────────────────────────────────────────────────────────────────

    def _build_service(self, credentials_path: str, token_path: str):
        import os
        creds: Optional[Credentials] = None

        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(token_path, "w") as f:
                f.write(creds.to_json())

        return build("sheets", "v4", credentials=creds)

    # ── Tab setup ─────────────────────────────────────────────────────────────

    def _ensure_tabs(self) -> None:
        """Create tabs and header rows if they don't exist yet."""
        sheet = self._service.spreadsheets()
        meta = sheet.get(spreadsheetId=self.spreadsheet_id).execute()
        existing_tabs = {s["properties"]["title"] for s in meta["sheets"]}

        requests = []
        if TAB_REJECTIONS not in existing_tabs:
            requests.append({"addSheet": {"properties": {"title": TAB_REJECTIONS}}})
        if TAB_APPLICATIONS not in existing_tabs:
            requests.append({"addSheet": {"properties": {"title": TAB_APPLICATIONS}}})

        if requests:
            sheet.batchUpdate(
                spreadsheetId=self.spreadsheet_id,
                body={"requests": requests},
            ).execute()
            # Write headers
            self._append(TAB_REJECTIONS,   [REJECTION_HEADERS])
            self._append(TAB_APPLICATIONS, [APPLICATION_HEADERS])
            logger.info("Created Sheets tabs and wrote headers")

    # ── Write helpers ─────────────────────────────────────────────────────────

    def _append(self, tab: str, rows: list[list]) -> None:
        try:
            self._service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=f"{tab}!A1",
                valueInputOption="USER_ENTERED",
                insertDataOption="INSERT_ROWS",
                body={"values": rows},
            ).execute()
        except HttpError as exc:
            logger.error("Sheets append failed | tab=%s | %s", tab, exc)
            raise

    def _fmt_date(self, iso: str) -> str:
        """Convert ISO-8601 to a human-readable date string."""
        try:
            return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            return iso

    # ── Public write methods ──────────────────────────────────────────────────

    def write_rejection(self, event: dict, enriched) -> None:
        """
        Append one row to the Rejections tab.

        Args:
            event:    Raw event dict (from Kafka message JSON).
            enriched: EnrichedFields instance from nlp.enricher.
        """
        row = [
            self._fmt_date(event.get("received_at", "")),
            enriched.company_name,
            enriched.role_title,
            enriched.rejection_type,
            event.get("source", ""),
            enriched.job_board,
            f"{enriched.confidence:.0%}",
            event.get("raw_subject", "")[:120],   # truncate long subjects
            event.get("event_id", ""),
        ]
        self._append(TAB_REJECTIONS, [row])
        logger.info(
            "Sheets ← rejection | company=%r role=%r type=%s",
            enriched.company_name, enriched.role_title, enriched.rejection_type,
        )

    def write_application(self, event: dict, enriched) -> None:
        """
        Append one row to the Applications tab.

        Args:
            event:    Raw event dict (from Kafka message JSON).
            enriched: EnrichedFields instance from nlp.enricher.
        """
        row = [
            self._fmt_date(event.get("received_at", "")),
            enriched.company_name,
            enriched.role_title,
            event.get("application_status", "applied"),
            event.get("source", ""),
            enriched.job_board,
            event.get("raw_subject", "")[:120],
            event.get("event_id", ""),
        ]
        self._append(TAB_APPLICATIONS, [row])
        logger.info(
            "Sheets ← application | company=%r role=%r board=%s",
            enriched.company_name, enriched.role_title, enriched.job_board,
        )