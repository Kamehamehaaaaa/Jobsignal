from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum



## ENUMS

class EmailSource(str, Enum):
    GMAIL = "gmail"
    OUTLOOK = "outlook"

class RejectionType(str, Enum):
    UNKNOWN          = "unknown"
    HARD_REJECTION   = "hard_rejection"    # explicit "we went with another candidate"
    SOFT_REJECTION   = "soft_rejection"    # "we'll keep your resume on file"
    NO_RESPONSE      = "no_response"

class ApplicationStatus(str, Enum):
    APPLIED      = "applied"
    INTERVIEWING = "interviewing"
    OFFER        = "offer"
    REJECTED     = "rejected"
    WITHDRAWN    = "withdrawn"

# ── Base event ─────────────────────────────────────────────────────────────────
 
@dataclass(frozen=True)
class BaseEmailEvent:
    """Fields shared by every event on every topic."""
    event_id:       str          # UUID4 — deduplication key for idempotent consumers
    source:         EmailSource
    raw_subject:    str
    raw_body_text:  str          # plain-text stripped; HTML not stored
    sender_email:   str
    received_at:    str          # ISO-8601 UTC
    ingested_at:    str          # ISO-8601 UTC — set by the producer
    schema_version: str = "1.0"
 
    def to_dict(self) -> dict:
        d = asdict(self)
        # convert enum values to their string representation
        d["source"] = self.source.value
        return d
 
    def to_json(self) -> bytes:
        return json.dumps(self.to_dict(), ensure_ascii=False).encode("utf-8")
 
 
# ── Topic-specific schemas ─────────────────────────────────────────────────────
 
@dataclass(frozen=True)
class EmailRejectionEvent(BaseEmailEvent):
    """
    Schema for  raw.emails.rejections
    The company / role / rejection_type fields start empty.
    Spark fills them after NLP classification.
    """
    company_name:    str = ""
    role_title:      str = ""
    rejection_type:  str = RejectionType.UNKNOWN.value
 
    def to_dict(self) -> dict:
        d = super().to_dict()
        d["rejection_type"] = self.rejection_type
        return d
 
 
@dataclass(frozen=True)
class EmailApplicationEvent(BaseEmailEvent):
    """
    Schema for  raw.emails.applications
    Confirmation emails from job boards / company ATS systems.
    """
    company_name:       str = ""
    role_title:         str = ""
    job_board:          str = ""   # e.g. "LinkedIn", "Indeed", "company_direct"
    application_status: str = ApplicationStatus.APPLIED.value
    job_posting_url:    str = ""
 
    def to_dict(self) -> dict:
        d = super().to_dict()
        d["application_status"] = self.application_status
        return d
 
 
# ── Dead-letter schema ─────────────────────────────────────────────────────────
 
@dataclass(frozen=True)
class DeadLetterEvent:
    """Wraps any failed event with enough context to debug and replay."""
    dlq_event_id:     str
    original_topic:   str
    failure_reason:   str
    raw_payload:      str    # the original bytes as a string (best-effort)
    failed_at:        str
    producer_name:    str
    schema_version:   str = "1.0"
 
    def to_json(self) -> bytes:
        return json.dumps(asdict(self), ensure_ascii=False).encode("utf-8")
 
 
# ── Factory helpers ────────────────────────────────────────────────────────────
 
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
 
def _new_id() -> str:
    return str(uuid.uuid4())
 
 
def make_rejection_event(
    source: EmailSource,
    subject: str,
    body_text: str,
    sender: str,
    received_at: str,
) -> EmailRejectionEvent:
    return EmailRejectionEvent(
        event_id=_new_id(),
        source=source,
        raw_subject=subject,
        raw_body_text=body_text,
        sender_email=sender,
        received_at=received_at,
        ingested_at=_now_iso(),
    )
 
 
def make_application_event(
    source: EmailSource,
    subject: str,
    body_text: str,
    sender: str,
    received_at: str,
    job_board: str = "",
) -> EmailApplicationEvent:
    return EmailApplicationEvent(
        event_id=_new_id(),
        source=source,
        raw_subject=subject,
        raw_body_text=body_text,
        sender_email=sender,
        received_at=received_at,
        ingested_at=_now_iso(),
        job_board=job_board,
    )
 
 
def make_dlq_event(
    original_topic: str,
    failure_reason: str,
    raw_payload: str,
    producer_name: str,
) -> DeadLetterEvent:
    return DeadLetterEvent(
        dlq_event_id=_new_id(),
        original_topic=original_topic,
        failure_reason=failure_reason,
        raw_payload=raw_payload,
        failed_at=_now_iso(),
        producer_name=producer_name,
    )
 