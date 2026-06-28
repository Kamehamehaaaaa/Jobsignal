"""
JobSignal — spaCy NLP enricher.

Takes a raw email event (subject + body) and returns enriched fields:
  - company_name
  - role_title
  - rejection_type  (hard / soft / unknown)
  - job_board       (LinkedIn, Indeed, etc.)

Design philosophy:
    This is the producer-side classifier's smarter sibling.  The classifier
    made a binary call (rejection vs application) with no ML.  This enricher
    runs inside Spark and does the actual information extraction.

    We use spaCy's en_core_web_sm model for NER (ORG entities → company name)
    combined with pattern-matching for role titles and rejection type signals.
    When Rohit is ready to swap in DistilBERT, replace _classify_rejection_type()
    with a model.predict() call — everything else stays the same.

Swap-in point for DistilBERT:
    The function _classify_rejection_type(text) is the only place that needs
    to change.  Return the same RejectionType string and Spark never knows.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import spacy

# ── Classifier backend flag ───────────────────────────────────────────────────
# Set USE_DISTILBERT=zero_shot  → DistilBERT zero-shot (no training data needed)
# Set USE_DISTILBERT=finetuned  → fine-tuned model (run --train first)
# Unset / any other value       → spaCy rules (default, fastest)
_DISTILBERT_MODE = os.getenv("USE_DISTILBERT", "").strip().lower()  # "" | "zero_shot" | "finetuned"

_distilbert_clf = None   # initialised lazily on first use

def _get_distilbert():
    global _distilbert_clf
    if _distilbert_clf is None:
        from nlp.distilbert_classifier import DistilBertClassifier
        _distilbert_clf = DistilBertClassifier(
            mode=_DISTILBERT_MODE,
            model_path=os.getenv("DISTILBERT_MODEL_PATH", "models/distilbert_rejection_classifier"),
        )
    return _distilbert_clf


# ── Lazy model load (cached — loaded once per Spark executor) ─────────────────

@lru_cache(maxsize=1)
def _get_nlp():
    """Load spaCy model once and cache it.  Thread-safe after first call."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        )


# ── Rejection type signals ────────────────────────────────────────────────────

_HARD_REJECTION_PATTERNS: list[str] = [
    r"not moving forward",
    r"will not be moving forward",
    r"decided not to (move|proceed)",
    r"chosen (to pursue|another) candidate",
    r"not selected",
    r"position has been filled",
    r"not able to offer",
    r"no longer (being )?considered",
    r"did not move forward",
    r"another candidate",
]

_SOFT_REJECTION_PATTERNS: list[str] = [
    r"keep your (resume|cv) on file",
    r"future opportunities",
    r"stay in touch",
    r"reach out (if|when) .{0,30} position",
    r"we'll keep you in mind",
    r"consider you for future",
]

_HARD_RE = re.compile("|".join(_HARD_REJECTION_PATTERNS), re.IGNORECASE)
_SOFT_RE = re.compile("|".join(_SOFT_REJECTION_PATTERNS), re.IGNORECASE)


# ── Role title patterns ───────────────────────────────────────────────────────
# Matches "Software Engineer", "Senior ML Engineer", "Data Scientist II", etc.

_ROLE_PATTERN = re.compile(
    r"""
    (?:
        (?:Senior|Junior|Lead|Staff|Principal|Associate|Mid[- ]?Level)\s+
    )?
    (?:
        Software\s+Engineer(?:ing)?     |
        Machine\s+Learning\s+Engineer   |
        ML\s+Engineer                   |
        Data\s+(?:Scientist|Engineer|Analyst) |
        MLOps\s+Engineer                |
        DevOps\s+Engineer               |
        Platform\s+Engineer             |
        Backend\s+Engineer              |
        Full[- ]?Stack\s+Engineer       |
        Research\s+(?:Scientist|Engineer)|
        Applied\s+Scientist             |
        Site\s+Reliability\s+Engineer   |
        SRE
    )
    (?:\s+(?:I{1,3}|IV|V|\d))?          # optional level suffix: I II III 2 etc.
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Job board inference from sender domain
_JOB_BOARD_MAP: dict[str, str] = {
    "linkedin":     "LinkedIn",
    "indeed":       "Indeed",
    "glassdoor":    "Glassdoor",
    "dice":         "Dice",
    "ziprecruiter": "ZipRecruiter",
    "lever":        "Lever",
    "greenhouse":   "Greenhouse",
    "workday":      "Workday",
    "icims":        "iCIMS",
    "smartrecruiters": "SmartRecruiters",
    "myworkdayjobs": "Workday",
}


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class EnrichedFields:
    company_name:   str
    role_title:     str
    rejection_type: str   # "hard_rejection" | "soft_rejection" | "unknown"
    job_board:      str
    confidence:     float  # 0.0–1.0 rough confidence score


# ── Main enricher ─────────────────────────────────────────────────────────────

def enrich(
    subject: str,
    body: str,
    sender_email: str,
    event_type: str,          # "rejection" or "application"
) -> EnrichedFields:
    """
    Run NLP enrichment on a single email event.

    Args:
        subject:      Email subject line.
        body:         Plain-text body (full text is fine; we slice internally).
        sender_email: From address — used for job board inference.
        event_type:   "rejection" or "application" (from Kafka topic).

    Returns:
        EnrichedFields with extracted metadata.
    """
    # Work on subject + first 600 chars of body for speed
    text = f"{subject}. {body[:600]}"

    company  = _extract_company(text, sender_email)
    role     = _extract_role(text)
    rej_type = _classify_rejection_type(text) if event_type == "rejection" else "n/a"
    board    = _infer_job_board(sender_email)

    # Rough confidence: higher if we found both company and role
    confidence = sum([
        0.4 if company else 0.0,
        0.4 if role    else 0.0,
        0.2 if board != "company_direct" else 0.0,
    ])

    return EnrichedFields(
        company_name=company,
        role_title=role,
        rejection_type=rej_type,
        job_board=board,
        confidence=confidence,
    )


# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_company(text: str, sender_email: str) -> str:
    """
    Use spaCy NER to find ORG entities, then pick the most likely company.

    Strategy:
      1. Run NER on the combined text.
      2. Filter ORG entities — exclude generic words and known non-company terms.
      3. Pick the first ORG that appears in the subject (highest signal).
      4. Fall back to the first ORG anywhere in the text.
      5. Fall back to extracting the domain from the sender email.
    """
    nlp = _get_nlp()
    doc = nlp(text[:500])   # NER on first 500 chars is sufficient

    _NOISE = frozenset({
        "linkedin", "indeed", "glassdoor", "greenhouse", "lever",
        "workday", "ziprecruiter", "dice", "the", "a", "an",
        "team", "company", "position", "role", "opportunity",
    })

    orgs = [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ == "ORG"
        and ent.text.strip().lower() not in _NOISE
        and len(ent.text.strip()) > 2
    ]

    if orgs:
        return orgs[0]

    # Fallback: parse sender domain  e.g. hr@stripe.com → Stripe
    return _company_from_email(sender_email)


def _company_from_email(email: str) -> str:
    """Extract a capitalised company name from an email domain."""
    match = re.search(r"@([\w.-]+)", email)
    if not match:
        return ""
    domain = match.group(1).lower()
    # Strip known non-company domains
    for noise in ("gmail", "outlook", "yahoo", "hotmail", "greenhouse-mail",
                  "lever", "workday", "myworkdayjobs"):
        if noise in domain:
            return ""
    # Take the second-level domain label and capitalise
    parts = domain.split(".")
    return parts[-2].capitalize() if len(parts) >= 2 else parts[0].capitalize()


def _extract_role(text: str) -> str:
    """Match role title patterns in the text."""
    match = _ROLE_PATTERN.search(text)
    if match:
        # Normalise whitespace
        return " ".join(match.group(0).split())
    return ""


def _classify_rejection_type(text: str) -> str:
    """
    Rejection type classifier — backend selected by USE_DISTILBERT env var.

    USE_DISTILBERT unset  → spaCy rules (fast, no dependencies)
    USE_DISTILBERT=zero_shot  → DistilBERT zero-shot classification
    USE_DISTILBERT=finetuned  → fine-tuned DistilBERT model

    All three return the same values: "hard_rejection" | "soft_rejection" | "unknown"
    """
    if _DISTILBERT_MODE in ("zero_shot", "finetuned"):
        return _get_distilbert().classify(text)

    # Default: spaCy rules
    if _HARD_RE.search(text):
        return "hard_rejection"
    if _SOFT_RE.search(text):
        return "soft_rejection"
    return "unknown"


def _infer_job_board(sender_email: str) -> str:
    sender = sender_email.lower()
    for keyword, board in _JOB_BOARD_MAP.items():
        if keyword in sender:
            return board
    return "company_direct"