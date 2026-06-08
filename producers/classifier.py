from __future__ import annotations
from enum import Enum


class EmailCategory(str, Enum):
    REJECTION    = "rejection"
    APPLICATION  = "application"
    UNRELATED    = "unrelated"


# ── Signal phrase lists ───────────────────────────────────────────────────────
# All lowercase; matched against subject + first 500 chars of body.

REJECTION_SIGNALS: frozenset[str] = frozenset({
    "we have decided",
    "we will not be moving forward",
    "not moving forward",
    "after careful consideration",
    "we've decided to pursue other candidates",
    "we have chosen to move forward with other candidates",
    "we won't be moving forward",
    "we regret to inform",
    "unfortunately",
    "we have filled the position",
    "the position has been filled",
    "we will keep your resume on file",
    "keep your resume on file",
    "we wish you the best",
    "thank you for your interest",               # only matches in combo
    "we are not able to offer you",
    "your application was not selected",
    "not selected for",
    "did not move forward",
    "no longer being considered",
})

APPLICATION_SIGNALS: frozenset[str] = frozenset({
    "application received",
    "application submitted",
    "application confirmation",
    "we received your application",
    "thank you for applying",
    "thanks for applying",
    "your application for",
    "you applied to",
    "you have applied",
    "application for the role",
    "application for the position",
    "we've received your application",
    "successfully submitted",
    "your submission",
    "applied to",
    "confirming your application",
})

# Subject-line patterns so strong they alone determine the category
STRONG_REJECTION_SUBJECTS: frozenset[str] = frozenset({
    "unfortunately",
    "update on your application",
    "your application status",
    "decision regarding your application",
    "your candidacy",
})

STRONG_APPLICATION_SUBJECTS: frozenset[str] = frozenset({
    "application received",
    "application confirmation",
    "application submitted",
    "thank you for applying",
    "thanks for applying",
})

# If none of these appear anywhere in subject + body preview, skip the email
JOB_CONTEXT_SIGNALS: frozenset[str] = frozenset({
    "position", "role", "job", "application", "candidate", "candidacy",
    "hiring", "recruiter", "talent", "opportunity", "interview",
    "resume", "cv", "team", "engineering", "scientist", "analyst",
    "developer", "manager", "offer", "onboard",
})


# ── Classifier ────────────────────────────────────────────────────────────────

def classify_email(subject: str, body: str) -> EmailCategory:
    """
    Classify an email into REJECTION, APPLICATION, or UNRELATED.

    Args:
        subject: Email subject line.
        body: Plain-text email body (truncated to first 1000 chars is fine).

    Returns:
        EmailCategory enum value.
    """
    subject_lower = subject.lower()
    # Only examine the first 1000 characters of the body for speed
    body_preview  = body[:1000].lower()
    combined      = f"{subject_lower} {body_preview}"

    # Fast gate: is this email job-related at all?
    if not any(signal in combined for signal in JOB_CONTEXT_SIGNALS):
        return EmailCategory.UNRELATED

    # Strong subject-line signals (single phrase is decisive)
    for phrase in STRONG_REJECTION_SUBJECTS:
        if phrase in subject_lower:
            return EmailCategory.REJECTION

    for phrase in STRONG_APPLICATION_SUBJECTS:
        if phrase in subject_lower:
            return EmailCategory.APPLICATION

    # Weighted scoring across full combined text
    rejection_score    = sum(1 for p in REJECTION_SIGNALS    if p in combined)
    application_score  = sum(1 for p in APPLICATION_SIGNALS  if p in combined)

    if rejection_score == 0 and application_score == 0:
        return EmailCategory.UNRELATED

    if rejection_score >= application_score:
        return EmailCategory.REJECTION

    return EmailCategory.APPLICATION


# ── Quick smoke-test (run: python -m jobsignal.producers.classifier) ──────────

if __name__ == "__main__":
    cases = [
        (
            "Update on your application at Acme Corp",
            "After careful consideration, we have decided not to move forward with your candidacy.",
            EmailCategory.REJECTION,
        ),
        (
            "Application received — Senior Data Engineer at Stripe",
            "Thank you for applying to Stripe. We've received your application for the Senior Data Engineer role.",
            EmailCategory.APPLICATION,
        ),
        (
            "Your Amazon order has shipped",
            "Your package is on its way. Track it here.",
            EmailCategory.UNRELATED,
        ),
        (
            "Thank you for your interest",
            "We appreciate you taking the time to apply. Unfortunately, we will not be moving forward with your application at this time.",
            EmailCategory.REJECTION,
        ),
    ]

    passed = 0
    for subject, body, expected in cases:
        result = classify_email(subject, body)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"{status} [{expected.value:>11}] expected | [{result.value:>11}] got | {subject[:50]}")

    print(f"\n{passed}/{len(cases)} passed")