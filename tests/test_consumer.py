"""
JobSignal — tests for NLP enricher and local sink.

Run with: pytest tests/ -v
No Kafka or Spark needed — these test the enrichment logic in isolation.
"""

import json
import os
import tempfile
import pytest

from jobsignal.nlp.enricher import (
    enrich,
    _classify_rejection_type,
    _extract_role,
    _company_from_email,
    _infer_job_board,
)
from jobsignal.sinks.local_sink import LocalSink


# ── Rejection type classifier ─────────────────────────────────────────────────

class TestRejectionClassifier:

    def test_hard_rejection_not_moving_forward(self):
        assert _classify_rejection_type(
            "After careful review we will not be moving forward with your application."
        ) == "hard_rejection"

    def test_hard_rejection_not_selected(self):
        assert _classify_rejection_type(
            "We regret to inform you that you were not selected for this role."
        ) == "hard_rejection"

    def test_hard_rejection_chosen_another_candidate(self):
        assert _classify_rejection_type(
            "We have chosen to pursue another candidate whose experience more closely matches."
        ) == "hard_rejection"

    def test_soft_rejection_keep_on_file(self):
        assert _classify_rejection_type(
            "We will keep your resume on file and reach out for future opportunities."
        ) == "soft_rejection"

    def test_soft_rejection_stay_in_touch(self):
        assert _classify_rejection_type(
            "We'd love to stay in touch and consider you for future openings."
        ) == "soft_rejection"

    def test_unknown_for_ambiguous(self):
        assert _classify_rejection_type(
            "Thank you for your time during the interview process."
        ) == "unknown"

    def test_hard_beats_soft_when_both_present(self):
        # "keep on file" + "not moving forward" → hard wins
        assert _classify_rejection_type(
            "We will not be moving forward but will keep your resume on file."
        ) == "hard_rejection"


# ── Role extraction ───────────────────────────────────────────────────────────

class TestRoleExtraction:

    def test_extracts_software_engineer(self):
        assert "Software Engineer" in _extract_role(
            "Your application for Software Engineer at Stripe has been received."
        )

    def test_extracts_senior_ml_engineer(self):
        role = _extract_role("We reviewed your application for Senior ML Engineer.")
        assert "Senior" in role and "ML Engineer" in role

    def test_extracts_data_scientist(self):
        assert "Data Scientist" in _extract_role(
            "Thank you for applying to the Data Scientist II position."
        )

    def test_returns_empty_for_no_match(self):
        assert _extract_role("Thank you for your interest in our company.") == ""


# ── Company from email ────────────────────────────────────────────────────────

class TestCompanyFromEmail:

    def test_stripe(self):
        assert _company_from_email("jobs@stripe.com") == "Stripe"

    def test_ignores_gmail(self):
        assert _company_from_email("user@gmail.com") == ""

    def test_ignores_greenhouse(self):
        assert _company_from_email("no-reply@greenhouse-mail.io") == ""

    def test_handles_subdomain(self):
        # hr@jobs.acme.com → Acme
        result = _company_from_email("hr@jobs.acme.com")
        assert result == "Acme"


# ── Job board inference ───────────────────────────────────────────────────────

class TestJobBoardInference:

    def test_linkedin(self):
        assert _infer_job_board("jobs@linkedin.com") == "LinkedIn"

    def test_indeed(self):
        assert _infer_job_board("noreply@indeed.com") == "Indeed"

    def test_greenhouse(self):
        assert _infer_job_board("no-reply@greenhouse.io") == "Greenhouse"

    def test_company_direct(self):
        assert _infer_job_board("hr@stripe.com") == "company_direct"


# ── Full enrich() integration ─────────────────────────────────────────────────

class TestEnrich:

    def test_rejection_enrichment(self):
        result = enrich(
            subject="Update on your application — Software Engineer at Stripe",
            body="After careful consideration we will not be moving forward with your candidacy.",
            sender_email="hr@stripe.com",
            event_type="rejection",
        )
        assert result.rejection_type == "hard_rejection"
        assert "Engineer" in result.role_title or result.role_title == ""
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_application_skips_rejection_type(self):
        result = enrich(
            subject="Application received — Data Scientist at Airbnb",
            body="Thank you for applying to Airbnb. We've received your application.",
            sender_email="jobs@greenhouse.io",
            event_type="application",
        )
        assert result.rejection_type == "n/a"
        assert result.job_board == "Greenhouse"

    def test_empty_input_does_not_crash(self):
        result = enrich(subject="", body="", sender_email="", event_type="rejection")
        assert result.company_name == ""
        assert result.confidence == 0.0


# ── Local sink ────────────────────────────────────────────────────────────────

class _FakeEnriched:
    def __init__(self):
        self.company_name   = "Stripe"
        self.role_title     = "Software Engineer"
        self.rejection_type = "hard_rejection"
        self.job_board      = "LinkedIn"
        self.confidence     = 0.8


class _FakeApplicationEnriched:
    def __init__(self):
        self.company_name   = "Airbnb"
        self.role_title     = "Data Scientist"
        self.rejection_type = "n/a"
        self.job_board      = "Greenhouse"
        self.confidence     = 0.6


class TestLocalSink:

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.sink = LocalSink(output_dir=self.tmpdir)

    def _fake_event(self, kind="rejection"):
        base = {
            "event_id": "test-123",
            "source": "gmail",
            "raw_subject": "Update on your application",
            "raw_body_text": "We will not be moving forward.",
            "sender_email": "hr@stripe.com",
            "received_at": "2025-06-01T10:00:00+00:00",
            "ingested_at": "2025-06-01T10:01:00+00:00",
        }
        if kind == "rejection":
            base["rejection_type"] = "unknown"
        else:
            base["application_status"] = "applied"
        return base

    def test_writes_rejection_jsonl(self):
        self.sink.write_rejection(self._fake_event("rejection"), _FakeEnriched())
        jsonl = (self.sink.output_dir / "rejections.jsonl").read_text()
        record = json.loads(jsonl.strip())
        assert record["company_name"] == "Stripe"
        assert record["rejection_type"] == "hard_rejection"
        assert record["nlp_confidence"] == 0.8

    def test_writes_application_jsonl(self):
        self.sink.write_application(
            self._fake_event("application"), _FakeApplicationEnriched()
        )
        jsonl = (self.sink.output_dir / "applications.jsonl").read_text()
        record = json.loads(jsonl.strip())
        assert record["company_name"] == "Airbnb"
        assert record["job_board"] == "Greenhouse"

    def test_csv_header_written_once(self):
        csv_path = self.sink.output_dir / "rejections.csv"
        lines = csv_path.read_text().splitlines()
        assert lines[0].startswith("date_received")

    def test_summary_counts(self):
        self.sink.write_rejection(self._fake_event("rejection"), _FakeEnriched())
        self.sink.write_rejection(self._fake_event("rejection"), _FakeEnriched())
        self.sink.write_application(
            self._fake_event("application"), _FakeApplicationEnriched()
        )
        s = self.sink.summary()
        assert s["rejections"] == 2
        assert s["applications"] == 1

    def test_multiple_writes_append_not_overwrite(self):
        for _ in range(3):
            self.sink.write_rejection(self._fake_event("rejection"), _FakeEnriched())
        lines = (self.sink.output_dir / "rejections.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3