"""
MLflow experiment tracking.

Wraps the NLP enricher with MLflow logging so every classification run
is tracked: which backend ran (spaCy / zero-shot / fine-tuned), confidence
distribution, label distribution, and per-batch latency.


What gets logged per Spark micro-batch:
    Params:
        - backend            ("spacy" | "zero_shot" | "finetuned")
        - batch_size
        - event_type         ("rejection" | "application")
    Metrics:
        - avg_confidence
        - min_confidence
        - max_confidence
        - company_extraction_rate   (% of rows where company_name was found)
        - role_extraction_rate      (% of rows where role_title was found)
        - batch_latency_seconds
        - label_distribution_*      (count per rejection_type)

Architecture note:
    Spark executors don't share Python state with the driver, so we log
    from the driver after collecting batch results, not inside the UDF.
    This keeps MLflow calls off the hot path and avoids serialisation
    issues with the MLflow client inside Spark workers.
"""

from __future__ import annotations
import logging
import os
import time
from collections import Counter
from contextlib import contextmanager
from typing import Iterable

logger = logging.getLogger(__name__)

# MLflow experiment name — all runs group under this in the UI
EXPERIMENT_NAME = "jobsignal-nlp-enrichment"

# Where MLflow stores tracking data (local file store by default)
DEFAULT_TRACKING_URI = "file:./mlruns"


def _get_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def _ensure_experiment():
    """Idempotent — creates the experiment once, reuses it on every call."""
    import mlflow

    mlflow.set_tracking_uri(_get_tracking_uri())
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
        logger.info("Created MLflow experiment: %s", EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_batch_run(
    backend: str,
    event_type: str,
    enriched_rows: list[dict],
    batch_latency_seconds: float,
) -> None:
    """
    Log one Spark micro-batch's enrichment results as an MLflow run.

    Args:
        backend: "spacy" | "zero_shot" | "finetuned"
        event_type: "rejection" | "application"
        enriched_rows: list of dicts, each with at minimum:
            {"company_name": str, "role_title": str,
             "rejection_type": str, "confidence": float}
        batch_latency_seconds: wall-clock time to process this batch.
    """
    if not enriched_rows:
        return   # nothing to log for an empty batch

    import mlflow

    try:
        _ensure_experiment()
    except Exception as exc:
        # MLflow tracking should never crash the pipeline — log and move on
        logger.warning("MLflow setup failed, skipping tracking: %s", exc)
        return

    n = len(enriched_rows)
    confidences = [r.get("confidence", 0.0) or 0.0 for r in enriched_rows]
    companies_found = sum(1 for r in enriched_rows if r.get("company_name"))
    roles_found = sum(1 for r in enriched_rows if r.get("role_title"))

    label_counts = Counter(
        r.get("rejection_type", "n/a") for r in enriched_rows
    )

    try:
        with mlflow.start_run(run_name=f"{event_type}_{backend}_batch"):
            # ── Params (categorical, identify the run) ─────────────────────
            mlflow.log_param("backend", backend)
            mlflow.log_param("event_type", event_type)
            mlflow.log_param("batch_size", n)

            # ── Metrics (numeric, trackable over time) ─────────────────────
            mlflow.log_metric("avg_confidence", sum(confidences) / n)
            mlflow.log_metric("min_confidence", min(confidences))
            mlflow.log_metric("max_confidence", max(confidences))
            mlflow.log_metric("company_extraction_rate", companies_found / n)
            mlflow.log_metric("role_extraction_rate", roles_found / n)
            mlflow.log_metric("batch_latency_seconds", batch_latency_seconds)
            mlflow.log_metric("throughput_events_per_sec", n / max(batch_latency_seconds, 0.001))

            # Label distribution — one metric per label, normalised counts
            for label, count in label_counts.items():
                safe_label = label.replace(" ", "_")
                mlflow.log_metric(f"label_count_{safe_label}", count)
                mlflow.log_metric(f"label_pct_{safe_label}", count / n)

        logger.info(
            "MLflow run logged | backend=%s event_type=%s rows=%d avg_conf=%.2f",
            backend, event_type, n, sum(confidences) / n,
        )
    except Exception as exc:
        # Never let tracking failures break the actual pipeline
        logger.warning("MLflow logging failed (pipeline continues): %s", exc)


@contextmanager
def timed_batch():
    """
    Context manager to measure batch processing latency.

    Usage:
        with timed_batch() as timer:
            # do the enrichment work
            ...
        log_batch_run(..., batch_latency_seconds=timer.elapsed)
    """
    class _Timer:
        elapsed: float = 0.0

    t = _Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - start


def log_model_comparison(
    backend_results: dict[str, list[dict]],
) -> None:
    """
    Compare multiple backends side-by-side in a single MLflow run.
    Useful for the "which classifier should I deploy" decision.

    Args:
        backend_results: {"spacy": [...rows...], "zero_shot": [...rows...]}
                          Each backend should have run on the SAME input data
                          for a fair comparison.
    """
    import mlflow

    try:
        _ensure_experiment()
    except Exception as exc:
        logger.warning("MLflow setup failed, skipping comparison: %s", exc)
        return

    try:
        with mlflow.start_run(run_name="backend_comparison"):
            for backend, rows in backend_results.items():
                if not rows:
                    continue
                confidences = [r.get("confidence", 0.0) or 0.0 for r in rows]
                n = len(rows)
                mlflow.log_metric(f"{backend}_avg_confidence", sum(confidences) / n)
                mlflow.log_metric(f"{backend}_sample_size", n)

        logger.info("MLflow backend comparison logged for: %s", list(backend_results.keys()))
    except Exception as exc:
        logger.warning("MLflow comparison logging failed: %s", exc)