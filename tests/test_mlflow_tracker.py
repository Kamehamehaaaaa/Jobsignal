from unittest.mock import MagicMock, patch, call
import pytest

from tracking.mlflow_tracker import (
    log_batch_run,
    log_model_comparison,
    timed_batch,
)


# ── log_batch_run ──────────────────────────────────────────────────────────────

class TestLogBatchRun:

    def _sample_rows(self):
        return [
            {"company_name": "Stripe",  "role_title": "SWE",  "rejection_type": "hard_rejection", "confidence": 0.9},
            {"company_name": "Airbnb",  "role_title": "ML Eng", "rejection_type": "soft_rejection", "confidence": 0.6},
            {"company_name": "",        "role_title": "",       "rejection_type": "unknown",        "confidence": 0.2},
        ]

    def test_empty_batch_skips_logging(self):
        with patch("mlflow.start_run") as mock_start:
            log_batch_run(
                backend="spacy", event_type="rejection",
                enriched_rows=[], batch_latency_seconds=1.0,
            )
        mock_start.assert_not_called()

    def test_logs_params_and_metrics(self):
        with patch("jobsignal.tracking.mlflow_tracker._ensure_experiment"), \
             patch("mlflow.start_run") as mock_start, \
             patch("mlflow.log_param") as mock_param, \
             patch("mlflow.log_metric") as mock_metric:

            mock_start.return_value.__enter__ = MagicMock()
            mock_start.return_value.__exit__ = MagicMock(return_value=False)

            log_batch_run(
                backend="spacy",
                event_type="rejection",
                enriched_rows=self._sample_rows(),
                batch_latency_seconds=2.5,
            )

        mock_param.assert_any_call("backend", "spacy")
        mock_param.assert_any_call("event_type", "rejection")
        mock_param.assert_any_call("batch_size", 3)

        # avg confidence = (0.9 + 0.6 + 0.2) / 3
        metric_calls = {c.args[0]: c.args[1] for c in mock_metric.call_args_list}
        assert abs(metric_calls["avg_confidence"] - (1.7 / 3)) < 1e-6
        assert metric_calls["min_confidence"] == 0.2
        assert metric_calls["max_confidence"] == 0.9
        assert metric_calls["batch_latency_seconds"] == 2.5

    def test_extraction_rates_computed_correctly(self):
        with patch("jobsignal.tracking.mlflow_tracker._ensure_experiment"), \
             patch("mlflow.start_run") as mock_start, \
             patch("mlflow.log_param"), \
             patch("mlflow.log_metric") as mock_metric:

            mock_start.return_value.__enter__ = MagicMock()
            mock_start.return_value.__exit__ = MagicMock(return_value=False)

            log_batch_run(
                backend="spacy", event_type="rejection",
                enriched_rows=self._sample_rows(),
                batch_latency_seconds=1.0,
            )

        metric_calls = {c.args[0]: c.args[1] for c in mock_metric.call_args_list}
        # 2 of 3 rows have company_name, 2 of 3 have role_title
        assert abs(metric_calls["company_extraction_rate"] - (2 / 3)) < 1e-6
        assert abs(metric_calls["role_extraction_rate"] - (2 / 3)) < 1e-6

    def test_label_distribution_logged_per_label(self):
        with patch("jobsignal.tracking.mlflow_tracker._ensure_experiment"), \
             patch("mlflow.start_run") as mock_start, \
             patch("mlflow.log_param"), \
             patch("mlflow.log_metric") as mock_metric:

            mock_start.return_value.__enter__ = MagicMock()
            mock_start.return_value.__exit__ = MagicMock(return_value=False)

            log_batch_run(
                backend="spacy", event_type="rejection",
                enriched_rows=self._sample_rows(),
                batch_latency_seconds=1.0,
            )

        metric_names = [c.args[0] for c in mock_metric.call_args_list]
        assert "label_count_hard_rejection" in metric_names
        assert "label_count_soft_rejection" in metric_names
        assert "label_count_unknown" in metric_names

    def test_mlflow_failure_does_not_raise(self):
        """Tracking must never crash the pipeline, even if MLflow is broken."""
        with patch(
            "jobsignal.tracking.mlflow_tracker._ensure_experiment",
            side_effect=RuntimeError("tracking server down"),
        ):
            # Should not raise
            log_batch_run(
                backend="spacy", event_type="rejection",
                enriched_rows=self._sample_rows(),
                batch_latency_seconds=1.0,
            )

    def test_start_run_failure_does_not_raise(self):
        with patch("jobsignal.tracking.mlflow_tracker._ensure_experiment"), \
             patch("mlflow.start_run", side_effect=RuntimeError("connection refused")):
            log_batch_run(
                backend="spacy", event_type="rejection",
                enriched_rows=self._sample_rows(),
                batch_latency_seconds=1.0,
            )

    def test_throughput_metric_uses_latency(self):
        with patch("jobsignal.tracking.mlflow_tracker._ensure_experiment"), \
             patch("mlflow.start_run") as mock_start, \
             patch("mlflow.log_param"), \
             patch("mlflow.log_metric") as mock_metric:

            mock_start.return_value.__enter__ = MagicMock()
            mock_start.return_value.__exit__ = MagicMock(return_value=False)

            log_batch_run(
                backend="zero_shot", event_type="application",
                enriched_rows=self._sample_rows(),
                batch_latency_seconds=3.0,
            )

        metric_calls = {c.args[0]: c.args[1] for c in mock_metric.call_args_list}
        # 3 rows / 3.0 seconds = 1.0 events/sec
        assert abs(metric_calls["throughput_events_per_sec"] - 1.0) < 1e-6


# ── log_model_comparison ──────────────────────────────────────────────────────

class TestLogModelComparison:

    def test_compares_multiple_backends(self):
        backend_results = {
            "spacy":     [{"confidence": 0.7}, {"confidence": 0.8}],
            "zero_shot": [{"confidence": 0.85}, {"confidence": 0.9}],
        }
        with patch("jobsignal.tracking.mlflow_tracker._ensure_experiment"), \
             patch("mlflow.start_run") as mock_start, \
             patch("mlflow.log_metric") as mock_metric:

            mock_start.return_value.__enter__ = MagicMock()
            mock_start.return_value.__exit__ = MagicMock(return_value=False)

            log_model_comparison(backend_results)

        metric_calls = {c.args[0]: c.args[1] for c in mock_metric.call_args_list}
        assert abs(metric_calls["spacy_avg_confidence"] - 0.75) < 1e-6
        assert abs(metric_calls["zero_shot_avg_confidence"] - 0.875) < 1e-6
        assert metric_calls["spacy_sample_size"] == 2
        assert metric_calls["zero_shot_sample_size"] == 2

    def test_skips_empty_backend_results(self):
        backend_results = {"spacy": [], "zero_shot": [{"confidence": 0.9}]}
        with patch("jobsignal.tracking.mlflow_tracker._ensure_experiment"), \
             patch("mlflow.start_run") as mock_start, \
             patch("mlflow.log_metric") as mock_metric:

            mock_start.return_value.__enter__ = MagicMock()
            mock_start.return_value.__exit__ = MagicMock(return_value=False)

            log_model_comparison(backend_results)

        metric_names = [c.args[0] for c in mock_metric.call_args_list]
        assert "spacy_avg_confidence" not in metric_names
        assert "zero_shot_avg_confidence" in metric_names


# ── timed_batch context manager ───────────────────────────────────────────────

class TestTimedBatch:

    def test_measures_elapsed_time(self):
        import time as time_module
        with timed_batch() as timer:
            time_module.sleep(0.05)
        assert timer.elapsed >= 0.05

    def test_elapsed_available_after_exception(self):
        import time as time_module
        with pytest.raises(ValueError):
            with timed_batch() as timer:
                time_module.sleep(0.02)
                raise ValueError("boom")
        assert timer.elapsed >= 0.02