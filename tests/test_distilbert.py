"""
JobSignal — tests for DistilBERT classifier and USE_DISTILBERT flag dispatch.

These tests mock the HuggingFace pipeline so they run without
downloading any model weights — fast, offline, CI-safe.

Run with: pytest tests/test_distilbert.py -v
"""

import os
import importlib
from unittest.mock import MagicMock, patch
import pytest


# ── DistilBertClassifier unit tests (mocked pipeline) ────────────────────────

class TestDistilBertClassifier:

    def _make_clf(self, mode="zero_shot", pipeline_output=None):
        """Build a classifier with the HuggingFace pipeline mocked out."""
        from jobsignal.nlp.distilbert_classifier import DistilBertClassifier

        clf = DistilBertClassifier(mode=mode)

        if mode == "zero_shot":
            mock_pipe = MagicMock(return_value={
                "labels": pipeline_output or ["hard_rejection", "soft_rejection", "unknown"],
                "scores": [0.85, 0.10, 0.05],
            })
            clf._zero_shot_pipe = mock_pipe
            # Patch the cached loader
            with patch("jobsignal.nlp.distilbert_classifier._load_zero_shot_pipeline",
                       return_value=mock_pipe):
                yield clf
        else:
            yield clf

    def test_zero_shot_returns_top_label(self):
        from jobsignal.nlp.distilbert_classifier import DistilBertClassifier, LABELS

        clf = DistilBertClassifier(mode="zero_shot")
        mock_pipe = MagicMock(return_value={
            "labels": ["hard_rejection", "soft_rejection", "unknown"],
            "scores": [0.88, 0.08, 0.04],
        })
        with patch("jobsignal.nlp.distilbert_classifier._load_zero_shot_pipeline",
                   return_value=mock_pipe):
            result = clf._classify_zero_shot("We will not be moving forward.")
        assert result == "hard_rejection"

    def test_zero_shot_falls_back_to_unknown_below_threshold(self):
        from jobsignal.nlp.distilbert_classifier import DistilBertClassifier

        clf = DistilBertClassifier(mode="zero_shot", confidence_threshold=0.9)
        mock_pipe = MagicMock(return_value={
            "labels": ["hard_rejection", "soft_rejection", "unknown"],
            "scores": [0.55, 0.30, 0.15],   # below 0.9 threshold
        })
        with patch("jobsignal.nlp.distilbert_classifier._load_zero_shot_pipeline",
                   return_value=mock_pipe):
            result = clf._classify_zero_shot("Some ambiguous text.")
        assert result == "unknown"

    def test_empty_text_returns_unknown_without_calling_pipeline(self):
        from jobsignal.nlp.distilbert_classifier import DistilBertClassifier

        clf = DistilBertClassifier(mode="zero_shot")
        mock_pipe = MagicMock()
        with patch("jobsignal.nlp.distilbert_classifier._load_zero_shot_pipeline",
                   return_value=mock_pipe):
            result = clf.classify("")
        mock_pipe.assert_not_called()
        assert result == "unknown"

    def test_whitespace_only_returns_unknown(self):
        from jobsignal.nlp.distilbert_classifier import DistilBertClassifier

        clf = DistilBertClassifier(mode="zero_shot")
        mock_pipe = MagicMock()
        with patch("jobsignal.nlp.distilbert_classifier._load_zero_shot_pipeline",
                   return_value=mock_pipe):
            result = clf.classify("   \n  ")
        mock_pipe.assert_not_called()
        assert result == "unknown"

    def test_finetuned_falls_back_to_zero_shot_if_model_missing(self):
        from jobsignal.nlp.distilbert_classifier import DistilBertClassifier

        clf = DistilBertClassifier(mode="finetuned", model_path="/nonexistent/path")
        # Should have fallen back to zero_shot
        assert clf.mode == "zero_shot"

    def test_text_truncated_to_400_words(self):
        """Pipeline should never receive more than 400 words."""
        from jobsignal.nlp.distilbert_classifier import DistilBertClassifier

        clf = DistilBertClassifier(mode="zero_shot")
        long_text = "word " * 600   # 600 words

        received_text = []
        def capture_pipe(text, **kwargs):
            received_text.append(text)
            return {
                "labels": ["unknown", "hard_rejection", "soft_rejection"],
                "scores": [0.60, 0.25, 0.15],
            }

        with patch("jobsignal.nlp.distilbert_classifier._load_zero_shot_pipeline",
                   return_value=capture_pipe):
            clf.classify(long_text)

        assert len(received_text[0].split()) <= 400


# ── USE_DISTILBERT flag dispatch (enricher) ───────────────────────────────────

class TestEnricherFlag:
    """
    Test that the USE_DISTILBERT environment variable correctly controls
    which classifier backend is used.

    We reload the enricher module for each test so the module-level
    _DISTILBERT_MODE constant picks up the patched env var.
    """

    def _reload_enricher(self):
        import jobsignal.nlp.enricher as m
        importlib.reload(m)
        return m

    def test_spacy_mode_by_default(self):
        """No USE_DISTILBERT set → spaCy rules run, DistilBERT not imported."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("USE_DISTILBERT", None)
            m = self._reload_enricher()
            assert m._DISTILBERT_MODE == ""

    def test_zero_shot_mode_set(self):
        with patch.dict(os.environ, {"USE_DISTILBERT": "zero_shot"}):
            m = self._reload_enricher()
            assert m._DISTILBERT_MODE == "zero_shot"

    def test_finetuned_mode_set(self):
        with patch.dict(os.environ, {"USE_DISTILBERT": "finetuned"}):
            m = self._reload_enricher()
            assert m._DISTILBERT_MODE == "finetuned"

    def test_spacy_rules_called_when_flag_unset(self):
        """When flag is unset, the regex matchers should run (not DistilBERT)."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("USE_DISTILBERT", None)
            m = self._reload_enricher()

            # Patch the regex matchers to verify they're called
            with patch.object(m, "_HARD_RE") as mock_hard:
                mock_hard.search.return_value = MagicMock()   # truthy → hard_rejection
                result = m._classify_rejection_type("We will not be moving forward.")
            mock_hard.search.assert_called_once()
            assert result == "hard_rejection"

    def test_distilbert_called_when_flag_is_zero_shot(self):
        """When USE_DISTILBERT=zero_shot, DistilBertClassifier.classify() is called."""
        with patch.dict(os.environ, {"USE_DISTILBERT": "zero_shot"}):
            m = self._reload_enricher()

            mock_clf = MagicMock()
            mock_clf.classify.return_value = "soft_rejection"

            with patch.object(m, "_get_distilbert", return_value=mock_clf):
                result = m._classify_rejection_type("We'll keep your resume on file.")

            mock_clf.classify.assert_called_once_with("We'll keep your resume on file.")
            assert result == "soft_rejection"

    def test_distilbert_not_imported_in_spacy_mode(self):
        """In default spaCy mode, DistilBERT should never be instantiated."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("USE_DISTILBERT", None)
            m = self._reload_enricher()

            with patch("jobsignal.nlp.distilbert_classifier.DistilBertClassifier") as mock_cls:
                m._classify_rejection_type("We will not be moving forward.")
            mock_cls.assert_not_called()