"""Unit tests for the Stage-2 normalization and fusion primitives (fuse_scores.py).

These cover the pure numeric core — Min-Max scaling, masked temperature Softmax,
the full preprocessing pipeline — plus the end-to-end dead-stream reweighting via
the real fuse_scores() entry point (strategy='equal', so no model/network is used).
No video I/O required.
"""
import json

import numpy as np
import pytest

from fuse_scores import (
    fuse_scores,
    min_max_scale,
    masked_temperature_softmax,
    preprocess_score_stream,
)


class TestMinMaxScale:
    def test_maps_to_unit_range(self):
        out = min_max_scale(np.array([0.0, 5.0, 10.0]))
        assert out.min() == 0.0
        assert out.max() == 1.0
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0])

    def test_constant_stream_maps_to_zero(self):
        # An all-identical stream carries no information -> mapped to all-zero
        # so it injects no noise into the fusion.
        out = min_max_scale(np.array([0.5, 0.5, 0.5]))
        np.testing.assert_array_equal(out, np.zeros(3))

    def test_empty_input(self):
        out = min_max_scale(np.array([]))
        assert len(out) == 0

    def test_shifts_negative_values_nonnegative(self):
        out = min_max_scale(np.array([-1.0, 0.0, 1.0]))
        assert out.min() >= 0.0
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


class TestMaskedTemperatureSoftmax:
    def test_output_is_probability_distribution(self):
        out = masked_temperature_softmax(np.array([0.1, 0.5, 0.9]), temperature=0.5)
        assert out.sum() == pytest.approx(1.0)
        assert np.all(out >= 0)

    def test_zeros_stay_zero(self):
        # Positions with a (near-)zero score keep zero probability (sparsity mask).
        out = masked_temperature_softmax(np.array([0.0, 0.8, 0.0, 0.2]), temperature=0.5)
        assert out[0] == 0.0
        assert out[2] == 0.0
        assert out.sum() == pytest.approx(1.0)

    def test_all_zero_input_returns_all_zero(self):
        out = masked_temperature_softmax(np.zeros(4), temperature=0.5)
        np.testing.assert_array_equal(out, np.zeros(4))

    def test_lower_temperature_sharpens(self):
        scores = np.array([0.3, 0.6, 1.0])
        sharp = masked_temperature_softmax(scores, temperature=0.1)
        smooth = masked_temperature_softmax(scores, temperature=2.0)
        # A sharper distribution puts more mass on the largest entry.
        assert sharp.max() > smooth.max()

    def test_nonpositive_temperature_does_not_crash(self):
        # Guarded against divide-by-zero -> NaN.
        out = masked_temperature_softmax(np.array([0.2, 0.8]), temperature=0.0)
        assert np.all(np.isfinite(out))
        assert out.sum() == pytest.approx(1.0)

    def test_empty_input(self):
        out = masked_temperature_softmax(np.array([]), temperature=0.5)
        assert len(out) == 0


class TestPreprocessScoreStream:
    def test_varied_stream_is_normalized_distribution(self):
        out = preprocess_score_stream([0.1, 0.4, 0.2, 0.9], temperature=0.5)
        assert out.sum() == pytest.approx(1.0)
        assert len(out) == 4

    def test_constant_stream_collapses_to_zero(self):
        # Min-Max sends a constant stream to all-zero, which the masked Softmax
        # keeps at zero. This is the "dead stream" signal Stage 2 reweights on.
        out = preprocess_score_stream([0.5, 0.5, 0.5, 0.5], temperature=0.5)
        assert out.sum() == 0.0

    def test_all_zero_stream_collapses_to_zero(self):
        out = preprocess_score_stream([0.0, 0.0, 0.0], temperature=0.5)
        assert out.sum() == 0.0

    def test_empty_stream(self):
        out = preprocess_score_stream([], temperature=0.5)
        assert len(out) == 0


class TestDeadStreamReweighting:
    """End-to-end Stage-2 behaviour via the real fuse_scores() entry point
    (strategy='equal', so no LLM/network). A stream whose normalized distribution
    is all-zero must have its gating weight redistributed to the active streams,
    and the fused distribution must never collapse to all-zero."""

    @staticmethod
    def _run(tmp_path, entry, strategy="equal", temperature=0.5):
        in_path = tmp_path / "raw.json"
        in_path.write_text(json.dumps([entry]))
        fuse_scores(str(in_path), str(tmp_path), strategy=strategy, temperature=temperature)
        out_path = tmp_path / f"raw_fusion_{strategy}_T{temperature}.json"
        return json.loads(out_path.read_text())[0]

    def test_constant_stream_weight_is_redistributed(self, tmp_path):
        # relevance is constant -> min-max collapses it to all-zero (a "dead" stream).
        entry = {
            "score_logic":     [0.1, 0.9, 0.2, 0.8, 0.5],
            "score_relevance": [0.5, 0.5, 0.5, 0.5, 0.5],
            "score_narrative": [0.2, 0.3, 0.9, 0.1, 0.6],
        }
        out = self._run(tmp_path, entry)
        # Dead stream gets zero weight; the two live streams absorb it and sum to 1.
        assert out["w_relevance"] == 0.0
        assert out["w_logic"] + out["w_narrative"] == pytest.approx(1.0)
        assert sum(out["score_fusion"]) == pytest.approx(1.0)

    def test_all_active_streams_keep_uniform_weights(self, tmp_path):
        entry = {
            "score_logic":     [0.1, 0.9, 0.2, 0.8, 0.5],
            "score_relevance": [0.3, 0.7, 0.4, 0.6, 0.5],
            "score_narrative": [0.2, 0.3, 0.9, 0.1, 0.6],
        }
        out = self._run(tmp_path, entry)
        # No dead stream -> the equal-strategy 1/3 weights are left untouched.
        assert out["w_logic"] == pytest.approx(0.333)
        assert out["w_relevance"] == pytest.approx(0.333)
        assert out["w_narrative"] == pytest.approx(0.334)
        assert sum(out["score_fusion"]) == pytest.approx(1.0)

    def test_fusion_never_collapses_to_all_zero(self, tmp_path):
        # Regression guard: even a degenerate mix (two dead streams) keeps the one
        # live stream's distribution instead of collapsing score_fusion to zeros.
        entry = {
            "score_logic":     [0.1, 0.9, 0.2, 0.8, 0.5],  # live
            "score_relevance": [0.5, 0.5, 0.5, 0.5, 0.5],  # constant -> dead
            "score_narrative": [0.0, 0.0, 0.0, 0.0, 0.0],  # all-zero -> dead
        }
        out = self._run(tmp_path, entry)
        assert sum(out["score_fusion"]) == pytest.approx(1.0)
        assert out["w_logic"] == pytest.approx(1.0)

    def test_all_weight_on_dead_stream_falls_back_to_active(self, tmp_path, monkeypatch):
        # The exact collapse case: the gater concentrates ALL weight on a dead
        # stream (e.g. a subtitle query on a subtitle-less video -> narrative=1.0).
        # Without the fallback, every weight zeroes out and score_fusion collapses.
        import fuse_scores as fs
        # Stub the backend (no API key / network) and force the pathological triple.
        monkeypatch.setattr(fs, "QGateGrounder", lambda *a, **k: object())
        monkeypatch.setattr(
            fs, "get_llm_weights",
            lambda *a, **k: {"w_logic": 0.0, "w_relevance": 0.0, "w_narrative": 1.0},
        )
        entry = {
            "question": "what is said?",
            "options": "A) x\nB) y",
            "score_logic":     [0.1, 0.9, 0.2, 0.8, 0.5],  # live
            "score_relevance": [0.2, 0.3, 0.9, 0.1, 0.6],  # live
            "score_narrative": [0.0, 0.0, 0.0, 0.0, 0.0],  # dead, yet gater weights it 1.0
        }
        out = self._run(tmp_path, entry, strategy="llm")
        # Weight falls back to a uniform split over the two live streams.
        assert out["w_narrative"] == 0.0
        assert out["w_logic"] == pytest.approx(0.5)
        assert out["w_relevance"] == pytest.approx(0.5)
        assert sum(out["score_fusion"]) == pytest.approx(1.0)  # not collapsed
