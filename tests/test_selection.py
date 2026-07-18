"""Unit tests for the Stage-3 keyframe selection and subtitle-timestamp helpers
(run_qa.py). Video duration is monkeypatched so no video files are needed.
"""
import pytest

import run_qa
from run_qa import select_keyframes, parse_time, clean_srt_line, format_time


@pytest.fixture
def fixed_duration(monkeypatch):
    """Force get_aligned_duration_and_fps to a known duration (no video I/O)."""
    def _set(duration, fps=1.0):
        monkeypatch.setattr(run_qa, "get_aligned_duration_and_fps", lambda _p: (duration, fps))
    return _set


class TestSelectKeyframes:
    def test_topk_picks_highest_scoring_seconds_sorted(self, fixed_duration):
        fixed_duration(10)
        entry = {"video_path": "x.mp4", "video_id": "v", "score_fusion": [0, 0, 9, 0, 8, 0, 0, 7, 0, 0]}
        out = select_keyframes(entry, k=3, score_key="score_fusion")
        assert out == [2, 4, 7]  # sorted, de-duplicated

    def test_zero_duration_returns_empty(self, fixed_duration):
        fixed_duration(0)
        entry = {"video_path": "x.mp4", "video_id": "v", "score_fusion": [1, 2, 3]}
        assert select_keyframes(entry, k=4, score_key="score_fusion") == []

    def test_nonpositive_k_returns_empty(self, fixed_duration):
        fixed_duration(10)
        entry = {"video_path": "x.mp4", "video_id": "v", "score_fusion": [1] * 10}
        assert select_keyframes(entry, k=0, score_key="score_fusion") == []

    def test_k_ge_duration_returns_all_seconds(self, fixed_duration):
        fixed_duration(3)
        entry = {"video_path": "x.mp4", "video_id": "v", "score_fusion": [1, 2, 3]}
        assert select_keyframes(entry, k=8, score_key="score_fusion") == [0, 1, 2]

    def test_empty_scores_falls_back_to_uniform(self, fixed_duration):
        fixed_duration(10)
        entry = {"video_path": "x.mp4", "video_id": "v", "score_fusion": []}
        out = select_keyframes(entry, k=4, score_key="score_fusion")
        assert out == [0, 3, 6, 9]  # empty scores -> uniform fallback

    def test_scores_shorter_than_duration_are_padded(self, fixed_duration):
        fixed_duration(10)
        # Only 4 scores for a 10s video; padding keeps the top-k within range.
        entry = {"video_path": "x.mp4", "video_id": "v", "score_fusion": [0.1, 0.9, 0.2, 0.8]}
        out = select_keyframes(entry, k=2, score_key="score_fusion")
        assert out == [1, 3]

    def test_result_is_sorted_and_deduplicated(self, fixed_duration):
        fixed_duration(6)
        entry = {"video_path": "x.mp4", "video_id": "v", "score_fusion": [5, 4, 3, 2, 1, 0]}
        out = select_keyframes(entry, k=3, score_key="score_fusion")
        assert out == sorted(set(out))
        assert out == [0, 1, 2]


class TestParseTime:
    def test_srt_comma_millis(self):
        assert parse_time("00:01:05,500") == pytest.approx(65.5)

    def test_colon_only(self):
        assert parse_time("01:00:00") == pytest.approx(3600.0)

    def test_dot_millis(self):
        assert parse_time("0:00:10.250") == pytest.approx(10.25)

    def test_malformed_returns_zero(self):
        assert parse_time("not-a-time") == 0.0


class TestCleanSrtLine:
    def test_strips_html_tags(self):
        assert clean_srt_line("<i>hello</i> world") == "hello world"

    def test_plain_text_unchanged(self):
        assert clean_srt_line("  plain text  ") == "plain text"


class TestFormatTime:
    def test_formats_hms(self):
        assert format_time(65) == "0:01:05"

    def test_truncates_fractional_seconds(self):
        assert format_time(3661.9) == "1:01:01"
