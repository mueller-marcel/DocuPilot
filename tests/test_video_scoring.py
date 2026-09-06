"""
The VLM stage: how a pair of frames is presented, how an answer becomes graded
evidence, and what the verdict cache promises. Frames are plain arrays.
"""

from dataclasses import asdict

import numpy as np
import pytest

from docupilot.segmentation import video_scoring as vs


def frame(seed: int, h: int = 120, w: int = 200) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def decode(jpeg: bytes) -> np.ndarray:
    import cv2

    return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)


class TestParse:
    def test_action_completed_maps_to_the_confidence(self):
        judgement = vs.parse(
            '{"observation": "Filter angewendet", "category": "ACTION_COMPLETED",'
            ' "confidence": 0.8}'
        )
        assert judgement.p_boundary == pytest.approx(0.8)
        assert judgement.reason == "Filter angewendet"

    def test_every_other_category_maps_to_the_complement(self):
        # A confident "this is only a menu" is strong evidence AGAINST a
        # boundary, which is what the fusion downstream needs to see.
        for category in ("TRANSIENT_UI", "IN_PROGRESS", "NO_CHANGE", "SYSTEM_INITIATED"):
            raw = f'{{"category": "{category}", "confidence": 0.9}}'
            assert vs.parse(raw).p_boundary == pytest.approx(0.1)

    def test_the_category_is_read_case_insensitively(self):
        assert vs.parse('{"category": "transient_ui", "confidence": 0.5}').category == "TRANSIENT_UI"

    def test_confidence_is_clamped_and_defaults_to_undecided(self):
        assert vs.parse('{"category": "NO_CHANGE", "confidence": 1.7}').p_boundary == 0.0
        assert vs.parse('{"category": "NO_CHANGE", "confidence": -2}').p_boundary == 1.0
        assert vs.parse('{"category": "NO_CHANGE", "confidence": "x"}').p_boundary == 0.5
        assert vs.parse('{"category": "NO_CHANGE"}').p_boundary == 0.5

    def test_json_embedded_in_prose_is_still_read(self):
        assert vs.parse('Sure: {"category": "NO_CHANGE", "confidence": 0.5} done') is not None

    def test_an_unusable_answer_is_none_rather_than_a_guess(self):
        assert vs.parse("") is None
        assert vs.parse("no json") is None
        assert vs.parse('{"category": "WHATEVER", "confidence": 0.9}') is None

    def test_the_observation_is_kept_so_a_guessed_label_can_be_spotted(self):
        judgement = vs.parse('{"category": "NO_CHANGE", "confidence": 0.5, "reason": "r"}')
        assert judgement.reason == "r"


class TestComposite:
    def test_both_states_go_into_one_image_side_by_side(self):
        before, after = frame(1), frame(2)
        canvas = decode(vs.encode_pair(before, after))
        # As two separate images the model swaps them, which is fatal for a
        # before/after question; one canvas makes the order unambiguous.
        assert canvas.shape[1] > before.shape[1] * 2
        assert canvas.shape[0] > before.shape[0]

    def test_a_small_region_adds_a_magnified_row(self):
        before, after = frame(1), frame(2)
        plain = decode(vs.encode_pair(before, after, None))
        zoomed = decode(vs.encode_pair(before, after, (0.0, 0.0, 0.2, 0.2)))
        assert zoomed.shape[0] > plain.shape[0]

    def test_a_large_region_gets_no_zoom_row(self):
        # The crop would be the frame again — pure token cost.
        before, after = frame(1), frame(2)
        boxed = decode(vs.encode_pair(before, after, (0.0, 0.0, 0.95, 0.95)))
        plain = decode(vs.encode_pair(before, after, None))
        assert boxed.shape[0] == plain.shape[0]

    def test_the_same_pair_encodes_byte_for_byte_the_same(self):
        before, after = frame(1), frame(2)
        assert vs.encode_pair(before, after) == vs.encode_pair(before, after)

    def test_encoding_does_not_modify_the_frames_it_was_given(self):
        before, after = frame(1), frame(2)
        original = before.copy()
        vs.encode_pair(before, after, (0.0, 0.0, 0.3, 0.3))
        np.testing.assert_array_equal(before, original)

    def test_downscale_shrinks_wide_frames_and_leaves_small_ones_alone(self):
        wide = frame(3, h=600, w=1800)
        assert vs.downscale(wide).shape[1] == vs._HALF_WIDTH
        assert vs.downscale(wide).shape[0] == round(600 * vs._HALF_WIDTH / 1800)
        small = frame(4, h=100, w=100)
        assert vs.downscale(small) is small

    def test_a_crop_keeps_a_little_context_around_the_region(self):
        image = frame(5, h=200, w=200)
        cropped = vs._crop(image, (0.4, 0.4, 0.6, 0.6))
        assert cropped.shape[0] > 0.2 * 200          # the region plus padding
        assert cropped.shape[0] < image.shape[0]

    def test_an_inverted_region_falls_back_to_the_whole_frame(self):
        image = frame(5)
        assert vs._crop(image, (0.9, 0.9, 0.1, 0.1)).shape == image.shape


class TestCache:
    def test_a_verdict_survives_a_round_trip(self, tmp_path):
        path = tmp_path / "vlm.json"
        cache = vs.Cache(path)
        cache.put("k", vs.Judgement("NO_CHANGE", 0.1, "nichts"))
        cache.flush()
        assert asdict(vs.Cache(path).get("k")) == {
            "category": "NO_CHANGE", "p_boundary": 0.1, "reason": "nichts"
        }

    def test_the_key_follows_the_image_content_the_model_and_the_prompt(self):
        a, b = vs.encode_pair(frame(1), frame(2)), vs.encode_pair(frame(1), frame(3))
        assert vs.Cache.key(a, "m") != vs.Cache.key(b, "m")
        assert vs.Cache.key(a, "m") != vs.Cache.key(a, "other")
        # Keyed on content, so re-encoding the video costs nothing.
        assert vs.Cache.key(a, "m") == vs.Cache.key(vs.encode_pair(frame(1), frame(2)), "m")

    def test_a_damaged_or_foreign_entry_is_a_miss(self, tmp_path):
        broken = tmp_path / "vlm.json"
        broken.write_text('{"k": "not a verdict"}', encoding="utf-8")
        assert vs.Cache(broken).get("k") is None
        assert vs.Cache(tmp_path / "absent.json").get("k") is None

    def test_a_hit_short_circuits_the_model(self, tmp_path):
        composite = vs.encode_pair(frame(1), frame(2))
        cache = vs.Cache(tmp_path / "vlm.json")
        cache.put(vs.Cache.key(composite, vs.MODEL), vs.Judgement("NO_CHANGE", 0.1))
        # ask() would need credentials; a hit must never reach it.
        assert vs.judge(composite, cache=cache).p_boundary == pytest.approx(0.1)
