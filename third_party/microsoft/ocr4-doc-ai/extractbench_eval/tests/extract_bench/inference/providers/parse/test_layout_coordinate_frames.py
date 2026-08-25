"""Tests for the ``bbox_scale`` config: absolute-pixel layout coordinates.

Some models ground well but are unreliable at re-normalizing bboxes into the
prompt's 0-1000 range, which conflates grounding quality with coordinate
compliance. Pipelines can now set ``bbox_scale=None`` (a number is the
coordinate grid, ``None`` means native pixels) to prompt for the pixel frame (the ``*_ABS`` prompt variants) and have
``build_layout_pages`` normalize by the image dimensions instead of 1000.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import pytest

pytest.importorskip("PIL", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")

from extract_bench.inference.providers.parse._layout_utils import (
    SYSTEM_PROMPT_LAYOUT,
    SYSTEM_PROMPT_LAYOUT_ABS,
    SYSTEM_PROMPT_LAYOUT_GEMINI,
    SYSTEM_PROMPT_LAYOUT_GEMINI_ABS,
    USER_PROMPT_LAYOUT,
    USER_PROMPT_LAYOUT_ABS,
    USER_PROMPT_LAYOUT_GEMINI,
    USER_PROMPT_LAYOUT_GEMINI_ABS,
    build_layout_pages,
    resolve_layout_prompts,
)


def _item(bbox: list, label: str = "Text", text: str = "t") -> dict:
    return {"bbox": bbox, "label": label, "text": text}


def _segs(items: list[dict], width: int = 850, height: int = 1100, scale: float | None = 1000):
    pages = build_layout_pages(items, width, height, "md", bbox_scale=scale)
    assert len(pages) == 1
    segs = []
    for it in pages[0].items:
        assert it.bbox is not None
        segs.append(it.bbox)
    return segs


class TestAbsPromptVariants(unittest.TestCase):
    def test_system_prompts_diverge_only_on_coordinate_frame(self) -> None:
        self.assertNotEqual(SYSTEM_PROMPT_LAYOUT_ABS, SYSTEM_PROMPT_LAYOUT)
        self.assertIn("ABSOLUTE PIXEL", SYSTEM_PROMPT_LAYOUT_ABS)
        self.assertNotIn("normalized 0-1000", SYSTEM_PROMPT_LAYOUT_ABS)
        # Everything except the coordinate bullet is retained.
        for token in ("Caption, Footnote, Formula", "reading order", "<div>"):
            self.assertIn(token, SYSTEM_PROMPT_LAYOUT_ABS)

    def test_gemini_system_prompts_diverge_only_on_coordinate_frame(self) -> None:
        self.assertNotEqual(SYSTEM_PROMPT_LAYOUT_GEMINI_ABS, SYSTEM_PROMPT_LAYOUT_GEMINI)
        self.assertIn("ABSOLUTE PIXEL", SYSTEM_PROMPT_LAYOUT_GEMINI_ABS)
        self.assertNotIn("normalized 0-1000", SYSTEM_PROMPT_LAYOUT_GEMINI_ABS)
        # Gemini's native coordinate order is preserved in the abs variant.
        self.assertIn("[y_min, x_min, y_max, x_max]", SYSTEM_PROMPT_LAYOUT_GEMINI_ABS)

    def test_user_prompts_mention_pixel_frame(self) -> None:
        self.assertIn("absolute pixel coordinates", USER_PROMPT_LAYOUT_ABS)
        self.assertNotIn("absolute pixel coordinates", USER_PROMPT_LAYOUT)
        self.assertIn("absolute pixel coordinates", USER_PROMPT_LAYOUT_GEMINI_ABS)
        self.assertNotIn("absolute pixel coordinates", USER_PROMPT_LAYOUT_GEMINI)


class TestNormalizedFrame(unittest.TestCase):
    def test_in_range_coords_unchanged(self) -> None:
        (seg,) = _segs([_item([100, 200, 600, 700])])
        self.assertAlmostEqual(seg.x, 0.1)
        self.assertAlmostEqual(seg.y, 0.2)
        self.assertAlmostEqual(seg.w, 0.5)
        self.assertAlmostEqual(seg.h, 0.5)

    def test_overshoot_clamped_into_unit_square(self) -> None:
        (seg,) = _segs([_item([0, 0, 1003, 998])])
        self.assertAlmostEqual(seg.w, 1.0)  # 1003 -> 1000
        self.assertAlmostEqual(seg.h, 0.998)

    def test_negative_coords_clamped(self) -> None:
        (seg,) = _segs([_item([-50, 0, 500, 500])])
        self.assertAlmostEqual(seg.x, 0.0)
        self.assertAlmostEqual(seg.w, 0.5)

    def test_pixel_output_warns_but_is_not_rescaled(self) -> None:
        # A model that emitted pixels under the normalized prompt gets a
        # compliance warning pointing at bbox_scale=None — the frame is
        # config-driven, never guessed from the values.
        with self.assertLogs("extract_bench.inference.providers.parse._layout_utils", "WARNING") as logs:
            (seg,) = _segs([_item([0, 990, 850, 1100])])
        self.assertTrue(any("bbox_scale" in m for m in logs.output))
        self.assertAlmostEqual(seg.y, 0.990)  # still the 0-1000 frame
        self.assertAlmostEqual(seg.h, 0.010)  # 1100 clamped to 1000


class TestPixelFrame(unittest.TestCase):
    def test_pixel_coords_normalized_by_image_dims(self) -> None:
        segs = _segs(
            [
                _item([85, 110, 425, 550]),
                _item([0, 990, 850, 1100]),
            ],
            scale=None,
        )
        body, footer = segs
        self.assertAlmostEqual(body.x, 0.1)
        self.assertAlmostEqual(body.y, 0.1)
        self.assertAlmostEqual(body.w, 0.4)
        self.assertAlmostEqual(body.h, 0.4)
        self.assertAlmostEqual(footer.y, 0.9)
        self.assertAlmostEqual(footer.h, 0.1)

    def test_pixel_overshoot_clamped_to_image(self) -> None:
        (seg,) = _segs([_item([0, 0, 900, 1120])], scale=None)
        self.assertAlmostEqual(seg.w, 1.0)  # 900 -> 850
        self.assertAlmostEqual(seg.h, 1.0)  # 1120 -> 1100


class TestRobustness(unittest.TestCase):
    def test_nan_bbox_skipped(self) -> None:
        segs = _segs([_item([float("nan"), 100, 500, 600]), _item([100, 200, 600, 700])])
        self.assertEqual(len(segs), 1)
        self.assertAlmostEqual(segs[0].x, 0.1)

    def test_string_bbox_skipped(self) -> None:
        segs = _segs([_item(["100", "200", "600", "700"]), _item([100, 200, 600, 700])])
        self.assertEqual(len(segs), 1)

    def test_short_bbox_skipped(self) -> None:
        segs = _segs([_item([1, 2, 3]), _item([100, 200, 600, 700])])
        self.assertEqual(len(segs), 1)


class TestPerceivedSizeResize(unittest.TestCase):
    """Pin the resize rule against the vision docs' published examples."""

    def test_docs_reference_vectors(self) -> None:
        from extract_bench.inference.providers.parse.anthropic import _resized_size

        # Worked examples from the vision-coordinates docs page.
        self.assertEqual(_resized_size(1075, 1520, 1568, 1568), (924, 1307))
        self.assertEqual(_resized_size(1920, 1080, 1568, 1568), (1456, 819))
        # Fits within limits: unchanged.
        self.assertEqual(_resized_size(800, 600, 1568, 1568), (800, 600))
        # Same scan on high-resolution-tier limits: no resize needed.
        self.assertEqual(_resized_size(1075, 1520, 2576, 4784), (1075, 1520))

    def test_pixel_mode_resizes_page_image(self) -> None:
        from PIL import Image

        from extract_bench.inference.providers.parse.anthropic import AnthropicProvider

        prev = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        try:
            provider = AnthropicProvider("anthropic", {"mode": "parse_with_layout", "bbox_scale": None})
            resized = provider._resize_to_perceived_size(Image.new("RGB", (1075, 1520)))
            self.assertEqual(resized.size, (924, 1307))
            small = Image.new("RGB", (800, 600))
            self.assertIs(provider._resize_to_perceived_size(small), small)
        finally:
            if prev is None:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            else:
                os.environ["ANTHROPIC_API_KEY"] = prev


class TestProviderConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._prev_key = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = "test-key"

    def tearDown(self) -> None:
        if self._prev_key is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = self._prev_key

    def _provider(self, **config):
        from extract_bench.inference.providers.parse.anthropic import AnthropicProvider

        return AnthropicProvider("anthropic", {"mode": "parse_with_layout", **config})

    def test_default_frame_uses_normalized_prompts(self) -> None:
        provider = self._provider()
        self.assertEqual(provider._layout_system_prompt, SYSTEM_PROMPT_LAYOUT)
        self.assertEqual(provider._layout_user_prompt, USER_PROMPT_LAYOUT)

    def test_pixel_frame_uses_abs_prompts(self) -> None:
        provider = self._provider(bbox_scale=None)
        self.assertEqual(provider._layout_system_prompt, SYSTEM_PROMPT_LAYOUT_ABS)
        self.assertEqual(provider._layout_user_prompt, USER_PROMPT_LAYOUT_ABS)

    def test_unsupported_scale_rejected(self) -> None:
        from extract_bench.inference.providers.base import ProviderConfigError

        with self.assertRaises(ProviderConfigError):
            self._provider(bbox_scale=500)

    def test_pixel_frame_rejected_for_file_mode(self) -> None:
        from extract_bench.inference.providers.base import ProviderConfigError
        from extract_bench.inference.providers.parse.anthropic import AnthropicProvider

        with self.assertRaises(ProviderConfigError):
            AnthropicProvider(
                "anthropic",
                {"mode": "parse_with_layout_file", "bbox_scale": None},
            )


class TestPixelFrameProviderGate(unittest.TestCase):
    """Pixel mode is only offered by providers that pin the perceived frame.

    ``build_layout_pages(bbox_scale=None)`` normalizes by the page dimensions
    recorded at inference time. That is only the model's coordinate frame if
    the sent image is never downscaled on the way in — which holds for
    Anthropic because ``_resize_to_perceived_size`` pre-resizes per the
    published rule, and does not hold for the other div-layout providers.
    Without the gate they would accept ``bbox_scale=None`` and report a
    silently shifted frame.
    """

    def test_gate_defaults_closed(self) -> None:
        from extract_bench.inference.providers.base import ProviderConfigError

        with self.assertRaises(ProviderConfigError) as ctx:
            resolve_layout_prompts(None, "parse_with_layout")
        self.assertIn("not supported by this provider", str(ctx.exception))

    def test_gate_open_returns_abs_prompts(self) -> None:
        self.assertEqual(
            resolve_layout_prompts(None, "parse_with_layout", pixel_frame_supported=True),
            (SYSTEM_PROMPT_LAYOUT_ABS, USER_PROMPT_LAYOUT_ABS),
        )

    def test_gate_does_not_affect_default_frame(self) -> None:
        # The 0-1000 frame is unchanged for every provider, gated or not.
        for supported in (False, True):
            self.assertEqual(
                resolve_layout_prompts(1000, "parse_with_layout", pixel_frame_supported=supported),
                (SYSTEM_PROMPT_LAYOUT, USER_PROMPT_LAYOUT),
            )

    def test_unsupported_scale_still_rejected_when_gate_open(self) -> None:
        from extract_bench.inference.providers.base import ProviderConfigError

        with self.assertRaises(ProviderConfigError):
            resolve_layout_prompts(500, "parse_with_layout", pixel_frame_supported=True)

    def _env(self, **kv: str) -> None:
        """Set provider API keys for the duration of one test."""
        patcher = mock.patch.dict(os.environ, kv)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_anthropic_opts_in(self) -> None:
        from extract_bench.inference.providers.parse.anthropic import AnthropicProvider

        self._env(ANTHROPIC_API_KEY="test-key")
        provider = AnthropicProvider("anthropic", {"mode": "parse_with_layout", "bbox_scale": None})
        self.assertEqual(provider._layout_system_prompt, SYSTEM_PROMPT_LAYOUT_ABS)

    def test_openai_rejects_pixel_frame(self) -> None:
        from extract_bench.inference.providers.base import ProviderConfigError
        from extract_bench.inference.providers.parse.openai import OpenAIProvider

        self._env(OPENAI_API_KEY="test-key")
        with self.assertRaises(ProviderConfigError):
            OpenAIProvider("openai", {"mode": "parse_with_layout", "bbox_scale": None})

    def test_google_rejects_pixel_frame(self) -> None:
        from extract_bench.inference.providers.base import ProviderConfigError
        from extract_bench.inference.providers.parse.google import GoogleProvider

        self._env(GEMINI_API_KEY="test-key", GOOGLE_API_KEY="test-key")
        with self.assertRaises(ProviderConfigError):
            GoogleProvider("google", {"mode": "parse_with_layout", "bbox_scale": None})

    def test_gemma4_rejects_pixel_frame(self) -> None:
        from extract_bench.inference.providers.base import ProviderConfigError
        from extract_bench.inference.providers.parse.gemma4 import Gemma4Provider

        with self.assertRaises(ProviderConfigError):
            Gemma4Provider("gemma4", {"server_url": "http://x", "prompt_mode": "layout", "bbox_scale": None})

    def test_gated_providers_still_accept_default_frame(self) -> None:
        from extract_bench.inference.providers.parse.gemma4 import Gemma4Provider
        from extract_bench.inference.providers.parse.openai import OpenAIProvider

        self._env(OPENAI_API_KEY="test-key")
        self.assertEqual(
            OpenAIProvider("openai", {"mode": "parse_with_layout"})._layout_system_prompt,
            SYSTEM_PROMPT_LAYOUT,
        )
        self.assertEqual(
            Gemma4Provider("gemma4", {"server_url": "http://x", "prompt_mode": "layout"})._system_prompt,
            SYSTEM_PROMPT_LAYOUT,
        )


if __name__ == "__main__":
    unittest.main()
