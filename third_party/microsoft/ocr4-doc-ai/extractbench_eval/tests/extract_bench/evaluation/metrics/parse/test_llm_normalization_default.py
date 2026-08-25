"""The LLM normalization switch must stay off unless asked for by name.

Turning it on issues billable judge calls during parse rule scoring. It used to
default to JUDGE, and fell back to JUDGE on any unrecognized value, so every
intuitive spelling of "off" (`0`, `false`, `no`) silently turned it *on*.
"""

import pytest

from extract_bench.evaluation.metrics.parse.llm_normalization.config import (
    NormalizationMode,
    get_normalization_mode,
)

ENV_VAR = "EXTRACT_BENCH_LLM_NORMALIZATION"


def test_unset_is_off(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert get_normalization_mode() == NormalizationMode.OFF


@pytest.mark.parametrize("value", ["off", "OFF", " off ", "0", "false", "no", "", "typo"])
def test_anything_but_judge_is_off(monkeypatch, value):
    monkeypatch.setenv(ENV_VAR, value)
    assert get_normalization_mode() == NormalizationMode.OFF


@pytest.mark.parametrize("value", ["judge", "JUDGE", " judge "])
def test_judge_requires_the_exact_word(monkeypatch, value):
    monkeypatch.setenv(ENV_VAR, value)
    assert get_normalization_mode() == NormalizationMode.JUDGE
