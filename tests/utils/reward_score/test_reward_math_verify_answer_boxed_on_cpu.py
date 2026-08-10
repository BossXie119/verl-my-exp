# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for the length-shaped math reward used by examples/grpo_trainer.

``math_verify`` is only imported lazily by the reward, so the fuzzy branch is
monkeypatched here and the whole file runs without extra dependencies.
"""

import importlib.util
from pathlib import Path

import pytest

REWARD_PATH = (
    Path(__file__).resolve().parents[3] / "examples" / "grpo_trainer" / "reward_math_verify_answer_boxed.py"
)

_spec = importlib.util.spec_from_file_location("reward_math_verify_answer_boxed", REWARD_PATH)
reward_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reward_mod)

MAX_LEN = 4096
LEN_TARGET = 1024


@pytest.fixture(autouse=True)
def no_math_verify(monkeypatch):
    """Fail loudly if a test relies on math_verify without declaring it."""
    monkeypatch.setattr(
        reward_mod,
        "_math_verify_score",
        lambda candidate, ground_truth: pytest.fail("unexpected math_verify call"),
    )


def score(solution, ground_truth="42", length=LEN_TARGET, max_length=MAX_LEN, **kwargs):
    return reward_mod.compute_score(
        data_source="dapo",
        solution_str=solution,
        ground_truth=ground_truth,
        extra_info={"response_length": length, "max_response_length": max_length},
        len_target=LEN_TARGET,
        **kwargs,
    )


STRICT = "Some reasoning.\nAnswer: \\boxed{42}"


def test_strict_format_earns_format_and_em():
    result = score(STRICT)
    assert result["em"] is True
    assert result["acc"] is True
    assert result["format_strict"] is True
    assert result["format_loose"] is True
    assert result["len_penalty"] == 0.0
    # acc_em (1.0) + format_strict (0.2), no length penalty at len_target
    assert result["score"] == pytest.approx(1.2)


def test_boxed_not_on_last_line_is_loose_only():
    result = score("Answer: \\boxed{42}\nHope this helps!")
    assert result["format_strict"] is False
    assert result["format_loose"] is True
    assert result["em"] is True
    assert result["score"] == pytest.approx(1.0)


def test_no_parseable_answer():
    result = score("I have no idea how to solve this.")
    assert result["score"] == pytest.approx(-1.0)
    assert result["acc"] is False
    assert result["format_loose"] is False
    assert result["pred"] == "[INVALID]"


def test_all_branches_return_the_same_keys():
    # The DAPO reward manager appends each key to a per-batch list, so a branch
    # returning fewer keys would desynchronize reward_extra_info.
    assert set(score(STRICT)) == set(score("nothing here"))


@pytest.mark.parametrize(
    "candidate,ground_truth",
    [
        ("Answer: \\boxed{1,234}", "1234"),
        ("Answer: \\boxed{42.}", "42"),
        ("Answer: \\boxed{ 42 }", "42"),
        ("Answer: \\boxed{\\text{42}}", "42"),
    ],
)
def test_exact_match_normalization(candidate, ground_truth):
    assert score(candidate, ground_truth=ground_truth)["em"] is True


def test_fuzzy_only_scores_lower_than_em(monkeypatch):
    monkeypatch.setattr(reward_mod, "_math_verify_score", lambda candidate, ground_truth: 1.0)
    result = score("Answer: \\boxed{\\frac{1}{2}}", ground_truth="0.5")
    assert result["em"] is False
    assert result["fuzzy"] is True
    assert result["score"] == pytest.approx(0.85 + 0.2)


def test_wrong_answer(monkeypatch):
    monkeypatch.setattr(reward_mod, "_math_verify_score", lambda candidate, ground_truth: 0.0)
    result = score("Answer: \\boxed{7}", ground_truth="42")
    assert result["acc"] is False
    assert result["len_penalty"] == 0.0  # not truncated -> no extra penalty
    assert result["score"] == pytest.approx(-0.5 + 0.2)


def test_length_penalty_is_quadratic_and_capped():
    at_target = score(STRICT, length=LEN_TARGET)["len_penalty"]
    below_target = score(STRICT, length=LEN_TARGET // 2)["len_penalty"]
    halfway = score(STRICT, length=(LEN_TARGET + MAX_LEN) // 2)["len_penalty"]
    at_max = score(STRICT, length=MAX_LEN)["len_penalty"]

    assert at_target == 0.0
    assert below_target == 0.0
    # Quadratic: half of the way costs a quarter of the full penalty.
    assert halfway == pytest.approx(-0.5 * 0.25)
    assert at_max == pytest.approx(-0.5)


def test_length_penalty_skipped_without_length_info():
    result = reward_mod.compute_score("dapo", STRICT, "42", extra_info={})
    assert result["len_penalty"] == 0.0
    assert result["truncated"] is False
    assert result["score"] == pytest.approx(1.2)


def test_truncated_wrong_answer_gets_flat_penalty(monkeypatch):
    monkeypatch.setattr(reward_mod, "_math_verify_score", lambda candidate, ground_truth: 0.0)
    result = score("Answer: \\boxed{7}", ground_truth="42", length=MAX_LEN)
    assert result["truncated"] is True
    assert result["len_penalty"] == pytest.approx(-0.2)


def test_brevity_is_never_rewarded_on_wrong_answers(monkeypatch):
    monkeypatch.setattr(reward_mod, "_math_verify_score", lambda candidate, ground_truth: 0.0)
    short_wrong = score("Answer: \\boxed{7}", ground_truth="42", length=64)["score"]
    long_correct = score(STRICT, length=MAX_LEN)["score"]
    assert short_wrong < long_correct


def test_nested_braces_in_boxed_answer():
    assert reward_mod.extract_final_answer("Answer: \\boxed{\\frac{1}{2}}") == "\\boxed{\\frac{1}{2}}"


def test_empty_boxed_is_not_strict_format():
    assert reward_mod.is_strict_format("Answer: \\boxed{}") is False

