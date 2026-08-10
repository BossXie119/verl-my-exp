#!/usr/bin/env python3
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

"""Length-shaped math reward for `Answer: \\boxed{...}` final answers.

The goal of this recipe is to keep accuracy while shortening responses, so the
score is decomposed into five components:

1. strict format  -- the last line is exactly ``Answer: \\boxed{...}``
2. loose format   -- an answer is parseable somewhere in the response
3. exact-match acc -- normalized string equality with the ground truth
4. fuzzy acc      -- ``math_verify`` judges the answer equivalent
5. length penalty -- quadratic decay applied *only to correct answers*

The length penalty is intentionally restricted to correct answers: penalizing
length on wrong answers would reward "short and wrong". Wrong answers that also
hit the truncation limit get a small extra penalty instead.

`response_length` / `max_response_length` are injected into ``extra_info`` by
``verl.workers.reward_manager.dapo``; without them the length term is skipped.

Requires the `math-verify` package: `pip install math-verify`.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# ---- score components (overridable via reward.custom_reward_function.reward_kwargs) ----
FORMAT_STRICT_SCORE = 0.2
ACC_EM_SCORE = 1.0
ACC_FUZZY_SCORE = 0.85
ACC_WRONG_SCORE = -0.5
NO_ANSWER_SCORE = -1.0
LEN_TARGET = 1024
LEN_PENALTY_COEF = 0.5
TRUNCATED_WRONG_PENALTY = 0.2

_ANSWER_TAG_RE = re.compile(r"(?i)Answer\s*:\s*")
_LATEX_NOISE_RE = re.compile(r"\\(?:left|right|,|;|!|:|\s|quad|qquad)|[\s$]")


def _math_verify_score(candidate: str, ground_truth: str) -> float:
    """Lazy import so the parsing/length logic stays unit-testable without verl deps."""
    from verl.utils.reward_score.math_verify import compute_score as math_verify_compute_score

    return math_verify_compute_score(candidate, ground_truth)


def _extract_boxed_span(text: str, start_idx: int) -> Optional[str]:
    left = "\\boxed{"
    if not text.startswith(left, start_idx):
        return None

    depth = 0
    for idx in range(start_idx, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx : idx + 1]
    return None


def _extract_after_answer_tag(solution_str: str) -> Optional[str]:
    matches = list(_ANSWER_TAG_RE.finditer(solution_str))
    if not matches:
        return None

    tail = solution_str[matches[-1].end() :].strip()
    if not tail:
        return None

    if tail.startswith("\\boxed{"):
        boxed = _extract_boxed_span(tail, 0)
        if boxed is not None:
            return boxed

    return tail.splitlines()[0].strip()


def _extract_last_boxed(solution_str: str) -> Optional[str]:
    last_idx = solution_str.rfind("\\boxed{")
    if last_idx < 0:
        return None
    return _extract_boxed_span(solution_str, last_idx)


def extract_final_answer(solution_str: str) -> Optional[str]:
    candidate = _extract_after_answer_tag(solution_str)
    if candidate:
        return candidate

    candidate = _extract_last_boxed(solution_str)
    if candidate:
        return candidate

    return None


def is_strict_format(solution_str: str) -> bool:
    """True when the last non-empty line is exactly ``Answer: \\boxed{...}``.

    This mirrors the instruction injected by
    ``examples/data_preprocess/prepare_dapo_en_aime24_for_grpo.py``.
    """
    lines = [line for line in solution_str.strip().splitlines() if line.strip()]
    if not lines:
        return False

    last = lines[-1].strip()
    if not last.startswith("Answer:"):
        return False

    tail = last[len("Answer:") :].strip()
    boxed = _extract_boxed_span(tail, 0)
    # The boxed span must consume the whole remainder, and must not be empty.
    return boxed is not None and boxed == tail and len(boxed) > len("\\boxed{}")


def strip_boxed(text: str) -> str:
    """Return the content of a ``\\boxed{...}`` wrapper, or the text unchanged."""
    boxed = _extract_boxed_span(text.strip(), 0)
    if boxed is None:
        return text.strip()
    return boxed[len("\\boxed{") : -1].strip()


def normalize_answer(text: str) -> str:
    """Aggressive normalization used for the exact-match component."""
    text = strip_boxed(text)
    text = _LATEX_NOISE_RE.sub("", text)
    text = text.replace("\\text{", "").replace("}", "").replace("{", "")
    text = text.rstrip(".")
    # Thousands separators: only strip commas that sit between digits.
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return text.lower()


def exact_match(candidate: str, ground_truth: str) -> bool:
    normalized = normalize_answer(candidate)
    return bool(normalized) and normalized == normalize_answer(ground_truth)


def length_penalty(
    response_length: Optional[int],
    max_response_length: Optional[int],
    len_target: int,
    len_penalty_coef: float,
) -> float:
    """Quadratic decay penalty in ``[-len_penalty_coef, 0]``.

    Zero at or below ``len_target``, ``-len_penalty_coef`` at ``max_response_length``.
    Returns 0.0 when the length information is unavailable or the target is not
    strictly below the maximum length.
    """
    if not response_length or not max_response_length:
        return 0.0
    if max_response_length <= len_target:
        return 0.0

    ratio = (response_length - len_target) / (max_response_length - len_target)
    ratio = min(max(ratio, 0.0), 1.0)
    return -len_penalty_coef * ratio * ratio


def _result(
    score: float,
    acc: bool,
    em: bool,
    fuzzy: bool,
    format_strict: bool,
    format_loose: bool,
    len_penalty: float,
    response_length: int,
    truncated: bool,
    pred: str,
) -> dict[str, Any]:
    """Every branch must return the same key set.

    The DAPO reward manager appends each key to a per-batch list, so a missing key
    would desynchronize ``reward_extra_info`` and break metrics/dumping.
    """
    return {
        "score": float(score),
        "acc": acc,
        "em": em,
        "fuzzy": fuzzy,
        "format_strict": format_strict,
        "format_loose": format_loose,
        "len_penalty": float(len_penalty),
        "response_length": int(response_length),
        "truncated": truncated,
        "pred": pred,
    }


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    format_strict_score: float = FORMAT_STRICT_SCORE,
    acc_em_score: float = ACC_EM_SCORE,
    acc_fuzzy_score: float = ACC_FUZZY_SCORE,
    acc_wrong_score: float = ACC_WRONG_SCORE,
    no_answer_score: float = NO_ANSWER_SCORE,
    len_target: int = LEN_TARGET,
    len_penalty_coef: float = LEN_PENALTY_COEF,
    truncated_wrong_penalty: float = TRUNCATED_WRONG_PENALTY,
    **_ignored,
) -> dict[str, Any]:
    del data_source

    extra_info = extra_info or {}
    response_length = int(extra_info.get("response_length") or 0)
    max_response_length = int(extra_info.get("max_response_length") or 0)
    truncated = bool(max_response_length) and response_length >= max_response_length

    candidate = extract_final_answer(solution_str)
    if not candidate:
        # No parseable answer at all: neither format component is earned and the
        # accuracy components are not evaluated.
        return _result(
            score=no_answer_score,
            acc=False,
            em=False,
            fuzzy=False,
            format_strict=False,
            format_loose=False,
            len_penalty=0.0,
            response_length=response_length,
            truncated=truncated,
            pred="[INVALID]",
        )

    format_strict = is_strict_format(solution_str)
    format_reward = format_strict_score if format_strict else 0.0

    em = exact_match(candidate, str(ground_truth))
    if em:
        fuzzy = True
    else:
        fuzzy = _math_verify_score(candidate, ground_truth) >= 1.0

    if em:
        acc_reward = acc_em_score
    elif fuzzy:
        acc_reward = acc_fuzzy_score
    else:
        acc_reward = acc_wrong_score

    acc = em or fuzzy
    if acc:
        # Shorten correct answers; never reward brevity on wrong ones.
        len_reward = length_penalty(response_length, max_response_length, len_target, len_penalty_coef)
    else:
        len_reward = -truncated_wrong_penalty if truncated else 0.0

    return _result(
        score=acc_reward + format_reward + len_reward,
        acc=acc,
        em=em,
        fuzzy=fuzzy,
        format_strict=format_strict,
        format_loose=True,
        len_penalty=len_reward,
        response_length=response_length,
        truncated=truncated,
        pred=candidate,
    )
