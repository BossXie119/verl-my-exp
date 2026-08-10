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

"""Custom math reward that prefers `Answer: \\boxed{...}` final answers.

Requires the `math-verify` package: `pip install math-verify`.
"""

from __future__ import annotations

import re
from typing import Optional

from verl.utils.reward_score.math_verify import compute_score as math_verify_compute_score

try:
    import math_verify  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "`math-verify` is required for this reward. Please install it first by running `pip install math-verify`."
    ) from exc


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
    matches = list(re.finditer(r"(?i)Answer\s*:\s*", solution_str))
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


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    del data_source, extra_info

    candidate = extract_final_answer(solution_str)
    if not candidate:
        return {
            "score": -1.0,
            "acc": False,
            "format": False,
            "pred": "[INVALID]",
        }

    score = math_verify_compute_score(candidate, ground_truth)
    acc = score >= 1.0

    return {
        "score": 1.0 if acc else -1.0,
        "acc": acc,
        "format": True,
        "pred": candidate,
    }
