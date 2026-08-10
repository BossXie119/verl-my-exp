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

"""Prepare DAPO-English train/test data for a shared GRPO format.

This script normalizes datasets to use the same `prompt` column and the same
final-answer instruction:

    Answer: \boxed{<final_answer>}

Two modes are supported:

1. Split mode (default in the provided launcher): with `--test-ratio`, the
   DAPO-English train set is randomly split into train/test (fixed seed), so the
   test set comes from the same distribution as training.
2. Legacy mode: with `--test-input` and no `--test-ratio`, the given test file
   (e.g. AIME24) is used as the test set.

The output parquet files are ready to be consumed by `verl` with
`data.prompt_key=prompt` and `data.return_raw_chat=True`.
"""

from __future__ import annotations

import argparse
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset


FINAL_ANSWER_INSTRUCTION = (
    "Please reason step by step. The last line of your response must be exactly:\n"
    "Answer: \\boxed{<final_answer>}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-input",
        default="/aiarena/group/rlgroup1/duanyuanrui/data/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet",
        help="Raw DAPO English parquet file.",
    )
    parser.add_argument(
        "--train-output",
        default="/aiarena/group/rlgroup1/duanyuanrui/data/DAPO-Math-17k-Processed/en/train_grpo_qwen3_4b.parquet",
        help="Output parquet for training.",
    )
    parser.add_argument(
        "--test-input",
        default="/aiarena/group/rlgroup1/duanyuanrui/data/RLVR-Linearity-Dataset/aime24.parquet",
        help="Raw test parquet file. Only used when --test-ratio is not set.",
    )
    parser.add_argument(
        "--test-output",
        default="/aiarena/group/rlgroup1/duanyuanrui/data/DAPO-Math-17k-Processed/en/test_grpo_qwen3_4b.parquet",
        help="Output parquet for evaluation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="Fraction of the train set held out as test set (e.g. 0.1). When set, "
        "--test-input is ignored and the DAPO-English train set is split instead.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/test split (only used with --test-ratio).",
    )
    return parser.parse_args()


def build_prompt(question: str) -> list[dict[str, str]]:
    question = question.strip()
    return [
        {
            "role": "user",
            "content": f"{question}\n\n{FINAL_ANSWER_INSTRUCTION}",
        }
    ]


def coerce_messages(value: Any) -> list[dict[str, Any]]:
    if value is None:
        raise ValueError("Expected a chat message list, but got None.")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise TypeError(f"Expected a list of messages, got {type(value).__name__}.")
    return value


def extract_question_from_dapo(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()

    source_prompt = coerce_messages(row.get("source_prompt"))
    content = source_prompt[0]["content"].strip()

    patterns = [
        r"^Solve the following math problem step by step\.\s*",
        r'The last line of your response should be of the form Answer: \$Answer \(without quotes\) where \$Answer is the answer to the problem\.\s*',
        r'Remember to put your answer on its own line after "Answer:"\.\s*$',
    ]
    for pattern in patterns:
        content = re.sub(pattern, "", content, flags=re.DOTALL)
    return content.strip()


def extract_question_from_aime24(row: dict[str, Any]) -> str:
    prompt = coerce_messages(row.get("prompt"))
    content = prompt[0]["content"].strip()
    content = re.sub(
        r"\s*Let's think step by step and output the final answer within \\boxed\{\}\.\s*$",
        "",
        content,
        flags=re.DOTALL,
    )
    return content.strip()


def normalize_rows(
    records: list[dict[str, Any]],
    split: str,
    question_extractor,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in records:
        item = deepcopy(row)
        question = question_extractor(item)
        item["prompt"] = build_prompt(question)

        extra_info = dict(item.get("extra_info") or {})
        extra_info["split"] = split
        extra_info["prompt_format"] = "answer_boxed"
        item["extra_info"] = extra_info

        normalized.append(item)
    return normalized


def write_dataset(records: list[dict[str, Any]], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(records)
    dataset.to_parquet(str(output))


def main() -> None:
    args = parse_args()

    train_df = pd.read_parquet(os.path.expanduser(args.train_input))

    if args.test_ratio is not None:
        # Split mode: hold out a fraction of the DAPO-English train set as the test set.
        # Deduplicate by the extracted question text first so the held-out set
        # contains no question that also appears in training (some raw DAPO prompts
        # differ only in formatting/instructions but normalize to the same question).
        n_before = len(train_df)
        train_df = train_df.assign(_qkey=train_df.apply(lambda r: extract_question_from_dapo(r), axis=1))
        train_df = train_df.drop_duplicates(subset="_qkey").drop(columns="_qkey")
        if len(train_df) < n_before:
            print(f"Removed {n_before - len(train_df)} duplicate prompts before splitting")

        test_df = train_df.sample(frac=args.test_ratio, random_state=args.seed)
        train_df = train_df.drop(test_df.index)
        test_extractor = extract_question_from_dapo
        print(
            f"Split train set with ratio={args.test_ratio}, seed={args.seed}: "
            f"{len(train_df)} train / {len(test_df)} test"
        )
    else:
        # Legacy mode: use an external test file (e.g. AIME24).
        test_df = pd.read_parquet(os.path.expanduser(args.test_input))
        test_extractor = extract_question_from_aime24

    train_rows = normalize_rows(train_df.to_dict(orient="records"), split="train", question_extractor=extract_question_from_dapo)
    test_rows = normalize_rows(test_df.to_dict(orient="records"), split="test", question_extractor=test_extractor)

    write_dataset(train_rows, os.path.expanduser(args.train_output))
    write_dataset(test_rows, os.path.expanduser(args.test_output))

    print(f"Wrote {len(train_rows)} train rows to {args.train_output}")
    print(f"Wrote {len(test_rows)} test rows to {args.test_output}")
    print("Example prompt:")
    print(train_rows[0]["prompt"][0]["content"])


if __name__ == "__main__":
    main()
