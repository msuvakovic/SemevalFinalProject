# utils_bio.py
"""
Utility functions for BIO tagging + wordpiece alignment + class weights.

This implements exactly what corrected_train_one_span_bio.py expects:
- to_bio_from_binary
- align_bio_to_wordpieces
- build_bio_label_maps
- compute_class_weights_for_ce
"""

from __future__ import annotations
from typing import List, Dict, Sequence

from collections import Counter

import numpy as np
from transformers import PreTrainedTokenizerBase


# ----------------------------------------------------
# 1. Binary -> BIO at word level
# ----------------------------------------------------
def to_bio_from_binary(
    words: Sequence[str],
    binary_tags: Sequence[str],
    role: str,
) -> List[str]:
    """
    Convert binary tags ["O" or role] into BIO tags:
    - "O" stays "O"
    - first token of a contiguous span with tag=role -> "B-role"
    - subsequent tokens in the span -> "I-role"
    """
    if len(words) != len(binary_tags):
        raise ValueError(
            f"Length mismatch words({len(words)}) vs tags({len(binary_tags)})"
        )

    bio_tags: List[str] = []
    prev_is_role = False
    for tag in binary_tags:
        if tag == role:
            if not prev_is_role:
                bio_tags.append(f"B-{role}")
            else:
                bio_tags.append(f"I-{role}")
            prev_is_role = True
        else:
            bio_tags.append("O")
            prev_is_role = False
    return bio_tags


# ----------------------------------------------------
# 2. Build label maps for BIO tags
# ----------------------------------------------------
def build_bio_label_maps(role: str):
    """
    Define label space for the given role.

    E.g. role="target" -> ["O", "B-target", "I-target"]
    """
    labels = ["O", f"B-{role}", f"I-{role}"]
    label2id: Dict[str, int] = {lab: i for i, lab in enumerate(labels)}
    id2label: Dict[int, str] = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


# ----------------------------------------------------
# 3. Align BIO labels to wordpieces
# ----------------------------------------------------
def align_bio_to_wordpieces(
    words: Sequence[str],
    bio_tags: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    label2id: Dict[str, int],
    label_all_tokens: bool = True,
):
    """
    Tokenize a list of words with a fast tokenizer (is_split_into_words=True)
    and project word-level BIO tags to wordpiece-level IDs.

    - Special tokens get label -100
    - If label_all_tokens=False, only the first wordpiece for each word keeps the label;
      others get -100.
    - If label_all_tokens=True, all wordpieces for a word get the same label id.
    """
    if len(words) != len(bio_tags):
        raise ValueError(
            f"Length mismatch words({len(words)}) vs BIO tags({len(bio_tags)})"
        )

    encoding = tokenizer(
        list(words),
        is_split_into_words=True,
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
    )

    word_ids = encoding.word_ids()  # list[Optional[int]]
    labels_wp: List[int] = []

    prev_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            # special token (CLS, SEP, padding, etc.)
            labels_wp.append(-100)
            continue

        # word-level tag
        tag_str = bio_tags[word_id]

        if not label_all_tokens:
            # only label the first subword, ignore the rest
            if word_id != prev_word_id:
                labels_wp.append(label2id[tag_str])
            else:
                labels_wp.append(-100)
        else:
            # all subwords get the word's label
            labels_wp.append(label2id[tag_str])

        prev_word_id = word_id

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels_wp,
        "_word_ids": word_ids,
    }


# ----------------------------------------------------
# 4. Class weights for CrossEntropy
# ----------------------------------------------------
def compute_class_weights_for_ce(
    flat_labels: Sequence[int],
    num_labels: int,
) -> List[float]:
    """
    Compute per-class weights for CrossEntropy from a flat list of label ids,
    ignoring -100 if present.

    Simple inverse-frequency scheme:
        weight_c = total_count / (num_labels * count_c)

    This makes rare classes get larger weights.
    """
    # Drop ignore index if present
    filtered = [int(x) for x in flat_labels if x != -100]

    if len(filtered) == 0:
        # degenerate, just return ones
        return [1.0] * num_labels

    counts = Counter(filtered)
    total = len(filtered)

    weights = []
    for label_id in range(num_labels):
        c = counts.get(label_id, 0)
        if c == 0:
            # if truly unseen, give it a modest weight instead of inf
            weights.append(0.0)
        else:
            w = total / (num_labels * float(c))
            weights.append(w)

    # normalize so mean weight is ~1.0 (optional, but nice)
    w_arr = np.array(weights, dtype=float)
    mean_w = w_arr[w_arr > 0].mean() if (w_arr > 0).any() else 1.0
    if mean_w > 0:
        w_arr = w_arr / mean_w

    return w_arr.tolist()
