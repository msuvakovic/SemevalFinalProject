"""
Train/val/test split for Huth-style decoding.

Huth use separate data dirs (train_response vs test_response); we have one dir
so we need an explicit split. Options:
  - Split file (JSON): lists train/val/test story names per subject or global.
  - Ratio-based: reproducible split by ratio (e.g. 0.8/0.1/0.1) with fixed seed.

Decoding must not use training stories (no memorization); val is for tuning, test for reporting.
"""

import os
import json
import numpy as np

import config

# Default ratios if no split file (reproducible with SPLIT_SEED)
SPLIT_TRAIN_RATIO = 0.8
SPLIT_VAL_RATIO = 0.1
SPLIT_TEST_RATIO = 0.1
SPLIT_SEED = 42


def _get_all_stories(subject):
    """Return sorted list of story names for subject (from .hf5 in preprocessed_data)."""
    subject_dir = os.path.join(config.DATA_TRAIN_DIR, "preprocessed_data", subject)
    if not os.path.exists(subject_dir):
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
    stories = []
    for f in os.listdir(subject_dir):
        if f.endswith(".hf5"):
            stories.append(f.replace(".hf5", ""))
    return sorted(stories)


def load_split(subject, split_file=None):
    """
    Return (train_list, val_list, test_list) for subject.

    If split_file is provided and exists:
      - If JSON has top-level key subject -> use that entry's train/val/test.
      - Else if JSON has top-level train/val/test -> use for all subjects.
    Else: build split by ratio (SPLIT_*_RATIO, SPLIT_SEED) over all stories.
    """
    all_stories = _get_all_stories(subject)
    if not all_stories:
        return [], [], []

    path = split_file or getattr(config, "SPLIT_FILE", None)
    if path and os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        if subject in data:
            s = data[subject]
        elif "train" in data and "test" in data:
            s = data
        else:
            raise ValueError(f"Split file {path}: expected key {subject!r} or top-level 'train'/'test'. Keys: {list(data.keys())}")
        train = list(s.get("train", []))
        val = list(s.get("val", []))
        test = list(s.get("test", []))
        # Validate: every story in file should be in all_stories; no duplicate across splits
        known = set(train) | set(val) | set(test)
        for story in known:
            if story not in all_stories:
                raise ValueError(f"Split file lists story {story!r} but subject {subject} has no {story}.hf5")
        return train, val, test

    # Ratio-based split (reproducible)
    rng = np.random.default_rng(SPLIT_SEED)
    n = len(all_stories)
    idx = np.arange(n)
    rng.shuffle(idx)
    t = int(n * SPLIT_TRAIN_RATIO)
    v = int(n * SPLIT_VAL_RATIO)
    # test gets the rest
    train = [all_stories[i] for i in idx[:t]]
    val = [all_stories[i] for i in idx[t : t + v]]
    test = [all_stories[i] for i in idx[t + v :]]
    return train, val, test


def save_split(split_file, subject, train, val, test, merge_with_existing=True):
    """
    Write split to JSON. If merge_with_existing and file exists, merge this subject
    into existing dict; else overwrite with single-subject dict.
    """
    entry = {"train": train, "val": val, "test": test}
    if merge_with_existing and os.path.exists(split_file):
        with open(split_file, "r") as f:
            data = json.load(f)
        data[subject] = entry
    else:
        data = {subject: entry}
    os.makedirs(os.path.dirname(split_file) or ".", exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(data, f, indent=2)
