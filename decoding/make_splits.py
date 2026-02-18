#!/usr/bin/env python3
"""
Generate a train/val/test split file for Huth-style decoding.

Usage:
  cd decoding && python make_splits.py --out data/Huth/derivative/splits.json
  cd decoding && python make_splits.py --out splits.json --test-stories life adollshouse

If --test-stories (and optionally --val-stories) are given, those are fixed;
the rest are train (and remaining val). Otherwise use ratio-based split (see splits.py).
"""

import os
import json
import argparse
import numpy as np

import config
from splits import SPLIT_SEED, SPLIT_TRAIN_RATIO, SPLIT_VAL_RATIO, SPLIT_TEST_RATIO


def _all_stories(subject):
    subject_dir = os.path.join(config.DATA_TRAIN_DIR, "preprocessed_data", subject)
    if not os.path.exists(subject_dir):
        return []
    stories = [f.replace(".hf5", "") for f in os.listdir(subject_dir) if f.endswith(".hf5")]
    return sorted(stories)


def _subjects():
    preproc = os.path.join(config.DATA_TRAIN_DIR, "preprocessed_data")
    if not os.path.exists(preproc):
        return []
    return sorted(d for d in os.listdir(preproc) if os.path.isdir(os.path.join(preproc, d)))


def main():
    parser = argparse.ArgumentParser(description="Write train/val/test splits to JSON.")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path (e.g. data/Huth/derivative/splits.json)")
    parser.add_argument("--test-stories", nargs="*", default=None,
        help="Fixed test story names; rest are train/val by ratio or --val-stories.")
    parser.add_argument("--val-stories", nargs="*", default=None,
        help="Fixed val story names (only used with --test-stories). Rest = train.")
    parser.add_argument("--train-ratio", type=float, default=SPLIT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=SPLIT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=SPLIT_TEST_RATIO)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    subjects = _subjects()
    if not subjects:
        print("No subjects found under", os.path.join(config.DATA_TRAIN_DIR, "preprocessed_data"))
        return

    out = {}
    for subject in subjects:
        all_s = _all_stories(subject)
        if not all_s:
            continue
        if args.test_stories is not None:
            test_set = set(args.test_stories)
            val_set = set(args.val_stories or [])
            train_set = [s for s in all_s if s not in test_set and s not in val_set]
            val_list = [s for s in all_s if s in val_set]
            test_list = [s for s in all_s if s in test_set]
            # if --val-stories not given, split train into train/val by ratio
            if not args.val_stories and train_set:
                rng = np.random.default_rng(args.seed)
                n = len(train_set)
                idx = np.arange(n)
                rng.shuffle(idx)
                nval = max(0, int(n * args.val_ratio))
                val_list = [train_set[i] for i in idx[:nval]]
                train_list = [train_set[i] for i in idx[nval:]]
            else:
                train_list = train_set
        else:
            rng = np.random.default_rng(args.seed)
            idx = np.arange(len(all_s))
            rng.shuffle(idx)
            t = int(len(all_s) * args.train_ratio)
            v = int(len(all_s) * args.val_ratio)
            train_list = [all_s[i] for i in idx[:t]]
            val_list = [all_s[i] for i in idx[t : t + v]]
            test_list = [all_s[i] for i in idx[t + v :]]
        out[subject] = {"train": train_list, "val": val_list, "test": test_list}
        print(f"{subject}: {len(train_list)} train, {len(val_list)} val, {len(test_list)} test")

    out_path = args.out if os.path.isabs(args.out) else os.path.join(config.REPO_DIR, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
