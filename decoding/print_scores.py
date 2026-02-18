#!/usr/bin/env python3
"""Print a readable summary of scores from evaluate_predictions.py output."""
import os
import sys
import numpy as np

# Allow running from decoding/ or repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
import config

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="UTS01")
    p.add_argument("--experiment", default="perceived_speech")
    p.add_argument("--task", default="treasureisland")
    args = p.parse_args()

    path = os.path.join(config.REPO_DIR, "scores", args.subject, args.experiment, args.task + ".npz")
    if not os.path.exists(path):
        path = os.path.join("scores", args.subject, args.experiment, args.task + ".npz")
    if not os.path.exists(path):
        print(f"Not found: {path}")
        return 1

    d = np.load(path, allow_pickle=True)
    story_scores = d["story_scores"].item()  # dict (ref, metric) -> array

    print(f"Scores: {args.subject} / {args.experiment} / {args.task}")
    print("-" * 40)
    for (ref, metric), arr in sorted(story_scores.items()):
        if ref == args.task:
            val = np.mean(arr)
            # WER is stored as score = 1 - WER in our code
            if metric == "WER":
                print(f"  {metric}: {val:.4f}  (score = 1 - WER; so WER = {1 - val:.4f})")
            else:
                print(f"  {metric}: {val:.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
