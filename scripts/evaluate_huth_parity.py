#!/usr/bin/env python3
"""
Evaluate a custom prediction (.npz with words/times) using Huth's metrics and null baselines.

This mirrors decoding/evaluate_predictions.py but allows a custom pred_path so we don't
overwrite Huth baseline results. It loads transcripts based on the specified experiment.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
FMRI_FLAMINGO_DIR = SCRIPT_DIR.parent
if str(FMRI_FLAMINGO_DIR) not in sys.path:
    sys.path.insert(0, str(FMRI_FLAMINGO_DIR))

# Import Huth evaluation utils (from decoding/)
DECODING_DIR = FMRI_FLAMINGO_DIR / "decoding"
if str(DECODING_DIR) not in sys.path:
    sys.path.insert(0, str(DECODING_DIR))

import config as dec_config
from utils_eval import (
    generate_null,
    load_transcript,
    windows,
    segment_data,
    WER,
    BLEU,
    METEOR,
    BERTSCORE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name for transcript lookup (e.g. imagined_speech)")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--pred-path", type=Path, required=True, help="Path to .npz with words/times")
    parser.add_argument("--metrics", nargs="+", type=str, default=["WER", "BLEU", "METEOR", "BERT"])
    parser.add_argument("--null", type=int, default=10)
    parser.add_argument("--score-suffix", type=str, default="fmri_flamingo", help="Suffix for score filename")
    args = parser.parse_args()

    # Load evaluation segments
    eval_segments_path = os.path.join(dec_config.DATA_TEST_DIR, "eval_segments.json")
    with open(eval_segments_path, "r") as f:
        eval_segments = json.load(f)

    # Load metrics
    metrics = {}
    if "WER" in args.metrics:
        metrics["WER"] = WER(use_score=True)
    if "BLEU" in args.metrics:
        metrics["BLEU"] = BLEU(n=1)
    if "METEOR" in args.metrics:
        metrics["METEOR"] = METEOR()
    if "BERT" in args.metrics:
        try:
            metrics["BERT"] = BERTSCORE(
                idf_sents=np.load(os.path.join(dec_config.DATA_TEST_DIR, "idf_segments.npy")),
                rescale=False,
                score="recall",
            )
        except ImportError:
            print("Warning: Skipping BERT (install bert-score for this metric).")
            args.metrics = [m for m in args.metrics if m != "BERT"]

    # Load predictions
    if not args.pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {args.pred_path}")
    pred_data = np.load(args.pred_path)
    pred_words, pred_times = pred_data["words"], pred_data["times"]

    # Null sequences (LM-only)
    if args.null > 0:
        gpt_checkpoint = "imagined" if args.experiment in ["imagined_speech"] else "perceived"
        null_word_list = generate_null(pred_times, gpt_checkpoint, args.null)
    else:
        null_word_list = []

    window_scores, window_zscores = {}, {}
    story_scores, story_zscores = {}, {}

    # Load reference transcript
    ref_data = load_transcript(args.experiment, args.task)
    ref_words, ref_times = ref_data["words"], ref_data["times"]

    # Segment into windows (Huth protocol)
    window_cutoffs = windows(*eval_segments[args.task], dec_config.WINDOW)
    ref_windows = segment_data(ref_words, ref_times, window_cutoffs)
    pred_windows = segment_data(pred_words, pred_times, window_cutoffs)
    null_window_list = [segment_data(null_words, pred_times, window_cutoffs) for null_words in null_word_list]

    for mname, metric in metrics.items():
        window_scores[(args.task, mname)] = metric.score(ref=ref_windows, pred=pred_windows)
        story_scores[(args.task, mname)] = metric.score(ref=ref_windows, pred=pred_windows)

        if len(null_window_list) > 0:
            window_null_scores = np.array([
                metric.score(ref=ref_windows, pred=null_windows) for null_windows in null_window_list
            ])
            story_null_scores = window_null_scores.mean(1)
            window_zscores[(args.task, mname)] = (
                window_scores[(args.task, mname)] - window_null_scores.mean(0)
            ) / (window_null_scores.std(0) + 1e-10)
            story_zscores[(args.task, mname)] = (
                story_scores[(args.task, mname)].mean() - story_null_scores.mean()
            ) / (story_null_scores.std() + 1e-10)
        else:
            window_zscores[(args.task, mname)] = window_scores[(args.task, mname)]
            story_zscores[(args.task, mname)] = np.array(story_scores[(args.task, mname)].mean())

    # Save scores (avoid overwriting baseline)
    save_location = os.path.join(dec_config.REPO_DIR, "scores", args.subject, args.experiment)
    os.makedirs(save_location, exist_ok=True)
    score_name = f"{args.task}__{args.score_suffix}"
    np.savez(
        os.path.join(save_location, score_name),
        window_scores=window_scores,
        window_zscores=window_zscores,
        story_scores=story_scores,
        story_zscores=story_zscores,
    )

    print(f"✅ Saved scores to {os.path.join(save_location, score_name)}.npz")
    print("Story-level scores:")
    for (ref, metric), score in story_scores.items():
        if isinstance(score, (np.ndarray, list)):
            score = np.mean(score)
        z = story_zscores[(ref, metric)]
        if isinstance(z, (np.ndarray, list)):
            z = np.mean(z)
        print(f"  {metric}: {score:.4f} (z={z:.3f})")


if __name__ == "__main__":
    main()
