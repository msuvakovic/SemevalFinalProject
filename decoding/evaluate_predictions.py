import os
import numpy as np
import json
import argparse

import config
from utils_eval import generate_null, load_transcript, windows, segment_data, WER, BLEU, METEOR, BERTSCORE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--experiment", type = str, required = True)
    parser.add_argument("--task", type = str, required = True)
    parser.add_argument("--metrics", nargs = "+", type = str, default = ["WER", "BLEU", "METEOR", "BERT"])
    parser.add_argument("--references", nargs = "+", type = str, default = [])
    parser.add_argument("--null", type = int, default = 10)
    parser.add_argument("--pred-file", type = str, default = None,
                        help = "Path to a custom prediction .npz (overrides default results/ lookup)")
    args = parser.parse_args()
    
    if len(args.references) == 0:
        args.references.append(args.task)
        
    eval_segments_path = os.path.join(config.DATA_TEST_DIR, "eval_segments.json")
    if os.path.exists(eval_segments_path):
        with open(eval_segments_path, "r") as f:
            eval_segments = json.load(f)
    else:
        eval_segments = {}

    # If task not in eval_segments, derive one segment from reference transcript (story-level only)
    if args.task not in eval_segments:
        ref_data = load_transcript(args.experiment, args.task)
        ref_times = ref_data["times"]
        if len(ref_times) == 0:
            raise ValueError(f"Reference transcript for {args.task} has no word times; cannot derive segment.")
        seg_start = int(np.floor(ref_times.min()))
        seg_end = int(np.ceil(ref_times.max()))
        if seg_end <= seg_start:
            seg_end = seg_start + 1
        eval_segments[args.task] = [seg_start, seg_end]
        print(f"Task {args.task!r} not in eval_segments.json; using derived segment [{seg_start}, {seg_end}] s (story-level only).")

    # load language similarity metrics
    metrics = {}
    if "WER" in args.metrics: metrics["WER"] = WER(use_score = False)
    if "BLEU" in args.metrics: metrics["BLEU"] = BLEU(n = 1)
    if "METEOR" in args.metrics: metrics["METEOR"] = METEOR()
    if "BERT" in args.metrics:
        try:
            metrics["BERT"] = BERTSCORE(
                idf_sents = np.load(os.path.join(config.DATA_TEST_DIR, "idf_segments.npy")),
                rescale = False,
                score = "recall")
        except ImportError:
            print("Warning: Skipping BERT (install bert-score for this metric).")
            args.metrics = [m for m in args.metrics if m != "BERT"]

    # load prediction transcript
    if args.pred_file:
        pred_path = args.pred_file
    else:
        pred_path = os.path.join(config.RESULT_DIR, args.subject, args.experiment, args.task + ".npz")
    pred_data = np.load(pred_path)
    pred_words, pred_times = pred_data["words"], pred_data["times"]

    # generate null sequences (skip if --null 0; requires data_lm/ when > 0)
    if args.null > 0:
        try:
            if args.experiment in ["imagined_speech"]: gpt_checkpoint = "imagined"
            else: gpt_checkpoint = "perceived"
            null_word_list = generate_null(pred_times, gpt_checkpoint, args.null)
        except FileNotFoundError as e:
            print(f"Warning: data_lm/ not found ({e}). Skipping null baselines (no z-scores). Use --null 0 to silence.")
            null_word_list = []
    else:
        null_word_list = []
        
    window_scores, window_zscores = {}, {}
    story_scores, story_zscores = {}, {}
    for reference in args.references:

        # load reference transcript
        ref_data = load_transcript(args.experiment, reference)
        ref_words, ref_times = ref_data["words"], ref_data["times"]

        # segment prediction and reference words into windows
        seg_start, seg_end = eval_segments[args.task]
        window_cutoffs = windows(seg_start, seg_end, config.WINDOW)
        if len(window_cutoffs) == 0:
            # story shorter than WINDOW; use single segment (story-level only)
            window_cutoffs = [(seg_start, seg_end)]
        ref_windows = segment_data(ref_words, ref_times, window_cutoffs)
        pred_windows = segment_data(pred_words, pred_times, window_cutoffs)
        null_window_list = [segment_data(null_words, pred_times, window_cutoffs) for null_words in null_word_list]

        for mname, metric in metrics.items():

            window_scores[(reference, mname)] = metric.score(ref = ref_windows, pred = pred_windows)
            story_scores[(reference, mname)] = metric.score(ref = ref_windows, pred = pred_windows)

            if len(null_window_list) > 0:
                window_null_scores = np.array([metric.score(ref = ref_windows, pred = null_windows)
                                               for null_windows in null_window_list])
                story_null_scores = window_null_scores.mean(1)
                window_zscores[(reference, mname)] = (window_scores[(reference, mname)]
                                                      - window_null_scores.mean(0)) / (window_null_scores.std(0) + 1e-10)
                story_zscores[(reference, mname)] = (story_scores[(reference, mname)].mean()
                                                      - story_null_scores.mean()) / (story_null_scores.std() + 1e-10)
            else:
                window_zscores[(reference, mname)] = window_scores[(reference, mname)]
                story_zscores[(reference, mname)] = np.array(story_scores[(reference, mname)].mean())
    
    # Print story-level scores (reference = task)
    print("Story-level scores:")
    for (ref, mname), vals in story_scores.items():
        if ref == args.task:
            raw = np.array(vals).mean()
            if (ref, mname) in story_zscores and len(null_word_list) > 0:
                z = story_zscores[(ref, mname)]
                print(f"  {mname}: {raw:.4f}  (z={float(z):.2f})")
            else:
                print(f"  {mname}: {raw:.4f}")

    save_location = os.path.join(config.REPO_DIR, "scores", args.subject, args.experiment)
    os.makedirs(save_location, exist_ok = True)
    np.savez(os.path.join(save_location, args.task), 
             window_scores = window_scores, window_zscores = window_zscores, 
             story_scores = story_scores, story_zscores = story_zscores)