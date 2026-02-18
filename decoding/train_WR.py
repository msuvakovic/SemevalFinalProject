import os
import numpy as np
import json
import argparse

import config
from utils_stim import get_story_wordseqs
from utils_resp import get_resp
from utils_ridge.DataSequence import DataSequence
from utils_ridge.util import make_delayed
from utils_ridge.ridge import bootstrap_ridge
from splits import load_split
np.random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--sessions", nargs = "+", type = int, 
        default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--split-file", type = str, default = None,
        help = "JSON with train/val/test. If not set, use encoding model's train list (run train_EM first).")
    args = parser.parse_args()

    # Use same training stories as encoding model (so we don't use val/test for word-rate fit)
    stories = None
    if not args.split_file:
        em_path = os.path.join(config.MODEL_DIR, args.subject, "encoding_model_perceived.npz")
        if not os.path.exists(em_path):
            em_dir = os.path.join(config.MODEL_DIR, args.subject)
            ems = [f for f in os.listdir(em_dir) if f.startswith("encoding_model")] if os.path.exists(em_dir) else []
            if ems:
                em_path = os.path.join(em_dir, ems[0])
        if os.path.exists(em_path):
            em_data = np.load(em_path, allow_pickle=True)
            if "stories" in em_data:
                stories = list(em_data["stories"])
    if stories is None:
        train_stories, _, _ = load_split(args.subject, getattr(config, "SPLIT_FILE", None))
        stories = train_stories
    if not stories:
        raise ValueError(f"No training stories for subject {args.subject}. Run train_EM.py first or pass --split-file.")
    print(f"Word-rate model: {len(stories)} training stories (same as encoding model)")

    # ROI voxels
    # Use voxels from encoding model if ROIs not available
    roi_path = os.path.join(config.DATA_TRAIN_DIR, "ROIs", "%s.json" % args.subject)
    if os.path.exists(roi_path):
        with open(roi_path, "r") as f:
            vox = json.load(f)
    else:
        print(f"ROIs not found at {roi_path}. Using top voxels from encoding model.")
        em_path = os.path.join(config.MODEL_DIR, args.subject, "encoding_model_perceived.npz") # Assuming default GPT
        if not os.path.exists(em_path):
             # Try to find any encoding model
             em_dir = os.path.join(config.MODEL_DIR, args.subject)
             ems = [f for f in os.listdir(em_dir) if f.startswith("encoding_model")]
             if ems:
                 em_path = os.path.join(em_dir, ems[0])
             else:
                 raise FileNotFoundError(f"No encoding model found for subject {args.subject}. Run train_EM.py first.")
        
        em_data = np.load(em_path)
        good_voxels = em_data["voxels"]
        # Use top 1000 voxels to avoid massive computation time in SVD
        # If we use all 10000, input dim is 10000 * 4 delays = 40000. 
        # SVD of (25000, 40000) is impossible.
        # We need to reduce dimensionality.
        
        # Take top 500 for now.
        if len(good_voxels) > 500:
            good_voxels = good_voxels[-500:] # argsort gives ascending, so end is best
            
        vox = {"speech": good_voxels, "auditory": good_voxels}

    # estimate word rate model
    save_location = os.path.join(config.MODEL_DIR, args.subject)
    os.makedirs(save_location, exist_ok = True)
    
    wordseqs = get_story_wordseqs(stories)
    rates = {}
    for story in stories:
        ds = wordseqs[story]
        words = DataSequence(np.ones(len(ds.data_times)), ds.split_inds, ds.data_times, ds.tr_times)
        rates[story] = words.chunksums("lanczos", window = 3)
    nz_rate = np.concatenate([rates[story][5+config.TRIM:-config.TRIM] for story in stories], axis = 0)
    nz_rate = np.nan_to_num(nz_rate).reshape([-1, 1])
    mean_rate = np.mean(nz_rate)
    rate = nz_rate - mean_rate
    
    for roi in ["speech", "auditory"]:
        resp = get_resp(args.subject, stories, stack = True, vox = vox[roi])
        # Average voxels to create a single coherent timecourse
        # This makes regression instantaneous (N=1 instead of N=10000)
        resp = resp.mean(axis=1, keepdims=True)
        
        delresp = make_delayed(resp, config.RESP_DELAYS)
        nchunks = int(np.ceil(delresp.shape[0] / 5 / config.CHUNKLEN))    
        weights, _, _ = bootstrap_ridge(delresp, rate, use_corr = False,
            alphas = config.ALPHAS, nboots = config.NBOOTS, chunklen = config.CHUNKLEN, nchunks = nchunks)
        np.savez(os.path.join(save_location, "word_rate_model_%s" % roi), 
            weights = weights, mean_rate = mean_rate, voxels = vox[roi])