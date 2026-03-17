import os
import numpy as np
import json
import argparse
from tqdm import tqdm

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim
from utils_resp import get_resp
from utils_ridge.ridge import ridge, bootstrap_ridge
from splits import load_split, save_split
np.random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--gpt", type = str, default = "perceived")
    parser.add_argument("--sessions", nargs = "+", type = int, 
        default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--split-file", type = str, default = None,
        help = "JSON with train/val/test per subject. If not set, use ratio-based split (splits.py).")
    parser.add_argument("--save-split", type = str, default = None,
        help = "If set, write computed split to this path (for reuse with --split-file).")
    args = parser.parse_args()

    # train/val/test split: train encoding model only on training stories
    train_stories, val_stories, test_stories = load_split(args.subject, args.split_file)
    if not train_stories:
        raise ValueError(f"No training stories for subject {args.subject}. Check --split-file or data.")
    print(f"Split: {len(train_stories)} train, {len(val_stories)} val, {len(test_stories)} test")
    if args.save_split:
        save_split(args.save_split, args.subject, train_stories, val_stories, test_stories)
    stories = train_stories  # use only train for fitting

    # load gpt
    gpt = GPT(path = "openai-gpt", vocab = None, device = config.GPT_DEVICE)
    features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)
    
    # estimate encoding model
    print("Estimating encoding model...")
    rstim, tr_stats, word_stats = get_stim(stories, features)
    rresp = get_resp(args.subject, stories, stack = True)
    print(f"rstim shape: {rstim.shape}, rresp shape: {rresp.shape}")
    print(f"rresp mean: {rresp.mean():.4f}, std: {rresp.std():.4f}, per-voxel std range: {rresp.std(0).min():.3f}-{rresp.std(0).max():.3f}")

    
    print(f"Stimulus shape: {rstim.shape}, Response shape: {rresp.shape}")
    
    nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))
    weights, alphas, bscorrs = bootstrap_ridge(rstim, rresp, use_corr = True, alphas = config.ALPHAS,
        nboots = config.NBOOTS, chunklen = config.CHUNKLEN, nchunks = nchunks)        
    bscorrs = bscorrs.mean(2).max(0)
    vox = np.sort(np.argsort(bscorrs)[-config.VOXELS:])
    del rstim, rresp
    import torch; torch.cuda.empty_cache()

    # estimate noise model
    stim_dict = {story : get_stim([story], features, tr_stats = tr_stats) for story in stories}
    resp_dict = get_resp(args.subject, stories, stack = False, vox = vox)
    noise_model = np.zeros([len(vox), len(vox)])
    for hstory in tqdm(stories, desc="Noise model"):
        tstim, hstim = np.vstack([stim_dict[tstory] for tstory in stories if tstory != hstory]), stim_dict[hstory]
        tresp, hresp = np.vstack([resp_dict[tstory] for tstory in stories if tstory != hstory]), resp_dict[hstory]
        bs_weights = ridge(tstim, tresp, alphas[vox])
        resids = hresp - hstim.dot(bs_weights)
        bs_noise_model = resids.T.dot(resids)
        noise_model += bs_noise_model / np.diag(bs_noise_model).mean() / len(stories)
    del stim_dict, resp_dict
    
    # save (stories = training stories only; run_decoder asserts task not in stories)
    save_location = os.path.join(config.MODEL_DIR, args.subject)
    os.makedirs(save_location, exist_ok = True)
    np.savez(os.path.join(save_location, "encoding_model_%s" % args.gpt),
        weights = weights, noise_model = noise_model, alphas = alphas, voxels = vox, stories = stories,
        test_stories = np.array(test_stories), val_stories = np.array(val_stories),
        tr_stats = np.array(tr_stats), word_stats = np.array(word_stats),
        bscorrs = bscorrs)