import os
import numpy as np
import json
import difflib

import config
from utils_ridge.stimulus_utils import TRFile, load_textgrids, load_simulated_trfiles
from utils_ridge.dsutils import make_word_ds
from utils_ridge.interpdata import lanczosinterp2D
from utils_ridge.util import make_delayed

def get_story_wordseqs(stories):
    """loads words and word times of stimulus stories
    """
    grids = load_textgrids(stories, config.DATA_TRAIN_DIR)
    with open(os.path.join(config.DATA_TRAIN_DIR, "respdict.json"), "r") as f:
        respdict = json.load(f)
    trfiles = load_simulated_trfiles(respdict)
    wordseqs = make_word_ds(grids, trfiles)
    return wordseqs

def get_stim(stories, features, tr_stats = None):
    """extract quantitative features of stimulus stories
    """
    word_seqs = get_story_wordseqs(stories)
    missing = [story for story in stories if story not in word_seqs]
    if missing:
        available = sorted(word_seqs.keys())
        lines = [
            "Missing story keys in word sequences.",
            f"Requested: {missing}",
        ]
        for story in missing:
            suggestions = difflib.get_close_matches(story, available, n=3)
            if suggestions:
                lines.append(f"Closest available for '{story}': {suggestions}")
        lines.append(
            "Common causes: missing/broken TextGrid files (e.g., git-annex symlinks not fetched), "
            "or story names absent from respdict.json."
        )
        raise KeyError(" ".join(lines))

    if tr_stats is None:
        word_sum = None
        word_sumsq = None
        word_count = 0
        tr_sum = None
        tr_sumsq = None
        tr_count = 0

        for story in stories:
            word_vec = features.make_stim(word_seqs[story].data).astype(np.float32, copy = False)
            if word_sum is None:
                word_sum = word_vec.sum(0, dtype = np.float64)
                word_sumsq = np.square(word_vec, dtype = np.float64).sum(0, dtype = np.float64)
            else:
                word_sum += word_vec.sum(0, dtype = np.float64)
                word_sumsq += np.square(word_vec, dtype = np.float64).sum(0, dtype = np.float64)
            word_count += word_vec.shape[0]

            ds_vec = lanczosinterp2D(
                word_vec, word_seqs[story].data_times, word_seqs[story].tr_times
            ).astype(np.float32, copy = False)
            trimmed = ds_vec[5 + config.TRIM : -config.TRIM]
            if tr_sum is None:
                tr_sum = trimmed.sum(0, dtype = np.float64)
                tr_sumsq = np.square(trimmed, dtype = np.float64).sum(0, dtype = np.float64)
            else:
                tr_sum += trimmed.sum(0, dtype = np.float64)
                tr_sumsq += np.square(trimmed, dtype = np.float64).sum(0, dtype = np.float64)
            tr_count += trimmed.shape[0]

        word_mean = (word_sum / word_count).astype(np.float32)
        word_var = np.maximum(word_sumsq / word_count - np.square(word_mean, dtype = np.float32), 0)
        word_std = np.sqrt(word_var, dtype = np.float32)

        r_mean = (tr_sum / tr_count).astype(np.float32)
        r_var = np.maximum(tr_sumsq / tr_count - np.square(r_mean, dtype = np.float32), 0)
        r_std = np.sqrt(r_var, dtype = np.float32)
        r_std[r_std == 0] = 1
    else:
        r_mean, r_std = tr_stats
        r_mean = np.asarray(r_mean, dtype = np.float32)
        r_std = np.asarray(r_std, dtype = np.float32)

    delayed = []
    for story in stories:
        word_vec = features.make_stim(word_seqs[story].data).astype(np.float32, copy = False)
        ds_vec = lanczosinterp2D(
            word_vec, word_seqs[story].data_times, word_seqs[story].tr_times
        ).astype(np.float32, copy = False)
        trimmed = ds_vec[5 + config.TRIM : -config.TRIM]
        normed = np.nan_to_num((trimmed - r_mean) / r_std, copy = False)
        delayed.append(make_delayed(normed, config.STIM_DELAYS).astype(np.float32, copy = False))

    del_mat = np.vstack(delayed)
    if tr_stats is None:
        return del_mat, (r_mean, r_std), (word_mean, word_std)
    return del_mat

def predict_word_rate(resp, wt, vox, mean_rate):
    """predict word rate at each acquisition time
    """
    # Check if we need to average voxels (compatibility with fast training)
    # wt shape is (n_features * n_delays, 1)
    # If wt is small (e.g. 4) but we have many voxels (e.g. 500), we trained on the average.
    
    selected_resp = resp[:, vox]
    n_delays = len(config.RESP_DELAYS)
    expected_features = wt.shape[0] // n_delays
    
    if selected_resp.shape[1] > 1 and expected_features == 1:
        # Model expects 1 feature but we have many -> Average them
        selected_resp = selected_resp.mean(axis=1, keepdims=True)
        
    delresp = make_delayed(selected_resp, config.RESP_DELAYS)
    rate = ((delresp.dot(wt) + mean_rate)).reshape(-1).clip(min = 0)
    return np.round(rate).astype(int)

def predict_word_times(word_rate, resp, starttime = 0, tr = 2):
    """predict evenly spaced word times from word rate
    """
    half = tr / 2
    trf = TRFile(None, tr)
    trf.soundstarttime = starttime
    trf.simulate(resp.shape[0])
    tr_times = trf.get_reltriggertimes() + half

    word_times = []
    for mid, num in zip(tr_times, word_rate):  
        if num < 1: continue
        word_times.extend(np.linspace(mid - half, mid + half, num, endpoint = False) + half / num)
    return np.array(word_times), tr_times
