import os
import numpy as np
import json
import argparse
import h5py
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", **kwargs):
        return iterable
import torch
import config
from GPT import     GPT
from Decoder import Decoder, Hypothesis
from LanguageModel import LanguageModel
from EncodingModel import EncodingModel
from StimulusModel import StimulusModel, get_lanczos_mat, affected_trs, LMFeatures
from utils_stim import predict_word_rate, predict_word_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--experiment", type = str, required = True)
    parser.add_argument("--task", type = str, required = True)
    parser.add_argument("--ignore-overlap", action="store_true", help="Allow decoding on training stories")
    args = parser.parse_args()
    
    # Encoding model: paper uses ONE model (trained on perceived speech) for both
    # perceived and imagined decoding. So we always load encoding_model_perceived.npz
    # unless you explicitly trained a separate "imagined" encoder (train_EM --gpt imagined).
    if args.experiment in ["imagined_speech"]:
        load_gpt = "imagined"  # try imagined first (if you trained it)
        load_location = os.path.join(config.MODEL_DIR, args.subject)
        if not os.path.exists(os.path.join(load_location, "encoding_model_imagined.npz")):
            load_gpt = "perceived"  # fallback: use perceived encoder (matches paper)
        gpt_checkpoint = load_gpt
    else:
        gpt_checkpoint = "perceived"

    # determine word rate model voxels based on experiment
    if args.experiment in ["imagined_speech", "perceived_movies"]: word_rate_voxels = "speech"
    else: word_rate_voxels = "auditory"

    # load responses
    # Try multiple locations for the response file
    resp_path = None
    possible_paths = [
        os.path.join(config.DATA_TEST_DIR, "preprocessed_data", args.subject, args.task + ".hf5"),
        os.path.join(config.DATA_TEST_DIR, "test_response", args.subject, args.experiment, args.task + ".hf5"),
    ]
    
    for p in possible_paths:
        if os.path.exists(p):
            resp_path = p
            break
            
    if resp_path is None:
        raise FileNotFoundError(f"Could not find response file for {args.subject}/{args.task} in {config.DATA_TEST_DIR}")
        
    hf = h5py.File(resp_path, "r")
    resp = np.nan_to_num(hf["data"][:])
    hf.close()
    
    # load gpt
    # with open(os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "vocab.json"), "r") as f:
    #     gpt_vocab = json.load(f)
    # with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
    #     decoder_vocab = json.load(f)
    gpt = GPT(path = "openai-gpt", vocab = None, device = config.GPT_DEVICE)
    gpt_vocab = gpt.vocab
    decoder_vocab = gpt.vocab # Use full vocab
    
    features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)
    lm = LanguageModel(gpt, decoder_vocab, nuc_mass = config.LM_MASS, nuc_ratio = config.LM_RATIO)

    # load models
    load_location = os.path.join(config.MODEL_DIR, args.subject)
    word_rate_model = np.load(os.path.join(load_location, "word_rate_model_%s.npz" % word_rate_voxels), allow_pickle = True)
    encoding_model = np.load(os.path.join(load_location, "encoding_model_%s.npz" % gpt_checkpoint), allow_pickle=True)
    weights = encoding_model["weights"]
    noise_model = encoding_model["noise_model"]
    tr_stats = encoding_model["tr_stats"]
    word_stats = encoding_model["word_stats"]
    em = EncodingModel(resp, weights, encoding_model["voxels"], noise_model, device = config.EM_DEVICE)
    em.set_shrinkage(config.NM_ALPHA)
    if not args.ignore_overlap:
        assert args.task not in encoding_model["stories"], f"Task {args.task} was in training set! Use --ignore-overlap to proceed anyway."
    
    # predict word times
    word_rate = predict_word_rate(resp, word_rate_model["weights"], word_rate_model["voxels"], word_rate_model["mean_rate"])
    print(f"Word rate: min={word_rate.min()}, max={word_rate.max()}, mean={word_rate.mean():.2f}, total words={word_rate.sum()}")
    if args.experiment == "perceived_speech": word_times, tr_times = predict_word_times(word_rate, resp, starttime = -10)
    else: word_times, tr_times = predict_word_times(word_rate, resp, starttime = 0)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)

    # decode responses
    decoder = Decoder(word_times, config.WIDTH)
    sm = StimulusModel(lanczos_mat, tr_stats, word_stats[0], device = config.SM_DEVICE)
    for sample_index in tqdm(range(len(word_times)), desc="Decoding"):
        trs = affected_trs(decoder.first_difference(), sample_index, lanczos_mat)
        ncontext = decoder.time_window(sample_index, config.LM_TIME, floor = 5)
        beam_nucs = lm.beam_propose(decoder.beam, ncontext)
        # for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
        #     nuc, logprobs = beam_nucs[c]
        #     if len(nuc) < 1: continue
        #     extend_words = [hyp.words + [x] for x in nuc]
        #     extend_embs = list(features.extend(extend_words))
        #     stim = sm.make_variants(sample_index, hyp.embs, extend_embs, trs)
        #     # ldata/Huth/derivative/preprocessed_data/UTS03/naked.hf5ikelihoods = em.prs(stim, trs)
        #     likelihoods = em.prs(stim, trs) 
        #     local_extensions = [Hypothesis(parent = hyp, extension = x) for x in zip(nuc, logprobs, extend_embs)]
        #     decoder.add_extensions(local_extensions, likelihoods, nextensions)

        # --- BATCHED INNER LOOP (replace lines 103-112) ---

        # 1. CPU: collect all extend_words across all hypotheses, track boundaries
        all_extend_words = []
        hyp_slices = []   # (start, end) index into all_extend_words per hypothesis
        valid_hyps = []   # (hyp, nextensions, nuc, logprobs)

        for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_nucs[c]
            if len(nuc) < 1:
                continue
            start = len(all_extend_words)
            all_extend_words.extend([hyp.words + [x] for x in nuc])
            hyp_slices.append((start, len(all_extend_words)))
            valid_hyps.append((hyp, nextensions, nuc, logprobs))

        if not valid_hyps:
            decoder.extend(verbose=False)
            continue

        # 2. GPU: ONE features.extend call for all hypotheses combined
        all_embs = list(features.extend(all_extend_words))

        # 3. GPU: make_variants still per-hypothesis (each has unique history)
        #    but batch em.prs across all hypotheses in one shot
        all_stims = []
        stim_meta = []  # track (hyp, nextensions, nuc, logprobs, emb slice)

        for (hyp, nextensions, nuc, logprobs), (s, e) in zip(valid_hyps, hyp_slices):
            extend_embs = all_embs[s:e]
            stim = sm.make_variants(sample_index, hyp.embs, extend_embs, trs)
            all_stims.append(stim)
            stim_meta.append((hyp, nextensions, nuc, logprobs, extend_embs))

        # 4. GPU: ONE em.prs call for everything (all hypotheses × all nucleus words)
        all_likelihoods = em.prs(torch.cat(all_stims, dim=0), trs)

        # 5. CPU: distribute likelihoods back and add extensions
        offset = 0
        for (hyp, nextensions, nuc, logprobs, extend_embs) in stim_meta:
            n = len(nuc)
            likelihoods = all_likelihoods[offset:offset + n]
            offset += n
            local_extensions = [Hypothesis(parent=hyp, extension=x) 
                                for x in zip(nuc, logprobs, extend_embs)]
            decoder.add_extensions(local_extensions, likelihoods, nextensions)

        decoder.extend(verbose=False)
        
    if args.experiment in ["perceived_movie", "perceived_multispeaker"]: decoder.word_times += 10
    save_location = os.path.join(config.RESULT_DIR, args.subject, args.experiment)
    os.makedirs(save_location, exist_ok = True)
    decoder.save(os.path.join(save_location, args.task))