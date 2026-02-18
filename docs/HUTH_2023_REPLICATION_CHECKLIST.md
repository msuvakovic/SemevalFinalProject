# Huth et al. 2023 (Nature Neuroscience) — Replication Checklist

**Paper:** Tang, LeBel, Jain & Huth. *Semantic reconstruction of continuous language from non-invasive brain recordings.* Nature Neuroscience 26, 858–866 (2023).  
**DOI:** 10.1038/s41593-023-01304-9  
**Code (paper):** https://github.com/HuthLab/semantic-decoding  

This checklist maps the **exact protocol** from the paper to our repo so you can replicate their experiment step by step.

---

## 1. Data (paper vs ours)

| Paper | Our repo |
|-------|----------|
| **Training:** 16 h narrative (82 stories, 5–15 min), The Moth / Modern Love; 15 story sessions after anat+localizers | `data/Huth/derivative/preprocessed_data/<subject>/*.hf5` + TextGrids; split train/val/test (splits or ratio) |
| **Test perceived:** “Where There’s Smoke” (Jenifer Hixson, Moth), held out | Put story in test split; use `--task wheretheressmoke` (or exact filename) |
| **Test imagined:** 5 × 1‑min Modern Love segments, cued twice each in 14‑min scan; refs = subjects told stories outside scanner | `test_response/<subject>/imagined_speech/` or preprocessed; refs from `test_stimulus/imagined_speech/` |
| **fMRI:** TR = 2 s; motion correction (FLIRT); Savitzky–Golay detrend (120 s); **trim 20 s (10 vols) start/end**; z‑score per voxel | Preprocessing may be pre-done in .hf5. Our **TRIM = 5** (10 s) in `decoding/config.py`; paper uses **10 volumes (20 s)** — set `TRIM = 10` for strict replication if your data include the buffer |

**Action:** Ensure “Where There’s Smoke” (and any other test stories) are in **test** split, not train. Use `decoding/splits.py` or a split file.

---

## 2. Encoding model (train_EM)

| Paper | Our code |
|-------|----------|
| **Features:** GPT (GPT‑1); for each word \(s_i\) use context \((s_{i-5},...,s_i)\) → **layer 9** 768‑dim embedding | `decoding/config.py`: `GPT_LAYER = 9`, `GPT_WORDS = 5` ✓ |
| **Stimulus → TRs:** Resample word features to fMRI acquisition times with **three‑lobe Lanczos** | `decoding/utils_stim.py`: `lanczosinterp2D` ✓ |
| **FIR:** 4 delays: **t−1, t−2, t−3, t−4** (2, 4, 6, 8 s before); concatenate → 768×4 = 3,072 features | `decoding/config.py`: `STIM_DELAYS = [1,2,3,4]` ✓; `utils_stim.get_stim` → `make_delayed` ✓ |
| **Regression:** L2 ridge; z‑score features; 50 bootstraps × 10 regularization coefficients (log 10–1,000); pick best α per voxel | `decoding/train_EM.py` + `utils_ridge.ridge.bootstrap_ridge`; `config.ALPHAS = np.logspace(1,3,10)` ✓; `NBOOTS = 5` in config (paper 50) — **increase to 50 for replication** |
| **Voxels:** 10,000 best (by cross‑val) used for decoding | `config.VOXELS = 10000` ✓ |
| **Noise covariance:** Bootstrap leave‑one‑story‑out residuals | In `train_EM.py` (noise_model from held‑out stories) ✓ |

**Commands (per subject):**

```bash
cd /home/dmarhoef/brain-model-alignment/decoding
# One encoding model per subject (trained on perceived-speech data)
python train_EM.py --subject UTS01
```

**Imagined speech:** The paper uses the **same** encoding model (trained on perceived speech) for both perceived and imagined decoding. You do **not** need to run `train_EM.py --gpt imagined`. `run_decoder.py` now loads `encoding_model_perceived.npz` for `--experiment imagined_speech` when `encoding_model_imagined.npz` is missing. If you did train `--gpt imagined`, it’s the same procedure and data—only the saved filename differs.

**Optional (strict replication):** In `decoding/config.py` set `NBOOTS = 50`, `TRIM = 10` (if your scans have 20 s buffer).

**Encoder match to paper:** Our encoder is set up to match Huth: GPT‑1 (openai-gpt), layer 9, 5 context words, Lanczos resampling to TRs, FIR delays [1,2,3,4], L2 ridge, 10k voxels, bootstrap voxel selection, leave-one-story noise covariance. Only `NBOOTS` (we use 5, paper 50) and `TRIM` (we use 5 vols, paper 10 vols) differ unless you change them.

---

## 3. Word rate model (train_WR)

| Paper | Our code |
|-------|----------|
| **Perceived:** Predict word rate from **auditory cortex** | `word_rate_voxels = "auditory"` for perceived ✓ |
| **Imagined / movie:** Broca + sPMv (speech) | `word_rate_voxels = "speech"` for imagined_speech ✓ |
| **FIR:** 4 delays **t+1, t+2, t+3, t+4** (responses 2, 4, 6, 8 s **later** → predict rate at t) | `config.RESP_DELAYS = [-4,-3,-2,-1]` ✓ (negative = future response) |
| **Word times:** Evenly divide TR interval by predicted (rounded) word counts | `utils_stim.predict_word_times` ✓ |

**Command:**

```bash
python train_WR.py --subject UTS01
```

Uses `encoding_model_perceived.npz` and ROI (e.g. auditory) to fit word rate; outputs `word_rate_model_auditory.npz` and `word_rate_model_speech.npz`.

---

## 4. Decoder (run_decoder)

| Paper | Our code |
|-------|----------|
| **Beam width** k = 200 | `config.WIDTH = 200` ✓ |
| **LM:** GPT fine‑tuned on Reddit + 240 Moth/Modern Love (not in train/test); context 8 s (≈ 100 tokens); **nucleus** (p, r) | We use `openai-gpt`; `config.LM_TIME = 8`, `LM_MASS = 0.9`, `LM_RATIO = 0.1` ✓ |
| **Decoder vocab:** 6,867 words (≥2 in encoding training) | Our run_decoder may use full GPT vocab or a subset; check `LanguageModel` / decoder vocab source |
| **At each word time:** LM proposes continuations; encoding model scores P(R|C); keep k best; max **5** extensions per candidate, scaled by LM quintile | `config.EXTENSIONS = 5` ✓ |
| **Output:** Single best sequence (highest encoding likelihood) | Decoder saves `words` and `times` ✓ |

**Commands:**

```bash
# Perceived speech — test story “Where There’s Smoke”
python run_decoder.py --subject UTS01 --task wheretheressmoke --experiment perceived_speech

# Imagined speech (need encoding_model_imagined.npz)
python run_decoder.py --subject UTS01 --task <story> --experiment imagined_speech
```

Results: `results/<subject>/<experiment>/<task>.npz` (words, times).

---

## 5. Evaluation (evaluate_predictions)

| Paper | Our code |
|-------|----------|
| **Metrics:** WER, BLEU‑1, METEOR, BERTScore (recall, IDF from training) | `decoding/evaluate_predictions.py` ✓ |
| **Windows:** 20 s around each second; story similarity = mean over windows | `config.WINDOW = 20` ✓ |
| **Null:** **200** LM‑only sequences (same word times, random likelihoods); one‑sided test; FDR | We support `--null N`; paper uses **200** — run with `--null 200` for replication |
| **Reference:** Transcript for perceived; for imagined, subjects’ spoken refs | `utils_eval.load_transcript(experiment, task)`; refs in `test_stimulus/<experiment>/` |

**Commands:**

```bash
# Perceived (Where There’s Smoke)
python evaluate_predictions.py --subject UTS01 --experiment perceived_speech --task wheretheressmoke --null 200

# Imagined
python evaluate_predictions.py --subject UTS01 --experiment imagined_speech --task <story> --null 200
```

**Note:** Our script currently expects `data_lm/` for null generation (vocab + LM). If missing, use `--null 0` and add nulls later, or set up `data_lm/` per HuthLab/semantic-decoding.

---

## 6. Paper numbers to match (Table 1, main text)

- **Perceived speech** (one test story, ~1,839 words):  
  WER ~0.92–0.94, BLEU‑1 ~0.23–0.25, METEOR ~0.16–0.17, BERTScore ~0.81.  
  Null (floor): WER 0.9637, BLEU 0.19, METEOR 0.13, BERT 0.79.  
  Report **raw** scores and **z‑scores** vs null (Fig. 1d).
- **Imagined speech:** 100% story ID; decoder predictions significantly more similar to correct ref than chance (Fig. 3a,b).

---

## 7. Order of operations (replication workflow)

1. **Data:** Place Huth (or compatible) data in `data/Huth/derivative/` (preprocessed_data, test_response, test_stimulus, TextGrids as needed). Ensure test stories (e.g. “Where There’s Smoke”) are in **test** split.
2. **Config tweaks (optional but recommended for strict replication):**  
   `decoding/config.py`: `NBOOTS = 50`, `TRIM = 10` (if your data have 20 s buffer).
3. **Encoding model:**  
   `train_EM.py --subject <id>` (perceived); `train_EM.py --subject <id> --gpt imagined` (for imagined_speech).
4. **Word rate model:**  
   `train_WR.py --subject <id>`.
5. **Decode:**  
   `run_decoder.py --subject <id> --task wheretheressmoke --experiment perceived_speech`  
   (and analogous for imagined_speech with your test story names).
6. **Evaluate:**  
   `evaluate_predictions.py --subject <id> --experiment perceived_speech --task wheretheressmoke --null 200`  
   (and for imagined_speech).
7. **Compare:** Raw WER/BLEU/METEOR/BERT and z‑scores vs null to Table 1 and Fig. 1d.

---

## 8. Gaps / notes

- **data_lm:** Null generation needs `data_lm/` (vocab + LM). Clone or mirror HuthLab/semantic-decoding’s `data_lm` setup, or run with `--null 0` and add nulls offline.
- **TRIM:** Paper 20 s (10 vol); we have 5 (10 s). Set `TRIM = 10` in decoding config for strict match if your preprocessing matches the paper.
- **NBOOTS:** Paper 50; we have 5. Set `NBOOTS = 50` for encoding model replication.
- **ROI for word rate:** Paper uses auditory (perceived) and Broca+sPMv (imagined). Our code uses config/ROI files; ensure ROIs match (e.g. `decoding` ROI selection or provided masks).

Once these are aligned, running the steps above replicates the Huth et al. 2023 experiment in our pipeline; then you can swap in FMRIFlamingo for the scoring step and compare fairly.
