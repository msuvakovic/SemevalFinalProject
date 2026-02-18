# Huth 2023 (imagined_speech) SOTA Audit vs FMRIFlamingo

Scope: **imagined_speech**; metrics **WER, BLEU-1, METEOR, BERTScore** and Huth-style **null baselines / z-scores**.

This audit documents what we currently do, what Huth did and measured, how we will compare, and the highest‑risk gaps/bugs that can explain 0% Telepathy performance or non-comparability.

---

## 1) Our exact tasks today (FMRIFlamingo pipeline)

### Training task
- **Data construction & alignment**: Each training sample is a fixed TR window with **context words + target word** derived from TextGrids, aligned to TRs with a **fixed hemodynamic delay** and a **±1s matching window**.
  - Windowing: `DEFAULT_WINDOW_SIZE`, `DEFAULT_STRIDE` (note dataset default stride is 5 unless overridden).
  - Prompt format: `"{pre_prompt} <image> {post_prompt} {answer} <|endofchunk|>"` with `post_prompt = "Predict the next word:"`.
  - File: [src/datasets/huth_fmri_dataset.py](/home/dmarhoef/brain-model-alignment/src/datasets/huth_fmri_dataset.py)

- **Objective**: CE loss + optional ranking loss (contrastive).
  - CE loss uses labels that **mask the prompt** and padding.
  - Ranking loss: **score correct sequence lower loss than distractors**, optionally **telepathy** (BOS-only prompt).
  - Text masking: randomly masks tokens to force fMRI usage.
  - Files: [scripts/train.py](/home/dmarhoef/brain-model-alignment/scripts/train.py), [src/models/fmri_flamingo.py](/home/dmarhoef/brain-model-alignment/src/models/fmri_flamingo.py), [config.py](/home/dmarhoef/brain-model-alignment/config.py)

### Evaluation tasks currently in repo
- **Ranking (telepathy)**: 1‑in‑100 candidate ranking using in‑story targets; BOS-only prompt; measures Top‑1/5/10.
  - File: [scripts/evaluate_ranking_telepathy.py](/home/dmarhoef/brain-model-alignment/scripts/evaluate_ranking_telepathy.py)

- **Constrained generation**: generate text, **take first word only**, and evaluate with Huth metrics via `decoding/evaluate_predictions.py`.
  - Uses **times = tr_end * TR** (not Huth word-time model), and **no null baseline** by default (`--null 0`).
  - File: [scripts/evaluate_constrained.py](/home/dmarhoef/brain-model-alignment/scripts/evaluate_constrained.py)

---

## 2) What Huth did and measured (imagined_speech)

### Encoding model (text → fMRI)
- Extract **GPT word embeddings** and **downsample** to TRs with Lanczos interpolation.
- Apply **FIR delays** (`STIM_DELAYS = [1,2,3,4]` TRs).
- Fit **ridge regression** to predict fMRI from stimulus features; select top voxels by bootstrap correlation.
- File: [decoding/train_EM.py](/home/dmarhoef/brain-model-alignment/decoding/train_EM.py)

### Word timing (from fMRI)
- Separate **word-rate model** predicts word rate from fMRI, then **predict word times** from TR grid.
- Files: [decoding/train_WR.py](/home/dmarhoef/brain-model-alignment/decoding/train_WR.py), [decoding/utils_stim.py](/home/dmarhoef/brain-model-alignment/decoding/utils_stim.py)

### Decoding (beam search)
- **LM proposals** (nucleus sampling) + **encoding-model likelihood** P(fMRI | text).
- Beam search over words using both LM log-prob and encoding likelihood; outputs predicted word sequence and word times.
- File: [decoding/run_decoder.py](/home/dmarhoef/brain-model-alignment/decoding/run_decoder.py)

### Metrics / evaluation
Huth metrics are computed by **`decoding/evaluate_predictions.py`**:
- **WER (as score = 1 − WER)**, **BLEU‑1**, **METEOR**, **BERTScore**.
- **Window-level** and **story-level** metrics; segmentation via `eval_segments.json` and `WINDOW=20` seconds.
- **Null baseline**: LM‑only decoding with random likelihoods; z‑scores computed against null distribution.
- Files: [decoding/evaluate_predictions.py](/home/dmarhoef/brain-model-alignment/decoding/evaluate_predictions.py), [decoding/utils_eval.py](/home/dmarhoef/brain-model-alignment/decoding/utils_eval.py)

---

## 3) How we will compare (apples‑to‑apples)

To compare to Huth SOTA on **imagined_speech**, we must:
1. **Generate a predicted word sequence + word times** using the same segmentation protocol (or explicitly state the deviation).
2. **Evaluate via `decoding/evaluate_predictions.py`** using the same metrics and **null baselines** (z‑scores).
3. **Match test stories / subjects** used in Huth 2023 imagined_speech.

### Comparison protocol (recommended)
- Use **Huth’s evaluation pipeline** for both:
  - Run Huth baseline decoder (ridge + beam) to establish SOTA target in our environment.
  - Generate Flamingo predictions into `.npz` with `words` and `times` arrays.
  - Evaluate both with `decoding/evaluate_predictions.py --null 10` and report **window-level and story-level** z-scores for WER/BLEU/METEOR/BERTScore.

This isolates whether performance gaps are **model** vs **evaluation** differences.

---

## 4) High‑risk bugs / mismatches (likely failure causes)

### A. Objective / evaluation mismatch
- **Training uses CE + ranking on fixed TR windows**, while Huth decodes **word-by-word beam search** with **word-time prediction** from fMRI.  
  This is not the same target. If we compare our outputs to Huth metrics without matching word timing, we are not comparable.

### B. Word timing mismatch
- Huth: word times are **predicted from fMRI** via the word-rate model.  
- Our constrained eval: word times = **`tr_end * TR`** per window.  
  This is a fundamental mismatch that affects window segmentation and metrics.

### C. Prompt construction mismatch
- Training prompt format includes **`post_prompt = "Predict the next word:"`**, which is not part of Huth decoding.
- Telepathy eval uses **BOS-only** prompt, but training uses full prompt + post_prompt.  
  This introduces a prompt distribution shift.

### D. Stride / window inconsistencies
- Config default `DEFAULT_STRIDE = 10` but dataset default `stride = 5` unless overridden.  
  This can change the effective training distribution and evaluation alignment.

### E. Evaluation not using null baseline
- `evaluate_constrained.py` runs Huth evaluation with `--null 0`.  
  Huth uses **null baselines** and reports **z‑scores**; skipping nulls is not comparable.

### F. First‑word extraction via whitespace
- `evaluate_constrained.py` uses `text.split()[0]`, which can be incorrect with subword tokenization or punctuation.  
  This can degrade metrics independent of model quality.

### G. Telepathy ranking ≠ Huth decoding
- Ranking Top‑1/5/10 is a useful ablation, but **not comparable** to Huth’s beam search decoding + WER/BLEU/METEOR/BERTScore.

---

## 5) Improvement checklist (no changes yet)

### Evaluation parity (highest priority)
1. **Use Huth evaluation pipeline with null baselines** for Flamingo outputs.  
2. **Use Huth word times** (word-rate model) or explicitly document deviations.
3. Ensure **same subject/story splits** as Huth imagined_speech (no leakage).

### Data alignment & timing
4. Validate TR alignment + hemodynamic delay assumptions; consider using Huth-style FIR delays or aligning windowing to word times.
5. Resolve stride mismatch (config vs dataset default) for consistent sampling.

### Training objective alignment
6. Add a decoding‑aligned objective (word-by-word scoring) or a two‑stage approach (fMRI → embedding → LM), since current CE+ranking is not aligned with Huth decoding.

### Baselines and sanity checks
7. Run Huth ridge baseline in this repo to establish the SOTA target for imagined_speech on your exact data/config.  
8. If ridge baseline is strong and Flamingo is near chance, the issue is **architecture/training**, not data.

---

## 6) Summary (what’s likely “fundamentally missing”)

The **biggest gap** is that we are **not matching Huth’s decoding protocol**:
1. Huth decodes **word-by-word** with **word times predicted from fMRI**, and scores candidates via **P(fMRI | text)**.  
2. We train a **direct fMRI → text** model on **fixed TR windows**, and evaluate with a mix of ranking and constrained generation without Huth-style timing or null baselines.

Until we **align evaluation and timing**, we cannot compare to Huth SOTA fairly.  
Once evaluation is matched, we can judge whether the **training objective itself** is fundamentally insufficient (likely yes, given 0% telepathy), and then prioritize objective/architecture changes.

