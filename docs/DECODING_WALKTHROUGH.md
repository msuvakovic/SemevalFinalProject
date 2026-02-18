# Decoding Walkthrough: Huth vs FMRIFlamingo, and What to Change

**Subjects:** We do per-subject training and decoding; **subject 9 (UTS09) is left out**. Use UTS01–UTS08 for decode/eval (see `config.TRAIN_SUBJECTS`).

## 1. What did Huth do? (Ridge + beam search)

The in-repo **decoding/** pipeline (Huth-style) does **not** learn "fMRI → text" directly.

- **Encoding model** (`train_EM.py`): **Text → fMRI**
  - Stimulus = text features (e.g. GPT embeddings of words).
  - Response = fMRI (voxel activity).
  - **Ridge regression**: learn weights so `predicted_fMRI = text_features @ W`.
  - So they model **P(fMRI | text)** (linear, simple).

- **Decoding** (`run_decoder.py`): **fMRI → text by search**
  - Given *observed* fMRI, they **score candidate texts** by how well the encoding model predicts that fMRI:
    - For each candidate continuation: `stim = text_features(candidate)`, then `likelihood = P(observed_fMRI | stim)` (using the ridge prediction and a noise model).
  - They combine with a **language model** prior and do **beam search** over text.
  - So **decoding = invert the encoding model by search**: no learned "fMRI → text" net; they use the learned "text → fMRI" and ask "which text best explains this fMRI?"

**Summary**: Huth = linear **encoding** (text → fMRI) + **decoding by search** (which text maximizes P(fMRI|text) × P(text)). Decoding is combinatorial search, not a forward fMRI→text map.

---

## 2. Can we get decoding with our model?

**Yes.** Our setup *is* decoding.

- We have a **direct** map: **fMRI → encoder → cross-attention → LLM**.
- At eval we do: fMRI → model → **scores for 100 candidate sequences** (by loss or log-prob) → pick argmax = decoded "next word."
- So we *are* decoding (fMRI → choice among candidates). The problem is **accuracy is 0%**: the model does not put the correct word first (or in top-5/top-10).

So the **capability** is there; the **training** is not yet producing a model that uses fMRI in a way that matches the right answer. Possible causes:

- Encoder / cross-attention not producing a representation the LLM uses.
- Training objective (CE + ranking) not aligned with 1-in-100 ranking.
- Not enough data, or hemodynamic/alignment issues.
- Telepathy (no text prompt) is very hard; the LLM was never pretrained on "brain only."

---

## 3. Possible improvements and why

### A. Sanity check: does the *data* support decoding?

- **Ridge-style baseline on the same task**
  - Train a **linear** (or shallow) map: fMRI → something useful (e.g. next-word embedding from frozen LLM, or logits for a small vocab).
  - Evaluate with the **same** 1-in-100 Telepathy ranking (same test set, same candidates).
  - If ridge gets **> 0%** and we get 0%, the signal is there and the bottleneck is our architecture/training. If ridge is also 0%, the issue may be data/alignment/task design.

### B. Use Huth-style decoding as a baseline

- Train the in-repo **encoding model** (text → fMRI) on the same data.
- At test time: given fMRI, **score the same 100 candidates** using P(fMRI | candidate) from the encoding model (like `run_decoder.py`), optionally × LM prior.
- If this baseline gets > chance on the same 1-in-100 eval, we know the data support decoding; then we can try to match or beat it with FMRIFlamingo.

### C. Easier task first (with prompt)

- Eval **with** text prompt: "given context + fMRI, rank the next word."
- If we get **> chance with prompt** but 0% Telepathy (no prompt), the model *can* use fMRI when combined with text; we can then try reducing the prompt or making it noisier to move toward Telepathy.

### D. Two-stage: fMRI → embedding, then LLM

- **Stage 1**: fMRI → predicted **text embedding** (e.g. next-word embedding from frozen LLM). Train with MSE or contrastive loss (e.g. correct embedding closer than distractors).
- **Stage 2**: Feed that embedding into the LLM (e.g. as a soft prompt or extra context) and generate/rank.
- This matches the idea "fMRI in **embedding space**" so the LLM sees something it already understands, instead of raw cross-attention from a modality it was never pretrained on.

### E. Train encoder to match LLM internal state

- Instead of CE on next token, train the encoder so that its **output** (after cross-attention) matches the LLM’s **internal representation** of the correct next word (e.g. last-layer hidden state or embedding of the correct token).
- Decoding = inject that predicted state and let the LLM generate. Closer to "fMRI → same space as text."

### F. Training and data

- **More contrastive**: `RANKING_LOSS_WEIGHT` and `RANKING_LOSS_FRAC` up (e.g. 0.7 and 1.0) so every batch has a strong "correct > distractors" signal.
- **Force fMRI usage**: higher `TEXT_MASKING_PROB` so the model can’t rely on text alone.
- **Alignment**: hemodynamic delay, voxel/ROI selection, subject-specific tuning—if our fMRI window or voxels are misaligned with "next word," the model never sees a clean signal.
- **Curriculum**: train first with prompt (easier), then gradually reduce prompt or add Telepathy.

---

## 4. Why we don't align like Huth (yet)

**Huth's alignment** (in `decoding/`):

| Step | What Huth does |
|------|----------------|
| Stimulus → TRs | **Lanczos interpolation**: word-level GPT features are resampled from word times onto the TR grid (smooth, band-limited). |
| Hemodynamic model | **FIR delays**: `STIM_DELAYS = [1, 2, 3, 4]` TRs. Stimulus at TR grid is concatenated with 4 delayed copies (1–4 TRs back). So each TR's fMRI is predicted from stimulus at t−1, t−2, t−3, t−4 TRs. |
| Word times at decode | **Word-rate model**: fMRI → word rate → predicted word times. Decoding uses these times + **Lanczos matrix** to map words ↔ TRs when scoring candidates. |
| Response side | `RESP_DELAYS = [-4, -3, -2, -1]` for word-rate model (fMRI lags). |

**Our alignment** (FMRIFlamingo dataset):

| Step | What we do |
|------|------------|
| Stimulus → TRs | **Single delay + window**: `stimulus_time = tr_time - HEMODYNAMIC_DELAY` (4 s), then a **±1 s window** to decide which words are "active" for that TR. No Lanczos. |
| Hemodynamic model | **One scalar delay** (4 s). No multi-TR FIR. |
| Training windows | **Fixed TR windows** (e.g. 10 TRs) with stride. We do not use Huth's word-rate-predicted word times for building training windows. |

So we use a **simplified** alignment: one global delay and a ±1 s word window, instead of Lanczos + 4-tap FIR. That was a design choice (simpler, fewer dependencies); it can **misalign** our training signal with the BOLD response shape Huth's pipeline is built for.

**To align like Huth we would:**

1. **Stimulus**: In our dataset or a preprocessing step, use **Lanczos interpolation** (e.g. `decoding/utils_ridge/interpdata.lanczosinterp2D`) to resample word features from word times to TR times, then apply **STIM_DELAYS** (concatenate 1–4 TR delayed copies) before building the fMRI→text training pairs.
2. **Training windows**: Optionally derive window boundaries from **word-rate-predicted word times** (as in Huth decoding) instead of fixed stride over TRs.
3. **Decoding**: We already use Huth's word times and Lanczos in `run_decoder_fmri_flamingo.py` when we call their decoder; the gap is mainly in **training** alignment.

Until we change the training pipeline to use Lanczos + FIR (or at least multi-delay), we are not fully aligning like Huth; that could be one reason decoding quality lags.

---

## 5. Summary

| Question | Answer |
|----------|--------|
| What did Huth do? | Linear **encoding** (text → fMRI). **Decoding** = score candidate text by P(fMRI\|text), beam search. No learned fMRI→text net. |
| Can we get decoding with our model? | **Yes.** We already decode (fMRI → scores over 100 options → argmax). The issue is 0% accuracy, not lack of a decoding path. |
| What to change? | (1) Sanity-check with ridge on same 1-in-100 task. (2) Huth-style encoding + score same 100 candidates as baseline. (3) Eval with prompt to see if fMRI helps when text is present. (4) Consider two-stage (fMRI → embedding → LLM) or representation-matching. (5) Stronger ranking + masking + alignment. |

The next concrete step is **run a ridge (or encoding-model) baseline on the same Telepathy 1-in-100 eval**; that tells you whether the bottleneck is "data/signal" vs "architecture/training."

---

## 6. Running Huth's experiment with our model

**Goal**: Same protocol as Huth (word-level beam search over text, same test stories/tasks) but **score candidates with FMRIFlamingo** instead of the ridge encoding model.

- **Huth**: At each step, score extension by **P(fMRI | candidate)** from the encoding model.
- **Us**: At each step, score extension by **P(candidate | fMRI)** from our model (e.g. negative CE loss or log-prob).

### What we need to change

1. **Scoring**
   - **Huth**: `likelihoods = em.prs(stim, trs)` — encoding model likelihood P(fMRI | text).
   - **Us**: For each candidate extension, build one batch: `{ time_series: fmri_segment, input_ids: context + candidate, prompt_len: len(context) }`, then `score = -model.compute_loss(batch)` (higher = more likely). We can batch multiple candidates (same fMRI, different continuations) for speed.

2. **fMRI segment**
   - **Huth**: `trs = affected_trs(decoder.first_difference(), sample_index, lanczos_mat)` gives TR indices for the word(s) being decoded. `resp[trs, :]` is (n_trs, voxels).
   - **Us**: Use the same `trs`; slice `resp[trs, :]` and **transpose to (voxels, n_trs)**. Use the **same voxel selection** as training (e.g. NUM_ROIS from our config, or the voxel indices our HuthFMRIDataset uses). Pad/truncate to a fixed window if needed; our `pad_and_apply_batch` handles variable TR length.

3. **Context + candidate**
   - **Huth**: Hypothesis words + candidate next word; they build text features (embeddings) for the encoding model.
   - **Us**: Build `input_ids` = tokenize(context + " <image> " + candidate) with our tokenizer; `prompt_len` = len(tokenize(context + " <image> ")) so labels mask the prompt. Context = hypothesis.words; candidate = one of the LM-proposed extensions.

4. **Data and voxels**
   - Load **same test fMRI** as Huth (e.g. from decoding config paths or our Huth dataset). Use **same voxel set** as FMRIFlamingo training (our dataset uses NUM_ROIS / ROI selection); if the decoding resp is full-brain, we need to apply the same voxel selection (e.g. from a saved encoding model voxels or our dataset’s ROI list).
   - **Word times**: Use Huth’s word_rate_model + predict_word_times so we have the same word-level alignment; or use our dataset’s alignment if we have word-level segments.

5. **Script**
   - New script (e.g. `scripts/run_decoder_fmri_flamingo.py`) that:
     - Loads FMRIFlamingo from checkpoint, tokenizer, our config.
     - Loads test fMRI and word times (same as `decoding/run_decoder.py` or from our dataset).
     - Reuses `Decoder`, `LanguageModel` (for beam proposals) and the same loop over sample_index.
     - Replaces `em.prs(stim, trs)` with: for each candidate, build batch → `score = -model.compute_loss(batch)` (or batched), pass scores as `likelihoods`.
     - Saves results in the same format as Huth (`decoder.save(...)`) so existing eval scripts work.

6. **Dependencies**
   - Decoding code lives under `decoding/` and uses its own config (paths, GPT, word rate model). We either:
     - Call from project root and add `decoding/` to path, use decoding config for paths/word_times and our config for model/voxels, or
     - Reimplement word-time prediction and beam structure in our script using our dataset/config only.

**Summary**: Replace the encoding-model likelihood with our model’s **P(candidate | fMRI)** (negative loss); keep beam search, word times, and data the same; ensure fMRI segments and voxel selection match our training. See `scripts/run_decoder_fmri_flamingo.py` for a concrete implementation.
