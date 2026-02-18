# Fundamental Checklist: Why Might Loss Not Go Down?

If tuning LR, dropout, and architecture doesn't help, something more fundamental may be wrong. Work through this list.

---

## 0. **How published fMRI→language decoders actually work (and why we may be thinking about it differently)**

### Huth 2023 (Nature Neuroscience) – encoding model + beam search

They **do not** train a single “fMRI encoder → LLM” like we do.

1. **Encoding model (forward):** Predict **BOLD from text**.  
   - Input: GPT-derived contextual features of the stimulus, aligned to TRs (with FIR delays, e.g. 1–4 TRs).  
   - Model: **Ridge regression** per voxel (or similar); voxels often selected by bootstrap correlation.  
   - Output: predicted BOLD for each voxel/TR.  
   So the “encoder” is **text → predicted fMRI**, not fMRI → text.

2. **Decoding (inversion):** At test time, **score candidate word sequences** by how well the encoding model predicts the **observed** BOLD from that candidate.  
   - Beam search: LM proposes candidates; encoding model gives **P(observed BOLD | candidate text)** (or a similarity score).  
   - Decoder picks the sequence that best “explains” the brain data under the encoding model.

So the brain is used as a **scoring signal**: “Which text, when fed through the encoding model, gives predicted BOLD closest to what we actually measured?”  
No end-to-end gradient from language loss through an fMRI encoder; the only thing trained on BOLD is the **encoding** (text→BOLD) model.

### Other published approaches

- **Ridge / linear:** Many older and some current pipelines use **ridge regression** to predict BOLD from semantic features (e.g. word embeddings, GPT hidden states), then decode by scoring candidates with that forward model. Same idea as Huth: forward model + search.
- **MindLLM (2025):** Uses a dedicated **fMRI encoder** (subject-agnostic, with attention over brain regions) whose output is fed into an LLM – i.e. **fMRI → encoder → LLM**, similar to our pipeline. So end-to-end fMRI→text is used in recent work, but often with strong inductive biases (e.g. subject/region handling, pretrained language representations).
- **Foundation-model style:** Some work uses contrastive **image–fMRI** or **text–fMRI** pretraining, then fine-tunes for decoding.

### What we do vs what works in Huth

| Aspect | Huth 2023 | Our FMRIFlamingo / ORPO |
|--------|-----------|--------------------------|
| Direction | **Text → BOLD** (encoding model) | **BOLD → text** (encoder + LLM) |
| Training objective | Fit BOLD from GPT features (ridge) | CE / ORPO on language given fMRI |
| Decoding | Beam search; score = encoding likelihood | Single forward: P(word \| fMRI) |
| Gradient | Only in encoding model (BOLD prediction) | End-to-end through encoder + perceiver + LLM |

So we are **inverting the problem**: we learn fMRI→text directly, while Huth learns text→fMRI and inverts at decode time. Their way has a simple, well-posed objective (predict BOLD); ours requires the gradient to pull a useful fMRI representation through a large LLM, which can be much harder.

**If loss stays at ~0.69 and accuracy at ~50%:** It’s plausible we’re hitting a **fundamental** limit of the “direct fMRI→LLM” setup (e.g. signal too weak, or optimization landscape too hard), and that a **two-stage, Huth-style** pipeline could be more reliable:  
(1) Train an **encoding model** (e.g. ridge: GPT features → BOLD) on your data.  
(2) Decode by **beam search** over candidates, scoring each with the encoding model (how well does this candidate explain the observed BOLD?).  
That would match the published recipe that actually decodes continuous language from non-invasive fMRI.

---

## 1. **Cross-attention gate starts at zero**

In Open Flamingo, the gated cross-attention uses `attn_gate = nn.Parameter(torch.tensor([0.0]))`, so **tanh(0) = 0**: the vision path is **completely off at init**. The model must learn to "open" the gate before fMRI can affect predictions. If updates are too small or the gradient to the gate is weak, the model may never use the brain signal.

**Fix:** Set `CROSS_ATTN_GATE_INIT` in config (e.g. `0.5`) so the gate starts partly open (see config and `fmri_flamingo.py`). Then re-run.

---

## 2. **Temporal alignment (fMRI ↔ word)**

BOLD peaks ~4–6 s after stimulus. The dataset uses `HEMODYNAMIC_DELAY` (default 4 s): for each TR we take `stimulus_time = tr_time - delay` and align words to that. So we associate **BOLD in window [t, t+window]** with **words heard at (t - delay) to (t+window - delay)**. The target word is the last word in that aligned set.

- **Check:** Is the target word the one that *elicited* the BOLD in this window? If your delay is wrong for this dataset (e.g. 2 s or 6 s), try `HEMODYNAMIC_DELAY` in `config.py` (e.g. 3.0, 5.0) or use `HEMODYNAMIC_DELAY_RANGE` if you add multi-delay support.
- **Sanity:** Plot a few (window, target_word) pairs and confirm they match the story transcript and timing.

---

## 3. **Voxel selection**

With `ROI_SELECTION_METHOD = "all_voxels"` you use every voxel (~50k+). That can dilute language-relevant signal with noise from non-language regions. Many decoding setups use **voxels that respond to language** (e.g. temporal cortex, TPJ) or a **semantic atlas** (Huth lab) to restrict to high semantic-tuning voxels.

**Fix:** Add subject- or atlas-based voxel selection (e.g. top-K by response to language) so the encoder sees a smaller, more informative set of voxels. Not wired yet; see config `ROI_SELECTION_METHOD` and `fmri_tokenizer.py` for `"anatomical"` / semantic-atlas TODOs.

---

## 4. **Perceiver bottleneck**

Encoder output (e.g. 50k tokens or 200 ROIs) is compressed to **num_latents** (default ~64 in open_flamingo) per `<image>`. If 64 slots can't carry enough information, the LLM simply doesn't get the signal.

**Fix:** Increase perceiver `num_latents` (e.g. 128 or 256) by passing it when constructing the perceiver (requires a small code change to open_flamingo or your wrapper). See `ARCHITECTURE.md` §3b and config comments.

---

## 5. **Gradients not reaching the fMRI path**

If the encoder or perceiver get zero (or tiny) gradients, they don't update and LR won't help.

**Fix:** Set `LOG_GRAD_NORMS_EVERY = 100` in config and run a short training. Check the logged **encoder** and **perceiver** grad norms. If they stay ~0, the vision path is effectively dead (e.g. gate stuck at 0, or wrong params in the optimizer).

---

## 6. **No decodable signal in the data**

If BOLD in your windows doesn't actually correlate with the target words (e.g. wrong alignment, wrong preprocessing, or task too hard), no model will learn.

**Sanity check:** Run a simple baseline, e.g. ridge regression from fMRI (averaged over the window) to word embeddings or to a one-hot of vocabulary. If this baseline is at or near chance, fix data/alignment first before investing in the full Flamingo pipeline.

---

## 7. **Objective / evaluation**

ORPO "accuracy" = fraction of (chosen vs rejected) where the model prefers the chosen word. Random = ~50% (or 1/(1+num_rejects) for strict ranking). If accuracy stays near chance for many steps, either the model isn’t using fMRI (check 1, 4, 5) or the signal isn’t there (check 2, 3, 6).

---

## Quick wins to try first

1. **Gate init:** Set `CROSS_ATTN_GATE_INIT = 0.5` in config so the vision path is used from step 1.
2. **Grad norms:** Set `LOG_GRAD_NORMS_EVERY = 100` and confirm encoder/perceiver norms are non-zero.
3. **Hemodynamic delay:** If you have a known delay for this dataset, set `HEMODYNAMIC_DELAY` accordingly; or try 3.0 and 5.0 in separate runs.
