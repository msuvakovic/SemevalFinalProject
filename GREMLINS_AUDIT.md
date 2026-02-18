# Gremlins Audit (No Code Changes)

Audit of the training and evaluation setup based on external review. **No fixes applied**—this document only identifies issues and their severity.

---

## 1. Mixed Precision: BF16 + GradScaler (CONFIRMED – sloppy)

**Location:** `scripts/train.py` ~840–841, 414–422.

**What happens:** We always create `torch.cuda.amp.GradScaler()` when CUDA + mixed precision are on, regardless of dtype. Config uses `MIXED_PRECISION_DTYPE = "bf16"`.

**Why it’s a gremlin:** GradScaler is for FP16 (to avoid underflow). BF16 has enough exponent range that scaling is usually unnecessary. Using GradScaler with BF16 is redundant and mixes old (`torch.cuda.amp`) and new (`torch.amp.autocast`) APIs. It may not crash but is inconsistent and can behave oddly across PyTorch versions.

**Severity:** Low for correctness; medium for clarity and future-proofing.

---

## 2. Scheduler Step Count Drift (CONFIRMED – minor)

**Location:** `scripts/train.py` ~835, 438–451, 485–494.

**What happens:**
- `num_training_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS`
- Scheduler is stepped every optimizer step (every `GRADIENT_ACCUMULATION_STEPS` batches) and again at end of epoch if `num_batches % GRADIENT_ACCUMULATION_STEPS != 0`.

**Why it’s a gremlin:** Integer division can undercount. The tail step can add one extra scheduler step per epoch. So total scheduler steps can be slightly more than `num_training_steps`. Cosine schedule may end a step or two late; warmup length can shift. Small but can make runs not perfectly reproducible across machines/dataloader lengths.

**Severity:** Low.

---

## 3. Text Masking: PAD Substitution + Unchanged attention_mask (FIXED)

**Location:** `scripts/train.py` ~385–418; `src/models/fmri_flamingo.py` ~531–549.

**Original issue:** We replaced masked token positions with `pad_token_id` but did not set `attention_mask` to 0 for those positions, so the model still attended to them.

**Fix applied:** We now clone `attention_mask` and set `masked_attention_mask[rand_mask] = 0` for masked positions, and pass `masked_attention_mask` into `compute_loss`. Masked positions are no longer attended to; labels at those positions are unchanged so the model is trained to predict the masked token from fMRI + context.

---

## 4. Validation Generation Brittleness (CONFIRMED)

**Location:** `scripts/train.py` validate_generation, validate; val_batch_size=1.

**What happens:**
- Assumes batch has `input_ids`, `prompt_len`, `time_series`; uses tokenizer eos/special ids; `max_new_tokens=10`.
- WER/BLEU on 4 samples with short generations are used as “Val Generation” metrics.

**Why it’s a gremlin:** Any refactor of batch keys or model.generate contract can break this silently. `max_new_tokens=10` plus greedy decoding often yields fragments (“The following is a…”) so WER/BLEU are noisy and not a reliable measure of task success. Useful for sanity checks only.

**Severity:** Low for correctness; medium for misinterpretation (don’t treat these numbers as the main quality metric).

---

## 5. Optimizer Param Groups by Name Substrings (CONFIRMED)

**Location:** `scripts/train.py` create_optimizer ~286–300.

**What happens:** Parameter groups are built by substring checks: `"vision_encoder"`, `"tokenizer"`, `"projector"`, `"perceiver"`, `"gated_cross_attn_layer"`, `"lang_encoder"`.

**Why it’s a gremlin:** Renaming or wrapping modules (e.g. `vision_encoder` → `encoder`) will silently reassign LRs. No static guarantee that every parameter lands in exactly one group.

**Severity:** Medium – “works until refactor” trap.

---

## 6. Tokenizer vs Model Embedding Resizing (VERIFIED – OK in our code)

**Location:** `src/models/fmri_flamingo.py` ~247: `lang_encoder.resize_token_embeddings(len(text_tokenizer))`.

**What happens:** FMRIFlamingo adds special tokens to `text_tokenizer` then calls `resize_token_embeddings(len(text_tokenizer))`. So the LLM’s embedding table is resized to match the tokenizer used by the model.

**Verdict:** In this repo, the **model** uses the same tokenizer and resizes; train script and evals also add the same special tokens. So we’re consistent. The only risk is if someone changes tokenizer or special tokens in one place and not the other.

**Severity:** None for current code; worth keeping in mind for future edits.

---

## 7. empty_cache Every 100 Batches (CONFIRMED)

**Location:** `scripts/train.py` ~474–475.

**What happens:** `torch.cuda.empty_cache()` is called every 100 batches when device is CUDA.

**Why it’s a gremlin:** empty_cache() returns cached memory to the driver; it does not reduce peak usage and can add sync points, which can hurt throughput. Better used only after OOM or at epoch boundaries if needed.

**Severity:** Low – might slightly slow training.

---

## 8. Collate / DataLoader (CONFIRMED – design choice)

**Location:** `scripts/train.py` collate_fn ~767–777.

**What happens:** Everything is collated as lists (no stacking for variable-length fields). `compute_loss` / `pad_and_apply_batch` then do padding on the fly.

**Verdict:** Intentional and consistent with the rest of the pipeline. No bug; just a design that can have more CPU overhead than a single stacked batch. Acceptable for research.

**Severity:** None.

---

## 9. local_files_only / trust_remote_code (NOTED)

**Location:** Various; e.g. `AutoTokenizer.from_pretrained(..., local_files_only=False, trust_remote_code=True)`.

**Verdict:** Fine for a lab repo; for stricter reproducibility or security, pin model revisions and consider `local_files_only=True` once assets are cached.

**Severity:** Low / policy.

---

## Summary Table

| # | Issue | Severity | Action (conceptual only; no code change) |
|---|--------|----------|------------------------------------------|
| 1 | BF16 + GradScaler | Low–Medium | Use GradScaler only when dtype is FP16. |
| 2 | Scheduler step count | Low | Use ceil(batches/grad_accum)*epochs or track optimizer steps. |
| 3 | Masking = PAD + same attention_mask | **Fixed** | Zero out attention_mask for masked positions (done). |
| 4 | Val generation brittle / misleading | Low–Medium | Don’t over-interpret WER/BLEU; consider making metrics optional or clearly “sanity only.” |
| 5 | Param groups by name | Medium | Prefer grouping by module references. |
| 6 | Tokenizer vs embeddings | None now | Keep tokenizer and model in sync when adding tokens. |
| 7 | empty_cache every 100 | Low | Remove or restrict to OOM/recovery. |
| 8 | Collate as lists | None | No change needed. |
| 9 | HF options | Low | Pin revisions if you need strict repro. |
| 10 | No global seed in train | Low–Medium | Set torch/numpy/random seed + worker_init_fn for full repro. |
| 11 | torch.load without weights_only | Low | Use weights_only=True when loading untrusted or shared checkpoints. |

**Highest-impact gremlin:** #3 (text masking semantics). Fixing that is the one change most likely to improve the “force fMRI” behavior without touching the rest of the setup.

---

## Additional Audits (beyond original GPT review)

### 10. No global RNG seed in training (CONFIRMED)

**Location:** `scripts/train.py` – no `torch.manual_seed`, `numpy.random.seed`, or `random.seed` at start of `train()`.

**What happens:** `create_splits()` uses `RANDOM_SEED` from config, so train/val/test story split is reproducible. The training loop does not set PyTorch/numpy/python random seeds, so DataLoader shuffle order, dropout, and any other randomness can differ run-to-run.

**Why it matters:** Same config can yield different loss curves and checkpoints across runs. For strict reproducibility, set `torch.manual_seed(RANDOM_SEED)`, `numpy.random.seed(RANDOM_SEED)`, `random.seed(RANDOM_SEED)` at the start of training, and for multi-worker loaders use a `worker_init_fn` (and optionally `generator=torch.Generator().manual_seed(RANDOM_SEED)`) so worker RNGs are deterministic.

**Severity:** Low–Medium – matters when comparing runs or debugging.

---

### 11. torch.load without weights_only (NOTED)

**Location:** All checkpoint loads in `scripts/train.py`, `evaluate_ranking*.py`, `evaluate_constrained.py`, `evaluate_flamingo.py`, `debug_inference.py`, etc.

**What happens:** `torch.load(path, map_location=...)` is used without `weights_only=True`. Default is `weights_only=False`, so loading can execute arbitrary pickle code.

**Why it matters:** For checkpoints you produce yourself, risk is low. For shared or downloaded checkpoints, `weights_only=True` is recommended (PyTorch 2.0+) to avoid pickle-based exploits. If the checkpoint stores only state dicts and you don’t need custom classes, `weights_only=True` is safer.

**Severity:** Low – policy/security best practice.

---

### 12. Eval scripts: tokenizer alignment (VERIFIED – OK)

**Location:** All eval scripts that load the Flamingo model.

**What happens:** Each script loads the tokenizer, then adds the same special tokens as training: `<|endofchunk|>`, `<image>`, and `<PAD>` if needed.

**Verdict:** Consistent with `train.py` and `FMRIFlamingo`; no extra audit finding.

---

### 13. Data leakage (already audited elsewhere)

**Location:** `DATA_LEAKAGE_AUDIT.md`.

**Verdict:** Story-level split via `create_splits()`; no train/val leakage. Test set held out. No additional findings here.
