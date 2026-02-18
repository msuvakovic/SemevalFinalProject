# FMRIFlamingo Architecture

This document describes the architecture as implemented in code. All shapes and components are taken from the codebase (see references below).

---

## 1. High-level data flow

```
Batch (list of samples)
    │  each sample: time_series (voxels, TRs), pre_prompt, post_prompt, answer
    ▼
pad_and_apply_batch
    │  pad_fmri_data → (B, voxels, TRs)   [pad/truncate to same voxels & TRs]
    │  flatten → (B, 1, 1, voxels×TRs)
    │  tokenize text → input_ids, attention_mask, labels (prompt masked with -100)
    ▼
model(vision_x, lang_x=input_ids, attention_mask, labels)
    │  vision_x: (B, 1, 1, voxels×TRs)
    │  lang_x:   (B, seq_len)
    ▼
_encode_vision_x(vision_x)
    │  reshape → (B, voxels, TRs)
    │  FMRITokenizer → (B, num_rois, 128)
    │  reshape → (B, 1, 1, num_rois, 128)
    │  Perceiver Resampler → (B, 1, 1, num_latents, dim)   [Open Flamingo]
    │  condition_vis_x(vision_x) on each LLM decoder layer
    ▼
LLM (e.g. Llama-3.2-1B) with gated cross-attention
    │  Q from language, K/V from vision latents; cross-attn every N=1 layer(s)
    ▼
LM head → logits → Cross-entropy loss on answer tokens only
```

---

## 2. Component diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INPUT (per batch)                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  fMRI windows: list of (voxels, TRs) tensors  │  Text: pre_prompt, post_prompt,   │
│  (variable voxels/TRs per sample)             │  answer (next-word target)        │
└───────────────────────┬───────────────────────┴────────────────┬─────────────────┘
                        │                                          │
                        ▼                                          ▼
┌───────────────────────────────────────┐    ┌─────────────────────────────────────┐
│  pad_fmri_data (in pad_and_apply_     │    │  Tokenizer (HuggingFace)            │
│  batch)                               │    │  Format: "pre <image> post answer    │
│  • Pad/truncate to max_voxels,        │    │         <|endofchunk|>"              │
│    max_TRs                            │    │  → input_ids, attention_mask,       │
│  • Stack → (B, voxels, TRs)           │    │    labels (prompt = -100)           │
│  • Flatten → (B, 1, 1, voxels×TRs)    │    │                                      │
└───────────────────────┬───────────────┘    └─────────────────────┬───────────────┘
                        │                                            │
                        │  vision_x: (B, 1, 1, voxels×TRs)           │  lang_x, labels
                        ▼                                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  FMRIFlamingoWithTrainableEncoder._encode_vision_x                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  1. Reshape vision_x → (B, voxels, TRs)   [TRs from config DEFAULT_WINDOW_SIZE]  │
│  2. FMRITokenizer (vision_encoder.visual)                                         │
└───────────────────────┬───────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  FMRITokenizer                                                                   │
│  Input: (B, voxels, TRs)                                                          │
│  • Optional per-ROI z-score norm                                                  │
│  • ROI grouping: voxels → num_rois (config NUM_ROIS=200; method: random/anatomical/│
│    all_voxels) → (B, num_rois, TRs)                                               │
│  • Temporal aggregation: mean | conv | attention → (B, num_rois, 1)               │
│  • Linear(1 → 128): roi_projection                                                 │
│  • + learnable pos_embed (1, max_rois, 128)                                       │
│  • LayerNorm, Dropout                                                             │
│  Output: (B, num_rois, 128)                                                        │
└───────────────────────┬───────────────────────────────────────────────────────────┘
                        │  (B, 1, 1, num_rois, 128) after reshape
                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Perceiver Resampler (Open Flamingo)                                              │
│  Variable-length encoder tokens → fixed number of latents for LLM.                 │
│  num_latents = PerceiverResampler default (e.g. 64 in open_flamingo; may be       │
│  smaller in older PyPI). Each <image> in text = this many K/V slots for LLM.      │
└───────────────────────┬───────────────────────────────────────────────────────────┘
                        │  vision_x: (B, 1, 1, num_latents, dim)
                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  LLM (e.g. meta-llama/Llama-3.2-1B)                                              │
│  • Frozen: decoder layers (except cross-attn), LM head                           │
│  • Trainable: gated_cross_attn_layers, (optionally) input embeddings              │
│  • Each decoder layer: condition_vis_x(vision_x) → gated cross-attention          │
│    (Q from language, K/V from vision latents); every CROSS_ATTN_EVERY_N_LAYERS=1  │
│  • Causal LM forward → logits                                                     │
└───────────────────────┬───────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Loss                                                                             │
│  Cross-entropy on answer tokens only (labels: prompt positions = -100).           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Shapes (as in code)

| Stage | Shape | Notes |
|-------|--------|------|
| Batch fMRI (per sample) | `(voxels, TRs)` | Variable; e.g. ~50k–110k voxels, 10 TRs |
| After pad_fmri_data | `(B, voxels, TRs)` | Padded/truncated to max in batch |
| To _encode_vision_x | `(B, 1, 1, voxels×TRs)` | Flattened |
| After reshape in _encode_vision_x | `(B, voxels, TRs)` | TRs from config (e.g. 10) |
| FMRITokenizer output | `(B, num_rois, 128)` | num_rois=200, 128=ENCODER_OUTPUT_DIM |
| After reshape for perceiver | `(B, 1, 1, num_rois, 128)` | T=1, F=1 in Flamingo terms |
| Perceiver output | `(B, 1, 1, num_latents, dim)` | PerceiverResampler.num_latents (default 64 in repo; package may differ) |
| Text | `input_ids` (B, seq_len), `labels` (prompt masked) | |

---

## 3b. Perceiver Resampler (multi-query bottleneck)

The **Perceiver Resampler** takes the encoder output (many tokens: `num_rois` or all voxels) and compresses it into a **fixed number of latent vectors** (`num_latents`). Those are the only “vision” tokens the LLM ever sees: at each `<image>` in the text, the decoder cross-attends to exactly `num_latents` keys/values.

- **So:** One `<image>` = one position in the sequence, but that position is backed by **num_latents** slots (the perceiver output). The “multi-query” change means: **increase num_latents** (e.g. to 16 or 32) so the LLM has more slots to attend to per brain window, instead of squeezing everything through a tiny bottleneck (e.g. 8 or fewer).
- **Where it’s set:** In `open_flamingo`, `Flamingo` builds the perceiver as `PerceiverResampler(dim=vis_dim)` only; `PerceiverResampler` in `helpers.py` has a default `num_latents=64`. The installed PyPI package (e.g. 0.0.2) may use an older default. To use 16 or 32 you’d need to pass `num_latents` into the resampler (e.g. by patching or subclassing so the Flamingo constructor accepts and forwards `num_latents` when creating the perceiver).

---

## 4. Trainable vs frozen

| Component | Trainable | Notes |
|-----------|-----------|------|
| FMRITokenizer (vision_encoder.visual) | Yes | ROI grouping, temporal agg, projection, pos_embed |
| Perceiver Resampler | Yes | |
| LLM gated_cross_attn_layers | Yes | |
| LLM input embeddings | Configurable (freeze_lm_embeddings) | Unfrozen by default |
| LLM decoder layers (rest) | No | |
| LLM LM head | No | |

---

## 5. Config values (from config.py)

- `NUM_ROIS = 200`
- `ROI_SELECTION_METHOD`: `"anatomical"` | `"random"` | `"all_voxels"`
- `ENCODER_OUTPUT_DIM = 128`, `TRANSFORMER_INPUT_DIM = 128`
- `CROSS_ATTN_EVERY_N_LAYERS = 1`
- `LLM_ID`: e.g. `meta-llama/Llama-3.2-1B`
- `DEFAULT_WINDOW_SIZE`: used to infer TRs when reshaping vision_x (see fmri_flamingo.py)

---

## 6. Source references

- **FMRIFlamingo, pad_and_apply_batch, compute_loss:** `src/models/fmri_flamingo.py`
- **FMRIFlamingoWithTrainableEncoder._encode_vision_x:** `src/models/fmri_flamingo.py` (lines 69–155)
- **FMRITokenizer:** `src/models/fmri_tokenizer.py`
- **TimeSeriesFlamingoWithTrainableEncoder, Perceiver:** `src/opentslm/model/llm/TimeSeriesFlamingoWithTrainableEncoder.py`; Perceiver from `open_flamingo.Flamingo`
- **Trainable parameters:** `src/models/fmri_flamingo.py` (requires_grad_ blocks)

---

*Generated from codebase; reflects implementation as of the last trace.*
