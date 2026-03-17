Here's every fix we applied:

1. **`SimpleNamespace` → `nn.Module` wrapper** (src/models/fmri_flamingo.py) — The encoder was wrapped in `SimpleNamespace(visual=encoder)` which is invisible to PyTorch's `named_parameters()`. Replaced with a proper `_VisionEncoderWrapper(nn.Module)`. **This was the critical bug — encoder had zero gradients.**

2. **Global → per-group gradient clipping** (scripts/train.py) — `clip_grad_norm_(model.parameters(), 1.0)` let the encoder's large gradients dominate, starving the perceiver to 0.00. Changed to per-param-group clipping.

3. **Positional embedding init scaling** (src/models/fmri_tokenizer.py) — `pos_embed` was initialized with `torch.randn(1, 120000, 128)` at std=1.0 (15.4M params with huge values). Scaled to `* 0.02`.

4. **Lazy → eager `roi_projection`** (src/models/fmri_tokenizer.py) — `roi_projection = None` was built lazily during first forward pass, AFTER optimizer creation. Its parameters were never in any optimizer group — unclipped and unoptimized. Changed to eager init with `DEFAULT_WINDOW_SIZE`.

5. **`CROSS_ATTN_GATE_INIT` 0.0 → 0.5** (config.py) — `tanh(0) = 0` meant fMRI signal was multiplied by zero at init. Set to 0.5 so vision path is active from step 1.

6. **`LR_PROJECTOR` 1e-4 → 5e-4** (config.py) — Perceiver was starved, needed higher LR.

7. **Per-parameter grad norm breakdown** (scripts/train.py) — Added diagnostic logging to show top-5 gradient contributors when encoder norm > 10.
