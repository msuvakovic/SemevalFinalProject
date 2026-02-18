# Study Log

## 2026-02-07 — Huth parity evaluation path

**Goal:** Compare FMRIFlamingo to Huth 2023 imagined_speech using the same evaluation pipeline and null baselines.

**Actions:**
- Added Huth SOTA audit: `docs/HUTH_SOTA_AUDIT.md`
- Added parity evaluator: `scripts/evaluate_huth_parity.py` (custom pred path + Huth metrics + null baselines)
- Updated `scripts/run_decoder_fmri_flamingo.py` voxel selection to respect `ROI_SELECTION_METHOD`
- Fixed `run_decoder_fmri_flamingo.py` crashing with `ValueError: Number of ROIs exceeds max_rois`. Added hard cap to 120,000 voxels (model limit) in scoring loop.

**Notes:**
- Huth-style evaluation requires **word times** (word-rate model) and the **decoding** pipeline’s transcripts.
- For imagined_speech, use Huth evaluation with null baselines (z-scores).
- Current FMRIFlamingo telepathy ranking is ~0% → parity evaluation likely near chance until training objective is aligned.

**Next:** Run Huth baseline decoding, then run FMRIFlamingo decoding with Huth word times and evaluate with null baselines.

---

## Where decoding results live

- **Huth baseline** (`decoding/run_decoder.py`):  
  `results/<subject>/<experiment>/<task>.npz`  
  Example: `results/UTS09/perceived_speech/againstthewind.npz`

- **FMRIFlamingo** (`scripts/run_decoder_fmri_flamingo.py`):  
  `results/<subject>/<experiment>/<task>_fmri_flamingo.npz`  
  Example: `results/UTS09/imagined_speech/adollshouse_fmri_flamingo.npz`

- **Evaluate Huth baseline** (expects `task.npz`):  
  `cd decoding && python evaluate_predictions.py --subject UTS09 --experiment perceived_speech --task againstthewind`

- **Evaluate FMRIFlamingo** (custom pred path):  
  `python scripts/evaluate_huth_parity.py --subject UTS09 --experiment imagined_speech --task adollshouse --pred-path results/UTS09/imagined_speech/adollshouse_fmri_flamingo.npz`
