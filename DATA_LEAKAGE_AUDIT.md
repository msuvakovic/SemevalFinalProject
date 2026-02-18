# Data Leakage Audit

**Date:** 2026-02-05  
**Scope:** Flamingo training (`scripts/train.py`), Huth dataset (`src/datasets/huth_fmri_dataset.py`), baseline decoding (`decoding/`).

---

## 1. Flamingo training (scripts/train.py)

**Verdict: No data leakage.**

- **Splits:** `create_splits()` from `huth_fmri_dataset` returns disjoint lists:
  - `train_stories` (80% of stories)
  - `val_stories` (10%)
  - `test_stories` (10%)
- **Implementation:** Stories are shuffled once with a fixed seed (`RANDOM_SEED`), then sliced. No story appears in more than one split.
- **Usage:** `train.py` builds:
  - `train_dataset` with `story_names=train_stories`
  - `val_dataset` with `story_names=val_stories`
- **Dataset behavior:** `HuthFMRIDataset._load_samples()` iterates only over `self.story_names`. So train (val) dataset contains only (subject, story) pairs where `story` is in `train_stories` (`val_stories`). Train and validation therefore see **disjoint sets of stories**.
- **Test set:** `test_stories` is computed but **not used** in `train.py`; it is held out for evaluation. No test data is used for training or validation.

**Config (config.py):** `TRAIN_FRAC=0.8`, `VAL_FRAC=0.1`, `TEST_FRAC=0.1`, `RANDOM_SEED` fixed.

---

## 2. Baseline decoding (decoding/)

**Encoding model (train_EM.py):**

- Trains on **all** stories present for the subject (e.g. 84 for UTS03). There is no train/val split inside this script; regularization is chosen via bootstrap.
- This is **per-subject** training only; no cross-subject leakage.

**Word-rate model (train_WR.py):**

- Same: uses all stories for the subject. No separate val set.

**Decoder (run_decoder.py):**

- **By default:** Asserts `args.task not in encoding_model["stories"]`. So you **cannot** decode on a story that was in the encoding-model training set unless you pass `--ignore-overlap`.
- **With `--ignore-overlap`:** You are explicitly allowing decoding on a story that was in the training set. That is **intentional** “training-set decoding” (e.g. for a quick sanity check), not an accidental leak. For a proper evaluation you should decode only on **held-out stories** (or another subject) and **not** use `--ignore-overlap`.

**Summary:** No accidental leakage. Using `--ignore-overlap` is an explicit choice to evaluate on training stories; for valid metrics, run without it on held-out stories only.

---

## 3. Cross-checks

| Check | Result |
|-------|--------|
| Same story in train and val (Flamingo) | No – splits are disjoint. |
| Test stories used in train/val (Flamingo) | No – test_stories unused in train.py. |
| Baseline decoding on training story by default | No – assertion blocks it unless `--ignore-overlap`. |
| Subject overlap (Flamingo) | Train and val both use `TRAIN_SUBJECTS` (UTS01–UTS08). Split is by **story**, not by subject; same subject can appear in train and val on different stories. This is story-level split, not subject-level. |

---

## 4. Recommendations

1. **Flamingo:** Keep using `create_splits()` and passing `train_stories` / `val_stories` into the datasets. No change needed for leakage.
2. **Baseline:** For a proper evaluation, either:
   - Hold out some stories when training the encoding model (e.g. train on 80% of stories, decode on the other 20%), or
   - Decode on a different subject than the one used for training.  
   Do **not** use `--ignore-overlap` when reporting results.
3. **Reproducibility:** `create_splits()` uses a fixed `RANDOM_SEED`; same machine will get the same train/val/test story split.
