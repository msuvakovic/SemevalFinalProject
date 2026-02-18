# Sharing This Repo on GitHub (For Project Partners)

Use this as a checklist before you push and when onboarding classmates/partners.

---

## 1. What to include in the repo (do push)

| Category | What | We have? |
|----------|------|-----------|
| **Code** | All Python under `scripts/`, `src/`, `decoding/`, `config.py` | ✓ Yes |
| **Config** | `config.py` (project), `decoding/config.py` | ✓ Yes |
| **Docs** | `README.md`, `docs/*.md`, root `*.md` (FUNDAMENTALS, EXPERIMENT, HUTH audit, replication checklist, study log, etc.) | ✓ Yes |
| **Structure** | `requirements.txt`, `CONTRIBUTORS.md`, `LICENSE.md` | ✓ Yes |
| **Small data/metadata** | `data/Huth/derivative/splits.json`, `eval_segments.json`, `respdict.json`, `data/.../TextGrids` (if small), `data/.gitkeep` | ✓ splits & eval_segments in derivative; data/ is gitignored |
| **Reproducibility** | `docs/HUTH_2023_REPLICATION_CHECKLIST.md`, `docs/DECODING_WALKTHROUGH.md`, `docs/HUTH_SOTA_AUDIT.md`, `docs/STUDY_LOG.md` | ✓ Yes |

---

## 2. What NOT to push (keep off GitHub)

| Item | Why | Current .gitignore? |
|------|-----|----------------------|
| **Raw / preprocessed fMRI data** | Large, often subject to data-use agreements; share via OpenNeuro/datasets instead | ✓ `**/data/*` (data dirs ignored) |
| **Model checkpoints** (e.g. `checkpoints/*.pt`) | Very large (~5–6 GB each); share via drive or separate artifact store | ⚠️ **Add** `checkpoints/` (see below) |
| **Trained encoding models** (`models/`) | Per-subject .npz; can be re-trained; large if many subjects | ⚠️ **Add** `models/` (see below) |
| **Decoding results** (`results/`) | Output .npz; can be re-run | ⚠️ **Add** `results/` (see below) |
| **Scores** (`scores/`) | Eval output; can be re-run | ⚠️ **Add** `scores/` (see below) |
| **Logs** (`logs/`) | Training logs; optional to share | ⚠️ **Add** `logs/` (see below) |
| **Virtual env** (`.venv`, `__pycache__`) | Environment-specific | ✓ Already in .gitignore |
| **Zip / large binaries** | e.g. `data_test.zip` | ✓ `*.zip` ignored |

---

## 3. Recommended .gitignore additions

Add these lines to `.gitignore` so you don’t accidentally push large or regenerable files:

```gitignore
# Trained models and outputs (re-trainable / re-runnable)
checkpoints/
models/
results/
scores/
logs/
```

---

## 4. What we have in the repo (quick audit)

- **Root:** `config.py`, `README.md`, `requirements.txt`, `CONTRIBUTORS.md`, `LICENSE.md`, plus `ARCHITECTURE.md`, `EXPERIMENT.md`, `FUNDAMENTALS.md`, `GREMLINS_AUDIT.md`, `DATA_LEAKAGE_AUDIT.md`, `STABILITY_FIXES.md`, `VERIFICATION.md`
- **docs/:** `DECODING_WALKTHROUGH.md`, `HUTH_2023_REPLICATION_CHECKLIST.md`, `HUTH_SOTA_AUDIT.md`, `STUDY_LOG.md`, and this file
- **scripts/:** Training (`train.py`), eval (ranking, constrained, Huth parity), decoder (`run_decoder_fmri_flamingo.py`), data/setup/diagnostic scripts
- **src/:** `models/` (fmri_flamingo, fmri_tokenizer), `datasets/` (huth_fmri_dataset), `utils/` (word_vocab), `opentslm/`
- **decoding/:** Huth-style pipeline (train_EM, train_WR, run_decoder, evaluate_predictions, LM, StimulusModel, utils_ridge, etc.)
- **data/:** Ignored by git (`**/data/*`). For partners: document where to get Huth data (e.g. OpenNeuro) and expected layout in README or `data/README.md`.

---

## 5. Partner onboarding (what to tell classmates)

1. **Clone the repo** (no data or checkpoints in it).
2. **Environment:**  
   `pip install -r requirements.txt`  
   (Plus decoding deps if they run Huth pipeline: scipy, jiwer, evaluate, datasets, etc. — consider adding a `requirements-decoding.txt` or listing in README.)
3. **Data:**  
   - Obtain Huth (or compatible) fMRI + TextGrids per your data-use agreement.  
   - Place under `data/Huth/derivative/` as in `docs/HUTH_2023_REPLICATION_CHECKLIST.md` (e.g. `preprocessed_data/<subject>/*.hf5`, TextGrids, `splits.json`, `eval_segments.json`).  
   - Do **not** commit the heavy data to git; share a link or internal drive.
4. **Checkpoints (optional):**  
   If you want partners to run eval without training, share `checkpoints/best_checkpoint.pt` (and any needed decoding outputs) via Google Drive, lab server, or similar; document the path in README or `docs/STUDY_LOG.md`.
5. **First runs:**  
   - Training: `python scripts/train.py` (see README / config).  
   - Huth decoding: `docs/HUTH_2023_REPLICATION_CHECKLIST.md` and `docs/DECODING_WALKTHROUGH.md`.

---

## 6. Before you push (final checklist)

- [ ] Add `checkpoints/`, `models/`, `results/`, `scores/`, `logs/` to `.gitignore` if not already.
- [ ] Ensure no large files are staged: `git status` and `git diff --cached`.
- [ ] Optionally run `git filter-branch` or `git filter-repo` if you ever committed large checkpoints (see GitHub “removing sensitive data” / “BFG” docs).
- [ ] In README, add a short “Data” section: where to get data, where to put it, and that it’s not in the repo.
- [ ] Create the GitHub repo (e.g. `brain-model-alignment` or your chosen name), add remote, push.

After that, share the repo link with partners and point them to this doc and the README.
