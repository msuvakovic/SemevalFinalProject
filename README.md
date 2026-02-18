# Brain–Model Alignment (244 Project)

fMRI-to-language decoding: FMRIFlamingo (tokenized cross-attention) and Huth-style ridge decoding. This repo supports training, evaluation, and replicating Huth et al. (Nature Neuroscience 2023).

---

## Table of contents

- [For classmates: step-by-step setup](#for-classmates-step-by-step-setup) ← **start here**
- [Project overview](#project-overview)
- [Repository structure](#repository-structure)
- [Data (where to get it and where it goes)](#data-where-to-get-it-and-where-it-goes)
- [Training](#training)
- [Decoding and evaluation](#decoding-and-evaluation)
- [Troubleshooting](#troubleshooting)
- [More documentation](#more-documentation)

---

# For classmates: step-by-step setup

Follow these steps in order. Every command is meant to be run from your **project root** (the folder that contains `config.py` and `scripts/`) unless we say otherwise.

---

## Step 1: Prerequisites

- **Python**: 3.10 or 3.11 (3.12 may work; we use 3.12 in dev).
- **Git**: to clone the repo.
- **Disk**: ~10 GB for code + env; **much more** if you download full Huth fMRI data (~100 GB). You can start with a subset.
- **GPU**: Optional but recommended for training. CPU works for small runs and decoding.
- **HuggingFace account**: Needed for the Llama model. Create one at https://huggingface.co/join.

---

## Step 2: Clone the repository

```bash
git clone https://github.com/Dom-Marhoefer/244Project.git
cd 244Project
```

You should see folders like `scripts/`, `src/`, `decoding/`, `config.py`, `requirements.txt`, and `README.md`.

---

## Step 3: Create a virtual environment (recommended)

Using a venv keeps the project’s dependencies separate from your system Python.

```bash
# Create the environment (Python 3.10 or 3.11)
python3 -m venv .venv

# Activate it
# On Linux/macOS:
source .venv/bin/activate
# On Windows (Command Prompt):
# .venv\Scripts\activate.bat
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

After activation, your prompt usually shows `(.venv)`. From now on, use `python` and `pip` from this environment.

---

## Step 4: Install Python dependencies

**4.1 – Main project (training, FMRIFlamingo):**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If the project is set up as installable:

```bash
pip install -e .
```

**4.2 – Decoding pipeline (Huth-style ridge + beam search):**

To run the scripts in `decoding/` (e.g. `train_EM.py`, `run_decoder.py`, `evaluate_predictions.py`), install:

```bash
pip install scipy jiwer evaluate datasets nltk
```

Then download NLTK data once:

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab'); nltk.download('omw-1.4')"
```

**4.3 – HuggingFace token (for Llama):**

Log in so the training script can load the language model:

```bash
huggingface-cli login
```

- Get a token: https://huggingface.co/settings/tokens  
- Accept the license for `meta-llama/Llama-3.2-1B`: https://huggingface.co/meta-llama/Llama-3.2-1B (click “Agree and access repository”).

---

## Step 5: Get the data (required for training and decoding)

**The repo does not contain fMRI data or checkpoints.** You need the data on your machine (or on a shared drive).

**5.1 – Where to get the data**

- **Huth et al. dataset** (OpenNeuro):  
  - ds003020: https://openneuro.org/datasets/ds003020  
  - ds004510: https://openneuro.org/datasets/ds004510  
- Or ask the course / project lead for a copy or link (e.g. preprocessed `.hf5` and TextGrids).

**5.2 – Where to put it**

Create this layout **inside the repo** (relative to the project root):

```
data/
└── Huth/
    └── derivative/
        ├── preprocessed_data/
        │   ├── UTS01/
        │   │   ├── adollshouse.hf5
        │   │   ├── buck.hf5
        │   │   └── ... (one .hf5 per story)
        │   ├── UTS02/
        │   └── ...
        ├── TextGrids/
        │   ├── adollshouse.TextGrid
        │   ├── buck.TextGrid
        │   └── ... (one .TextGrid per story)
        ├── splits.json          ← train/val/test story names per subject (ask team or see docs)
        ├── eval_segments.json   ← optional; for evaluation windows (ask team or see docs)
        └── respdict.json        ← optional; for some decoding utils
```

- **preprocessed_data**: one folder per subject (e.g. `UTS01`, `UTS02`), each containing one `.hf5` file per story.  
- **TextGrids**: one `.TextGrid` per story (word-level transcripts and timing).  
- **splits.json**: defines which stories are train / val / test per subject. If you don’t have it, see `decoding/splits.py` (ratio-based split) or `docs/HUTH_2023_REPLICATION_CHECKLIST.md`.

**5.3 – Point the code at the data**

The project’s `config.py` and `decoding/config.py` expect data under `data/Huth/derivative/` relative to the **repository root**. So from your clone:

- Repo root = directory that contains `data/`, `scripts/`, `config.py`.  
- Run all commands from this directory (e.g. `python scripts/train.py`).

No extra environment variables are needed if the layout above is correct.

---

## Step 6: Verify the setup

**6.1 – Imports**

```bash
python -c "from src.datasets.huth_fmri_dataset import HuthFMRIDataset; print('Dataset OK')"
python -c "from src.models.fmri_flamingo import FMRIFlamingo; print('Model OK')"
```

**6.2 – Data (after placing data as in Step 5)**

```bash
python scripts/inspect_huth_data.py --summary-only
```

If data is missing or paths are wrong, you’ll see errors or empty counts; fix the paths or add the missing files.

**6.3 – Training dry-run (optional)**

```bash
python scripts/test_training_setup.py
```

This checks data loading, config, and model creation (and may download the Llama model the first time).

---

## Step 7: Run training (FMRIFlamingo)

From the **project root**:

```bash
python scripts/train.py
```

- Uses settings from `config.py` (batch size, learning rate, data splits, etc.).  
- Saves checkpoints under `checkpoints/` (e.g. `best_checkpoint.pt`).  
- Training can take hours/days depending on data size and hardware. Use `--device cuda` if you have a GPU.

To resume from a checkpoint:

```bash
python scripts/train.py --resume checkpoints/best_checkpoint.pt
```

---

## Step 8: Run the Huth-style decoding pipeline (optional)

If you want to run ridge encoding + beam-search decoding (as in Huth et al.):

**8.1 – Train the encoding model (per subject)**

From the project root, with `decoding` on `PYTHONPATH` (or run from `decoding/` and set `PYTHONPATH` to include the repo root):

```bash
cd decoding
python train_EM.py --subject UTS01
```

This creates `models/UTS01/encoding_model_perceived.npz`. Repeat for other subjects (e.g. `UTS02`, …).

**8.2 – Train the word-rate model (per subject)**

```bash
python train_WR.py --subject UTS01
```

This creates `models/UTS01/word_rate_model_auditory.npz` and `word_rate_model_speech.npz`.

**8.3 – Run the decoder**

Use a **test** story (one that is **not** in the encoding model’s training set). For the default splits, examples:

```bash
# Subjects UTS01–UTS03: test story "buck"
python run_decoder.py --subject UTS01 --task buck --experiment perceived_speech

# Subjects UTS04–UTS08: test story "tildeath"
python run_decoder.py --subject UTS04 --task tildeath --experiment perceived_speech
```

Outputs go to `results/<subject>/perceived_speech/<task>.npz`.

**8.4 – Evaluate predictions**

```bash
python evaluate_predictions.py --subject UTS01 --experiment perceived_speech --task buck --null 0
```

Use `--null 0` unless you have the optional `data_lm/` set up for null baselines. Scores are written under `scores/`.

---

## Step 9: Run FMRIFlamingo decoding (optional)

If you have a trained FMRIFlamingo checkpoint and want to score candidates with it instead of the ridge encoding model:

```bash
# From project root
python scripts/run_decoder_fmri_flamingo.py \
  --checkpoint checkpoints/best_checkpoint.pt \
  --subject UTS01 \
  --task buck \
  --experiment perceived_speech
```

Requires the same data and word-rate model as in Step 8. Output format matches the Huth decoder so you can use the same evaluation script with a custom pred path (see `scripts/evaluate_huth_parity.py`).

---

## Step 10: Troubleshooting

| Problem | What to do |
|--------|------------|
| `ModuleNotFoundError: open_flamingo` | `pip install open-flamingo` (or re-run `pip install -r requirements.txt`). |
| `ModuleNotFoundError: scipy` / `jiwer` / `evaluate` | Install decoding deps: `pip install scipy jiwer evaluate datasets`. |
| HuggingFace “access denied” for Llama | Log in with `huggingface-cli login` and accept the license at https://huggingface.co/meta-llama/Llama-3.2-1B. |
| “No such file or directory” for `.hf5` or `.TextGrid` | Check that `data/Huth/derivative/preprocessed_data/` and `TextGrids/` exist and contain the expected subject/story files. Run `scripts/inspect_huth_data.py --summary-only`. |
| “Task … was in training set” when decoding | Choose a story that is in the **test** set for that subject (see `splits.json` or `docs/HUTH_2023_REPLICATION_CHECKLIST.md`). |
| Out-of-memory (OOM) during training | Reduce `BATCH_SIZE` in `config.py`, or use `--device cpu` (slower). |
| Decoding script can’t find `encoding_model_imagined.npz` | For imagined_speech we use the **perceived** encoder; the script falls back to `encoding_model_perceived.npz`. Train with `train_EM.py --subject <id>` (no `--gpt imagined`). |

If something isn’t covered here, check **docs/** (e.g. `docs/HUTH_2023_REPLICATION_CHECKLIST.md`, `docs/SHARING_REPO_WITH_PARTNERS.md`) or open an issue on the repo.

---

# Project overview

- **Goal**: Map fMRI (BOLD) to language using (1) **FMRIFlamingo** (tokenized fMRI + cross-attention to an LLM) and (2) **Huth-style** ridge encoding + beam-search decoding.
- **Data**: Huth et al.–style dataset (listening/imagined speech, narrative stories), with preprocessed `.hf5` and word-level TextGrids.
- **Training**: Next-word prediction (and optional ranking/telepathy objectives) on fixed TR windows; checkpoints in `checkpoints/`.
- **Evaluation**: Ranking metrics, Huth-style WER/BLEU/METEOR/BERTScore via `decoding/evaluate_predictions.py`, and custom parity eval in `scripts/evaluate_huth_parity.py`.

See the rest of the README and **docs/** for architecture details, replication steps, and references.

---

# Repository structure

```
.
├── config.py                 # Main project config (paths, model, training)
├── requirements.txt          # Python deps for training/FMRIFlamingo
├── scripts/
│   ├── train.py             # Train FMRIFlamingo
│   ├── run_decoder_fmri_flamingo.py  # Decode using FMRIFlamingo scoring
│   ├── evaluate_ranking_telepathy.py  # Telepathy ranking eval
│   ├── evaluate_huth_parity.py        # Eval with Huth metrics + custom pred path
│   ├── inspect_huth_data.py # Inspect data layout and stats
│   └── ...                  # Other utils and evals
├── src/
│   ├── models/              # FMRIFlamingo, FMRITokenizer
│   ├── datasets/            # Huth fMRI dataset
│   └── utils/               # Helpers (e.g. word vocab)
├── decoding/                # Huth-style pipeline (ridge + beam search)
│   ├── config.py            # Decoding-specific config
│   ├── train_EM.py          # Train encoding model
│   ├── train_WR.py          # Train word-rate model
│   ├── run_decoder.py       # Run beam-search decoder
│   ├── evaluate_predictions.py
│   └── ...
├── data/                    # Not in git; you add Huth data here (see Step 5)
├── docs/                    # Extra docs (replication, walkthrough, study log)
├── checkpoints/             # Not in git; training writes here
├── models/                  # Not in git; decoding writes encoding models here
├── results/                 # Not in git; decoding writes .npz here
└── scores/                  # Not in git; evaluation writes here
```

---

# Data (where to get it and where it goes)

- **Source**: OpenNeuro (ds003020, ds004510) or a copy provided by the course/team.  
- **Layout**: See [Step 5](#step-5-get-the-data-required-for-training-and-decoding) above.  
- **Not in the repo**: Raw and preprocessed fMRI, checkpoints, and large outputs are gitignored. Share data and big checkpoints via a separate drive or link and document the location (e.g. in README or `docs/STUDY_LOG.md`).

---

# Training

- **Command**: `python scripts/train.py` from the project root.  
- **Config**: Edit `config.py` (data paths, batch size, learning rate, number of epochs, etc.).  
- **Resume**: `python scripts/train.py --resume checkpoints/best_checkpoint.pt`.  
- **Device**: `python scripts/train.py --device cuda` (or `cpu`).

See [Step 7](#step-7-run-training-fmriflamingo) for the minimal workflow.

---

# Decoding and evaluation

- **Huth pipeline**: Train encoding + word-rate models, then run `decoding/run_decoder.py` on a **test** story; evaluate with `decoding/evaluate_predictions.py`. See [Step 8](#step-8-run-the-huth-style-decoding-pipeline-optional).  
- **FMRIFlamingo decoding**: `scripts/run_decoder_fmri_flamingo.py`; then use `scripts/evaluate_huth_parity.py` with the output path.  
- **Telepathy ranking**: `scripts/evaluate_ranking_telepathy.py` with a checkpoint.

Details and replication protocol: **docs/HUTH_2023_REPLICATION_CHECKLIST.md** and **docs/DECODING_WALKTHROUGH.md**.

---

# Troubleshooting

See the table in [Step 10](#step-10-troubleshooting). For data and split issues, use `scripts/inspect_huth_data.py` and `docs/HUTH_2023_REPLICATION_CHECKLIST.md`.

---

# More documentation

| Doc | Contents |
|-----|----------|
| **docs/SHARING_REPO_WITH_PARTNERS.md** | What to push to git, what to exclude, how to onboard partners. |
| **docs/HUTH_2023_REPLICATION_CHECKLIST.md** | Exact Huth et al. 2023 replication steps and config. |
| **docs/DECODING_WALKTHROUGH.md** | Huth vs FMRIFlamingo decoding and alignment. |
| **docs/HUTH_SOTA_AUDIT.md** | Audit vs Huth SOTA and improvement checklist. |
| **docs/STUDY_LOG.md** | Log of studies and decisions. |

---

# License and contributors

See **LICENSE.md** and **CONTRIBUTORS.md**.
