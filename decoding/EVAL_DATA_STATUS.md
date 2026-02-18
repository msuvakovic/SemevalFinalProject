# Baseline evaluation: required files (Huth protocol)

`decoding/evaluate_predictions.py` expects the following under **DATA_TEST_DIR** (`data/Huth/derivative/`). They are **not** in this repo; you get them by following the Huth README.

## Data we already have (no need to re-download)

**All training and decoding data lives in `data/Huth/derivative/`** — the code uses only this:

| Location | Contents |
|----------|----------|
| `data/Huth/derivative/preprocessed_data/` | 304 `.hf5` (fMRI) — used by train_EM, train_WR, run_decoder |
| `data/Huth/derivative/TextGrids/` | 84 `.TextGrid` (word timings) |
| `data/Huth/derivative/respdict.json` | Story → TR counts |

**You do not need to download OpenNeuro ds003020 for training or for transcripts.**  
- `data/ds003020/` (if present) is largely **duplicate**: same dataset, overlapping TextGrids/respdict, plus FreeSurfer derivatives we don’t use. Safe to remove or ignore.  
- `Huth/` at repo root is another BIDS copy; the pipeline uses `data/Huth/derivative/` only.

## What we need for evaluation (missing)

| File or directory | Used for |
|-------------------|----------|
| `eval_segments.json` | Task → (start_time, end_time) for evaluation windows |
| `idf_segments.npy` | BERTScore (optional if you skip BERT metric) |
| `test_stimulus/[EXPERIMENT]/[TASK].TextGrid` | Reference transcripts (e.g. `test_stimulus/perceived_speech/life.TextGrid`) |

We **do** have: `data/Huth/derivative/TextGrids/<story>.TextGrid`, `respdict.json`, `preprocessed_data/`.

## Train / val / test split (no memorization)

The encoding model and word-rate model must be trained only on **training** stories; decoding and evaluation run on **test** (and optionally **val**). Huth use separate data dirs (train_response vs test_response); we have one dir, so we use an explicit split.

- **`decoding/splits.py`**: `load_split(subject, split_file)` returns `(train_list, val_list, test_list)`. If no split file: build by **ratio** (default 80% train, 10% val, 10% test) with fixed seed.
- **`decoding/config.py`**: Optional `SPLIT_FILE` path to a JSON like `{ "UTS01": { "train": [...], "val": [...], "test": [...] }, ... }` or top-level `{"train": [...], "val": [...], "test": [...]}` for all subjects.
- **`train_EM.py`**: Uses only **train** stories for fitting; saves `stories` (train), `test_stories`, `val_stories` in the encoding model npz. Options: `--split-file`, `--save-split`.
- **`train_WR.py`**: Uses the same train list (from encoding model npz or `--split-file`).
- **`run_decoder.py`**: Asserts the decoded task is **not** in the encoding model’s `stories` (train); decoding on val or test is allowed.
- **Generate a split file**:  
  `cd decoding && python make_splits.py --out data/Huth/derivative/splits.json`  
  Or fix test (and val) by name:  
  `python make_splits.py --out data/Huth/derivative/splits.json --test-stories life adollshouse`

Then point config at it: in `decoding/config.py` set `SPLIT_FILE = os.path.join(DATA_TRAIN_DIR, "splits.json")`, or pass `--split-file ...` to `train_EM.py` / `train_WR.py`.

## Easiest: get eval files from Huth Box (test data zip)

**Direct link (Huth README):**  
https://utexas.box.com/shared/static/ae5u0t3sh4f46nvmrd3skniq0kk2t5uh.zip

1. Download the zip (browser or `wget`/`curl`).
2. Unzip into `data/` and copy into `data/Huth/derivative/`:

```bash
cd ~/brain-model-alignment
unzip /path/to/ae5u0t3sh4f46nvmrd3skniq0kk2t5uh.zip -d data
# The zip creates a folder inside data/ (e.g. data/data_test or data/<folder_name>)
# Copy eval files from that folder into data/Huth/derivative/:
cp data/*/eval_segments.json data/Huth/derivative/ 2>/dev/null || cp data/eval_segments.json data/Huth/derivative/
cp data/*/idf_segments.npy data/Huth/derivative/ 2>/dev/null || cp data/idf_segments.npy data/Huth/derivative/
# If you need test_stimulus/ (optional; we already have TextGrids/):
cp -r data/*/test_stimulus data/Huth/derivative/ 2>/dev/null || cp -r data/test_stimulus data/Huth/derivative/ 2>/dev/null || true
```

**If the test data is in `data/data_test/`** (Box zip unzipped there), copy into derivative:

```bash
cd ~/brain-model-alignment
cp data/data_test/eval_segments.json data/Huth/derivative/
cp data/data_test/idf_segments.npy data/Huth/derivative/
# Optional: test_stimulus (we already have TextGrids/)
cp -r data/data_test/test_stimulus data/Huth/derivative/ 2>/dev/null || true
```

Then run evaluation: `cd decoding && python evaluate_predictions.py --subject UTS03 --experiment perceived_speech --task life`

**Note:** OpenNeuro ds004510 is the same test dataset but the DataLad clone often has empty/pointer content; the Box zip has the actual files.

---

## Alternative (follow Huth README – OpenNeuro)

From [HuthLab/semantic-decoding](https://github.com/HuthLab/semantic-decoding):

> Download **test data** and extract contents into new **data_test/** directory. Stimulus data for **test_stimulus/[EXPERIMENT]** and response data for **test_response/[SUBJECT_ID]** can be downloaded from **OpenNeuro** (ds004510).

1. **OpenNeuro:** Go to [OpenNeuro ds003020](https://openneuro.org/datasets/ds003020), accept the data use agreement, and download the **test data** (or the derivative that contains `test_stimulus/`, `test_response/`, and any eval metadata). The dataset has a `stimuli` folder; test stimulus materials may be there or in a separate “test” / derivative package.

2. **Extract** so that you have a `data_test/` layout (e.g. next to semantic-decoding or in a clone):
   - `data_test/test_stimulus/[EXPERIMENT]/` (e.g. `perceived_speech/life.TextGrid`)
   - If the OpenNeuro package includes `eval_segments.json` and `idf_segments.npy`, they will be in that same tree.

3. **Point our repo at it:** Our code uses `DATA_TEST_DIR = data/Huth/derivative/`. Either:
   - **Option A:** Copy/link from your `data_test/` into `data/Huth/derivative/`:
     - `data_test/test_stimulus/` → `data/Huth/derivative/test_stimulus/`
     - `data_test/eval_segments.json` → `data/Huth/derivative/eval_segments.json`
     - `data_test/idf_segments.npy` → `data/Huth/derivative/idf_segments.npy`
   - **Option B:** If you already have `TextGrids/` in derivative and only need eval metadata, copy just `eval_segments.json` and `idf_segments.npy` into `data/Huth/derivative/`, and change `decoding/utils_eval.py` `load_transcript()` to use `TextGrids/` instead of `test_stimulus/` (so we don’t duplicate transcripts).

There is **no** `semantic-decoding` or `data_test` directory inside this repo; the Huth README expects you to download test data from OpenNeuro and place it (or a copy) where your scripts look — for us that’s `data/Huth/derivative/`.
