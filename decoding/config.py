import os
import numpy as np

# paths

REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_LM_DIR = os.path.join(REPO_DIR, "data_lm")
DATA_TRAIN_DIR = os.path.join(REPO_DIR, "data", "Huth", "derivative")
DATA_TEST_DIR = os.path.join(REPO_DIR, "data", "Huth", "derivative")
MODEL_DIR = os.path.join(REPO_DIR, "models")
RESULT_DIR = os.path.join(REPO_DIR, "results")
SCORE_DIR = os.path.join(REPO_DIR, "scores")

# GPT encoding model parameters

TRIM = 5
STIM_DELAYS = [1, 2, 3, 4]
RESP_DELAYS = [-4, -3, -2, -1]
ALPHAS = np.logspace(1, 3, 10)
NBOOTS = 5
VOXELS = 10000
CHUNKLEN = 40
GPT_LAYER = 9
GPT_WORDS = 5

# decoder parameters

RANKED = True
WIDTH = 200
NM_ALPHA = 2/3
LM_TIME = 8
LM_MASS = 0.9
LM_RATIO = 0.1
EXTENSIONS = 5

# evaluation parameters

WINDOW = 20

# train/val/test split (so we don't decode on training stories)
# If set, splits.py loads this JSON: { "<subject>": {"train": [...], "val": [...], "test": [...]} }
# or top-level {"train": [...], "val": [...], "test": [...]} for all subjects.
# If unset, split is built by ratio (see splits.SPLIT_TRAIN_RATIO, SPLIT_VAL_RATIO, SPLIT_TEST_RATIO, SPLIT_SEED).
SPLIT_FILE = None  # e.g. os.path.join(DATA_TRAIN_DIR, "splits.json")

# devices

GPT_DEVICE = "cuda"
EM_DEVICE = "cuda"
SM_DEVICE = "cuda"