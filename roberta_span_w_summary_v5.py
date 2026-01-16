"""
train_span_multilabel_roberta_with_decoding_v5.py

Span-based MULTI-LABEL marker detection (Actor/Action/Effect/Evidence/Victim) with:
  ✅ vectorized span mean pooling (fast)
  ✅ boundary-aware span features (start + end + mean + width emb)
  ✅ context features (before/after token embeddings) - V5
  ✅ smarter candidate sampling:
        - always keep spans that overlap gold (easy positives / near-positives)
        - fill remaining with random negatives, seeded per-example (SEED+idx)
  ✅ improved supervision:
        - candidate labeled positive using IoU(cand,gold) >= POS_IOU_THR
  ✅ decoding step:
        - per-role containment-based NMS (better for nested spans)
        - optional merging of adjacent/overlapping spans (MERGE_GAP_CHARS=2)
        - role-specific decoding parameters - V5
  ✅ decoded span extraction metrics (role-wise + micro)
  ✅ AUTOMATIC threshold tuning on validation set (enabled by default)
  ✅ saves best checkpoint by decoded_micro_f1

V5 Improvements:
- Focal Loss for better hard example learning (USE_FOCAL_LOSS = True, gamma=2.0)
- Context features: before/after token embeddings (6H features instead of 4H)
- More epochs (8 instead of 5)
- Lower default thresholds for rare roles in tuning
- Role-specific decoding parameters (stricter for low-precision roles, more aggressive merging for low-recall roles)
- TIER 1 FIXES:
  * Tightened POS_IOU_THR: 0.30 → 0.50 (cleaner supervision, better calibration)
  * Reduced MAX_SPANS_PER_EX: 256 → 128 (less noise, better training signal)
  * Hard negative mining: prioritize spans with IoU in [0.05, 0.25] (teaches boundary discrimination)

Input JSONL format (one example per line):
{
  "text": "...",
  "markers": [
    {"type": "Actor", "startIndex": 10, "endIndex": 25},
    ...
  ]
}
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm


# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = "roberta-base"
SEED = 42
TRAIN_JSONL = "train_rehydrated.jsonl"
OUTPUT_DIR = "./outputs_span_multilabel_roberta_decode_v5"

ROLES = ["Actor", "Action", "Effect", "Evidence", "Victim"]

# Candidate span enumeration (token-space)
MAX_SPAN_TOKENS = 20          # max token length for candidate spans
MAX_SPANS_PER_EX = 128        # V5: Reduced from 256 to reduce noise and improve calibration

# Supervision: label candidate positive if IoU(cand,gold) >= thr
POS_IOU_THR = 0.50            # V5: Tightened from 0.30 to 0.50 for cleaner supervision
HARD_NEG_IOU_MIN = 0.05       # V5: Hard negatives: IoU in [0.05, 0.25] for boundary learning
HARD_NEG_IOU_MAX = 0.25

# Training
BATCH_SIZE = 4
LR = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 8                    # V5: Increased from 5
WARMUP_RATIO = 0.1
GRAD_CLIP = 1.0
FP16 = torch.cuda.is_available()

# Imbalance handling
POS_WEIGHT_CLIP_MAX = 20.0    # clip extreme pos_weight

# V5: Focal Loss parameters
USE_FOCAL_LOSS = False         # Changing from True to False for ablation studies
FOCAL_GAMMA = 1.0             # focusing parameter

# Logging / Eval
LOG_EVERY_STEPS = 25
EVAL_EVERY_STEPS = 500        # set None to eval only at epoch end

# Decoding params
DEFAULT_THRESH = 0.5

# NMS: containment-based suppression (better for nested spans)
# Default values (can be overridden per-role)
DEFAULT_CONTAIN_THR = 0.75
DEFAULT_NMS_IOU_THR = 0.45
DEFAULT_MERGE_GAP_CHARS = 2

# V5: Role-specific decoding parameters
# Note: Lower contain_thr/nms_iou + higher merge_gap = more permissive (keeps more, merges more) → increases recall
ROLE_DECODE_PARAMS = {
    "Actor": {"contain_thr": 0.75, "nms_iou": 0.45, "merge_gap": 2},  # Balanced
    "Action": {"contain_thr": 0.70, "nms_iou": 0.40, "merge_gap": 3},  # More permissive for low recall
    "Effect": {"contain_thr": 0.70, "nms_iou": 0.40, "merge_gap": 3},  # More permissive (helps with low recall, may hurt precision)
    "Evidence": {"contain_thr": 0.65, "nms_iou": 0.35, "merge_gap": 3},  # Most permissive for low recall
    "Victim": {"contain_thr": 0.75, "nms_iou": 0.45, "merge_gap": 2},  # Balanced
}

# Decoded evaluation matching
MATCH_IOU_THR = 0.3

# V5: Threshold tuning enabled by default
TUNE_THRESHOLDS = True
THRESH_GRID = [round(x, 2) for x in np.arange(0.15, 0.86, 0.05)]


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Span geometry helpers (char-space)
# -------------------------
def inter_len(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def union_len(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, (a[1] - a[0]) + (b[1] - b[0]) - inter_len(a, b))


def iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    u = union_len(a, b)
    if u <= 0:
        return 0.0
    return inter_len(a, b) / u


def contain_score(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Returns intersection / min_len. High when one span nearly contains the other.
    """
    inter = inter_len(a, b)
    la = max(1, a[1] - a[0])
    lb = max(1, b[1] - b[0])
    return inter / min(la, lb)


# -------------------------
# Dataset
# -------------------------
class SpanMultiLabelDataset(Dataset):
    """
    Per example returns:
      input_ids: [T]
      attention_mask: [T]
      offsets: [T,2] char offsets for each token (special tokens -> (0,0))
      spans_tok: [S,2] inclusive token indices
      labels: [S,R] multi-hot per candidate span
      gold_char_by_role: Dict[str, List[(s,e)]]
      text_len: int
    """

    def __init__(self, jsonl_path: str, tokenizer, split: str):
        self.tokenizer = tokenizer
        self.examples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

        # deterministic 90/10 split
        rng = np.random.RandomState(SEED)
        idx = np.arange(len(self.examples))
        rng.shuffle(idx)

        cut = int(0.9 * len(idx))
        keep = idx[:cut] if split == "train" else idx[cut:]
        self.examples = [self.examples[i] for i in keep]

    def __len__(self):
        return len(self.examples)

    def _tokenize(self, text: str):
        return self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
            return_attention_mask=True,
        )

    def _enumerate_all_spans(self, offsets: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        valid = [i for i, (s, e) in enumerate(offsets) if not (s == 0 and e == 0)]
        spans = []
        for i_pos, i in enumerate(valid):
            for j_pos in range(i_pos, min(i_pos + MAX_SPAN_TOKENS, len(valid))):
                j = valid[j_pos]
                spans.append((i, j))
        return spans

    def _tok_span_to_char(self, offsets, tok_span: Tuple[int, int]) -> Tuple[int, int]:
        i, j = tok_span
        return (int(offsets[i][0]), int(offsets[j][1]))

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        text = ex["text"]
        markers = ex.get("markers", [])

        gold_char_by_role: Dict[str, List[Tuple[int, int]]] = {r: [] for r in ROLES}
        for m in markers:
            r = m.get("type")
            if r in gold_char_by_role:
                gold_char_by_role[r].append((int(m["startIndex"]), int(m["endIndex"])))

        enc = self._tokenize(text)
        offsets = enc["offset_mapping"]

        # Enumerate all spans, then smart-select up to MAX_SPANS_PER_EX
        all_spans = self._enumerate_all_spans(offsets)

        # Label ALL spans first (needed so we can preferentially keep positives / near positives)
        labels_all = np.zeros((len(all_spans), len(ROLES)), dtype=np.float32)

        for si, tok_span in enumerate(all_spans):
            cand_char = self._tok_span_to_char(offsets, tok_span)
            if cand_char[1] <= cand_char[0]:
                continue
            for ri, role in enumerate(ROLES):
                for gold_char in gold_char_by_role[role]:
                    if iou(cand_char, gold_char) >= POS_IOU_THR:
                        labels_all[si, ri] = 1.0
                        break

        # V5: Smart keep with hard negative mining:
        # 1) keep positives (IoU >= POS_IOU_THR)
        # 2) keep hard negatives (IoU in [HARD_NEG_IOU_MIN, HARD_NEG_IOU_MAX]) for boundary learning
        # 3) fill remaining with random negatives
        if len(all_spans) > MAX_SPANS_PER_EX:
            positives = []  # IoU >= POS_IOU_THR
            hard_negatives = []  # IoU in [HARD_NEG_IOU_MIN, HARD_NEG_IOU_MAX]
            
            for si, tok_span in enumerate(all_spans):
                cand_char = self._tok_span_to_char(offsets, tok_span)
                if cand_char[1] <= cand_char[0]:
                    continue
                
                # Check IoU with all gold spans
                max_iou = 0.0
                for role in ROLES:
                    for gold_char in gold_char_by_role[role]:
                        iou_val = iou(cand_char, gold_char)
                        max_iou = max(max_iou, iou_val)
                
                if max_iou >= POS_IOU_THR:
                    positives.append(si)
                elif max_iou >= HARD_NEG_IOU_MIN and max_iou < HARD_NEG_IOU_MAX:
                    hard_negatives.append(si)

            # Unique lists
            positives = list(dict.fromkeys(positives))
            hard_negatives = list(dict.fromkeys(hard_negatives))

            rng = np.random.RandomState(SEED + idx)  # per-example deterministic
            keep = []

            # Priority 1: Keep all positives (up to limit)
            if positives:
                if len(positives) >= MAX_SPANS_PER_EX:
                    keep = rng.choice(positives, size=MAX_SPANS_PER_EX, replace=False).tolist()
                else:
                    keep = positives[:]
                    remaining = MAX_SPANS_PER_EX - len(keep)
                    
                    # Priority 2: Add hard negatives
                    if hard_negatives and remaining > 0:
                        hard_neg_pool = [x for x in hard_negatives if x not in keep]
                        if hard_neg_pool:
                            n_hard = min(remaining, len(hard_neg_pool))
                            add_hard = rng.choice(hard_neg_pool, size=n_hard, replace=False).tolist()
                            keep += add_hard
                            remaining -= n_hard
                    
                    # Priority 3: Fill with random negatives
                    if remaining > 0:
                        all_indices = set(range(len(all_spans)))
                        exclude = set(keep) | set(hard_negatives)  # Don't re-sample hard negs
                        pool = list(all_indices - exclude)
                        if pool:
                            n_random = min(remaining, len(pool))
                            add_random = rng.choice(pool, size=n_random, replace=False).tolist()
                            keep += add_random
            else:
                # No positives: prioritize hard negatives, then random
                if hard_negatives:
                    if len(hard_negatives) >= MAX_SPANS_PER_EX:
                        keep = rng.choice(hard_negatives, size=MAX_SPANS_PER_EX, replace=False).tolist()
                    else:
                        keep = hard_negatives[:]
                        remaining = MAX_SPANS_PER_EX - len(keep)
                        pool = np.setdiff1d(np.arange(len(all_spans)), np.array(keep), assume_unique=False)
                        if len(pool) > 0 and remaining > 0:
                            add = rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist()
                            keep += add
                else:
                    keep = rng.choice(len(all_spans), size=MAX_SPANS_PER_EX, replace=False).tolist()

            keep = sorted(keep)
            spans_tok = [all_spans[k] for k in keep]
            labels = labels_all[keep]
        else:
            spans_tok = all_spans
            labels = labels_all

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "offsets": torch.tensor(offsets, dtype=torch.long),          # [T,2]
            "spans": torch.tensor(spans_tok, dtype=torch.long),          # [S,2]
            "labels": torch.tensor(labels, dtype=torch.float32),         # [S,R]
            "gold_char_by_role": gold_char_by_role,                      # python object
            "text_len": len(text),
        }


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    offsets: torch.Tensor
    spans: torch.Tensor
    labels: torch.Tensor
    span_mask: torch.Tensor
    gold_char_by_role: List[Dict[str, List[Tuple[int, int]]]]
    text_len: torch.Tensor


def make_collate_fn(tokenizer):
    """Create collate_fn with tokenizer for proper padding token."""
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        # V5: Better fallback - use EOS or 0, not UNK (padding != unknown token)
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    
    def collate_fn(batch_list: List[Dict]) -> Batch:
        max_t = max(x["input_ids"].shape[0] for x in batch_list)
        max_s = max(x["spans"].shape[0] for x in batch_list)
        R = batch_list[0]["labels"].shape[1]

        input_ids, attention_mask, offsets = [], [], []
        spans, labels, span_mask = [], [], []
        gold_char_by_role = []
        text_len = []

        for x in batch_list:
            t = x["input_ids"].shape[0]
            s = x["spans"].shape[0]

            pad_t = max_t - t
            # V5: Use tokenizer's pad_token_id instead of hardcoded 1
            input_ids.append(torch.cat([x["input_ids"], torch.full((pad_t,), pad_token_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([x["attention_mask"], torch.zeros((pad_t,), dtype=torch.long)]))
            offsets.append(torch.cat([x["offsets"], torch.zeros((pad_t, 2), dtype=torch.long)], dim=0))

            pad_s = max_s - s
            spans.append(torch.cat([x["spans"], torch.full((pad_s, 2), -1, dtype=torch.long)], dim=0))
            labels.append(torch.cat([x["labels"], torch.zeros((pad_s, R), dtype=torch.float32)], dim=0))

            sm = torch.cat([torch.ones((s,), dtype=torch.bool), torch.zeros((pad_s,), dtype=torch.bool)], dim=0)
            span_mask.append(sm)

            gold_char_by_role.append(x["gold_char_by_role"])
            text_len.append(x["text_len"])

        return Batch(
            input_ids=torch.stack(input_ids, dim=0),
            attention_mask=torch.stack(attention_mask, dim=0),
            offsets=torch.stack(offsets, dim=0),
            spans=torch.stack(spans, dim=0),
            labels=torch.stack(labels, dim=0),
            span_mask=torch.stack(span_mask, dim=0),
            gold_char_by_role=gold_char_by_role,
            text_len=torch.tensor(text_len, dtype=torch.long),
        )
    
    return collate_fn


# -------------------------
# Model (boundary-aware span pooling + context features)
# -------------------------
class SpanMultiLabelModel(nn.Module):
    def __init__(self, base_model_name: str, num_roles: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        H = self.encoder.config.hidden_size

        self.max_width = MAX_SPAN_TOKENS
        self.width_emb = nn.Embedding(self.max_width + 1, H)

        # V5: start + end + mean + width_emb + before + after => 6H input
        self.classifier = nn.Sequential(
            nn.Linear(H * 6, H),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(H, num_roles),
        )

    def forward(self, input_ids, attention_mask, spans):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # [B,T,H]
        B, T, H = h.shape
        _, S, _ = spans.shape

        # prefix sums for fast mean pooling
        prefix = torch.zeros((B, T + 1, H), device=h.device, dtype=h.dtype)
        prefix[:, 1:, :] = torch.cumsum(h, dim=1)

        start = spans[:, :, 0].clamp(min=0)
        end = spans[:, :, 1].clamp(min=0)
        end_ex = end + 1

        # mean pooling
        sum_vec = prefix.gather(1, end_ex.unsqueeze(-1).expand(-1, -1, H)) - \
                  prefix.gather(1, start.unsqueeze(-1).expand(-1, -1, H))
        lengths = (end - start + 1).clamp(min=1).to(h.dtype).unsqueeze(-1)
        mean_vec = sum_vec / lengths

        # boundary vectors
        h_start = h.gather(1, start.unsqueeze(-1).expand(-1, -1, H))
        h_end = h.gather(1, end.unsqueeze(-1).expand(-1, -1, H))

        # V5: Context features (before/after tokens)
        # Fixed: Zero out invalid boundary positions instead of collapsing to boundary token
        start_before_idx = (start - 1).clamp(min=0)
        end_after_idx = (end + 1).clamp(max=T - 1)
        
        # Check if before/after positions are valid (not at boundaries)
        valid_before = (start > 0).unsqueeze(-1).expand(-1, -1, H)  # [B, S, H]
        valid_after = (end < T - 1).unsqueeze(-1).expand(-1, -1, H)  # [B, S, H]
        
        h_before = h.gather(1, start_before_idx.unsqueeze(-1).expand(-1, -1, H))
        h_after = h.gather(1, end_after_idx.unsqueeze(-1).expand(-1, -1, H))
        
        # Zero out invalid positions (at boundaries)
        h_before = h_before * valid_before.float()
        h_after = h_after * valid_after.float()

        width = (end - start + 1).clamp(min=1, max=self.max_width)
        wemb = self.width_emb(width)

        # V5: 6H features instead of 4H
        feat = torch.cat([h_start, h_end, mean_vec, wemb, h_before, h_after], dim=-1)  # [B,S,6H]
        logits = self.classifier(feat)  # [B,S,R]
        return logits


# -------------------------
# Imbalance handling (BCE pos_weight) + CLIP
# -------------------------
def compute_pos_weight(train_loader) -> torch.Tensor:
    pos = np.zeros((len(ROLES),), dtype=np.float64)
    neg = np.zeros((len(ROLES),), dtype=np.float64)

    for batch in train_loader:
        y = batch.labels.numpy()
        m = batch.span_mask.numpy()
        y = y[m]
        pos += y.sum(axis=0)
        neg += (1.0 - y).sum(axis=0)

    pos = np.maximum(pos, 1.0)
    pw = neg / pos
    pw = np.minimum(pw, POS_WEIGHT_CLIP_MAX)  # clip
    return torch.tensor(pw, dtype=torch.float32)


# -------------------------
# Focal Loss for multi-label classification
# -------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Fixed V5: Compute probabilities correctly, apply pos_weight separately.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # pos_weight tensor [R] for class imbalance
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # inputs: [N, R] logits
        # targets: [N, R] binary labels
        # alpha: [R] pos_weight (neg/pos ratio per role)
        
        # Compute probabilities from logits
        p = torch.sigmoid(inputs)  # [N, R]
        
        # p_t = probability of true class: p if target=1, else (1-p)
        p_t = p * targets + (1 - p) * (1 - targets)  # [N, R]
        
        # Focal weight: (1 - p_t)^gamma (downweight easy examples)
        focal_weight = (1 - p_t) ** self.gamma  # [N, R]
        
        # Standard BCE loss (unweighted)
        bce = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )  # [N, R]
        
        # Apply focal weighting
        focal_loss = focal_weight * bce  # [N, R]
        
        # Apply pos_weight for class imbalance (upweight positive examples for rare classes)
        if self.alpha is not None:
            # pos_weight increases loss for positive labels: weight = alpha when target=1, else 1.0
            pos_weight_mask = targets * (self.alpha.unsqueeze(0) - 1.0) + 1.0  # [N, R]
            focal_loss = focal_loss * pos_weight_mask
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# -------------------------
# Decoding helpers
# -------------------------
def tok_span_to_char(offsets: np.ndarray, tok_span: Tuple[int, int]) -> Tuple[int, int]:
    i, j = tok_span
    return (int(offsets[i, 0]), int(offsets[j, 1]))


def nms_char_spans_containment(
    spans_with_scores: List[Tuple[Tuple[int, int], float]],
    contain_thr: float,
    iou_thr: float,
) -> List[Tuple[Tuple[int, int], float]]:
    """
    Greedy NMS with containment suppression:
      suppress if contain_score >= contain_thr OR IoU >= iou_thr
    """
    spans_with_scores = sorted(spans_with_scores, key=lambda x: x[1], reverse=True)
    kept: List[Tuple[Tuple[int, int], float]] = []

    for span, score in spans_with_scores:
        suppress = False
        for kspan, kscore in kept:
            if contain_score(span, kspan) >= contain_thr or iou(span, kspan) >= iou_thr:
                suppress = True
                break
        if not suppress:
            kept.append((span, score))

    return kept


def merge_char_spans(spans: List[Tuple[int, int]], gap_chars: int) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s <= pe + gap_chars:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def decode_example(
    probs_ex: np.ndarray,           # [S,R]
    spans_tok_ex: np.ndarray,       # [S,2]
    offsets_ex: np.ndarray,         # [T,2]
    span_mask_ex: np.ndarray,       # [S]
    thresholds: Dict[str, float],
) -> Dict[str, List[Tuple[int, int]]]:
    decoded: Dict[str, List[Tuple[int, int]]] = {r: [] for r in ROLES}

    valid_idxs = np.where(span_mask_ex)[0].tolist()
    if not valid_idxs:
        return decoded

    for ri, role in enumerate(ROLES):
        thr = thresholds.get(role, DEFAULT_THRESH)
        
        # V5: Get role-specific decoding parameters
        decode_params = ROLE_DECODE_PARAMS.get(role, {
            "contain_thr": DEFAULT_CONTAIN_THR,
            "nms_iou": DEFAULT_NMS_IOU_THR,
            "merge_gap": DEFAULT_MERGE_GAP_CHARS
        })
        contain_thr = decode_params["contain_thr"]
        nms_iou_thr = decode_params["nms_iou"]
        merge_gap = decode_params["merge_gap"]
        
        candidates: List[Tuple[Tuple[int, int], float]] = []

        for si in valid_idxs:
            p = float(probs_ex[si, ri])
            if p < thr:
                continue

            tok_span = (int(spans_tok_ex[si, 0]), int(spans_tok_ex[si, 1]))
            if tok_span[0] < 0 or tok_span[1] < 0:
                continue

            char_span = tok_span_to_char(offsets_ex, tok_span)
            if char_span[1] <= char_span[0]:
                continue

            candidates.append((char_span, p))

        kept = nms_char_spans_containment(candidates, contain_thr=contain_thr, iou_thr=nms_iou_thr)
        kept_spans = [s for (s, _) in kept]
        merged = merge_char_spans(kept_spans, gap_chars=merge_gap)
        decoded[role] = merged

    return decoded


# -------------------------
# Decoded span evaluation (soft overlap matching)
# -------------------------
def match_pred_to_gold(
    pred_spans: List[Tuple[int, int]],
    gold_spans: List[Tuple[int, int]],
    match_iou_thr: float,
) -> Tuple[int, int, int]:
    if not pred_spans and not gold_spans:
        return (0, 0, 0)
    if not pred_spans:
        return (0, 0, len(gold_spans))
    if not gold_spans:
        return (0, len(pred_spans), 0)

    gold_used = [False] * len(gold_spans)
    tp = 0
    fp = 0

    for p in pred_spans:
        best_j = None
        best_iou = 0.0
        for j, g in enumerate(gold_spans):
            if gold_used[j]:
                continue
            v = iou(p, g)
            if v > best_iou:
                best_iou = v
                best_j = j

        if best_j is not None and best_iou >= match_iou_thr:
            tp += 1
            gold_used[best_j] = True
        else:
            fp += 1

    fn = gold_used.count(False)
    return (tp, fp, fn)


def prf_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return (p, r, f)


# -------------------------
# Evaluation (candidate + decoded)
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device, thresholds: Dict[str, float]) -> Dict:
    model.eval()

    cand_true = []
    cand_pred = []

    role_counts = {r: {"tp": 0, "fp": 0, "fn": 0} for r in ROLES}

    n_spans_eval = 0
    n_examples = 0

    for batch in loader:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        spans = batch.spans.to(device)
        labels = batch.labels.to(device)
        span_mask = batch.span_mask.to(device)

        logits = model(input_ids, attention_mask, spans)
        probs = torch.sigmoid(logits)

        # Candidate metrics (fixed 0.5 threshold)
        y_true = labels[span_mask]
        y_prob = probs[span_mask]
        y_hat = (y_prob >= 0.5).float()

        cand_true.append(y_true.cpu().numpy())
        cand_pred.append(y_hat.cpu().numpy())

        # Decoded metrics
        probs_np = probs.detach().cpu().numpy()
        spans_np = batch.spans.detach().cpu().numpy()
        offsets_np = batch.offsets.detach().cpu().numpy()
        mask_np = batch.span_mask.detach().cpu().numpy()
        gold_list = batch.gold_char_by_role

        B = probs_np.shape[0]
        for b in range(B):
            n_examples += 1
            decoded = decode_example(
                probs_ex=probs_np[b],
                spans_tok_ex=spans_np[b],
                offsets_ex=offsets_np[b],
                span_mask_ex=mask_np[b],
                thresholds=thresholds,
            )
            gold_by_role = gold_list[b]
            for role in ROLES:
                pred_spans = decoded[role]
                gold_spans = gold_by_role.get(role, [])
                tp, fp, fn = match_pred_to_gold(pred_spans, gold_spans, match_iou_thr=MATCH_IOU_THR)
                role_counts[role]["tp"] += tp
                role_counts[role]["fp"] += fp
                role_counts[role]["fn"] += fn

        n_spans_eval += int(y_true.shape[0])

    y_true_all = np.concatenate(cand_true, axis=0) if cand_true else np.zeros((0, len(ROLES)))
    y_pred_all = np.concatenate(cand_pred, axis=0) if cand_pred else np.zeros((0, len(ROLES)))

    cand_micro_f1 = float(f1_score(y_true_all.reshape(-1), y_pred_all.reshape(-1), zero_division=0.0))
    cand_role_f1 = {ROLES[i]: float(f1_score(y_true_all[:, i], y_pred_all[:, i], zero_division=0.0))
                    for i in range(len(ROLES))}

    decoded_role_prf = {}
    micro_tp = micro_fp = micro_fn = 0
    for role in ROLES:
        tp = role_counts[role]["tp"]
        fp = role_counts[role]["fp"]
        fn = role_counts[role]["fn"]
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        p, r, f = prf_from_counts(tp, fp, fn)
        decoded_role_prf[role] = {"p": p, "r": r, "f1": f}

    decoded_p, decoded_r, decoded_micro_f1 = prf_from_counts(micro_tp, micro_fp, micro_fn)

    return {
        "n_spans_eval": int(n_spans_eval),
        "n_examples_eval": int(n_examples),
        "cand_micro_f1": cand_micro_f1,
        "cand_role_f1": cand_role_f1,
        "decoded_micro_f1": float(decoded_micro_f1),
        "decoded_micro_p": float(decoded_p),
        "decoded_micro_r": float(decoded_r),
        "decoded_role_prf": decoded_role_prf,
        "thresholds": dict(thresholds),
    }


# -------------------------
# Threshold tuning (per role) on decoded F1
# -------------------------
@torch.no_grad()
def tune_thresholds(model, loader, device) -> Dict[str, float]:
    """
    Coordinate-descent-ish tuning:
      for each role, sweep THRESH_GRID while holding others fixed
      maximize decoded_role_f1[role]
    """
    # V5: Start with even lower default thresholds for rare roles
    thresholds = {
        "Actor": 0.40,      # V5: lowered from 0.45
        "Action": 0.35,     # V5: lowered from 0.45 (low recall)
        "Effect": 0.35,     # V5: lowered from 0.45 (low precision)
        "Evidence": 0.35,   # V5: lowered from 0.45 (low recall)
        "Victim": 0.30,     # V5: lowered from 0.35
    }

    model.eval()
    cached = []
    for batch in loader:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        spans = batch.spans.to(device)
        logits = model(input_ids, attention_mask, spans)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        cached.append({
            "probs": probs,
            "spans": batch.spans.detach().cpu().numpy(),
            "offsets": batch.offsets.detach().cpu().numpy(),
            "mask": batch.span_mask.detach().cpu().numpy(),
            "gold": batch.gold_char_by_role,
        })

    def eval_one_role(role: str, thr: float, fixed: Dict[str, float]) -> float:
        role_tp = role_fp = role_fn = 0
        temp_thr = dict(fixed)
        temp_thr[role] = thr

        for pack in cached:
            probs = pack["probs"]
            spans = pack["spans"]
            offsets = pack["offsets"]
            mask = pack["mask"]
            gold_list = pack["gold"]

            B = probs.shape[0]
            for b in range(B):
                decoded = decode_example(probs[b], spans[b], offsets[b], mask[b], temp_thr)
                pred_spans = decoded[role]
                gold_spans = gold_list[b].get(role, [])
                tp, fp, fn = match_pred_to_gold(pred_spans, gold_spans, match_iou_thr=MATCH_IOU_THR)
                role_tp += tp
                role_fp += fp
                role_fn += fn

        _, _, f1v = prf_from_counts(role_tp, role_fp, role_fn)
        return float(f1v)

    # sweep per role
    for role in ROLES:
        best_thr = thresholds[role]
        best_f1 = -1.0
        for thr in THRESH_GRID:
            f1v = eval_one_role(role, thr, thresholds)
            if f1v > best_f1:
                best_f1 = f1v
                best_thr = thr
        thresholds[role] = best_thr
        print(f"[tune] {role}: best_thr={best_thr:.2f} decoded_f1={best_f1:.4f}")
    
    # V5: Diagnostic: Check if thresholds are well-calibrated (should be 0.45-0.75, not all 0.85)
    max_thr = max(thresholds.values())
    min_thr = min(thresholds.values())
    thr_range = max_thr - min_thr
    print(f"[tune] Threshold range: [{min_thr:.2f}, {max_thr:.2f}] (range={thr_range:.2f})")
    if max_thr >= 0.80:
        print(f"[tune] WARNING: High thresholds ({max_thr:.2f}) suggest poor calibration - model may be underconfident")
    if thr_range < 0.10:
        print(f"[tune] WARNING: All thresholds similar (range={thr_range:.2f}) - roles may not be well-separated")

    return thresholds


# -------------------------
# Summary printing
# -------------------------
def print_training_summary(config: Dict, best_metrics: Dict, pos_weight: torch.Tensor, total_time_s: float):
    print("\n" + "=" * 90)
    print("TRAINING SUMMARY - ALL MARKERS (SPAN MULTI-LABEL + DECODING) [V5]")
    print("=" * 90)

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\nBest Validation Metrics (by decoded_micro_f1):")
    print(f"  decoded_micro_f1: {best_metrics.get('decoded_micro_f1', 0.0):.4f}")
    print(f"  decoded_micro_p:  {best_metrics.get('decoded_micro_p', 0.0):.4f}")
    print(f"  decoded_micro_r:  {best_metrics.get('decoded_micro_r', 0.0):.4f}")
    print(f"  cand_micro_f1:    {best_metrics.get('cand_micro_f1', 0.0):.4f}")
    print(f"  n_examples_eval:  {best_metrics.get('n_examples_eval', 0)}")
    print(f"  n_spans_eval:     {best_metrics.get('n_spans_eval', 0)}")

    print("\nThresholds (decoded):")
    th = best_metrics.get("thresholds", {})
    for r in ROLES:
        print(f"  {r:<8} {th.get(r, DEFAULT_THRESH):.2f}")

    print("\nRole-wise metrics:")
    print(f"{'Role':<10} {'CandF1':<10} {'DecP':<10} {'DecR':<10} {'DecF1':<10} {'PosWeight':<10}")
    print("-" * 70)

    pw = pos_weight.detach().cpu().numpy().tolist()
    cand_role = best_metrics.get("cand_role_f1", {})
    dec_role = best_metrics.get("decoded_role_prf", {})

    for i, r in enumerate(ROLES):
        cf1 = cand_role.get(r, 0.0)
        dp = dec_role.get(r, {}).get("p", 0.0)
        dr = dec_role.get(r, {}).get("r", 0.0)
        df1 = dec_role.get(r, {}).get("f1", 0.0)
        print(f"{r:<10} {cf1:<10.4f} {dp:<10.4f} {dr:<10.4f} {df1:<10.4f} {pw[i]:<10.2f}")

    print("\nRuntime:")
    print(f"  total_seconds: {total_time_s:.2f}")
    print("=" * 90)
    print(f"Training completed. Models saved to: {OUTPUT_DIR}")
    print("=" * 90)


# -------------------------
# Main training loop
# -------------------------
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, add_prefix_space=True)

    train_ds = SpanMultiLabelDataset(TRAIN_JSONL, tokenizer, split="train")
    val_ds = SpanMultiLabelDataset(TRAIN_JSONL, tokenizer, split="val")

    # V5: Create collate_fn with tokenizer for proper padding token
    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = SpanMultiLabelModel(MODEL_NAME, num_roles=len(ROLES)).to(device)

    pos_weight = compute_pos_weight(train_loader).to(device)
    
    # V5: Use Focal Loss if enabled, otherwise BCE
    if USE_FOCAL_LOSS:
        loss_fct = FocalLoss(alpha=pos_weight, gamma=FOCAL_GAMMA) # alpha changed from pos_weight to None for ablation studies, focal_gamma changed to be lower for ablation studies
        print(f"[loss] Using Focal Loss (gamma={FOCAL_GAMMA})")
        max_pw = pos_weight.max().item()
        if max_pw > 10.0:
            print(f"[loss] WARNING: Combining focal loss (gamma={FOCAL_GAMMA}) with high pos_weight (max={max_pw:.1f})")
            print(f"[loss]         This can be unstable - may over-emphasize rare classes early in training")
            print(f"[loss]         If training is unstable, try: gamma=1.0 or disable focal loss")
    else:
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"[loss] Using BCE with pos_weight")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=FP16)

    config = {
        "Model": MODEL_NAME,
        "Roles": ROLES,
        "MaxSpanTokens": MAX_SPAN_TOKENS,
        "MaxSpansPerEx": MAX_SPANS_PER_EX,
        "PosIoUThr": POS_IOU_THR,
        "HardNegIoUMin": HARD_NEG_IOU_MIN,
        "HardNegIoUMax": HARD_NEG_IOU_MAX,
        "BatchSize": BATCH_SIZE,
        "LR": LR,
        "WeightDecay": WEIGHT_DECAY,
        "Epochs": EPOCHS,
        "WarmupRatio": WARMUP_RATIO,
        "FP16": FP16,
        "Seed": SEED,
        "LogEverySteps": LOG_EVERY_STEPS,
        "EvalEverySteps": EVAL_EVERY_STEPS,
        "Dec_DefaultThresh": DEFAULT_THRESH,
        "Dec_ContainThr": DEFAULT_CONTAIN_THR,
        "Dec_NMS_IoU": DEFAULT_NMS_IOU_THR,
        "Dec_MergeGapChars": DEFAULT_MERGE_GAP_CHARS,
        "Eval_MatchIoU": MATCH_IOU_THR,
        "PosWeightClipMax": POS_WEIGHT_CLIP_MAX,
        "TuneThresholds": TUNE_THRESHOLDS,
        "UseFocalLoss": USE_FOCAL_LOSS,
        "FocalGamma": FOCAL_GAMMA if USE_FOCAL_LOSS else None,
        "RoleSpecificDecoding": True,
    }

    print("[config]")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"[pos_weight] {pos_weight.detach().cpu().numpy().round(2).tolist()}")

    # V5: Start with lower default thresholds
    thresholds = {
        "Actor": 0.40,
        "Action": 0.35,
        "Effect": 0.35,
        "Evidence": 0.35,
        "Victim": 0.30,
    }

    best_decoded = -1.0
    best_metrics: Dict = {}

    global_step = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True)

        running_loss = 0.0
        running_count = 0

        for batch in pbar:
            global_step += 1

            # V5: Sanity check - catch collate bugs early
            if global_step == 1 or global_step % 100 == 0:
                assert batch.spans.shape[:2] == batch.labels.shape[:2], \
                    f"Shape mismatch: spans {batch.spans.shape[:2]} vs labels {batch.labels.shape[:2]}"
                assert batch.span_mask.shape == batch.spans.shape[:2], \
                    f"Shape mismatch: span_mask {batch.span_mask.shape} vs spans {batch.spans.shape[:2]}"

            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            spans = batch.spans.to(device)
            labels = batch.labels.to(device)
            span_mask = batch.span_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=FP16):
                logits = model(input_ids, attention_mask, spans)
                real_logits = logits[span_mask]
                real_labels = labels[span_mask]
                loss = loss_fct(real_logits, real_labels)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            lr_now = scheduler.get_last_lr()[0]

            running_loss += float(loss.item())
            running_count += 1

            pbar.set_postfix(loss=float(loss.item()), lr=lr_now)

            if LOG_EVERY_STEPS and (global_step % LOG_EVERY_STEPS == 0):
                epoch_frac = (epoch - 1) + (global_step - (epoch - 1) * steps_per_epoch) / steps_per_epoch
                log = {
                    "loss": round(float(running_loss / max(1, running_count)), 4),
                    "grad_norm": float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                    "learning_rate": lr_now,
                    "epoch": round(epoch_frac, 2),
                    "step": global_step,
                }
                print(log)
                running_loss = 0.0
                running_count = 0

            if EVAL_EVERY_STEPS and (global_step % EVAL_EVERY_STEPS == 0):
                if TUNE_THRESHOLDS:
                    print(f"[eval @ step {global_step}] Tuning thresholds...")
                    thresholds = tune_thresholds(model, val_loader, device)

                metrics = evaluate(model, val_loader, device, thresholds=thresholds)

                print(
                    f"[eval @ step {global_step}] "
                    f"decoded_micro_f1={metrics['decoded_micro_f1']:.4f} "
                    f"cand_micro_f1={metrics['cand_micro_f1']:.4f} "
                    f"n_ex={metrics['n_examples_eval']} n_spans={metrics['n_spans_eval']}"
                )

                for r in ROLES:
                    df1 = metrics["decoded_role_prf"][r]["f1"]
                    cf1 = metrics["cand_role_f1"][r]
                    print(f"  {r:<8} decodedF1={df1:.4f} candF1={cf1:.4f} thr={metrics['thresholds'][r]:.2f}")

                if metrics["decoded_micro_f1"] > best_decoded:
                    best_decoded = metrics["decoded_micro_f1"]
                    best_metrics = metrics

                    ckpt_dir = os.path.join(OUTPUT_DIR, "best")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
                    tokenizer.save_pretrained(ckpt_dir)
                    with open(os.path.join(ckpt_dir, "metrics.json"), "w", encoding="utf-8") as f:
                        json.dump(metrics, f, indent=2)
                    # Save thresholds separately for easy access
                    with open(os.path.join(ckpt_dir, "best_thresholds.json"), "w", encoding="utf-8") as f:
                        json.dump(thresholds, f, indent=2)

                    print(f"[save] new best checkpoint -> {ckpt_dir}")

        # End-of-epoch eval
        if TUNE_THRESHOLDS:
            print(f"\n[epoch {epoch}] Tuning thresholds...")
            thresholds = tune_thresholds(model, val_loader, device)

        metrics = evaluate(model, val_loader, device, thresholds=thresholds)

        print(
            f"\n[epoch {epoch}] "
            f"decoded_micro_f1={metrics['decoded_micro_f1']:.4f} "
            f"cand_micro_f1={metrics['cand_micro_f1']:.4f} "
            f"n_ex={metrics['n_examples_eval']} n_spans={metrics['n_spans_eval']}"
        )
        for r in ROLES:
            df1 = metrics["decoded_role_prf"][r]["f1"]
            cf1 = metrics["cand_role_f1"][r]
            print(f"  {r:<8} decodedF1={df1:.4f} candF1={cf1:.4f} thr={metrics['thresholds'][r]:.2f}")

        if metrics["decoded_micro_f1"] > best_decoded:
            best_decoded = metrics["decoded_micro_f1"]
            best_metrics = metrics

            ckpt_dir = os.path.join(OUTPUT_DIR, "best")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
            tokenizer.save_pretrained(ckpt_dir)
            with open(os.path.join(ckpt_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            # Save thresholds separately for easy access
            with open(os.path.join(ckpt_dir, "best_thresholds.json"), "w", encoding="utf-8") as f:
                json.dump(thresholds, f, indent=2)

            print(f"[save] new best checkpoint -> {ckpt_dir}")

    total_time = time.time() - start_time
    if not best_metrics:
        best_metrics = metrics

    print_training_summary(config, best_metrics, pos_weight, total_time)


if __name__ == "__main__":
    main()