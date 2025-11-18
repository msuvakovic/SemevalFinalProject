# corrected_train_one_span_bio.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import json
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from transformers.modeling_outputs import TokenClassifierOutput

from sklearn.metrics import precision_recall_fscore_support, f1_score
from seqeval.metrics import f1_score as span_f1

# ---- utils you already have ----
from utils_bio import (
    to_bio_from_binary,
    align_bio_to_wordpieces,
    build_bio_label_maps,
    compute_class_weights_for_ce,
)

# ------------------
# CONFIG
# ------------------
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_ROOT = "./outputs_bio"
LABEL_ALL_TOKENS = True
SEED = 42


# ------------------
# TOKENIZATION PIPELINE
# ------------------
def make_tokenize_fn(tokenizer, label2id, role: str):
    def _fn(batch):
        all_enc = {"input_ids": [], "attention_mask": [], "labels": [], "_word_ids": []}
        for words, bin_tags in zip(batch["tokens"], batch["tags"]):
            # 1) Binary → BIO at word-level
            bio_tags = to_bio_from_binary(words, bin_tags, role)

            # 2) Word-level BIO → wordpieces aligned labels
            enc = align_bio_to_wordpieces(
                words,
                bio_tags,
                tokenizer,
                label2id,
                label_all_tokens=LABEL_ALL_TOKENS,
            )
            for k in ("input_ids", "attention_mask", "labels", "_word_ids"):
                all_enc[k].append(enc[k])
        return all_enc

    return _fn


# ------------------
# METRICS
# ------------------
def make_compute_metrics(id2label: Dict[int, str]):
    def _compute_metrics(eval_pred):
        preds, labels = eval_pred
        pred_ids = np.argmax(preds, axis=-1)

        # Token-level (sklearn): drop -100
        mask = labels != -100
        y_true = labels[mask].ravel()
        y_pred = pred_ids[mask].ravel()

        token_f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0.0)
        prec, rec, token_f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0.0
        )

        # Span-level (seqeval): derive string sequences by removing -100
        y_true_seqs, y_pred_seqs = [], []
        B, T = labels.shape
        for i in range(B):
            true_seq_ids, pred_seq_ids = [], []
            for j in range(T):
                if labels[i, j] == -100:
                    continue
                true_seq_ids.append(labels[i, j])
                pred_seq_ids.append(pred_ids[i, j])
            y_true_seqs.append([id2label[int(x)] for x in true_seq_ids])
            y_pred_seqs.append([id2label[int(x)] for x in pred_seq_ids])

        span_f1_score = span_f1(y_true_seqs, y_pred_seqs)

        return {
            "token_f1_macro": float(token_f1_macro),
            "token_prec_micro": float(prec),
            "token_rec_micro": float(rec),
            "span_f1": float(span_f1_score),
        }

    return _compute_metrics


# ------------------
# INSPECTION
# ------------------
def inspect_examples(trainer: Trainer, id2label: Dict[int, str], tokenizer, ds_eval, num_examples=5):
    model = trainer.model
    model.eval()

    # Get device from any parameter (works with CPU, single GPU, DataParallel, etc.)
    device = next(model.parameters()).device
    count = 0

    print("\n[inspect] Showing a few eval examples (first-subword tokens, gold vs pred BIO):")
    for ex in ds_eval.select(range(min(len(ds_eval), 200))):
        inputs = {
            "input_ids": torch.tensor([ex["input_ids"]], device=device),
            "attention_mask": torch.tensor([ex["attention_mask"]], device=device),
        }
        with torch.no_grad():
            outputs = model(**inputs)
            # Our model.forward will return a dict; but be defensive:
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs.logits
            pred_ids = logits.argmax(-1)[0].cpu().tolist()

        gold_ids = [lid for lid in ex["labels"] if lid != -100]
        pred_ids_effective = [pid for pid, lid in zip(pred_ids, ex["labels"]) if lid != -100]

        gold_bio = [id2label[i] for i in gold_ids]
        pred_bio = [id2label[i] for i in pred_ids_effective]

        # Reconstruct readable first-subword tokens
        tokens = []
        prev_wid = None
        for tid, wid in zip(ex["input_ids"], ex["_word_ids"]):
            if wid is None:
                continue
            if wid != prev_wid:
                tokens.append(tokenizer.convert_ids_to_tokens([tid])[0])
            prev_wid = wid

        print("=" * 80)
        print("TEXT (approx):", " ".join(tokens[:120]))
        print("GOLD BIO:     ", " ".join(gold_bio[:120]))
        print("PRED BIO:     ", " ".join(pred_bio[:120]))

        count += 1
        if count >= num_examples:
            break

# ------------------
# MODEL WITH CLASS WEIGHTS
# ------------------
class WeightedTokenClassificationModel(nn.Module):
    def __init__(self, base_model_name: str, num_labels: int, class_weights: List[float] | None):
        super().__init__()
        self.base = AutoModelForTokenClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
            problem_type="single_label_classification",
        )
        self.register_buffer(
            "loss_weights",
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None,
            persistent=False,
        )
        self.loss_fct = nn.CrossEntropyLoss(
            weight=self.loss_weights if class_weights is not None else None,
            ignore_index=-100,
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Ignore extra kwargs from Trainer (e.g., num_items_in_batch)
        outputs = self.base.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (B, T, H)
        logits = self.base.classifier(sequence_output)  # (B, T, C)

        loss = None
        if labels is not None:
            loss = self.loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

        # Return a plain dict so Trainer/DataParallel are happy
        return {"loss": loss, "logits": logits}


# ------------------
# TRAIN ONE ROLE
# ------------------
def train_one_role(
    role: str,
    datasets: DatasetDict,
    output_dir: str,
    learning_rate: float = 2e-5,
    epochs: int = 5,
    per_device_batch_size: int = 8,
):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    label2id, id2label = build_bio_label_maps(role)
    print(f"[{role}] label2id:", label2id)

    tokenize_fn = make_tokenize_fn(tokenizer, label2id, role)
    ds_train = datasets["train"].map(
        tokenize_fn, batched=True, remove_columns=datasets["train"].column_names
    )
    ds_eval = datasets["validation"].map(
        tokenize_fn, batched=True, remove_columns=datasets["validation"].column_names
    )

    # Class weights from TRAIN labels
    flat_train_labels = [lid for row in ds_train["labels"] for lid in row if lid != -100]
    class_weights = compute_class_weights_for_ce(flat_train_labels, num_labels=len(label2id))
    print(f"[{role}] class weights:", class_weights)

    model = WeightedTokenClassificationModel(
        MODEL_NAME, num_labels=len(label2id), class_weights=class_weights
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    metrics_fn = make_compute_metrics(id2label)

    args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"{role}_bio"),
        seed=SEED,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=25,          # log every N steps
        save_steps=1000,           # save every N steps (you can tune this)
        fp16=torch.cuda.is_available(),  # mixed precision if GPU supports it
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics_fn,
    )

    print(f"\n--- Training (BIO) for role: {role} ---")
    trainer.train()
    print(f"[{role}] best model loaded. Evaluating...")
    eval_out = trainer.evaluate()
    print(f"[{role}] eval metrics:", eval_out)

    inspect_examples(trainer, id2label, tokenizer, ds_eval, num_examples=3)

    # Save best model
    best_dir = os.path.join(output_dir, f"{role}_bio", "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"[{role}] saved to: {best_dir}")

    return trainer


# ------------------
# JSONL → DATASET HELPERS
# ------------------
def _detect_roles(json_path: str) -> List[str]:
    roles = set()
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            for m in ex.get("markers", []):
                t = m.get("type")
                if isinstance(t, str) and len(t.strip()) > 0:
                    roles.add(t.strip())
    roles = sorted(roles)
    if not roles:
        raise RuntimeError("No roles detected in JSONL (markers[].type).")
    return roles


def _load_from_jsonl_to_binary(role: str, filepath: str) -> DatasetDict:
    """
    Convert 'markers' to token-level binary tags (O or role).
    Tokenization is regex word/punct; tags assigned by span overlap.
    """
    tokens_list, tags_list = [], []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            text = ex["text"]
            markers = ex.get("markers", [])

            # regex tokenize: words or punctuation as standalone
            words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

            # build char spans for each token (left-to-right search)
            spans = []
            cursor = 0
            for w in words:
                start = text.find(w, cursor)
                if start < 0:
                    # extremely rare fallback: skip token
                    continue
                end = start + len(w)
                spans.append((start, end))
                cursor = end

            # assign O / role by overlap with marker spans
            tags = ["O"] * len(spans)
            for m in markers:
                if m.get("type") != role:
                    continue
                m_start, m_end = m["startIndex"], m["endIndex"]
                for i, (s, e) in enumerate(spans):
                    if not (e <= m_start or s >= m_end):  # any overlap
                        tags[i] = role

            tokens_list.append(words)
            tags_list.append(tags)

    dataset = Dataset.from_dict({"tokens": tokens_list, "tags": tags_list})
    split = dataset.train_test_split(test_size=0.1, seed=SEED)
    return DatasetDict(train=split["train"], validation=split["test"])


# ------------------
# MAIN
# ------------------
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    json_path = "train_rehydrated.jsonl"  # <- make sure this exists in cwd
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Could not find {json_path}. Put your rehydrated JSONL in the working directory."
        )

    print("[main] scanning roles in:", json_path)
    roles = _detect_roles(json_path)
    print("[main] detected roles:", roles)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for role in roles:
        print("\n" + "=" * 70)
        print(f"[main] Training role: {role}")
        print("=" * 70)

        datasets = _load_from_jsonl_to_binary(role, json_path)
        print(f"[{role}] #train={len(datasets['train'])}  #val={len(datasets['validation'])}")

        _ = train_one_role(
            role=role,
            datasets=datasets,
            output_dir=OUTPUT_ROOT,
            learning_rate=2e-5,
            epochs=5,                    # adjust as needed
            per_device_batch_size=8,     # adjust for GPU RAM
        )

    print("\n[main] All roles finished training.")
