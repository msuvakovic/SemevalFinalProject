# corrected_train_one_span_bio_no_crf_roberta.py
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
MODEL_NAME = "roberta-base"     # <-- RoBERTa
OUTPUT_ROOT = "./outputs_bio_roberta"
LABEL_ALL_TOKENS = True       # try True if span-F1 is stuck
SEED = 42


# ------------------
# TOKENIZATION PIPELINE
# ------------------
def make_tokenize_fn(tokenizer, label2id, role: str):
    def _fn(batch):
        all_enc = {"input_ids": [], "attention_mask": [], "labels": [], "_word_ids": []}
        for words, bin_tags in zip(batch["tokens"], batch["tags"]):
            bio_tags = to_bio_from_binary(words, bin_tags, role)

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
def _fix_bio_sequence(tags):
    fixed = []
    prev = "O"
    prev_type = None
    for t in tags:
        if t == "O":
            fixed.append("O")
            prev = "O"
            prev_type = None
            continue

        prefix, typ = t.split("-", 1)

        if prefix == "B":
            fixed.append(t)
            prev = t
            prev_type = typ
            continue

        # prefix == "I"
        if prev == "O" or prev_type != typ:
            fixed.append("B-" + typ)
            prev = "B-" + typ
            prev_type = typ
        else:
            fixed.append(t)
            prev = t
            prev_type = typ
    return fixed


def make_compute_metrics(id2label: Dict[int, str]):
    def _compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.asarray(preds)
        labels = np.asarray(labels)

        # logits (B,T,C) -> argmax
        if preds.ndim == 3:
            pred_ids = np.argmax(preds, axis=-1)
        # already ids (B,T)
        elif preds.ndim == 2:
            pred_ids = preds
        else:
            raise ValueError(f"Unexpected preds shape: {preds.shape}")

        # Token-level: drop -100
        mask = labels != -100
        y_true = labels[mask].ravel()
        y_pred = pred_ids[mask].ravel()

        o_id = 0
        gold_o_rate = float((y_true == o_id).mean())
        pred_o_rate = float((y_pred == o_id).mean())

        token_f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0.0)
        prec, rec, _, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0.0
        )

        # Span-level (seqeval): remove -100 positions
        y_true_seqs, y_pred_seqs = [], []
        B, T = labels.shape
        for i in range(B):
            true_seq_ids, pred_seq_ids = [], []
            for j in range(T):
                if labels[i, j] == -100:
                    continue
                true_seq_ids.append(labels[i, j])
                pred_seq_ids.append(pred_ids[i, j])

            true_tags = [id2label[int(x)] for x in true_seq_ids]
            pred_tags = [id2label[int(x)] for x in pred_seq_ids]

            y_true_seqs.append(true_tags)
            y_pred_seqs.append(_fix_bio_sequence(pred_tags))

        span_f1_score = span_f1(y_true_seqs, y_pred_seqs)

        return {
            "token_f1_macro": float(token_f1_macro),
            "token_prec_micro": float(prec),
            "token_rec_micro": float(rec),
            "span_f1": float(span_f1_score),
            "gold_o_rate": gold_o_rate,
            "pred_o_rate": pred_o_rate,
        }

    return _compute_metrics


# ------------------
# INSPECTION
# ------------------
def inspect_examples(trainer: Trainer, id2label: Dict[int, str], tokenizer, ds_eval, num_examples=5):
    model = trainer.model
    model.eval()
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
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            pred_ids = logits.argmax(-1)[0].cpu().tolist()

        gold_ids = [lid for lid in ex["labels"] if lid != -100]
        pred_ids_effective = [pid for pid, lid in zip(pred_ids, ex["labels"]) if lid != -100]

        gold_bio = [id2label[i] for i in gold_ids]
        pred_bio = _fix_bio_sequence([id2label[i] for i in pred_ids_effective])

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
# DATA COLLATOR WRAPPER
# ------------------
class CollatorDropWordIds:
    def __init__(self, base_collator):
        self.base = base_collator

    def __call__(self, features):
        for f in features:
            f.pop("_word_ids", None)
        return self.base(features)


# ------------------
# MODEL WITH CLASS WEIGHTS (NO CRF)
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
        base_out = self.base.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = base_out[0]                   # (B,T,H)
        logits = self.base.classifier(sequence_output)  # (B,T,C)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits)


def trunc_stats(ds, tokenizer, role):
    maxlen = getattr(tokenizer, "model_max_length", 512)
    lens = [sum(1 for x in row if x != 0) for row in ds["attention_mask"]]
    hit = sum(l >= maxlen for l in lens)
    print(f"[{role}] avg_len={np.mean(lens):.1f}  max_len={np.max(lens)}  hit_max={hit}/{len(lens)}")


# ------------------
# TRAIN ONE ROLE
# ------------------
def train_one_role(
    role: str,
    datasets: DatasetDict,
    output_dir: str,
    learning_rate: float = 3e-5,
    epochs: int = 5,
    per_device_batch_size: int = 8,
):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True, use_fast=True)


    label2id, id2label = build_bio_label_maps(role)
    print(f"[{role}] label2id:", label2id)

    tokenize_fn = make_tokenize_fn(tokenizer, label2id, role)
    ds_train = datasets["train"].map(tokenize_fn, batched=True, remove_columns=datasets["train"].column_names)
    ds_eval = datasets["validation"].map(tokenize_fn, batched=True, remove_columns=datasets["validation"].column_names)

    trunc_stats(ds_train, tokenizer, role)
    trunc_stats(ds_eval, tokenizer, role)

    # Class weights from TRAIN labels
    flat_train_labels = [lid for row in ds_train["labels"] for lid in row if lid != -100]
    class_weights = compute_class_weights_for_ce(flat_train_labels, num_labels=len(label2id))
    print(f"[{role}] class weights (USED):", class_weights)

    model = WeightedTokenClassificationModel(
        MODEL_NAME,
        num_labels=len(label2id),
        class_weights=None,  # IMPORTANT: remove weights
    )


    base_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    data_collator = CollatorDropWordIds(base_collator)

    metrics_fn = make_compute_metrics(id2label)

    args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"{role}_bio_no_crf"),
        seed=SEED,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=25,

        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="span_f1",
        greater_is_better=True,
        save_total_limit=1,

        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,

        # usually helps stability
        warmup_ratio=0.1,
        max_grad_norm=1.0,
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

    print(f"\n--- Training (BIO, NO CRF) for role: {role} ---")
    trainer.train()

    print(f"[{role}] best model loaded. Evaluating...")
    eval_out = trainer.evaluate()
    print(f"[{role}] eval metrics:", eval_out)

    inspect_examples(trainer, id2label, tokenizer, ds_eval, num_examples=3)

    best_dir = os.path.join(output_dir, f"{role}_bio_no_crf", "best")
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
    tokens_list, tags_list = [], []
    pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def overlap_ratio(s, e, ms, me):
        inter = max(0, min(e, me) - max(s, ms))
        tok_len = max(1, e - s)
        return inter / tok_len

    def span_stats(tags, positive_tag):
        # count spans + average span length + max span length
        spans = 0
        lengths = []
        i = 0
        while i < len(tags):
            if tags[i] == positive_tag:
                spans += 1
                j = i
                while j < len(tags) and tags[j] == positive_tag:
                    j += 1
                lengths.append(j - i)
                i = j
            else:
                i += 1
        avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0
        max_len = max(lengths) if lengths else 0
        return spans, avg_len, max_len

    # Aggregate sanity stats across file (so it's not just random one-offs)
    total_spans = 0
    total_span_len = 0.0
    span_len_count = 0
    max_span_len_global = 0
    examples_with_any_span = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            ex = json.loads(line)
            text = ex["text"]
            markers = ex.get("markers", [])

            words, spans = [], []
            for m_tok in pattern.finditer(text):
                words.append(m_tok.group(0))
                spans.append((m_tok.start(), m_tok.end()))

            tags = ["O"] * len(words)
            for m in markers:
                if m.get("type") != role:
                    continue
                ms, me = m["startIndex"], m["endIndex"]
                for i, (s, e) in enumerate(spans):
                    if overlap_ratio(s, e, ms, me) >= 0.5:
                        tags[i] = role

            # ---- per-example span sanity (occasionally print) ----
            s_cnt, s_avg, s_max = span_stats(tags, role)
            if s_cnt > 0:
                examples_with_any_span += 1
                total_spans += s_cnt
                # weight avg span length by number of spans in this example
                total_span_len += s_avg * s_cnt
                span_len_count += s_cnt
                max_span_len_global = max(max_span_len_global, s_max)

            if np.random.rand() < 0.002:
                print(f"[gold sanity {role}] spans={s_cnt} avg_span_len={s_avg:.2f} max_span_len={s_max} words={len(words)}")

            tokens_list.append(words)
            tags_list.append(tags)

    # ---- aggregate sanity print ----
    avg_span_len_global = (total_span_len / span_len_count) if span_len_count else 0.0
    avg_spans_per_pos_ex = (total_spans / examples_with_any_span) if examples_with_any_span else 0.0
    pos_rate = examples_with_any_span / max(1, len(tokens_list))

    print(
        f"[gold summary {role}] "
        f"pos_examples={examples_with_any_span}/{len(tokens_list)} ({pos_rate:.3f}) "
        f"avg_spans_per_pos_ex={avg_spans_per_pos_ex:.2f} "
        f"avg_span_len={avg_span_len_global:.2f} "
        f"max_span_len={max_span_len_global}"
    )

    dataset = Dataset.from_dict({"tokens": tokens_list, "tags": tags_list})
    split = dataset.train_test_split(test_size=0.1, seed=SEED)
    return DatasetDict(train=split["train"], validation=split["test"])



# ------------------
# MAIN
# ------------------
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    json_path = "train_rehydrated.jsonl"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find {json_path}. Put your rehydrated JSONL in the working directory.")

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
            learning_rate=3e-5,
            epochs=5,
            per_device_batch_size=8,
        )

    print("\n[main] All roles finished training.")
