# corrected_train_one_span_bio.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import json
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torchcrf import CRF


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
LABEL_ALL_TOKENS = False
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
# ------------------
# METRICS (FULL, CORRECTED)
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

        # Handle tags like 'B-Action' or 'I-Action'
        if "-" in t:
            prefix, typ = t.split("-", 1)
        else:
            # Should not happen in standard BIO, but for safety:
            fixed.append(t)
            prev = t
            prev_type = None
            continue

        if prefix == "B":
            fixed.append(t)
            prev = t
            prev_type = typ
            continue

        # prefix == "I"
        if prev == "O" or prev_type != typ:
            # An 'I-' tag cannot follow 'O' or a different type. Correct it to 'B-'.
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
        # eval_pred is (preds, labels).
        # preds: np.ndarray (B, T, C) [from Logits] OR np.ndarray (B, T) [from fixed CRFTrainer]
        # labels: np.ndarray (B, T)
        preds, labels = eval_pred

        # Convert labels to numpy array for indexing/masking
        labels = np.asarray(labels)

        # 1. Determine Predicted IDs (B, T)
        preds = np.asarray(preds)
        
        if preds.ndim == 3:
            # Path for standard token classification (Logits B, T, C)
            pred_ids = np.argmax(preds, axis=-1)
        elif preds.ndim == 2:
            # Path for CRFTrainer (Padded IDs B, T) or direct ID output
            pred_ids = preds 
        else:
            raise ValueError(f"Unexpected preds shape: {preds.shape}. Expected 2D or 3D array.")

        # 2. Extract non-padded ID sequences based on the labels mask
        B, T = labels.shape
        y_pred_seqs_ids = [] # List[List[int]]: non-padded predicted ID sequences
        y_true_seqs_ids = [] # List[List[int]]: non-padded true ID sequences
        
        for i in range(B):
            # The mask is defined by labels != -100 (token has a gold label)
            mask_indices = [j for j in range(T) if labels[i, j] != -100]
            
            true_seq_ids = [labels[i, j] for j in mask_indices]
            pred_seq_ids = [pred_ids[i, j] for j in mask_indices] 
            
            y_pred_seqs_ids.append(pred_seq_ids)
            y_true_seqs_ids.append(true_seq_ids)


        # --- 1. Token-level Metrics ---
        # Flatten all non-padded true and predicted IDs
        y_true_token = np.array([tid for seq in y_true_seqs_ids for tid in seq])
        y_pred_token = np.array([pid for seq in y_pred_seqs_ids for pid in seq])

        # Sanity check: lengths MUST match!
        if len(y_true_token) != len(y_pred_token):
             raise ValueError(
                 f"Token sequence length mismatch! True: {len(y_true_token)}, Pred: {len(y_pred_token)}. "
                 "CRF prediction step or CE ID extraction is faulty."
             )
            
        o_id = 0
        gold_o_rate = float((y_true_token == o_id).mean())
        pred_o_rate = float((y_pred_token == o_id).mean())

        token_f1_macro = f1_score(y_true_token, y_pred_token, average="macro", zero_division=0.0)
        prec, rec, _, _ = precision_recall_fscore_support(
             y_true_token, y_pred_token, average="micro", zero_division=0.0
        )

        # --- 2. Span-level Metrics (Seqeval) ---
        y_true_seqs = []
        for seq_ids in y_true_seqs_ids:
             y_true_seqs.append([id2label[int(x)] for x in seq_ids])

        # Map predicted IDs to tags and apply BIO fix
        y_pred_seqs = []
        for seq_ids in y_pred_seqs_ids:
             pred_tags = [id2label[int(x)] for x in seq_ids]
             # Apply the BIO fix to ensure correct span counting for seqeval
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
# ------------------
# INSPECTION (CORRECTED)
# ------------------
def inspect_examples(trainer: Trainer, id2label: Dict[int, str], tokenizer, ds_eval, num_examples=5):
    model = trainer.model
    model.eval()

    # Ensure we use the CRF model if it's the one we trained
    if not isinstance(model, DistilBertTokenCRF):
        print("[inspect] Model is not DistilBertTokenCRF. Falling back to Argmax.")
    
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
            
            # Extract emissions (logits) and mask
            emissions = outputs.logits
            attention_mask = inputs["attention_mask"].bool()
            
            # --- VITERBI DECODING ---
            if isinstance(model, DistilBertTokenCRF):
                # Perform Viterbi decoding using the CRF layer
                # model.crf.decode returns a list of list[int] (non-padded IDs)
                pred_seq_ids = model.crf.decode(emissions=emissions, mask=attention_mask)[0] 
                # pad pred_ids back to sequence length for zip/alignment below
                pred_ids = [0] * len(ex["input_ids"]) # Initialize to 'O'
                idx = torch.where(attention_mask[0])[0].cpu().tolist()
                for i_seq, i_idx in enumerate(idx):
                     if i_idx < len(pred_ids) and i_seq < len(pred_seq_ids):
                         pred_ids[i_idx] = pred_seq_ids[i_seq]
            else:
                # Fallback to simple argmax (CE model)
                pred_ids = emissions.argmax(-1)[0].cpu().tolist()
                
        # The remaining logic extracts the non-padded tokens for comparison
        gold_ids = [lid for lid in ex["labels"] if lid != -100]
        # Align predicted IDs with gold non-padded labels
        pred_ids_effective = [pid for pid, lid in zip(pred_ids, ex["labels"]) if lid != -100]

        gold_bio = [id2label[i] for i in gold_ids]
        pred_bio = _fix_bio_sequence([id2label[i] for i in pred_ids_effective])


        # Reconstruct readable first-subword tokens
        # ... (rest of inspect_examples remains the same)

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
        # Run the underlying transformer
        base_out = self.base.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = base_out[0]  # (B, T, H)
        logits = self.base.classifier(sequence_output)  # (B, T, C)

        loss = None
        if labels is not None:
            loss = self.loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

        # Return TokenClassifierOutput so Trainer is perfectly happy
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=base_out.hidden_states if hasattr(base_out, "hidden_states") else None,
            attentions=base_out.attentions if hasattr(base_out, "attentions") else None,
        )

class DistilBertTokenCRF(nn.Module):
    def __init__(self, base_model_name: str, num_labels: int):
        super().__init__()
        self.base = AutoModelForTokenClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels,
            problem_type="single_label_classification",
        )
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        base_out = self.base.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = base_out[0]                    # (B,T,H)
        emissions = self.base.classifier(sequence_output) # (B,T,C)

        loss = None
        if labels is not None:
            # CRF cannot consume -100 labels; mask them out and replace with 0
            mask = attention_mask.bool()



            labels_fixed = labels.clone()
            labels_fixed[labels_fixed == -100] = 0

            # torchcrf returns log-likelihood; we minimize negative log-likelihood
            loss = -self.crf(emissions, labels_fixed, mask=mask, reduction="mean")

        return TokenClassifierOutput(loss=loss, logits=emissions)



class CRFTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            labels = inputs.get("labels")
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels, # Note: loss is still calculated, but not used for pred step
            )
            loss = outputs.loss # Scalar tensor

            emissions = outputs.logits  # (B,T,C)
            mask = inputs["attention_mask"].bool()

            # 1. Viterbi decoding
            # pred_seqs: List[List[int]] (non-padded)
            pred_seqs = model.crf.decode(emissions, mask=mask) 
            
            # --- START FIX ---

            # 2. Pad the Viterbi output sequences to the batch's max sequence length (T)
            # Find the max sequence length (T) in this batch
            max_len = emissions.size(1) 
            
            # Pad sequences to max_len. Use -100 as the padding ID, 
            # which is what the labels tensor uses for masked values.
            padded_preds = []
            for seq in pred_seqs:
                # Pad with -100 to match the length of the batch's padded tensors
                padded_seq = seq + [-100] * (max_len - len(seq))
                padded_preds.append(padded_seq)

            # 3. Convert the list of padded lists back to a PyTorch Tensor
            # Convert to a tensor on the same device as the model/inputs
            preds_tensor = torch.tensor(padded_preds, dtype=torch.long, device=emissions.device)

            # --- END FIX ---


        if prediction_loss_only:
            return (loss, None, None)
        
        # Return the PADDED PREDICTIONS TENSOR and the true labels tensor
        return (loss.detach(), preds_tensor, labels)

# Note: The logic in make_compute_metrics will need a slight adjustment 
# to now handle pred_ids as a padded tensor instead of a list of lists. 
# It's an array of (B, T) now, but the standard CE path already handles this!

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
    # Class weights from TRAIN labels  (TEMP: disable to debug span F1)
    flat_train_labels = [lid for row in ds_train["labels"] for lid in row if lid != -100]
    class_weights = compute_class_weights_for_ce(flat_train_labels, num_labels=len(label2id))
    class_weights[0] = max(class_weights[0], 0.2)
    #if len(class_weights) == 3:
    #    class_weights[2] = class_weights[1]
    print(f"[{role}] class weights (USED):", class_weights)


    # DEBUG: pass None to disable class weighting
    #model = WeightedTokenClassificationModel(
    #    MODEL_NAME, num_labels=len(label2id), class_weights=None
    #)

    # Class weights from TRAIN labels
    #flat_train_labels = [lid for row in ds_train["labels"] for lid in row if lid != -100]
    #class_weights = compute_class_weights_for_ce(flat_train_labels, num_labels=len(label2id))
    #print(f"[{role}] class weights:", class_weights)

    model = DistilBertTokenCRF(
        MODEL_NAME, num_labels=len(label2id)
    )



    base_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    data_collator = CollatorDropWordIds(base_collator)

    metrics_fn = make_compute_metrics(id2label)

    args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"{role}_bio"),
        seed=SEED,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=25,

        # Evaluate regularly & select best based on span F1
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="span_f1",
        greater_is_better=True,
        save_total_limit=1,

        fp16=torch.cuda.is_available(),
        # Optional but nice with custom columns:
        remove_unused_columns=False,
    )

    trainer = CRFTrainer(
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
    Tokenization is regex word/punct; tags assigned by token overlap with marker spans.
    Uses re.finditer so token spans are exact (no text.find cursor issues).
    """
    tokens_list, tags_list = [], []

    pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def overlap_ratio(s, e, ms, me):
        inter = max(0, min(e, me) - max(s, ms))
        tok_len = max(1, e - s)
        return inter / tok_len

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            ex = json.loads(line)
            text = ex["text"]
            markers = ex.get("markers", [])

            # 1) Tokenize + exact char spans
            words = []
            spans = []
            for m_tok in pattern.finditer(text):
                words.append(m_tok.group(0))
                spans.append((m_tok.start(), m_tok.end()))

            # 2) Assign binary tags aligned to tokens
            tags = ["O"] * len(words)

            for m in markers:
                if m.get("type") != role:
                    continue
                ms, me = m["startIndex"], m["endIndex"]

                # Conservative overlap: require at least 50% of token covered
                for i, (s, e) in enumerate(spans):
                    if overlap_ratio(s, e, ms, me) >= 0.5:
                        tags[i] = role

            # 3) Sanity check (optional but nice)
            if len(words) != len(tags):
                raise ValueError(f"Length mismatch words({len(words)}) vs tags({len(tags)})")

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
            learning_rate=1e-5,
            epochs=5,                    # adjust as needed
            per_device_batch_size=8,     # adjust for GPU RAM
        )

    print("\n[main] All roles finished training.")
    print("\n ")