"""  
Train a binary conspiracy detector with pre‑computed qwen3-embeddings.  

If the embeddings file (`embeddings.npy`) is missing, it will be created  
automatically by calling `embed_text()` on each example in the training data.  
"""  
import json  
import os  
import random  
from pathlib import Path  
import numpy as np  
import torch  
from torch import nn  
from torch.nn import TransformerEncoderLayer, TransformerEncoder  
from torch.utils.data import DataLoader  
from torch.optim.lr_scheduler import OneCycleLR  
from transformers import EarlyStoppingCallback  
from datasets import Dataset, DatasetDict  
from transformers import TrainingArguments, Trainer  
from sklearn.metrics import accuracy_score, f1_score  
from tqdm import tqdm

# ---------------------------------------------------------------------------- #  
# 1️⃣  Reproducibility  
# ---------------------------------------------------------------------------- #  
SEED = 42  
random.seed(SEED)  
np.random.seed(SEED)  
torch.manual_seed(SEED)  
if torch.cuda.is_available():  
    torch.cuda.manual_seed_all(SEED)  
  
# ---------------------------------------------------------------------------- #  
# 2️⃣  Load the Ollama client (still used for *embedding* if we need to  
#      recompute them)  
# ---------------------------------------------------------------------------- #  
try:  
    import ollama  
except ImportError as exc:  
    raise RuntimeError(  
        "You need the `ollama` python package to talk to your local Ollama daemon.\n"  
        "Install it with: pip install ollama\n"  
    ) from exc  
  
ollama_client = ollama.Client()  
  
# ---------------------------------------------------------------------------- #  
# 3️⃣  Helper: compute an embedding for a single string (kept for reference)  
# ---------------------------------------------------------------------------- #  
def embed_text(text: str) -> np.ndarray:  
    """  
    Ask Ollama to embed *text* with the 'qwen3-embedding' model.  
  
    Returns  
    -------  
    np.ndarray of shape (768,)  
    """  
    try:  
        response = ollama_client.embeddings(  
            model="qwen3-embedding",  
            prompt=text,  
        )  
    except Exception as exc:          # pragma: no cover  
        raise RuntimeError(  
            "Failed to get embeddings from Ollama.\n"  
            "Make sure the Ollama daemon is running and the model "  
            "'qwen3-embedding' is downloaded.\n"  
        ) from exc  
  
    return np.array(response.embedding, dtype=np.float32)  
  
# ---------------------------------------------------------------------------- #  
# 4️⃣  Load & filter the raw JSON‑lines data  
# ---------------------------------------------------------------------------- #  
def load_and_filter_data(file_path: str) -> list[dict]:  
    """  
    Load a JSON‑lines file and keep only examples where  
    `"conspiracy"` is `"Yes"` or `"No"`.  
    """  
    data = []  
    with open(file_path, "r", encoding="utf-8") as f:  
        for line in f:  
            try:  
                item = json.loads(line)  
                if "conspiracy" in item and item["conspiracy"] in ("Yes", "No"):  
                    data.append(item)  
            except json.JSONDecodeError:  
                print(f"⚠️  Skipping invalid JSON: {line[:30]!r}…")  
    return data  
  
# ---------------------------------------------------------------------------- #  
# 5️⃣  Create a HuggingFace Dataset, embed, encode, and split  
# ---------------------------------------------------------------------------- #  
def prepare_datasets(  
    raw_data: list[dict],  
    label_map: dict[str, int],  
    embeddings: np.ndarray | None = None,  
    test_ratio: float = 0.1,  
    seed: int = SEED,  
) -> DatasetDict:  
    """  
    1. Convert to Dataset.  
    2. Attach pre‑computed embeddings (if provided).  
    3. Detect the label column (conspiracy / label / etc.) and  
       map it to the integer column called `label`.  
    4. Shuffle & split into train / validation.  
    """  
    ds = Dataset.from_list(raw_data)  
  
    # 1️⃣ Attach embeddings ------------------------------------------------ #  
    if embeddings is not None:  
        if embeddings.shape[0] != len(raw_data):  
            raise RuntimeError(  
                f"Number of embeddings ({embeddings.shape[0]}) does not match "  
                f"the number of examples ({len(raw_data)})."  
            )  
        ds = ds.add_column("embedding", embeddings.tolist())  
    else:  
        # Fall back to live Ollama embeddings (rarely used)  
        def embed_fn(example):  
            example["embedding"] = embed_text(example["text"])  
            return example  
  
        ds = ds.map(embed_fn, batched=False)  
  
    # 2️⃣ Detect & encode the label ---------------------------------------- #  
    LABEL_KEY_VARIANTS = ("conspiracy", "label", "is_conspiracy", "class")  
  
    def encode_fn(example):  
        for key in LABEL_KEY_VARIANTS:  
            if key in example:  
                raw = example[key]  
                if isinstance(raw, str):  
                    if raw not in label_map:  
                        raise KeyError(  
                            f"Unknown label value '{raw}' in field '{key}'. "  
                            f"Expected one of {list(label_map.keys())}."  
                        )  
                    example["label"] = label_map[raw]  
                else:  
                    example["label"] = int(raw)  
                break  
        else:  
            raise KeyError(  
                f"No known label field found in example. "  
                f"Expected one of {LABEL_KEY_VARIANTS}."  
            )  
        return example  
  
    ds = ds.map(encode_fn, batched=False)  
  
    if "label" not in ds.column_names:  
        raise RuntimeError(  
            "prepare_datasets() failed to add a 'label' column. "  
            "Check your data – it might not contain any of "  
            f"{LABEL_KEY_VARIANTS}."  
        )  
  
    # 3️⃣ Shuffle & split ------------------------------------------------- #  
    return ds.train_test_split(test_size=test_ratio, seed=seed, shuffle=True)  

# ---------------------------------------------------------------------------- #  
# 6️⃣  Tiny PyTorch classifier  
# ---------------------------------------------------------------------------- #  
class EmbeddingClassifier(nn.Module):  
    """Linear layer: 768‑dim (or whatever) embedding ➜ 2 logits."""  
  
    def __init__(self, embedding_dim: int, num_labels: int):  
        super().__init__()  
        self.fc = nn.Linear(embedding_dim, num_labels)  
        self.loss_fct = nn.CrossEntropyLoss()  
  
    def forward(self, embedding: torch.Tensor, labels=None):  
        logits = self.fc(embedding)  
        output = {"logits": logits}  
        if labels is not None:  
            loss = self.loss_fct(logits, labels)  
            output["loss"] = loss  
        return output  

class MLPClassifier(nn.Module):
    def __init__(self, embedding_dim, num_labels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, embedding, labels=None):
        logits = self.net(embedding)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = self.loss_fct(logits, labels)
        return out

class TinyTransformerHead(nn.Module):
    def __init__(self, embedding_dim, num_labels):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim, nhead=8, dim_feedforward=256
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embedding_dim, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, embedding, labels=None):
        # reshape (B,D) → (1,B,D) → (1,B,D) → (B,D)
        seq = embedding.unsqueeze(0)      # (1, B, D)
        encoded = self.encoder(seq).squeeze(0)
        logits = self.fc(encoded)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = self.loss_fct(logits, labels)
        return out

class BiggerMLP(nn.Module):  
    def __init__(self, embedding_dim, num_labels):  
        super().__init__()  
        self.net = nn.Sequential(  
            nn.Linear(embedding_dim, 512),  
            nn.ReLU(),  
            nn.Dropout(0.3),  
            nn.Linear(512, 256),  
            nn.ReLU(),  
            nn.Dropout(0.3),  
            nn.Linear(256, num_labels)  
        )  
        self.loss_fct = nn.CrossEntropyLoss()  
  
    def forward(self, embedding, labels=None):  
        logits = self.net(embedding)  
        out = {"logits": logits}  
        if labels is not None:  
            out["loss"] = self.loss_fct(logits, labels)  
        return out  

# ---------------------------------------------------------------------------- #  
# 7️⃣  Data collator – turn list of examples into a batch  
# ---------------------------------------------------------------------------- #  
def collate_fn(batch):  
    embeddings = torch.tensor(  
        [b["embedding"] for b in batch], dtype=torch.float32  
    )  
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)  
    return {"embedding": embeddings, "labels": labels}  
  
# ---------------------------------------------------------------------------- #  
# 8️⃣  Metric calculation – weighted‑F1 & accuracy  
# ---------------------------------------------------------------------------- #  
def compute_metrics(p):  
    preds = np.argmax(p.predictions, axis=1)  
    labels = p.label_ids  
  
    f1_w = f1_score(labels, preds, average="weighted", zero_division=0)  
    acc = accuracy_score(labels, preds)  
  
    return {"accuracy": acc, "f1_weighted": f1_w}  
  
# ---------------------------------------------------------------------------- #  
# 9️⃣  Save predictions in the same JSON format as eval_binary.py  
# ---------------------------------------------------------------------------- #  
def save_predictions(  
    trainer: Trainer,  
    dataset: Dataset,  
    out_file: str,  
    id2label: dict[int, str],  
):  
    preds = trainer.predict(dataset)  
    pred_ids = np.argmax(preds.predictions, axis=1)  
    true_ids = dataset["label"]  
  
    results = []  
    for i in range(len(true_ids)):  
        results.append(  
            {  
                "predicted_label": id2label[pred_ids[i]],  
                "true_label": id2label[true_ids[i]],  
                "text": dataset["text"][i],  
            }  
        )  
  
    Path(out_file).write_text(json.dumps(results, indent=4, ensure_ascii=False))  
    print(f"✓ Predictions written to {out_file}")  
  
# ---------------------------------------------------------------------------- #  
# 🔟  Main training routine  
# ---------------------------------------------------------------------------- #  
if __name__ == "__main__":  
    # ------------------------------------------------------------  
    # 1️⃣  Hyper‑parameters  
    # ------------------------------------------------------------  
    TRAIN_FILE = "train_rehydrated.jsonl"  
    EMBED_FILE = "embeddings.npy"          # <-- pre‑computed file  
    OUTPUT_DIR = "qwen3-conspiracy-classification"  
    BATCH_SIZE = 16  
    LEARNING_RATE = 2e-4  
    NUM_EPOCHS = 20  
    TEST_RATIO = 0.15  # 15 % validation split  
  
    # 1️⃣  Label mapping  
    LABEL_MAP = {"No": 0, "Yes": 1}  
    ID2LABEL = {v: k for k, v in LABEL_MAP.items()}  
    NUM_LABELS = len(LABEL_MAP)  
  
    # ------------------------------------------------------------  
    # 2️⃣  Load data  
    # ------------------------------------------------------------  
    print("📥 Loading & filtering data …")  
    raw_examples = load_and_filter_data(TRAIN_FILE)  
    print(f"✅ {len(raw_examples)} examples ready.")  
  
    # ------------------------------------------------------------  
    # 3️⃣  Load (or create) the pre‑computed embeddings  
    # ------------------------------------------------------------  
    if os.path.exists(EMBED_FILE):  
        print(f"📥 Loading pre‑computed embeddings from {EMBED_FILE} …")  
        embeddings_arr = np.load(EMBED_FILE)  
        if embeddings_arr.shape[0] != len(raw_examples):  
            raise RuntimeError(  
                f"The embeddings file contains {embeddings_arr.shape[0]} rows, "  
                f"but there are {len(raw_examples)} examples.  The file "  
                f"must match the order of the data."  
            )  
        print(f"✅ Embeddings loaded – shape {embeddings_arr.shape}")
    else:  
        # ---- Compute and persist embeddings ---- #  
        print(f"⚠️  Embeddings file '{EMBED_FILE}' not found.  Computing embeddings …")  
        embeddings_list = []  
        for idx, example in enumerate(tqdm(raw_examples, desc="Computing embeddings")):
            # Optionally: print progress or use tqdm  
            emb = embed_text(example["text"])  
            if emb is None:  
                raise RuntimeError(f"Embedding failed for example #{idx}")  
            embeddings_list.append(emb)  
  
        embeddings_arr = np.array(embeddings_list, dtype=np.float32)  
        np.save(EMBED_FILE, embeddings_arr)  
        print(f"✅ {len(raw_examples)} embeddings computed and saved to {EMBED_FILE}")  
  
    # ------------------------------------------------------------  
    # 4️⃣  Build datasets (the embeddings are now attached instead of  
    #      computed on the fly)  
    # ------------------------------------------------------------  
    datasets = prepare_datasets(  
        raw_examples,  
        LABEL_MAP,  
        embeddings=embeddings_arr,  
        test_ratio=TEST_RATIO,  
    )  
    train_ds, val_ds = datasets["train"], datasets["test"]  
    print(f"✅ Train: {len(train_ds)} | Val: {len(val_ds)}")  
  
    # ------------------------------------------------------------  
    # 5️⃣  Instantiate the tiny classifier (use the actual dimensionality)  
    # ------------------------------------------------------------  
    model = MLPClassifier(  
        embedding_dim=embeddings_arr.shape[1],  # <- use real width  
        num_labels=NUM_LABELS,  
    )
  
    # ------------------------------------------------------------  
    # 6️⃣  Training arguments  
    # ------------------------------------------------------------  
    training_args = TrainingArguments(  
        output_dir=OUTPUT_DIR,  
        eval_strategy="epoch",  
        save_strategy="epoch",  
        learning_rate=LEARNING_RATE,  
        per_device_train_batch_size=BATCH_SIZE,  
        per_device_eval_batch_size=BATCH_SIZE,  
        num_train_epochs=NUM_EPOCHS,  
        weight_decay=0.01,  
        logging_dir="./logs",  
        report_to="none",  
        load_best_model_at_end=True,  
        metric_for_best_model="f1_weighted",  
        fp16=True,  
    )  
  
    # ------------------------------------------------------------  
    # 7️⃣  Trainer  
    # ------------------------------------------------------------  
    optimizer = torch.optim.AdamW(  
        model.parameters(),  
        lr=LEARNING_RATE,  
        weight_decay=0.01,  
    )  
  
    scheduler = OneCycleLR(  
        optimizer,  
        max_lr=LEARNING_RATE,  
        steps_per_epoch=len(train_ds) // BATCH_SIZE,  
        epochs=NUM_EPOCHS,  
    )  
  
    trainer = Trainer(  
        model=model,  
        args=training_args,  
        train_dataset=train_ds,  
        eval_dataset=val_ds,  
        data_collator=collate_fn,  
        compute_metrics=compute_metrics,  
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  
        # optimizers=(optimizer, scheduler),  
    )  
  
    # ------------------------------------------------------------  
    # 8️⃣  Train  
    # ------------------------------------------------------------  
    print("\n🚀  Starting training …")  
    trainer.train()  
    print("\n🏁  Training finished.")  
  
    # ------------------------------------------------------------  
    # 9️⃣  Save the best model (optional)  
    # ------------------------------------------------------------  
    best_path = os.path.join(OUTPUT_DIR, "best_model")  
    trainer.save_model(best_path)  
    print(f"📦  Best model saved to {best_path}")  
  
    # ------------------------------------------------------------  
    # 🔟  Export predictions on the validation set  
    # ------------------------------------------------------------  
    pred_file = os.path.join(OUTPUT_DIR, "validation_predictions.json")  
    save_predictions(trainer, val_ds, pred_file, ID2LABEL)  
  
    # ------------------------------------------------------------  
    # 🔟  Final metrics  
    # ------------------------------------------------------------  
    print("\nFinal evaluation on the validation set:")  
    final_metrics = trainer.evaluate()  
    for k, v in final_metrics.items():  
        print(f"  {k:>12}: {v:.4f}")