#!/usr/bin/env python
# infer_binary.py
# -------------------------------------------------------------
# Predict “conspiracy” labels for dev_rehydrated.jsonl
# using the best checkpoint saved during training.
# -------------------------------------------------------------

import json
import os
import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ollama import Client as OllamaClient
from safetensors.torch import load_file as load_safetensors

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

# ------------------------------------------------------------------
# 2. Label mapping
# ------------------------------------------------------------------
ID2LABEL = {0: "No", 1: "Yes"}

# ------------------------------------------------------------------
# 3. Main inference routine
# ------------------------------------------------------------------
def main(
    dev_file: str = "dev_rehydrated.jsonl",
    submission_file: str = "submission.jsonl",
    model_dir: str = "qwen3-conspiracy-classification/best_model",
    embed_model: str = "qwen3-embedding",
):
    # 3.1 Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(embedding_dim=4096, num_labels=2).to(device)

    # Find the checkpoint file (either .safetensors or .bin)
    ckpt_path = Path(model_dir) / "model.safetensors"
    if not ckpt_path.exists():
        ckpt_path = Path(model_dir) / "pytorch_model.bin"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Could not find a checkpoint in {model_dir}. "
            f"Looked for {ckpt_path}"
        )

    # Load state‑dict
    if ckpt_path.suffix == ".safetensors":
        state_dict = load_safetensors(str(ckpt_path))
    else:
        state_dict = torch.load(str(ckpt_path), map_location=device)

    model.load_state_dict(state_dict)
    model.eval()

    # 3.2 Ollama embedding client
    ollama_client = OllamaClient()

    # 3.3 Predict on every dev example
    results = []

    with open(dev_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Predicting", unit="ex"):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed line: {line}")
                continue

            example_id = entry.get("_id")
            if not example_id:
                print(f"Skipping entry without _id: {entry}")
                continue

            # Grab the text – fallback to a few common keys
            text = (
                entry.get("text")
                or entry.get("body")
                or entry.get("content")
                or ""
            )
            if not isinstance(text, str):
                print(f"Skipping entry with non‑string text: {entry}")
                continue

            # 1) Embed via Ollama
            embed_res = ollama_client.embeddings(
                model=embed_model,
                prompt=text,
            )
            embedding = torch.tensor(
                embed_res["embedding"], dtype=torch.float32
            ).unsqueeze(0).to(device)

            # 2) Predict
            with torch.no_grad():
                logits = model(embedding)["logits"]
            pred_id = int(torch.argmax(logits, dim=1).item())
            pred_label = ID2LABEL[pred_id]

            results.append(
                {"_id": example_id, "conspiracy": pred_label}
            )

    # 3.4 Write results to JSON‑L
    Path(submission_file).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n"
    )
    print(f"✓ Finished. {len(results)} predictions written to {submission_file}")


# ------------------------------------------------------------------
# 4. CLI entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Infer conspiracy labels for the dev set."
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        default="dev_rehydrated.jsonl",
        help="Path to the dev JSON‑L file.",
    )
    parser.add_argument(
        "--submission_file",
        type=str,
        default="submission.jsonl",
        help="Output path for the JSON‑L submission.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="qwen3-conspiracy-classification/best_model",
        help="Directory containing the trained checkpoint.",
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        default="qwen3-embedding",
        help="Ollama embedding model name.",
    )

    args = parser.parse_args()
    main(
        dev_file=args.dev_file,
        submission_file=args.submission_file,
        model_dir=args.model_dir,
        embed_model=args.embed_model,
    )