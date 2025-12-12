# SemEval 2026 Task 10 Experiments
## Ryan King

## The Goal
The goal of my experimentation for this project was to test the usefulness of embedding models other than BERT in the conspiracy analysis task. I started with nomic-embed-text, but the current implementation uses the qwen3-embedding model to create embeddings which are then classified using a variety of basic classifiers.

## Prerequisites
Setup for the repository is detailed in the README.md file located in the semeval26_task10_starter_pack folder. \
You will rehydrate the training and dev test data and run the training to build the embeddings and train the model. \
For this code to work, ollama must be running on your machine, and the qwen3-embedding model must be available. (If `ollama list` shows `qwen3-embedding:latest` but the code still errors, try running `ollama run qwen3-embedding test` in your terminal (to get ollama to initialize the model), and then run `train_binary.py` again).

## Model Explanation
### Training
`train_binary.py` follows a three‑step pipeline: \

1. **Load data** – `load_and_filter_data()` reads the raw JSON‑Lines file, keeps only the examples that have a `"conspiracy"` field equal to `"Yes"` or `"No"`, and turns the list into a HuggingFace `datasets.Dataset`.  

2. **Add embeddings** – If an `embeddings.npy` file already exists, it is loaded and attached to the dataset as a new column called `"embedding"`. If it doesn’t exist, the script falls back to calling `embed_text()` for every example on‑the‑fly.  Pre‑computing and storing the embeddings removes the need to hit the local Ollama daemon every time we run training.  

3. **Encode labels & split** – `prepare_datasets()` scans for the label column (e.g. `"conspiracy"`, `"label"`, `"is_conspiracy"`, `"class"`), maps the string value to an integer via `LABEL_MAP`, and then shuffles & splits the data into a training and validation set.  

After the data is ready, the script builds a very small **PyTorch classifier** on top of the embeddings.  Four heads are available:

* `EmbeddingClassifier` – a single linear layer (embedding → 2 logits).  
* `MLPClassifier` – a 2‑layer MLP with ReLU and dropout.  
* `BiggerMLP` – a deeper MLP (512 → 256 hidden units).  
* `TinyTransformerHead` – a one‑layer TransformerEncoder followed by a linear head.

The default in the script is `MLPClassifier` because it offers a good trade‑off: the embeddings already capture most of the semantic signal, so a shallow MLP is enough to learn the binary decision boundary while keeping the risk of over‑fitting low.

**Training loop**

* A custom data collator (`collate_fn`) stacks the raw `embedding` tensors and `label` integers into a batch dictionary.  
* `compute_metrics()` reports accuracy and weighted F1 to the `Trainer`.  
* Training arguments use a one‑cycle learning‑rate scheduler (`OneCycleLR`), early stopping (`EarlyStoppingCallback`), and FP16 mixed‑precision training.  
* The `Trainer` runs `train()` and `evaluate()`, automatically saving the best checkpoint to `OUTPUT_DIR/best_model`.  
* After training finishes, `save_predictions()` writes the validation predictions in the same JSON‑Lines format that is required for codabench submission.  

Run the code with `python3 train_binary.py`

### Inference
`infer_binary.py` is a small helper that uses the best checkpoint saved by `train_binary.py` to label every example in the dev set (`dev_rehydrated.jsonl`).  

1. It loads the best checkpoint and the same classifier architecture (`MLPClassifier` by default).  
2. For each line in the dev file it calls Ollama’s `qwen3-embedding` model to produce an embedding, feeds that embedding through the classifier, and records the predicted `"conspiracy"` label.  
3. The results are written to a JSON‑Lines file (`submission.jsonl`) that can be uploaded as a Codabench submission.  

## Experiments/Changes to try
### Different Embedding Models
To try a different embedding model, change the `model` parameter around line 60 of `train_binary.py`. You will also have to change `EMBEDDING_DIM` on line 99 of `infer_binary.py` to matching the embedding dimension size of the new model. \
I have found larger models to be more accurate, I originally tried with `nomic-embed-text` and switched to `qwen3-embedding` as it has better performance.

### Different Classifier Model
To try a different classifier head, write a class for it and replace the current model used on line 359 of `train_binary.py`. I have written 4 small PyTorch classifiers, `EmbeddingClassifier` (a single Linear layer with CrossEntropyLoss), `MLPClassifer` (a simple MLP with 2 linear layers and ReLU), `BiggerMLP` (a slightly larger MLP with a 3rd larger hidden layer), and `TinyTransformerHead` (a TransformerEncoder model followed by a Linear layer). \
Make sure to also add your model to `infer_binary.py` and change the model used for inference on line 112. \
I found my basic `MLPClassifier` to be the most accurate, as larger models tend to overfit since the embedding model is already quite large and has done much of the work already. The single Linear Layer in the `EmbeddingClassifier` was too small to capture all the information from the embeddings, however.

### Hyperparameters
In addition to adjusting the values in the model classifier code, general hyperparamters for batch size, learning rate, number of epochs, and train-validation split ratio can be adjusted around line 300 in `train_binary.py`. \
These values are currently at what I found to be generally most effective.
