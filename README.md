# NeoBERT Conspiracy Detection & Ablation Study

This project implements a detection pipeline for conspiracy theories using **NeoBERT** (chandar-lab) augmented with psycholinguistic features.

## Prerequisites
1. Install dependencies:
   `pip install -r requirements.txt`
2. Download the dataset from Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
3. Extract `True.csv` and `Fake.csv` into a folder named `data/` in this directory.

## How to Run the Ablation Study

We test 4 different configurations to see which linguistic factors help NeoBERT detect conspiracies.

**1. Baseline (NeoBERT Only)**
```bash
python run_ablation.py

python run_ablation.py --sentiment
python run_ablation.py --complexity
python run_ablation.py --pronouns
python run_ablation.py --sentiment --complexity
python run_ablation.py --sentiment --pronouns
python run_ablation.py --complexity --pronouns
python run_ablation.py --sentiment --complexity --pronouns
```
