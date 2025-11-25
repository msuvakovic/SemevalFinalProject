# NeoBERT Conspiracy Detection & Ablation Study

This project implements a detection pipeline for conspiracy theories using **NeoBERT** (chandar-lab) augmented with psycholinguistic features.

## Prerequisites
1. source ~/myenv/bin/activate
2. Install dependencies:
   `pip install -r requirements.txt`
   `pip install --upgrade torch`
3. Download the dataset from Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
4. Extract `True.csv` and `Fake.csv` into a folder named `data/` in this directory.

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

Results:

NeoBert only:
Accuracy: 0.7975
F1 Score: 0.7897
------------------------------
              precision    recall  f1-score   support

        True       0.97      0.61      0.75       196
        Fake       0.72      0.98      0.83       204

    accuracy                           0.80       400
   macro avg       0.84      0.79      0.79       400
weighted avg       0.84      0.80      0.79       400

Sentiment:
Accuracy: 0.7675
F1 Score: 0.7673
------------------------------
              precision    recall  f1-score   support

        True       0.79      0.74      0.76       201
        Fake       0.75      0.80      0.77       199

    accuracy                           0.77       400
   macro avg       0.77      0.77      0.77       400
weighted avg       0.77      0.77      0.77       400

All 3:
Accuracy: 0.7550
F1 Score: 0.7547
------------------------------
              precision    recall  f1-score   support

        True       0.79      0.71      0.75       207
        Fake       0.72      0.80      0.76       193

    accuracy                           0.76       400
   macro avg       0.76      0.76      0.75       400
weighted avg       0.76      0.76      0.75       400