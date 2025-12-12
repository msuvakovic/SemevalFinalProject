# NeoBERT Conspiracy Detection & Ablation Study

This project implements a detection pipeline for conspiracy theories using **NeoBERT** (chandar-lab) augmented with psycholinguistic features.

## Prerequisites
1. source ~/myenv/bin/activate
2. Install dependencies:
   `pip install -r requirements.txt`
   `pip install --upgrade torch`
3. Download the "train_redacted.jsonl" file from https://zenodo.org/records/17065240 and place the files in the /data folder located within the current file directory level 
4. run "rehydrate_data.py" from https://github.com/hide-ous/semeval26_task10_starter_pack to rehydrate the dataset.
5. place the rehydrated dataset in /data


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

Results(innaccurate at the moment with new changes):

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


## Notes: changes by Milos (origianlly Glenn's repository)

Added the following:
- Added "xformers==0.0.28.post3" to requirements.txt
- included freezing of the last three transformer encoders and the layer normalization of the NeoBERT model
- changed learning rate (to accomodate larger batch size)
- changed weight decay to 0.01
- changed batch size to 128 (may need to reduce this batch size on lower end gpu/Macintosh A-series chips)
- adapted the model to no longer use NeoBERT specific configs and instead using "AutoModel" from transformers library. with added arugment " trust_remote_code=True"

