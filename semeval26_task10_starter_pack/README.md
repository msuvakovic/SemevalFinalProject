# SemEval 2026 Task 10  starter pack
Scripts to facilitate participation in the 2026 Semeval Task 10: PsyCoMark -- Psycholinguistic Conspiracy Marker Extraction and Detection

--------
## Download and rehydrate plain text data
1. `git clone https://github.com/hide-ous/semeval26_starter_pack.git`
2. `cd semeval26_starter_pack`
3. `pip install -r requirements.txt`
4. place `train_redacted.jsonl` in the folder (available on [zenodo](https://doi.org/10.5281/zenodo.15114171))
5. run `python rehydrate_data.py` to generate `train_rehydrated.jsonl` containing plain texts
6. `submission_example.py` provides you with the boilerplate script to prepare a submission in the right format (by default, no model is used: the submission attributes no markers and "No" conspiracy to each comment)

## Conspiracy detection baseline
1. run `train_binary.py` (~6 minutes on gpu) 
2. run `infer_binary.py`
3. zip the submission `submission.jsonl` --> `submission.zip`
4. go to the [detection task on codabench](https://www.codabench.org/competitions/10749/)
   1. go to the "my submissions" tab 
   2. upload the zip file
   3. wait a few minutes for evaluation (tip: if the page does not reload, switch back and forth to the test phase)
   4. add the result to the leaderboard and make it public!
   5. you should score ~ 0.76 weighted F1 on the dev set

## Changes
Ollama must be installed and running locally, with `qwen3-embedding` model loaded in memory. \
Running train_binary.py the first time will generate the embeddings using the embedding model and save them to embeddings.npy. \
Future trainings will be much faster since they won't have to recalculate the embeddings. \
Running infer_binary.py will create a submissions.jsonl as mentioned above. \
Files that didn't undergo modification and aren't necessary to the binary classifier training and inference pipeline like the `one_span` files have been removed to avoid confusion. \
For more instructions on experimentation and explanation of the code, please refer to the README.md file in the root directory of this repository.
