# Mid-Quarter Update — Brain–Model Alignment (244 Project)

**For your presentation:** This document explains our project in plain language so classmates can follow. Use it to build slides and to explain what we’ve done and what the results mean.

---

## In plain English — TL;DR

**What we’re doing:** We’re trying to **read language from brain activity**. People listen to (or imagine) stories while we scan their brains with fMRI. We train models to turn that brain data into text—either by picking the right word from a list (“ranking”) or by generating full sentences (“decoding”).

**What we built:**  
1. **FMRIFlamingo** — Our main model. It takes brain scan data, feeds it into a language model (Llama), and learns to predict what word comes next. We can test it by “which of these 100 words did the person hear?” (ranking) or “write the next word” (generation).  
2. **Huth pipeline** — We also implemented the method from a famous 2023 paper (Huth et al.) so we can compare ourselves to that baseline. Their approach: first learn “what brain activity looks like for each piece of text,” then use that to score possible sentences and pick the best one.

**Results so far:**  
- **Ranking:** We observed **~42% Top-1** in one setup (Run 5, UTS03, story *avatar*, 1-in-100 with in-story distractors). **Important:** We ran a **blind control** (same eval but with the fMRI signal zeroed out) and got similar accuracy. That means the 42% is **largely or solely due to the language prior**—the LLM ranking words by how plausible they are in a story context, not from decoding brain activity. So we **cannot** claim the model is using fMRI for that task; we need to see a big drop in the blind condition to claim brain signal. In other setups (e.g. strict “telepathy” evals) we’ve seen ~0%.  
- **Generation / full-sentence decoding:** Still very weak. The model often says generic or wrong words; we’re not yet “reading minds” into full sentences.  
- **Huth baseline:** We got their pipeline running on our data. Their decoder also struggles on our test stories (low scores), which tells us the task is genuinely hard and we’re not missing something obvious.

**Bottom line:** We have a working pipeline. The earlier 42% ranking result does **not** hold up as evidence of fMRI use once we control for the language prior (blind test). Turning that into reliable, fMRI-driven “brain reading” is the next step.

---

## 1. Project goal and research question (explained simply)

### The big idea

**Goal:** Decode **continuous language** (full sentences or streams of words) from **non-invasive** brain recordings.

- **Decode** = turn brain activity into text (what the person heard or is thinking).
- **Continuous language** = not just single words, but sequences of words like in a story.
- **Non-invasive** = we use fMRI (a brain scan that doesn’t require surgery). No implants.

**In one sentence:** Can we train a model that takes a person’s fMRI data while they listen to (or imagine) speech and outputs the words they’re hearing or thinking?

### How we’re testing that

We compare our approach to a strong baseline from the literature:

- **Baseline: Huth et al. (2023), *Nature Neuroscience*.**  
  - **In simple terms:** They first train a model that predicts “what does the brain look like when you hear this text?” (that’s the **encoding model**). Then, to **decode**, they take real brain data and search over many possible sentences; they score each sentence by “how well does this sentence explain the brain activity?” and combine that with a language model so the output sounds like real language. They evaluate with standard text metrics: WER (word error rate), BLEU, METEOR, BERTScore (see jargon section below).

- **Our approach: FMRIFlamingo.**  
  - **In simple terms:** We don’t build a separate “text → brain” model. Instead we feed **brain data directly** into a large language model (Llama) via a special “brain tokenizer” and cross-attention. The model learns to predict the next word from brain + previous words. So it’s “brain in, language out” in one network. We use the same decoding (beam search over sentence candidates) and the same metrics (WER, BLEU, etc.) so we can compare fairly.

**Research question:** Can FMRIFlamingo decode perceived or imagined speech from fMRI **at least as well as** the Huth (ridge + beam-search) baseline?

---

## 2. Jargon cheat sheet (so everyone’s on the same page)

| Term | Plain-English meaning |
|------|------------------------|
| **fMRI** | A brain scan that measures blood flow (BOLD) in tiny 3D cubes (voxels). More active brain areas “light up.” We use it to see which regions are active while someone listens to or imagines speech. |
| **BOLD** | Blood-oxygen-level dependent signal—what fMRI actually measures. Indirect measure of neural activity. |
| **Voxel** | One small 3D “cube” in the brain scan. We have tens or hundreds of thousands of voxels per scan; each has a number (activity level). |
| **Encoding model** | A model that predicts “what would the brain look like if the person heard this text?” So: **text in → predicted brain activity out.** |
| **Decoding** | The reverse: **brain activity in → text out.** We’re trying to recover what the person heard or thought. |
| **Beam search** | A way to generate a sentence: at each step we keep several best partial sentences and extend them word by word, scoring with our model + language model. |
| **WER (Word Error Rate)** | How many word substitutions, insertions, or deletions are needed to turn the model’s output into the correct text. Lower is better. Often we report “score = 1 − WER” so higher is better. |
| **BLEU** | A metric that measures how much the model’s output overlaps with the reference (correct) text (n-gram overlap). Higher is better. |
| **METEOR / BERTScore** | Other metrics that compare model output to reference; they capture meaning and wording. Higher is better. |
| **Ranking (e.g. 1-in-100)** | We give the model one correct word and 99 wrong ones; can it rank the correct word higher? **Top-1** = did the correct word get the highest score? Chance = 1%. |
| **Language prior** | The tendency of the LLM to prefer some words over others (e.g. likely next words in a story) even without brain input. If we don’t control for it, we can mistake “LLM guessed a plausible word” for “decoded from fMRI.” |
| **Blind control** | We run the same eval but **zero out the fMRI** input. If accuracy stays high, the result is from the language prior, not from brain decoding. We need the blind condition to drop toward chance to claim fMRI use. |
| **Perceived vs imagined speech** | **Perceived** = person listens to audio. **Imagined** = person imagines saying/hearing speech without sound. We use the same encoder for both (as in Huth). |
| **Per-subject** | We train and evaluate **separately for each person** (UTS01, UTS02, …). We don’t mix different people’s brains in one model. |
| **Train / val / test** | **Train** = data we learn from. **Val** = we check during training to avoid overfitting. **Test** = we never train on this; we only evaluate here to report results. |

---

## 3. What we built (and what each piece does)

Here’s what exists in the repo and what it’s for, in plain language.

| Component | What it does (simple) | Where it lives |
|-----------|------------------------|----------------|
| **FMRIFlamingo** | Takes brain scan data, converts it into “tokens” the language model can use, and feeds it into Llama so the model can predict the next word (and optionally learn to rank the right word). | `src/models/fmri_flamingo.py`, `src/models/fmri_tokenizer.py` |
| **Huth dataset loader** | Loads our fMRI data and the correct text, aligned in time (we know which words were heard at which scan times). Handles train/val/test splits and a simple delay between hearing and brain response. | `src/datasets/huth_fmri_dataset.py` |
| **Training script** | Trains FMRIFlamingo: next-word prediction + optional “rank the right word” loss. Uses tricks like masking some text so the model can’t cheat and must use the brain. Saves checkpoints and does early stopping. | `scripts/train.py` |
| **Telepathy ranking eval** | Evaluates “can the model pick the right word from 100 options using only brain data?” (no text clue). Reports Top-1, Top-5, Top-10 accuracy. | `scripts/evaluate_ranking_telepathy.py` |
| **Blind ranking eval** | Same as telepathy ranking but **zeros out the fMRI** so the model only has the LLM prior. If accuracy stays high, the “brain” result was really the language prior. | `scripts/evaluate_ranking_blind.py` |
| **Huth decoding pipeline** | The full Huth-style pipeline: (1) train encoding model “text → brain,” (2) train word-rate model for timing, (3) run beam-search decoder that scores sentences with that encoding model. | `decoding/` (train_EM.py, train_WR.py, run_decoder.py) |
| **FMRIFlamingo decoder** | Same beam-search idea as Huth, but we score candidate sentences using FMRIFlamingo instead of the Huth encoding model. Lets us compare “Huth-style decoding” vs “FMRIFlamingo decoding” on the same stories. | `scripts/run_decoder_fmri_flamingo.py` |
| **Parity evaluator** | Computes the same metrics as Huth (WER, BLEU, METEOR, BERTScore) on our own predictions (e.g. from FMRIFlamingo), so we can compare apples to apples. | `scripts/evaluate_huth_parity.py` |
| **Replication checklist** | A step-by-step list of how to reproduce Huth 2023 (data, encoding, word rate, decoder, evaluation) so we know we’re doing the same thing they did. | `docs/HUTH_2023_REPLICATION_CHECKLIST.md` |

**Config:** All main settings (paths, model choice, training options) are in `config.py` and `decoding/config.py`. We use one model per subject; we left subject 9 (UTS09) out of the default setup and use subjects 1–8.

---

## 4. What we did (progress so far, in order)

- **Trained FMRIFlamingo** on our Huth-style dataset (subjects 1–8). The model sees brain data + previous words and learns to predict the next word. We use fixed time windows (each “step” is one TR of brain data) and we mask 50% of the text sometimes so the model has to rely on the brain.

- **Evaluated with ranking.** We ran “1-in-100” ranking: give the model 100 words, one of which is correct; can it score the correct one highest using only brain data (BOS-only prompt)? In one run (Run 5, UTS03, story *avatar*, 99 in-story distractors) we got **42.86% Top-1**. We then ran a **blind control** (`evaluate_ranking_blind.py`): same setup but **fMRI zeroed out**. Accuracy stayed similar, so the 42% is **attributable to the language prior** (LLM preferring plausible story words), not to decoding from brain data. In other settings we’ve seen ~0%. So we do *not* have solid evidence yet that the model uses fMRI for ranking; we need a control that shows a drop when fMRI is removed.

- **Evaluated with generation.** We tried “generate the next word (or sentence) from brain + context.” Results were **very poor**—WER/BLEU near zero, lots of generic or wrong words. So at this stage we treat **ranking** as the more reliable way to measure whether the model is using brain data; generation is a harder task we’re working toward.

- **Integrated the Huth pipeline.** We got their encoding model (train_EM), word-rate model (train_WR), and beam decoder (run_decoder) running in our repo. We use the same evaluation script (WER, BLEU, METEOR, BERTScore). We also fixed a technical issue (voxel limit) so our FMRIFlamingo decoder can run on full-sized brain data (we cap at 120k voxels to avoid memory errors).

- **Ran Huth baseline evaluation.** We ran their decoder on “perceived speech” for subjects 1–8 on test stories (e.g. *buck*, *tildeath*). We got **negative WER score** (meaning raw WER > 1—the decoder’s output was very far from the correct text) and **low BLEU/METEOR**; BERTScore was a bit higher. So their method also struggles on our test stories—the task is hard, and we’re not comparing against a “perfect” baseline.

- **Audited against Huth.** We wrote down exactly how our setup matches or differs from theirs (encoding, word timing, decoding, metrics) and made a replication checklist. We also noted that we use a **simplified alignment** (one delay + a time window) while they use a fancier one (Lanczos + FIR). We might align that later to make the comparison fairer.

- **Repo and onboarding.** We wrote a detailed README so classmates can clone the repo, set up the environment, understand where data goes, and run training and decoding. The repo is on GitHub; we don’t put the actual brain data or big model checkpoints in git (see `docs/SHARING_REPO_WITH_PARTNERS.md`).

---

## 5. Results so far — what to report and what it means

### FMRIFlamingo

- **Ranking**
  - **What we did:** 1-in-100 word choice using only brain data (BOS-only prompt), with 1 correct + 99 in-story distractors. We also ran a **blind control**: same eval but with the fMRI input **zeroed out** (`scripts/evaluate_ranking_blind.py`) to measure how much the LLM prior alone contributes.
  - **What we got:** With real fMRI we saw **42.86% Top-1** in one run (Run 5, UTS03, *avatar*). In the **blind** condition (no fMRI signal), accuracy was **similar**. So the 42% is **not** evidence that the model uses brain data—it’s largely or solely the **language prior** (the LLM ranking words by story plausibility). In other evals we’ve seen ~0%.
  - **What it means:** We cannot claim the model is decoding from fMRI based on that 42% result. To show fMRI use we need the blind condition to drop toward chance (e.g. ~1%) while the real-fMRI condition stays high. Next steps: strengthen training/alignment so that removing fMRI actually hurts ranking, and keep reporting blind controls.

- **Generation (full sentences or next word)**
  - **What we did:** Ask the model to generate the next word or a sentence given brain data (and maybe previous words).
  - **What we got:** Very poor—WER/BLEU near zero, output often generic or wrong.
  - **What it means:** The model isn’t yet “reading” the brain well enough to produce the exact words. Ranking suggests there *is* some useful signal in the brain data; generation is a harder task we’re targeting next.

- **Training**
  - Loss goes down with strong regularization (dropout, weight decay) and cross-attention every 8 layers. We use 50% text masking so the model can’t ignore the brain. So the training pipeline is working; the bottleneck is turning that into good generation.

### Huth baseline (ridge + beam search) in our setup

- **What we did:** Trained their encoding model and word-rate model, then ran their beam decoder on test stories for subjects 1–8 (perceived speech; we can also run imagined with the same encoder).
- **What we got:** Decoder runs successfully. Story-level scores: **negative WER score** (raw WER > 1), **low BLEU/METEOR**, **higher BERTScore**. We skipped null baselines when we didn’t have the extra data (`--null 0`).
- **What it means:** Their method also produces poor open decoding on our test stories. So we’re not failing alone—the task is genuinely difficult. Next we want to run FMRIFlamingo decoding on the *same* stories and compare with the same metrics.

### Takeaway for the presentation

- We have **two working pipelines**: FMRIFlamingo (train + rank + decode) and Huth (encode + decode + evaluate).
- **Ranking:** We saw ~42% Top-1 in one setup, but a **blind control** (fMRI zeroed out) gave similar accuracy → the result is **due to the language prior**, not fMRI decoding. We do *not* have evidence yet that the model uses brain data for ranking; we need blind to drop toward chance.
- **Generation** and **open decoding** are still weak for both our model and the Huth baseline on our data.
- **Next step:** Run blind controls whenever we report ranking; improve training/alignment so that removing fMRI actually hurts performance; run FMRIFlamingo decoding on the same test stories as Huth and compare with the same metrics.

### Our results and experiments vs Huth (at a glance)

**Experiment setup (what each row is):**

| | **Huth et al. 2023 (paper)** | **Our Huth pipeline (our data)** | **Our FMRIFlamingo** |
|---|------------------------------|----------------------------------|----------------------|
| **Method** | Encoding model (ridge) + beam search | Same: encoding model + beam search | fMRI → Llama via cross-attention; next-word + ranking |
| **Data** | 16 h narrative; test story “Where There’s Smoke” held out | Our Huth-style data; test stories e.g. *buck*, *tildeath* (Where There’s Smoke in train for us) | Same dataset; ranking on *avatar* (UTS03); generation on constrained “first word” |
| **Eval type** | Full-sentence decoding → WER, BLEU, METEOR, BERTScore | Same metrics on decoder output | **Ranking:** 1-in-100 word choice (Top-1/5/10). **Generation:** WER/BLEU/METEOR/BERTScore on generated text |

**Reported results (numbers):**

| Metric | **Huth et al. 2023 (paper)**<br>Perceived, “Where There’s Smoke” | **Our Huth pipeline**<br>Perceived, our test stories | **Our FMRIFlamingo (ranking)**<br>UTS03, story *avatar*, 1-in-100 | **Our FMRIFlamingo (generation)**<br>Constrained first word |
|--------|------------------------------------------------------------------|-----------------------------------------------------|------------------------------------------------------------------|-----------------------------------------------------------|
| **WER** (lower better) | ~0.92–0.94 | Poor (raw WER > 1) | — | ~1.0 (0% correct) |
| **BLEU-1** (higher better) | ~0.23–0.25 | Low | — | 0.00 |
| **METEOR** (higher better) | ~0.16–0.17 | Low | — | ~0.0007 |
| **BERTScore** (higher better) | ~0.81 | Relatively higher | — | ~0.62 |
| **Top-1 accuracy** (1-in-100 ranking) | — | — | **42.86%** (but **blind ≈ same** → language prior) | — |
| **Top-5 / Top-10** | — | — | 42.86% / 54.29% (blind control: similar) | — |

**Notes:** (1) Huth’s numbers are from their paper (Table 1 / Fig. 1d); ours are from our runs. (2) The **42% ranking is not evidence of fMRI use**: we ran a blind control (fMRI zeroed out) and got similar accuracy, so that result is **attributable to the language prior**. (3) Ranking (Top-1) is a different task than full-sentence decoding (WER/BLEU). (4) Our Huth pipeline on our data gives poor decoding (WER > 1); we’re not yet doing a strict replication.

---

## 6. Decisions we made (and why)

| Decision | Why we did it |
|----------|----------------|
| **One model per subject; leave out subject 9** | Standard in this literature: each person’s brain is different, so we train and test per subject. We use subjects 1–8 to keep things simple and match the “per-subject” setup. |
| **Same encoding model for “heard” and “imagined”** | Huth et al. do the same: train on “heard” speech and use that model for “imagined” too. So we only need one encoding model unless we explicitly want a separate one for imagined. |
| **Keep “Where There’s Smoke” in train for now** | Our data splits have that story in training. For a strict Huth replication we’d move it to test later; we didn’t change splits yet so we could keep going with current evals. |
| **Use ranking as our main metric** | Generation is all over the place and gives bad WER/BLEU. Ranking (“pick 1 of 100”) is clearer and shows whether the model is using brain data at all. |
| **Simple alignment (one delay + window)** | Faster to build. Huth uses a more complex alignment (Lanczos + FIR); we can add that later if we need a fairer comparison. |
| **Don’t put data or big checkpoints in git** | Data and model files are huge and not in the repo. Classmates get code and instructions; they get data separately so the repo stays usable. |

---

## 7. Challenges we hit (and what we learned)

- **Model “babbling” or repeating:** Early on the model produced nonsense or repetitive text. We fixed it with stronger regularization and text masking, and we focused evaluation on ranking instead of generation for now.

- **Too many voxels:** Our FMRIFlamingo tokenizer has a maximum number of brain regions (120k). Some subjects have more voxels, so we cap the input in the decoder script so it doesn’t crash.

- **Alignment difference from Huth:** We use a simpler way to align words in time with brain scans. Huth uses a more sophisticated method (Lanczos + FIR). That might make our decoding a bit worse; we wrote it down and can align with them later if needed.

- **Null baselines:** Huth’s full evaluation uses “null” (language-model-only) sequences to compute z-scores. We don’t have that set up yet, so we run with `--null 0` and report raw metrics. We can add nulls later for a stricter comparison.

- **Apples-to-apples comparison:** To compare FMRIFlamingo fairly to Huth we need the same test stories, same metrics, and ideally the same null setup. We built a replication checklist and a parity evaluator so we can do that.

---

## 8. Current status and next steps

### What’s working

- FMRIFlamingo: training, saving checkpoints, ranking evaluation (1-in-100).
- Huth pipeline: encoding model training, word-rate model, beam decoder, evaluation script (WER, BLEU, METEOR, BERTScore).
- FMRIFlamingo decoder (beam search with our model), with voxel capping.
- Parity evaluator for our own predictions.
- Repo with README and partner docs; ready for classmates to clone and run.

### What we’re doing next

- Run Huth baseline decoding on all subjects 1–8 on the same test stories and write down the numbers.
- Run FMRIFlamingo decoding on those same stories and evaluate with the parity script.
- (Optional) Move “Where There’s Smoke” to test and run a full Huth replication (with null baselines if we set up `data_lm/`).
- (Optional) Match Huth’s alignment (Lanczos + FIR) and re-run to see if decoding improves.
- (Optional) Set up `data_lm/` so we can report z-scores and null baselines like in the paper.

---

## 9. Repo and how to run (for slides or handout)

- **Repo:** https://github.com/Dom-Marhoefer/244Project (or your current URL).
- **Setup:** Clone repo → create virtual environment → `pip install -r requirements.txt` → install decoding deps (scipy, jiwer, evaluate, datasets) → `huggingface-cli login`.
- **Data:** Not in the repo. Put Huth (or compatible) data under `data/Huth/derivative/` (see README for exact folders: preprocessed_data, TextGrids, splits.json).
- **Train FMRIFlamingo:** `python scripts/train.py` from project root.
- **Run Huth pipeline:** `cd decoding`, then run `train_EM.py`, `train_WR.py`, `run_decoder.py` (see README and `docs/HUTH_2023_REPLICATION_CHECKLIST.md`).
- **Evaluate:** `decoding/evaluate_predictions.py` for Huth decoder outputs; `scripts/evaluate_huth_parity.py` for FMRIFlamingo outputs (with path to your predictions).

---

## 10. References and where to read more

- **Huth et al. (2023):** Tang, LeBel, Jain & Huth, “Semantic reconstruction of continuous language from non-invasive brain recordings,” *Nature Neuroscience* 26, 858–866. DOI: 10.1038/s41593-023-01304-9. Code: https://github.com/HuthLab/semantic-decoding.
- **In this repo:**  
  - `docs/HUTH_2023_REPLICATION_CHECKLIST.md` — step-by-step replication.  
  - `docs/HUTH_SOTA_AUDIT.md` — how we compare to Huth and what we could improve.  
  - `docs/DECODING_WALKTHROUGH.md` — alignment and Huth vs FMRIFlamingo.  
  - `docs/STUDY_LOG.md` — study log and where results are saved.  
  - `docs/SHARING_REPO_WITH_PARTNERS.md` — what to share and how to onboard partners.  
  - `EXPERIMENT.md` — training runs and ranking vs generation results.

---

Use this doc as your main source for the presentation. You can copy sections into slides and adjust numbers (e.g. exact Top-1 %) as you run more experiments.
