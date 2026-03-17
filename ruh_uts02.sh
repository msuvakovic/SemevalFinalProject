#!/bin/bash

# Define the subject
SUBJ="UTS02"
LOGFILE="pipeline_${SUBJ}.log"

echo "--- Starting Pipeline for $SUBJ at $(date) ---" | tee -a $LOGFILE

# 1. EM Training
echo "[1/4] Training EM..." | tee -a $LOGFILE
python decoding/train_EM.py --subject $SUBJ >> $LOGFILE 2>&1

# 2. WR Training
echo "[2/4] Training WR..." | tee -a $LOGFILE
python decoding/train_WR.py --subject $SUBJ >> $LOGFILE 2>&1

# 3. Run Decoder
echo "[3/4] Running Decoder (Task: buck, Exp: perceived_speech)..." | tee -a $LOGFILE
python decoding/run_decoder.py --subject $SUBJ --task buck --experiment perceived_speech >> $LOGFILE 2>&1

# 4. Evaluate
echo "[4/4] Evaluating Predictions..." | tee -a $LOGFILE
python decoding/evaluate_predictions.py --subject $SUBJ --experiment perceived_speech --task buck >> $LOGFILE 2>&1

echo "--- Pipeline Finished for $SUBJ at $(date) ---" | tee -a $LOGFILE
