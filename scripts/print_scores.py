import numpy as np
import sys
from pathlib import Path

# Path to score file
score_path = Path("/home/dmarhoef/brain-model-alignment/scores/UTS03/fmri_flamingo/wheretheressmoke.npz")

if score_path.exists():
    print(f"📈 Scores loaded from {score_path}")
    scores = np.load(score_path, allow_pickle=True)
    story_scores = scores['story_scores'].item()
    
    print("\n🏆 Results:")
    print("-" * 40)
    print(f"{'Metric':<10} | {'Score':<10}")
    print("-" * 40)
    
    for (ref, metric), score in story_scores.items():
        if isinstance(score, (np.ndarray, list)):
            score = np.mean(score)
        print(f"{metric:<10} | {score:.4f}")
    print("-" * 40)
else:
    print(f"❌ Score file not found at {score_path}")
