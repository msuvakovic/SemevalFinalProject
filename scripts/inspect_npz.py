import numpy as np
import sys

path = "/home/dmarhoef/brain-model-alignment/results/UTS03/fmri_flamingo/wheretheressmoke.npz"
data = np.load(path)
print("Words:", data['words'])
print("Times:", data['times'])
