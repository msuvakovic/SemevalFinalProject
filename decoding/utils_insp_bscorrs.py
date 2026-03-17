import numpy as np
import matplotlib.pyplot as plt




em = np.load("models/UTS03/encoding_model_perceived.npz", allow_pickle=True)
bscorrs = em["bscorrs"]  # shape: (n_voxels,) — Pearson r per voxel

print(f"Shape: {bscorrs.shape}")
print(f"Max: {bscorrs.max():.3f}, Mean: {bscorrs.mean():.3f}, Median: {np.median(bscorrs):.3f}")
print(f"Voxels with r > 0.1: {(bscorrs > 0.1).sum()}")
print(f"Voxels with r > 0.2: {(bscorrs > 0.2).sum()}")

plt.hist(bscorrs, bins=100)
plt.xlabel("Bootstrap Pearson r")
plt.ylabel("Voxel count")
plt.title("Encoding model voxel correlations")
plt.axvline(0, color='r', linestyle='--')
plt.show()
