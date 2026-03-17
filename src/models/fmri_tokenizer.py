"""
FMRITokenizer: Tokenizes fMRI data for Flamingo-style cross-attention.

Converts fMRI signals (voxels × TRs) into tokenized representations (tokens × embed_dim).
Uses ROI-based tokenization strategy: groups voxels into regions, each ROI becomes one token.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import importlib.util

# Define minimal base class (to avoid OpenTSLM import issues)
# This matches TimeSeriesEncoderBase from OpenTSLM
class TimeSeriesEncoderBase(nn.Module):
    """Minimal base class matching OpenTSLM's TimeSeriesEncoderBase."""
    def __init__(self, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.output_dim = output_dim
        self.dropout = dropout

# Import config
FMRI_FLAMINGO_DIR = Path(__file__).parent.parent.parent
config_path = FMRI_FLAMINGO_DIR / "config.py"
spec = importlib.util.spec_from_file_location("fmri_config", config_path)
fmri_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmri_config)

# Get config values
NUM_ROIS = fmri_config.NUM_ROIS
ROI_SELECTION_METHOD = fmri_config.ROI_SELECTION_METHOD
ENCODER_OUTPUT_DIM = fmri_config.ENCODER_OUTPUT_DIM
TRANSFORMER_INPUT_DIM = fmri_config.TRANSFORMER_INPUT_DIM
MAX_PATCHES = fmri_config.MAX_PATCHES


class FMRITokenizer(TimeSeriesEncoderBase):
    """
    Tokenizes fMRI data using ROI-based strategy.
    
    Input: (voxels, TRs) or (num_rois, TRs) if already grouped
    Output: (num_rois, embed_dim) token embeddings
    
    Process:
    1. ROI grouping (if needed): Group voxels into ROIs
    2. Temporal aggregation: Average or convolve across TRs
    3. Projection: Linear layer to embed_dim
    4. Positional embeddings: Learnable positional encodings
    5. Normalization: LayerNorm + dropout
    """
    
    def __init__(
        self,
        num_rois: int = NUM_ROIS,
        roi_selection_method: str = ROI_SELECTION_METHOD,
        output_dim: int = None,  # Will use ENCODER_OUTPUT_DIM from config if None
        transformer_input_dim: int = None,  # Will use TRANSFORMER_INPUT_DIM from config if None
        dropout: float = 0.0,
        max_rois: int = MAX_PATCHES,
        temporal_aggregation: str = "mean",  # "mean", "conv", "attention"
        normalize_per_roi: bool = True,  # Z-score normalization per ROI
    ):
        """
        Args:
            num_rois: Number of ROIs (tokens) to create
            roi_selection_method: How to select/group voxels into ROIs
                - "all_voxels": Use all voxels (no grouping, num_rois = num_voxels)
                - "random": Random grouping
                - "learned": Learnable grouping (future)
                - "anatomical": Use anatomical atlases (future)
            output_dim: Output embedding dimension
            transformer_input_dim: Input dimension to transformer layers
            dropout: Dropout probability
            max_rois: Maximum number of ROIs (for positional embedding)
            temporal_aggregation: How to aggregate across TRs
                - "mean": Simple average
                - "conv": Temporal convolution
                - "attention": Temporal attention
            normalize_per_roi: Whether to z-score normalize each ROI
        """
        # Use defaults from config if not provided
        if output_dim is None:
            output_dim = ENCODER_OUTPUT_DIM
        if transformer_input_dim is None:
            transformer_input_dim = TRANSFORMER_INPUT_DIM
        
        super().__init__(output_dim, dropout)
        
        self.num_rois = num_rois
        self.roi_selection_method = roi_selection_method
        self.temporal_aggregation = temporal_aggregation
        self.normalize_per_roi = normalize_per_roi
        
        # ROI grouping will be done on-the-fly based on input shape
        # We'll store voxel-to-ROI mapping if needed
        self.voxel_to_roi = None  # Will be set during first forward pass
        
        # Projection from ROI temporal profile to embedding dimension.
        # We keep ALL TRs as input features instead of averaging to a scalar,
        # so each ROI contributes its full temporal profile (e.g. 10 dims for
        # window_size=10) rather than a single number.  This avoids the rank-1
        # collapse that results from Linear(1, dim).
        #
        # Build eagerly using DEFAULT_WINDOW_SIZE from config so the parameters
        # exist before the optimizer is created. If the actual TR count differs
        # at forward time, _build_projection will rebuild (rare).
        self._transformer_input_dim = transformer_input_dim
        default_window = getattr(fmri_config, "DEFAULT_WINDOW_SIZE", 10)
        self._temporal_dim = default_window
        self.roi_projection = nn.Linear(default_window, transformer_input_dim)
        
        # Positional embeddings for ROIs (small init to avoid exploding gradients)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_rois, transformer_input_dim) * 0.02
        )
        
        # Normalization and dropout
        self.input_norm = nn.LayerNorm(transformer_input_dim)
        self.input_dropout = nn.Dropout(self.dropout)
        
    def _create_roi_mapping(self, num_voxels: int, num_rois: int, method: str) -> torch.Tensor:
        """
        Create mapping from voxels to ROIs.
        
        Returns:
            voxel_to_roi: Tensor of shape (num_voxels,) with ROI indices
        """
        if method == "all_voxels":
            # Each voxel is its own ROI
            return torch.arange(num_voxels, dtype=torch.long)
        
        elif method == "random":
            # Randomly assign voxels to ROIs
            voxel_to_roi = torch.randint(0, num_rois, (num_voxels,))
            return voxel_to_roi
        
        elif method == "learned":
            # TODO: Implement learned ROI assignment
            # For now, fall back to random
            return torch.randint(0, num_rois, (num_voxels,))
        
        elif method == "anatomical":
            # No atlas files available, so use deterministic contiguous blocks:
            # Shuffle voxel indices with a fixed seed (so same voxel count always
            # gives the same mapping), then assign equal-sized contiguous blocks
            # to each ROI. This is reproducible across runs and ensures balanced
            # ROI sizes, unlike torch.randint which varies per run.
            generator = torch.Generator()
            generator.manual_seed(42)
            perm = torch.randperm(num_voxels, generator=generator)
            # Assign each voxel in the permuted order to an ROI round-robin
            voxel_to_roi = torch.zeros(num_voxels, dtype=torch.long)
            voxel_to_roi[perm] = torch.arange(num_voxels, dtype=torch.long) % num_rois
            return voxel_to_roi
        
        else:
            raise ValueError(f"Unknown ROI selection method: {method}")
    
    def _group_voxels_to_rois(
        self, 
        x: torch.Tensor, 
        voxel_to_roi: torch.Tensor
    ) -> torch.Tensor:
        """
        Group voxels into ROIs by averaging.
        
        Args:
            x: (num_voxels, TRs) tensor
            voxel_to_roi: (num_voxels,) tensor mapping voxels to ROI indices
        
        Returns:
            roi_data: (num_rois, TRs) tensor
        """
        num_rois = voxel_to_roi.max().item() + 1
        num_voxels, num_trs = x.shape
        
        # Initialize ROI data
        roi_data = torch.zeros(num_rois, num_trs, device=x.device, dtype=x.dtype)
        
        # Vectorized sum: Accumulate voxels into ROIs
        # This replaces the slow Python loop over 60k voxels
        roi_data.index_add_(0, voxel_to_roi, x)
        
        # Compute counts per ROI for averaging
        ones = torch.ones(num_voxels, device=x.device, dtype=x.dtype)
        roi_counts = torch.zeros(num_rois, device=x.device, dtype=x.dtype)
        roi_counts.index_add_(0, voxel_to_roi, ones)
        
        # Average (avoid division by zero)
        roi_counts = roi_counts.clamp(min=1)
        roi_data = roi_data / roi_counts.unsqueeze(1)
        
        return roi_data
    
    def _build_projection(self, temporal_dim: int):
        """Lazily build the projection layer once we know the TR count."""
        self._temporal_dim = temporal_dim
        self.roi_projection = nn.Linear(temporal_dim, self._transformer_input_dim)
        # Move to same device/dtype as pos_embed
        self.roi_projection = self.roi_projection.to(
            device=self.pos_embed.device, dtype=self.pos_embed.dtype
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize fMRI data.
        
        Args:
            x: FloatTensor of shape [B, voxels, TRs] or [B, num_rois, TRs]
               For compatibility with OpenTSLM's batch processing, 
               we also support [B, L] where L = voxels * TRs (flattened)
        
        Returns:
            FloatTensor of shape [B, num_rois, embed_dim]
        """
        # Handle different input shapes
        if x.ndim == 2:
            # [B, L] - flattened by _encode_vision_x rearrange.
            # Unflatten back to (B, voxels, TRs) using DEFAULT_WINDOW_SIZE.
            B, L = x.shape
            num_trs = self._temporal_dim  # from config DEFAULT_WINDOW_SIZE
            num_voxels = L // num_trs
            if num_voxels * num_trs != L:
                raise ValueError(
                    f"Cannot unflatten 2D input (B={B}, L={L}) into (B, voxels, TRs={num_trs}). "
                    f"L must be divisible by TRs."
                )
            x = x.view(B, num_voxels, num_trs)
        
        elif x.ndim == 3:
            # [B, voxels, TRs] or [B, num_rois, TRs]
            B, num_features, num_trs = x.shape
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected [B, voxels, TRs] or [B, num_rois, TRs]")
        
        # Normalize per ROI/voxel if requested
        if self.normalize_per_roi:
            # Z-score normalization: (x - mean) / std per feature
            x_mean = x.mean(dim=2, keepdim=True)  # (B, features, 1)
            x_std = x.std(dim=2, keepdim=True) + 1e-8  # (B, features, 1)
            x = (x - x_mean) / x_std
        
        # Group voxels into ROIs if needed (skip if already pre-grouped to num_rois)
        if num_features != self.num_rois:
            # Need to group voxels into ROIs
            if self.voxel_to_roi is None or self.voxel_to_roi.shape[0] != num_features:
                # Create ROI mapping on first pass
                self.voxel_to_roi = self._create_roi_mapping(
                    num_features, 
                    self.num_rois, 
                    self.roi_selection_method
                ).to(x.device)
            
            # Group voxels to ROIs for each sample in batch
            roi_data_list = []
            for b in range(B):
                roi_data = self._group_voxels_to_rois(x[b], self.voxel_to_roi)
                roi_data_list.append(roi_data)
            roi_data = torch.stack(roi_data_list, dim=0)  # (B, num_rois, TRs)
        else:
            # Already in ROI format
            roi_data = x  # (B, num_rois, TRs)
        
        # Project full temporal profile to embedding dimension.
        # roi_data is (B, num_rois, TRs) — we keep all TRs as input features
        # so that Linear(TRs, dim) can learn which temporal patterns matter.
        num_trs = roi_data.shape[2]
        if self.roi_projection is None or self._temporal_dim != num_trs:
            self._build_projection(num_trs)

        # (B, num_rois, TRs) -> (B, num_rois, transformer_input_dim)
        embeddings = self.roi_projection(roi_data)
        
        # Add positional embeddings
        num_rois_actual = embeddings.size(1)
        if num_rois_actual > self.pos_embed.size(1):
            raise ValueError(
                f"Number of ROIs ({num_rois_actual}) exceeds max_rois ({self.pos_embed.size(1)}). "
                f"Increase max_rois parameter."
            )
        pos = self.pos_embed[:, :num_rois_actual, :]  # (1, num_rois, transformer_input_dim)
        embeddings = embeddings + pos  # (B, num_rois, transformer_input_dim)
        
        # Normalize and dropout
        embeddings = self.input_norm(embeddings)
        embeddings = self.input_dropout(embeddings)
        
        return embeddings  # (B, num_rois, transformer_input_dim)
