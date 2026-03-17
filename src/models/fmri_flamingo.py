"""
FMRIFlamingo: Flamingo architecture adapted for fMRI data.

Wraps OpenTSLM's Flamingo architecture, replacing CNNTokenizer with FMRITokenizer.
Handles fMRI-specific data format: (voxels, TRs) instead of 1D time series.
"""

import math
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any, Optional
from types import SimpleNamespace

from transformers import AutoTokenizer, AutoModelForCausalLM
from open_flamingo.src.utils import extend_instance
from open_flamingo.src.flamingo_lm import FlamingoLMMixin

# Import OpenTSLM components directly to avoid package initialization issues
import sys
from pathlib import Path
import importlib.util

FMRI_FLAMINGO_DIR = Path(__file__).parent.parent.parent
BASE_DIR = FMRI_FLAMINGO_DIR.parent
# Use in-repo OpenTSLM (src/opentslm) so no separate clone is needed
OPENTSLM_DIR = FMRI_FLAMINGO_DIR / "src"

# Import TimeSeriesFlamingoWithTrainableEncoder directly
flamingo_path = OPENTSLM_DIR / "opentslm" / "model" / "llm" / "TimeSeriesFlamingoWithTrainableEncoder.py"
spec_flamingo = importlib.util.spec_from_file_location("TimeSeriesFlamingoWithTrainableEncoder", flamingo_path)
flamingo_module = importlib.util.module_from_spec(spec_flamingo)
spec_flamingo.loader.exec_module(flamingo_module)
TimeSeriesFlamingoWithTrainableEncoder = flamingo_module.TimeSeriesFlamingoWithTrainableEncoder

# Import config
config_path_opentslm = OPENTSLM_DIR / "opentslm" / "model_config.py"
spec_config = importlib.util.spec_from_file_location("opentslm.model_config", config_path_opentslm)
opentslm_config = importlib.util.module_from_spec(spec_config)
spec_config.loader.exec_module(opentslm_config)
ENCODER_OUTPUT_DIM_DEFAULT = opentslm_config.ENCODER_OUTPUT_DIM
from einops import rearrange

# Import our tokenizer
from src.models.fmri_tokenizer import FMRITokenizer

# Import config
import importlib.util
config_path = FMRI_FLAMINGO_DIR / "config.py"
spec = importlib.util.spec_from_file_location("fmri_config", config_path)
fmri_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmri_config)

# Get config values
LLM_ID = fmri_config.LLM_ID
CROSS_ATTN_EVERY_N_LAYERS = fmri_config.CROSS_ATTN_EVERY_N_LAYERS
GRADIENT_CHECKPOINTING = fmri_config.GRADIENT_CHECKPOINTING
NUM_ROIS = fmri_config.NUM_ROIS
ROI_SELECTION_METHOD = fmri_config.ROI_SELECTION_METHOD
ENCODER_OUTPUT_DIM = fmri_config.ENCODER_OUTPUT_DIM
DROPOUT = fmri_config.DROPOUT
ATTN_IMPLEMENTATION = getattr(fmri_config, "ATTN_IMPLEMENTATION", "eager")
CROSS_ATTN_GATE_INIT = getattr(fmri_config, "CROSS_ATTN_GATE_INIT", 0.0)


class FMRIFlamingoWithTrainableEncoder(TimeSeriesFlamingoWithTrainableEncoder):
    """
    Custom Flamingo encoder that handles fMRI data format.
    
    Overrides _encode_vision_x to handle (B, voxels, TRs) format
    instead of 1D time series.
    """
    
    def _encode_vision_x(self, vision_x):
        """
        Handle fMRI data format: (B, T, F, voxels*TRs) -> reshape and tokenize.
        
        Args:
            vision_x: Tensor of shape (b, T, F, voxels*TRs) where voxels*TRs is flattened
        """
        if vision_x.ndim == 4:
            b, T, F, flattened = vision_x.shape
            
            # Flatten batch, time and frame dimensions
            vision_x_flat = rearrange(vision_x, "b T F c -> (b T F) c")
            
            # Infer dimensions from flattened size
            # We know from pad_and_apply_batch that we flatten (voxels, TRs)
            # Try to infer TRs from config first, then use heuristic
            trs = None
            try:
                # Try to get default window size from config
                DEFAULT_WINDOW_SIZE = fmri_config.DEFAULT_WINDOW_SIZE
                if flattened % DEFAULT_WINDOW_SIZE == 0:
                    trs = DEFAULT_WINDOW_SIZE
            except:
                pass
            
            if trs is None:
                # Heuristic: try common TR window sizes (typical: 5-30 TRs)
                possible_trs = [10, 15, 20, 25, 30, 5, 8, 12]
                for tr in possible_trs:
                    if flattened % tr == 0:
                        voxels_candidate = flattened // tr
                        # Check if voxels count is reasonable (typical: 50k-100k for Huth)
                        if 10000 <= voxels_candidate <= 200000:
                            trs = tr
                            break
            
            if trs is None:
                # Fallback: try all reasonable TR values
                import math
                for tr in range(5, 35):
                    if flattened % tr == 0:
                        voxels_candidate = flattened // tr
                        if 10000 <= voxels_candidate <= 200000:
                            trs = tr
                            break
                
                if trs is None:
                    raise ValueError(
                        f"Cannot infer TRs from flattened size {flattened}. "
                        f"Expected to be divisible by a reasonable TR count (5-30). "
                        f"This suggests a mismatch in data format."
                    )
            
            voxels = flattened // trs
            
            if voxels * trs != flattened:
                raise ValueError(
                    f"Cannot infer dimensions: {flattened} is not divisible by {trs}. "
                    f"Expected flattened = voxels * TRs. "
                    f"Consider storing shape information explicitly."
                )
            
            # Reshape to (b*T*F, voxels, TRs)
            vision_x_reshaped = vision_x_flat.view(-1, voxels, trs)
            
            # Process through encoder (FMRITokenizer)
            # Input: (b*T*F, voxels, TRs) -> Output: (b*T*F, num_rois, embed_dim)
            # vision_encoder might be a SimpleNamespace wrapper, so get the actual encoder
            if hasattr(self.vision_encoder, 'visual'):
                # It's a SimpleNamespace wrapper (from OpenTSLM pattern)
                encoder = self.vision_encoder.visual
            else:
                # It's the encoder directly
                encoder = self.vision_encoder
            vision_x = encoder(vision_x_reshaped)
            
            # Reshape back to (b, T, F, num_rois, embed_dim)
            vision_x = rearrange(vision_x, "(b T F) p d -> b T F p d", b=b, T=T, F=F)
            
            # Process through perceiver
            vision_x = self.perceiver(vision_x)
        else:
            # Fall back to parent implementation for other formats
            return super()._encode_vision_x(vision_x)
        
        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)


class FMRIFlamingo(nn.Module):
    """
    Flamingo model adapted for fMRI data.
    
    Reuses OpenTSLM's Flamingo architecture but:
    - Uses FMRITokenizer instead of CNNTokenizer
    - Handles fMRI data format: (voxels, TRs) instead of 1D time series
    - Adapts pad_and_apply_batch for fMRI-specific batching
    """
    
    def __init__(
        self,
        device: str = "cuda",
        llm_id: str = LLM_ID,
        cross_attn_every_n_layers: int = CROSS_ATTN_EVERY_N_LAYERS,
        freeze_lm_embeddings: bool = False,
        num_rois: int = NUM_ROIS,
        roi_selection_method: str = ROI_SELECTION_METHOD,
        temporal_aggregation: str = "mean",
        gradient_checkpointing: bool = GRADIENT_CHECKPOINTING,
        **flamingo_kwargs,
    ):
        """
        Args:
            device: Device to run on ("cuda" or "cpu")
            llm_id: HuggingFace model ID for language model
            cross_attn_every_n_layers: Cross-attention every N decoder layers
            freeze_lm_embeddings: Whether to freeze LLM input embeddings
            num_rois: Number of ROIs for tokenization
            roi_selection_method: ROI selection method ("random", "all_voxels", etc.)
            temporal_aggregation: Temporal aggregation method ("mean", "conv", "attention")
            gradient_checkpointing: Whether to use gradient checkpointing
            **flamingo_kwargs: Additional arguments for Flamingo
        """
        super().__init__()
        
        self.device = device
        print(f"FMRIFlamingo Using device: {self.device}")
        
        # Create FMRITokenizer instead of CNNTokenizer
        # Note: We disable per-ROI normalization because Huth data is already 
        # globally Z-scored. Local window normalization would destroy amplitude info.
        time_series_encoder = FMRITokenizer(
            num_rois=num_rois,
            roi_selection_method=roi_selection_method,
            temporal_aggregation=temporal_aggregation,
            output_dim=ENCODER_OUTPUT_DIM,
            normalize_per_roi=False,
            dropout=DROPOUT,
        ).to(device)
        
        # Load text tokenizer
        text_tokenizer = AutoTokenizer.from_pretrained(
            llm_id,
            local_files_only=False,
            trust_remote_code=True,
            cache_dir=None,
        )
        
        # Load language model
        # Use low_cpu_mem_usage=True to prevent OOM during loading
        # This avoids creating two copies in RAM (random init + pretrained weights)
        # Without this, loading can use 2x model size in RAM before moving to GPU
        lang_encoder = AutoModelForCausalLM.from_pretrained(
            llm_id,
            local_files_only=False,
            trust_remote_code=True,
            cache_dir=None,
            device_map={"": device},
            attn_implementation=ATTN_IMPLEMENTATION,
            low_cpu_mem_usage=True,  # Critical: Prevents OOM during model loading
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,  # Reduce RAM usage
        )
        
        # Add Flamingo special tokens
        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
        )
        if text_tokenizer.pad_token is None:
            text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            text_tokenizer.pad_token = "<PAD>"
        
        # Convert LM to FlamingoLM
        extend_instance(lang_encoder, FlamingoLMMixin)
        
        # Infer decoder layers attribute name
        decoder_layers_attr_name = self._infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        lang_encoder.resize_token_embeddings(len(text_tokenizer))
        
        # Fix compatibility for Gemma3Config
        if hasattr(lang_encoder.config, "text_config") and hasattr(
            lang_encoder.config.text_config, "hidden_size"
        ):
            if not hasattr(lang_encoder.config, "hidden_size"):
                lang_encoder.config.hidden_size = (
                    lang_encoder.config.text_config.hidden_size
                )
        
        # Remove gradient_checkpointing from flamingo_kwargs if present to avoid duplicate
        flamingo_kwargs.pop('gradient_checkpointing', None)
        
        # Wrap encoder in nn.Module (not SimpleNamespace) so PyTorch discovers its parameters
        class _VisionEncoderWrapper(nn.Module):
            def __init__(self, visual):
                super().__init__()
                self.visual = visual
            def forward(self, x):
                return self.visual(x)

        # Create Flamingo model with custom fMRI encoder
        model = FMRIFlamingoWithTrainableEncoder(
            _VisionEncoderWrapper(time_series_encoder),
            lang_encoder,
            text_tokenizer.encode("<|endofchunk|>")[-1],
            text_tokenizer.encode("<image>")[-1],
            vis_dim=ENCODER_OUTPUT_DIM if 'ENCODER_OUTPUT_DIM' in dir() else ENCODER_OUTPUT_DIM_DEFAULT,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            gradient_checkpointing=gradient_checkpointing,
            **flamingo_kwargs,
        )
        
        # Freeze all parameters initially
        model.requires_grad_(False)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0
        
        # Unfreeze trainable components
        model.perceiver.requires_grad_(True)
        model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        if not freeze_lm_embeddings:
            model.lang_encoder.get_input_embeddings().requires_grad_(True)
        
        # Unfreeze encoder (FMRITokenizer)
        # vision_encoder might be a SimpleNamespace with .visual attribute, or the encoder directly
        if hasattr(model.vision_encoder, 'visual'):
            # It's a SimpleNamespace wrapper
            model.vision_encoder.visual.requires_grad_(True)
        else:
            # It's the encoder directly
            model.vision_encoder.requires_grad_(True)
        
        # Optional: start with cross-attn gate partly open so vision path is used from step 1 (default Flamingo gate=0 => tanh(0)=0)
        if CROSS_ATTN_GATE_INIT != 0.0:
            init_val = min(0.99, max(0.01, float(CROSS_ATTN_GATE_INIT)))
            gate_val = math.atanh(init_val)
            for layer in model.lang_encoder.gated_cross_attn_layers:
                if layer is not None and hasattr(layer, "attn_gate") and layer.attn_gate is not None:
                    with torch.no_grad():
                        layer.attn_gate.data.fill_(gate_val)
            print(f"FMRIFlamingo: cross-attn gate init set to tanh(gate)={init_val} (vision path on from start)")
        
        # Ensure entire model is on correct device
        # (Some components like perceiver might not be moved by device_map)
        model = model.to(device)
        
        self.model = model
        self.llm = model
        self.text_tokenizer = text_tokenizer
    
    def _infer_decoder_layers_attr_name(self, model) -> str:
        """Infer the attribute name for decoder layers."""
        __KNOWN_DECODER_LAYERS_ATTR_NAMES = {
            "opt": "model.decoder.layers",
            "gptj": "transformer.h",
            "gpt-j": "transformer.h",
            "pythia": "gpt_neox.layers",
            "llama": "model.layers",
            "gptneoxforcausallm": "gpt_neox.layers",
            "mpt": "transformer.blocks",
            "mosaicgpt": "transformer.blocks",
            "gemma": "model.layers",
            "gemma2": "model.layers",
            "gemma3": "model.layers",
            "medgemma": "model.layers",
        }
        
        model_class_name = model.__class__.__name__
        if "gemma3" in model_class_name.lower():
            if "ConditionalGeneration" in model_class_name:
                return "language_model.layers"
            else:
                return "model.layers"
        
        for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
            if k.lower() in model_class_name.lower():
                return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]
        
        raise ValueError(
            f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. "
            f"Please supply this string manually."
        )
    
    def pad_and_apply_batch(
        self, batch: List[Dict[str, Any]], include_labels: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Process batch of fMRI samples for Flamingo.
        
        Args:
            batch: List of dicts with keys:
                - 'time_series': List of 1D arrays (one per voxel) - from HuthFMRIDataset
                - 'pre_prompt': str
                - 'post_prompt': str
                - 'answer': str
            include_labels: Whether to include labels for training
        
        Returns:
            input_ids: Tokenized text inputs
            images: Tokenized fMRI data (ready for vision encoder)
            attention_mask: Attention mask
            labels: Labels for training (if include_labels=True)
        """
        def pad_fmri_data(batch, max_length=None):
            """
            Convert fMRI data from list of 1D arrays to tensor format.
            
            Input format: 
            - If DataLoader collated: dict with 'time_series' key containing list of lists
            - If list of dicts: list of dicts with 'time_series' key
            Output format: (B, voxels, TRs) tensor
            """
            fmri_data_list = []
            
            # Handle DataLoader collated format (dict of lists)
            if isinstance(batch, dict) and "time_series" in batch:
                # DataLoader has collated: batch['time_series'] is a list of time_series (one per sample)
                time_series_list = batch["time_series"]
            else:
                # List of dicts format (one dict per sample)
                time_series_list = [item["time_series"] for item in batch]
            
            for time_series in time_series_list:
                # Fast path: time_series is already (voxels, TRs) tensor from dataset
                if isinstance(time_series, torch.Tensor) and time_series.dim() == 2:
                    fmri_data_list.append(time_series.float())
                    continue
                # Legacy: time_series is list of 1D arrays (one per voxel)
                voxel_tensors = []
                for ts in time_series:
                    if isinstance(ts, torch.Tensor):
                        if ts.dim() == 0:
                            ts = ts.unsqueeze(0)
                        voxel_tensors.append(ts.float())
                    elif isinstance(ts, (list, tuple)):
                        voxel_tensors.append(torch.tensor(ts, dtype=torch.float32))
                    else:
                        voxel_tensors.append(torch.as_tensor(ts, dtype=torch.float32))
                fmri_data_list.append(torch.stack(voxel_tensors, dim=0))
            
            # Determine max TR length and max voxel count
            if max_length is None:
                max_length = max(ts.shape[1] for ts in fmri_data_list)
            max_voxels = max(ts.shape[0] for ts in fmri_data_list)
            
            # Pad all to same (voxels, TRs)
            padded_list = []
            for ts in fmri_data_list:
                voxels, trs = ts.shape
                # Pad or truncate TR dimension
                if trs < max_length:
                    padding = torch.zeros(voxels, max_length - trs, dtype=ts.dtype, device=ts.device)
                    padded = torch.cat([ts, padding], dim=1)
                else:
                    padded = ts[:, :max_length]
                # Pad or truncate voxel dimension so all samples stack
                if voxels < max_voxels:
                    pad_rows = torch.zeros(max_voxels - voxels, max_length, dtype=padded.dtype, device=padded.device)
                    padded = torch.cat([padded, pad_rows], dim=0)
                elif voxels > max_voxels:
                    padded = padded[:max_voxels, :]
                padded_list.append(padded)
            
            # Stack into (B, voxels, TRs)
            return torch.stack(padded_list, dim=0)
        
        cast_dtype = None
        tokenizer = self.text_tokenizer
        media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]
        
        # Process fMRI data
        # Output: (B, voxels, TRs)
        images = pad_fmri_data(batch).to(
            self.device, dtype=cast_dtype, non_blocking=True
        )
        # Reshape for FMRIFlamingoWithTrainableEncoder
        # It expects (b, T, F, features) where features is flattened voxels*TRs
        B, voxels, trs = images.shape
        # Flatten voxels and TRs: (B, voxels*TRs)
        images_flat = images.view(B, voxels * trs)
        # Add time and frame dimensions: (B, 1, 1, voxels*TRs)
        images = images_flat.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, voxels*TRs)
        
        # Process text inputs
        text_inputs = []
        prompt_lengths = []
        
        # Check if batch is pre-tokenized (from updated HuthFMRIDataset)
        is_pretokenized = False
        if isinstance(batch, dict) and "input_ids" in batch:
            is_pretokenized = True
        elif isinstance(batch, list) and len(batch) > 0 and "input_ids" in batch[0]:
            is_pretokenized = True
            
        if is_pretokenized:
            # Pre-tokenized path
            if isinstance(batch, dict):
                # collated
                text_inputs = batch["input_ids"]
                # prompt_len might be tensor or list
                p_lens = batch["prompt_len"]
                if isinstance(p_lens, torch.Tensor):
                    prompt_lengths = p_lens.tolist()
                else:
                    prompt_lengths = p_lens
            else:
                # list of dicts
                text_inputs = [item["input_ids"] for item in batch]
                prompt_lengths = [item["prompt_len"] for item in batch]
        else:
            # Legacy path: Tokenize on the fly
            # Handle DataLoader collated format (dict of lists) or list of dicts
            if isinstance(batch, dict) and "pre_prompt" in batch:
                # DataLoader has collated: batch['pre_prompt'] is a list (one per sample)
                batch_size = len(batch.get("time_series", batch.get("pre_prompt", [])))
                pre_prompts = batch.get("pre_prompt", [""] * batch_size)
                post_prompts = batch.get("post_prompt", [""] * batch_size)
                answers = batch.get("answer", [""] * batch_size)
                items = zip(pre_prompts, post_prompts, answers)
            else:
                # List of dicts format
                items = [(item.get("pre_prompt", ""), item.get("post_prompt", ""), item.get("answer", "")) for item in batch]
            
            for pre_prompt, post_prompt, answer in items:
                # Format: pre_prompt <image> post_prompt answer <|endofchunk|>
                if pre_prompt:
                    text = f"{pre_prompt} <image> {post_prompt} {answer} <|endofchunk|>"
                else:
                    text = f"<image> {post_prompt} {answer} <|endofchunk|>"
                
                # Tokenize
                tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                text_inputs.append(tokens)
                
                # Track prompt length (before answer)
                prompt_text = f"{pre_prompt} <image> {post_prompt}" if pre_prompt else f"<image> {post_prompt}"
                prompt_tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                prompt_lengths.append(len(prompt_tokens))
        
        # Pad text inputs
        max_text_len = max(len(t) for t in text_inputs)
        padded_text = []
        attention_masks = []
        
        for tokens, prompt_len in zip(text_inputs, prompt_lengths):
            # Pad to max length (use tokens.device so padding works when input_ids are on GPU)
            _device = getattr(tokens, "device", None)
            if len(tokens) < max_text_len:
                padding = torch.full(
                    (max_text_len - len(tokens),),
                    tokenizer.pad_token_id,
                    dtype=tokens.dtype,
                    device=_device,
                )
                padded = torch.cat([tokens, padding])
                mask = torch.cat([
                    torch.ones(len(tokens), dtype=torch.bool, device=_device),
                    torch.zeros(max_text_len - len(tokens), dtype=torch.bool, device=_device),
                ])
            else:
                padded = tokens
                mask = torch.ones(len(tokens), dtype=torch.bool, device=_device)
            
            padded_text.append(padded)
            attention_masks.append(mask)
        
        input_ids = torch.stack(padded_text).to(self.device)
        attention_mask = torch.stack(attention_masks).to(self.device)
        
        # Create labels if needed
        labels = None
        if include_labels:
            labels = input_ids.clone()
            # Mask out prompt tokens (set to -100)
            for i, prompt_len in enumerate(prompt_lengths):
                labels[i, :prompt_len] = -100
            
            # Mask out padding tokens (set to -100)
            labels[attention_mask == 0] = -100
        
        return input_ids, images, attention_mask, labels
    
    def compute_loss(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        """Compute loss for a batch."""
        # Use provided lang_x and labels if available (e.g. from masked training)
        if isinstance(batch, dict) and "lang_x" in batch and "labels" in batch:
            # Manual batch construction from training loop
            # We still need to process vision_x
            _, images, _, _ = self.pad_and_apply_batch(batch, include_labels=False)
            
            input_ids = batch["lang_x"]
            labels = batch["labels"]
            attention_mask = batch.get("attention_mask")
            
            if attention_mask is None:
                # Infer mask from input_ids (assuming pad is 0 or tokenizer.pad_token_id)
                # We don't have easy access to tokenizer pad id here without self.text_tokenizer
                # But pad_and_apply_batch usually handles this.
                # If manual batch, we expect attention_mask to be provided.
                # Fallback: assume non-zero is valid
                attention_mask = (input_ids != self.text_tokenizer.pad_token_id).long()
                
        else:
            # Standard path
            input_ids, images, attention_mask, labels = self.pad_and_apply_batch(
                batch, include_labels=True
            )
        
        # The model expects vision_x in format (B, T, F, ...)
        # Our images are (B, 1, voxels, TRs)
        # TimeSeriesFlamingoWithTrainableEncoder will handle the reshaping
        
        output = self.model(
            vision_x=images,
            lang_x=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return output[0]
    
    def generate(
        self, batch: List[Dict[str, Any]], max_new_tokens: int = 50, **generate_kwargs
    ) -> List[str]:
        """Generate predictions for a batch."""
        # Temporarily disable compilation to avoid data-dependent operation issues
        original_disable = torch._dynamo.config.disable
        torch._dynamo.config.disable = True

        try:
            with torch.inference_mode():
                input_ids, images, attention_mask, _ = self.pad_and_apply_batch(
                    batch, include_labels=False
                )

                gen_ids = self.llm.generate(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    **generate_kwargs,
                )

                # Remove input ids from generation
                answer_only_ids = gen_ids[:, input_ids.shape[1] :]

                return self.text_tokenizer.batch_decode(
                    answer_only_ids, skip_special_tokens=True
                )
        finally:
            # Restore original compilation setting
            torch._dynamo.config.disable = original_disable
    
    def get_eos_token(self) -> str:
        """Get end-of-sequence token."""
        return self.text_tokenizer.eos_token
