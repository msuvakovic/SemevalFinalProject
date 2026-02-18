#!/usr/bin/env python3
"""
Debug inference issues:
1. Verify special tokens and IDs.
2. Test text-only generation.
3. Test generation with dummy fMRI input.
4. Inspect real data statistics.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
FMRI_FLAMINGO_DIR = SCRIPT_DIR.parent
if str(FMRI_FLAMINGO_DIR) not in sys.path:
    sys.path.insert(0, str(FMRI_FLAMINGO_DIR))

from src.models.fmri_flamingo import FMRIFlamingo
import config as fmri_config

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    device = get_device()
    print(f"🔍 Debugging on {device}")
    
    # 1. Load Tokenizer
    print("\n1️⃣  Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        fmri_config.LLM_ID,
        local_files_only=False,
        trust_remote_code=True,
    )
    
    # Add special tokens (same as in FMRIFlamingo.__init__)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        tokenizer.pad_token = "<PAD>"
        
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"   <image> ID: {tokenizer.convert_tokens_to_ids('<image>')}")
    print(f"   <|endofchunk|> ID: {tokenizer.convert_tokens_to_ids('<|endofchunk|>')}")
    
    # 2. Load Model
    print("\n2️⃣  Loading Model...")
    checkpoint_path = FMRI_FLAMINGO_DIR / "checkpoints" / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = FMRIFlamingo(
        device=device,
        llm_id=fmri_config.LLM_ID,
        num_rois=fmri_config.NUM_ROIS,
        cross_attn_every_n_layers=fmri_config.CROSS_ATTN_EVERY_N_LAYERS,
        gradient_checkpointing=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    
    # 3. Test Text-Only Generation (Bypassing fMRI)
    print("\n3️⃣  Test Text-Only Generation (Direct LLM)...")
    prompt = "The quick brown fox jumps over the lazy"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    try:
        # We access the internal LLM directly
        # Note: The LLM has Flamingo layers injected. If we don't provide vision_x,
        # we need to ensure the cross-attention layers handle it (usually they expect it).
        # However, we can try to generate without vision_x if the forward pass allows None.
        # Looking at OpenTSLM code, it might fail if vision_x is missing.
        # Let's try passing dummy vision_x.
        
        # Create dummy vision_x: (B, 1, 1, voxels*TRs) -> (1, 1, 1, 128*10)?
        # No, FMRIFlamingo expects (B, 1, voxels, TRs) in pad_and_apply_batch
        # But here we are calling model.llm.generate directly?
        # model.llm is FMRIFlamingoWithTrainableEncoder
        # It expects vision_x processed by encoder?
        
        # Let's use model.generate with dummy input
        dummy_batch = [{
            'time_series': torch.zeros(fmri_config.NUM_ROIS, 10), # Dummy fMRI
            'input_ids': inputs['input_ids'][0],
            'prompt_len': len(inputs['input_ids'][0])
        }]
        
        # But we want to see if the LLM ITSELF produces garbage.
        # If we pass zeros as fMRI, it should produce something reasonable if the prompt is strong.
        
        print(f"   Prompt: '{prompt}'")
        outputs = model.generate(
            dummy_batch, 
            max_new_tokens=10,
            do_sample=False
        )
        print(f"   Output with Zero fMRI: '{outputs[0]}'")
        
    except Exception as e:
        print(f"   ❌ Generation failed: {e}")

    # 4. Inspect Real Data
    print("\n4️⃣  Inspect Real Data Sample...")
    from src.datasets.huth_fmri_dataset import HuthFMRIDataset
    dataset = HuthFMRIDataset(
        split="test",
        subject_ids=["UTS03"],
        story_names=["wheretheressmoke"],
        window_size=fmri_config.DEFAULT_WINDOW_SIZE,
        stride=fmri_config.DEFAULT_STRIDE,
        tokenizer=tokenizer,
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        fmri_data = sample['time_series']
        print(f"   fMRI Shape: {fmri_data.shape}")
        print(f"   fMRI Mean: {fmri_data.mean():.4f}, Std: {fmri_data.std():.4f}")
        print(f"   fMRI Min: {fmri_data.min():.4f}, Max: {fmri_data.max():.4f}")
        
        # Decode input_ids
        input_ids = sample['input_ids']
        decoded = tokenizer.decode(input_ids)
        print(f"   Decoded Input: '{decoded}'")
        
        # Try generating from this sample
        print("   Generating from real sample...")
        real_batch = [{
            'time_series': fmri_data,
            'input_ids': input_ids[:sample['prompt_len']], # Slice to prompt
            'prompt_len': sample['prompt_len']
        }]
        
        outputs = model.generate(
            real_batch,
            max_new_tokens=10,
            do_sample=False
        )
        print(f"   Output: '{outputs[0]}'")
        
        # Check if output contains special tokens
        out_ids = tokenizer(outputs[0], add_special_tokens=False)['input_ids']
        print(f"   Output Token IDs: {out_ids}")

if __name__ == "__main__":
    main()
