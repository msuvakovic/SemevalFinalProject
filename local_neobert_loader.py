import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download
import types # Make sure this is imported at the top!
import importlib.util

# ... (setup_xformers_mock and other constants remain the same) ...

def patch_and_load_neobert(manual_feature_dim, device):
    """Downloads, patches, and loads NeoBERT into a safe environment."""
    
    # ... (Step 0, 1, 2, 3 remain the same) ...

    # 4. Dynamically load the model from the patched local file
    try:
        # Load config and pass trust_remote_code=True
        config = AutoConfig.from_pretrained(REPO_ID, trust_remote_code=True)
        config.model_type = "neobert"
        
        # --- FIX: New Module Execution Logic ---
        module_name = "transformers_local_neobert"
        
        # 4a. Load the module spec
        spec = importlib.util.spec_from_file_location(module_name, local_path)
        local_module = importlib.util.module_from_spec(spec)
        
        # 4b. Temporarily register the module in sys.modules
        sys.modules[module_name] = local_module
        
        # 4c. Execute the module (This is where the patched code runs)
        spec.loader.exec_module(local_module)

        # 4d. Use the module to define the class and load the weights
        # Identify the main model class
        ModelClass = getattr(local_module, 'NeoBERTModel', None)
        if not ModelClass:
             ModelClass = getattr(local_module, 'NeoBertModel', None)
        
        if ModelClass is None:
             raise ImportError("Could not find NeoBERTModel or NeoBertModel class in patched file.")

        # Instantiate the patched model class and load the original weights
        # trust_remote_code=True is crucial here to suppress the final interactive prompt
        neobert_model = ModelClass.from_pretrained(
            REPO_ID, 
            config=config, 
            trust_remote_code=True,
            # We explicitly tell transformers to look in the module we just loaded!
            _from_pipeline=module_name 
        )
        
        # 4e. Clean up the temporary module
        del sys.modules[module_name]
        
        return neobert_model

    except Exception as e:
        # Log the specific error that is preventing successful loading
        print(f"🛑 Error loading patched model: {e}")
        return None