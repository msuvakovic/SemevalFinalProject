import torch
import torch.nn as nn
from transformers import AutoModel

class NeoBERTDetector(nn.Module):
    def __init__(self, manual_feature_dim, num_labels=2):
        super(NeoBERTDetector, self).__init__()
        
        # Load NeoBERT directly as requested
        # Note: We use the base model to get embeddings, not MaskedLM
        self.neobert = AutoModel.from_pretrained(
            "chandar-lab/NeoBERT", 
            trust_remote_code=True
        )
        
        # Freeze NeoBERT layers for faster ablation study (optional, unfreeze for best performance)
        for param in self.neobert.parameters():
            param.requires_grad = False
            
        # NeoBERT hidden size is 768
        self.bert_hidden_size = 768
        
        # Classification Head: BERT Embedding + Psycholinguistic Features
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_hidden_size + manual_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, manual_features):
        # 1. Get NeoBERT Embeddings
        outputs = self.neobert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the representation of the [CLS] token (first token)
        # Check if model outputs pooler_output, if not use last_hidden_state[:,0,:]
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            cls_embedding = outputs.pooler_output
        else:
            cls_embedding = outputs.last_hidden_state[:, 0, :]

        # 2. Concatenate with Psycholinguistic Features
        if manual_features.shape[1] > 0:
            combined = torch.cat((cls_embedding, manual_features), dim=1)
        else:
            combined = cls_embedding

        # 3. Classify
        logits = self.classifier(combined)
        return logits