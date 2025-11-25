import pandas as pd
import torch
from torch.utils.data import Dataset
from features import PsychoLinguisticExtractor

class ConspiracyDataset(Dataset):
    def __init__(self, tokenizer, data_path_true, data_path_fake, max_len=256, 
                 use_sentiment=True, use_complexity=True, use_pronouns=True):
        
        print("Loading datasets...")
        # Load Kaggle Dataset
        df_true = pd.read_csv(data_path_true)
        df_true['label'] = 0 # Not Conspiracy
        
        df_fake = pd.read_csv(data_path_fake)
        df_fake['label'] = 1 # Conspiracy/Fake
        
        # Balance dataset for speed/demo purposes (optional: remove .head() for full run)
        df = pd.concat([df_true.head(1000), df_fake.head(1000)]).sample(frac=1).reset_index(drop=True)
        
        self.texts = df['text'].astype(str).tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Initialize Feature Extractor
        self.extractor = PsychoLinguisticExtractor()
        self.use_sentiment = use_sentiment
        self.use_complexity = use_complexity
        self.use_pronouns = use_pronouns
        
        # Pre-calculate manual features to save time during training
        print("Extracting psycholinguistic features...")
        self.manual_features = [
            self.extractor.get_features(t, use_sentiment, use_complexity, use_pronouns) 
            for t in self.texts
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        manual_feat = self.manual_features[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'manual_features': torch.tensor(manual_feat, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }