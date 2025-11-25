# import torch
# import copy
# import numpy as np
# import argparse
# import warnings
# from torch.utils.data import DataLoader, random_split
# from transformers import AutoTokenizer
# from sklearn.metrics import accuracy_score, classification_report, f1_score
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# # Import your modules
# from dataset_loader import ConspiracyDataset
# from features import PsychoLinguisticExtractor
# from model import NeoBERTDetector

# warnings.filterwarnings("ignore")

# class EarlyStopping:
#     """Stops training if validation loss doesn't improve after a given patience."""
#     def __init__(self, patience=3, delta=0):
#         self.patience = patience
#         self.counter = 0
#         self.best_loss = None
#         self.early_stop = False
#         self.delta = delta
#         self.best_model_wts = None

#     def __call__(self, val_loss, model):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#             self.best_model_wts = copy.deepcopy(model.state_dict())
#         elif val_loss > self.best_loss + self.delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_loss = val_loss
#             self.best_model_wts = copy.deepcopy(model.state_dict())
#             self.counter = 0

# def train(args):
#     print(f"\n--- Starting Ablation Run ---")
#     print(f"Features :: Sentiment: {args.sentiment} | Complexity: {args.complexity} | Pronouns: {args.pronouns}")

#     # 1. Setup Device (MPS for Mac)
#     if torch.backends.mps.is_available():
#         device = torch.device('mps')
#         print("Using device: MPS (Apple Silicon GPU)")
#     elif torch.cuda.is_available():
#         device = torch.device('cuda')
#         print("Using device: CUDA")
#     else:
#         device = torch.device('cpu')
#         print("Using device: CPU")

#     # 2. Load Tokenizer
#     tokenizer = AutoTokenizer.from_pretrained("chandar-lab/NeoBERT")

#     # 3. Prepare Dataset
#     full_dataset = ConspiracyDataset(
#         tokenizer, 
#         'data/True.csv', 
#         'data/Fake.csv', 
#         use_sentiment=args.sentiment,
#         use_complexity=args.complexity,
#         use_pronouns=args.pronouns
#     )

#     # 80/20 Split
#     train_size = int(0.8 * len(full_dataset))
#     test_size = len(full_dataset) - train_size
#     train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#     # We use the test set as validation for early stopping in this ablation context
#     val_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#     # 4. Feature Dimension
#     extractor = PsychoLinguisticExtractor()
#     feat_dim = extractor.get_feature_dim(args.sentiment, args.complexity, args.pronouns)

#     # 5. Initialize Model
#     model = NeoBERTDetector(manual_feature_dim=feat_dim).to(device)
    
#     # Optimizer & Loss
#     optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5) # Added weight_decay
#     criterion = torch.nn.CrossEntropyLoss()
    
#     # Scheduler: Reduce LR if validation loss stops dropping
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
#     # Early Stopping
#     early_stopper = EarlyStopping(patience=3)

#     # 6. Training Loop
#     print(f"Training for up to {args.epochs} epochs with Early Stopping...")
    
#     for epoch in range(args.epochs):
#         # --- Training Phase ---
#         model.train()
#         train_loss = 0
#         for batch in train_loader:
#             input_ids = batch['input_ids'].to(device)
#             mask = batch['attention_mask'].to(device)
#             feats = batch['manual_features'].to(device)
#             labels = batch['label'].to(device)

#             optimizer.zero_grad()
#             outputs = model(input_ids, mask, feats)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
        
#         avg_train_loss = train_loss / len(train_loader)

#         # --- Validation Phase ---
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for batch in val_loader:
#                 input_ids = batch['input_ids'].to(device)
#                 mask = batch['attention_mask'].to(device)
#                 feats = batch['manual_features'].to(device)
#                 labels = batch['label'].to(device)

#                 outputs = model(input_ids, mask, feats)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
        
#         avg_val_loss = val_loss / len(val_loader)
        
#         print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

#         # Update Scheduler
#         scheduler.step(avg_val_loss)

#         # Check Early Stopping
#         early_stopper(avg_val_loss, model)
#         if early_stopper.early_stop:
#             print("🛑 Early stopping triggered. Loading best model weights.")
#             break

#     # Load best model weights
#     if early_stopper.best_model_wts:
#         model.load_state_dict(early_stopper.best_model_wts)

#     # 7. Final Evaluation
#     print("\n--- Final Evaluation on Test Set ---")
#     model.eval()
#     preds = []
#     true_labels = []
    
#     with torch.no_grad():
#         for batch in val_loader:
#             input_ids = batch['input_ids'].to(device)
#             mask = batch['attention_mask'].to(device)
#             feats = batch['manual_features'].to(device)
#             labels = batch['label'].to(device)

#             outputs = model(input_ids, mask, feats)
#             predictions = torch.argmax(outputs, dim=1)
            
#             preds.extend(predictions.cpu().numpy())
#             true_labels.extend(labels.cpu().numpy())

#     acc = accuracy_score(true_labels, preds)
#     f1 = f1_score(true_labels, preds, average='weighted')
    
#     print(f"Accuracy: {acc:.4f}")
#     print(f"F1 Score: {f1:.4f}")
#     print("-" * 30)
#     print(classification_report(true_labels, preds, target_names=['True', 'Fake']))
    
#     return acc

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--sentiment", action="store_true", help="Include Sentiment Factors")
#     parser.add_argument("--complexity", action="store_true", help="Include Complexity Factors")
#     parser.add_argument("--pronouns", action="store_true", help="Include Pronoun Factors")
#     parser.add_argument("--epochs", type=int, default=10) # Default upped to 10
    
#     args = parser.parse_args()
    
#     train(args)
import torch
import copy
import numpy as np
import argparse
import warnings
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import your modules
from dataset_loader import ConspiracyDataset
from features import PsychoLinguisticExtractor
from model import NeoBERTDetector

warnings.filterwarnings("ignore")

class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0

def train(args):
    print(f"\n--- Starting Ablation Run ---")
    print(f"Features :: Sentiment: {args.sentiment} | Complexity: {args.complexity} | Pronouns: {args.pronouns}")

    # 1. Setup Device (MPS for Mac)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using device: CUDA")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("chandar-lab/NeoBERT")

    # 3. Prepare Dataset
    # CHANGED: Loading from single JSON file instead of separate CSVs
    # Note: 'train_redacted.json' needs to be in your project root or data folder
    full_dataset = ConspiracyDataset(
        tokenizer, 
        'data/train_redacted.jsonl', 
        use_sentiment=args.sentiment,
        use_complexity=args.complexity,
        use_pronouns=args.pronouns
    )

    if len(full_dataset) == 0:
        print("🛑 No data loaded. Check if 'train_redacted.json' exists and has valid 'conspiracy' labels.")
        return

    # 80/20 Split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # We use the test set as validation for early stopping in this ablation context
    val_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 4. Feature Dimension
    extractor = PsychoLinguisticExtractor()
    feat_dim = extractor.get_feature_dim(args.sentiment, args.complexity, args.pronouns)

    # 5. Initialize Model
    model = NeoBERTDetector(manual_feature_dim=feat_dim).to(device)
    
    # Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5) 
    criterion = torch.nn.CrossEntropyLoss()
    
    # Scheduler: Reduce LR if validation loss stops dropping
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    # Early Stopping
    early_stopper = EarlyStopping(patience=3)

    # 6. Training Loop
    print(f"Training for up to {args.epochs} epochs with Early Stopping...")
    
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            feats = batch['manual_features'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, mask, feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                feats = batch['manual_features'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, mask, feats)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Update Scheduler
        scheduler.step(avg_val_loss)

        # Check Early Stopping
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print("🛑 Early stopping triggered. Loading best model weights.")
            break

    # Load best model weights
    if early_stopper.best_model_wts:
        model.load_state_dict(early_stopper.best_model_wts)

    # 7. Final Evaluation
    print("\n--- Final Evaluation on Test Set ---")
    model.eval()
    preds = []
    true_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            feats = batch['manual_features'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, mask, feats)
            predictions = torch.argmax(outputs, dim=1)
            
            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='weighted')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("-" * 30)
    print(classification_report(true_labels, preds, target_names=['No (Not Conspiracy)', 'Yes (Conspiracy)']))
    
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment", action="store_true", help="Include Sentiment Factors")
    parser.add_argument("--complexity", action="store_true", help="Include Complexity Factors")
    parser.add_argument("--pronouns", action="store_true", help="Include Pronoun Factors")
    parser.add_argument("--epochs", type=int, default=10)
    
    args = parser.parse_args()
    
    train(args)