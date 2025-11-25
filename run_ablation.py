import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from dataset_loader import ConspiracyDataset
from model import NeoBERTDetector
from features import PsychoLinguisticExtractor
import argparse
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

def train(args):
    print(f"--- Starting Ablation Run ---")
    print(f"Variables :: Sentiment: {args.sentiment} | Complexity: {args.complexity} | Pronouns: {args.pronouns}")

    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load NeoBERT Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("chandar-lab/NeoBERT", trust_remote_code=True)

    # 3. Prepare Dataset
    # Ensure you have 'True.csv' and 'Fake.csv' in the data/ folder
    full_dataset = ConspiracyDataset(
        tokenizer, 
        'data/True.csv', 
        'data/Fake.csv', 
        use_sentiment=args.sentiment,
        use_complexity=args.complexity,
        use_pronouns=args.pronouns
    )

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 4. Calculate Feature Dimension
    extractor = PsychoLinguisticExtractor()
    feat_dim = extractor.get_feature_dim(args.sentiment, args.complexity, args.pronouns)

    # 5. Initialize Model
    model = NeoBERTDetector(manual_feature_dim=feat_dim).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # 6. Training Loop (Shortened for demo)
    print("Training Model...")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
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
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    # 7. Evaluation
    print("Evaluating...")
    model.eval()
    preds = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            feats = batch['manual_features'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, mask, feats)
            predictions = torch.argmax(outputs, dim=1)
            
            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    print(f"Accuracy with current factors: {acc:.4f}")
    print("-" * 30)
    
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment", action="store_true", help="Include Sentiment Factors")
    parser.add_argument("--complexity", action="store_true", help="Include Complexity Factors")
    parser.add_argument("--pronouns", action="store_true", help="Include Pronoun Factors")
    parser.add_argument("--epochs", type=int, default=3)
    
    args = parser.parse_args()
    
    train(args)