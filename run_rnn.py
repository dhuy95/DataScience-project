import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

# Constants
TRAIN_INPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\input_training.csv'
TRAIN_OUTPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\output_training_gmEd6Zt.csv'
BATCH_SIZE = 1024  # Large batch size for speed
EPOCHS = 8  # Few epochs as data is large-ish
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReturnsDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, 53) -> Reshape to (N, 53, 1) for LSTM
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        # y: map -1,0,1 to 0,1,2
        self.y = torch.tensor(y + 1, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x: (Batch, Seq, Feat)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def load_and_preprocess():
    print("Loading data...")
    X_train = pd.read_csv(TRAIN_INPUT_PATH)
    y_train = pd.read_csv(TRAIN_OUTPUT_PATH)
    data = pd.merge(X_train, y_train, on='ID')
    
    return_cols = [f'r{i}' for i in range(53)]
    X = data[return_cols].fillna(0).values
    y = data['reod'].values
    days = data['day'].values
    
    # Scale returns globally (standard scale)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, days

def train_rnn(X, y, groups):
    print(f"\n--- Training RNN on {DEVICE} ---")
    gkf = GroupKFold(n_splits=5)
    
    metrics = {'acc': [], 'macro_f1': []}
    
    fold = 1
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        print(f"\nFold {fold} processing...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Datasets
        train_ds = ReturnsDataset(X_train, y_train)
        val_ds = ReturnsDataset(X_val, y_val)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
        
        # Model
        model = LSTMModel().to(DEVICE)
        
        # Class Weights for loss
        # Count classes in current fold
        u, c = np.unique(y_train, return_counts=True)
        class_weights = 1.0 / c
        class_weights = class_weights / class_weights.sum() * 3 # Normalize somewhat
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training Loop
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(bx)
                loss = criterion(outputs, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    outputs = model(bx)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(by.cpu().numpy())
            
            val_acc = accuracy_score(all_targets, all_preds)
            print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f}")
            
        # Final Evaluation for fold
        final_preds = np.array(all_preds)
        final_targets = np.array(all_targets)
        # Shift back from 0,1,2 to -1,0,1 for reporting if needed, but metrics function handles it fine as categorical
        acc = accuracy_score(final_targets, final_preds)
        f1 = f1_score(final_targets, final_preds, average='macro')
        
        metrics['acc'].append(acc)
        metrics['macro_f1'].append(f1)
        
        if fold == 1:
            # Shift targets back to -1, 0, 1 for consistency in report
            print("\n--- Fold 1 Classification Report (RNN) ---")
            print(classification_report(final_targets, final_preds, target_names=['-1', '0', '1']))
            
        fold += 1
        
    print("\n--- RNN Results (Average 5 Folds) ---")
    print(f"Accuracy: {np.mean(metrics['acc']):.4f}")
    print(f"Macro F1: {np.mean(metrics['macro_f1']):.4f}")

if __name__ == "__main__":
    X, y, days = load_and_preprocess()
    train_rnn(X, y, days)
