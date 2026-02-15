"""
Stock Trend Prediction - Recurrent Neural Network (LSTM)
========================================================

This script implements a Long Short-Term Memory (LSTM) network using PyTorch.
It is designed to explicitly model the sequential nature of the returns, which 
aggregate-based models (LightGBM) may miss.

Key Corrections & Improvements:
-------------------------------
1.  **Leakage-Free Scaling**: The scaler is fitted INSIDE the cross-validation loop.
    `scaler.fit_transform(X_train)` and `scaler.transform(X_val)` ensures that 
    validation data statistics do not contaminate the training process.
2.  **Deeper Training**: We train for 20 epochs to allow the LSTM to converge.
3.  **Manual Best Model Saving**: Instead of complex early stopping callbacks, we track
    the validation loss/accuracy manually and save the state dictionary of the best epoch.
4.  **Actionable Metrics**: We explicitly calculate and report Recall for classes -1 and 1.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

# --- Constants & Configuration ---
TRAIN_INPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\input_training.csv'
TRAIN_OUTPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\output_training_gmEd6Zt.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, num_classes=3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True expects input shape (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(self.dropout(out[:, -1, :]))
        return out

def load_data():
    print("Loading data...")
    X_train = pd.read_csv(TRAIN_INPUT_PATH)
    y_train = pd.read_csv(TRAIN_OUTPUT_PATH)
    data = pd.merge(X_train, y_train, on='ID')
    return data

def train_rnn(data):
    print(f"\n--- Training LSTM on {DEVICE} ---")
    
    # Features (Raw Returns)
    return_cols = [f'r{i}' for i in range(53)]
    X = data[return_cols].fillna(0).values
    y = data['reod'].values
    groups = data['day'].values
    
    # Map target -1, 0, 1 to 0, 1, 2 for CrossEntropyLoss
    # -1 -> 0, 0 -> 1, 1 -> 2
    y_mapped = y + 1 
    
    gkf = GroupKFold(n_splits=5)
    
    metrics = {
        'acc': [], 
        'macro_f1': [],
        'recall_minus1': [],
        'recall_1': []
    }
    
    fold = 1
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        print(f"\nFold {fold} processing...")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_mapped[train_idx], y_mapped[val_idx]
        
        # --- 1. Scaling (Inside Loop to Prevent Leakage) ---
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Reshape for LSTM: (Samples, Time Steps, Features per step)
        # Here we have 53 steps, 1 feature per step (return)
        X_train_t = torch.tensor(X_train.reshape(-1, 53, 1), dtype=torch.float32).to(DEVICE)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(DEVICE) # CrossEntropy expects Long
        
        X_val_t = torch.tensor(X_val.reshape(-1, 53, 1), dtype=torch.float32).to(DEVICE)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(DEVICE)
        
        # DataLoader
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
        
        # --- 2. Model Initialization ---
        model = LSTMModel(input_size=1, hidden_size=64, num_layers=1, num_classes=3).to(DEVICE)
        
        # Loss and Optimizer
        # Class weights could be added here if needed: weight=torch.tensor([wt0, wt1, wt2])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # --- 3. Training Loop (20 Epochs) ---
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(20):
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            # Validation Step
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                _, val_preds = torch.max(val_outputs, 1)
                val_acc = (val_preds == y_val_t).float().mean().item()
                
                # Save best model logic
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict()
                    
            if (epoch+1) % 5 == 0:
                print(f"  Epoch {epoch+1}/20 | Val Acc: {val_acc:.4f}")

        # --- 4. Final Evaluation (Best Model) ---
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_t)
            _, preds = torch.max(outputs, 1)
            
            # Move to CPU for metrics
            y_true_cpu = y_val_t.cpu().numpy()
            y_pred_cpu = preds.cpu().numpy()
            
            # Remap back to -1, 0, 1 for consistent reporting with other scripts
            # 0 -> -1, 1 -> 0, 2 -> 1
            y_true_orig = y_true_cpu - 1
            y_pred_orig = y_pred_cpu - 1
            
            acc = accuracy_score(y_true_orig, y_pred_orig)
            f1 = f1_score(y_true_orig, y_pred_orig, average='macro')
            # labels=[-1, 1] for recall
            recalls = recall_score(y_true_orig, y_pred_orig, labels=[-1, 1], average=None)
            
            metrics['acc'].append(acc)
            metrics['macro_f1'].append(f1)
            metrics['recall_minus1'].append(recalls[0])
            metrics['recall_1'].append(recalls[1])
            
            print(f"-> Best Acc: {acc:.4f}, Macro F1: {f1:.4f}")
            print(f"   Recall (-1): {recalls[0]:.4f}, Recall (1): {recalls[1]:.4f}")
            
        fold += 1

    print("\n" + "="*40)
    print("     LSTM RESULTS (5-FOLD CV)")
    print("="*40)
    print(f"Average Accuracy:      {np.mean(metrics['acc']):.4f}")
    print(f"Average Macro F1:      {np.mean(metrics['macro_f1']):.4f}")
    print(f"Average Recall (-1):   {np.mean(metrics['recall_minus1']):.4f}")
    print(f"Average Recall (1):    {np.mean(metrics['recall_1']):.4f}")
    print("="*40)

if __name__ == "__main__":
    df = load_data()
    train_rnn(df)
