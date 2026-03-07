"""
Stock Trend Prediction - Final Submission Generator
===================================================

This script generates the final predictions for the test dataset (`input_test.csv`).
Unlike the validation scripts (`run_lightgbm.py`, etc.), this script focuses on **deployment**.

Strategy:
---------
1.  **Full Data Utilization**: We retrain the best performing model (LightGBM) on the 
    **entire training dataset**. We do not use Cross-Validation here because our goal is 
    no longer estimation, but maximizing predictive power. The more data the model sees, 
    the better it generalizes.
2.  **Internal Validation**: We still reserve a small slice (5%) of the training data 
    as an internal validation set. This is strictly for **Early Stopping** to determine 
    the optimal number of trees and prevent overfitting.
3.  **Consistent Preprocessing**: We apply the exact same feature engineering steps 
    (aggregates like mean, std) to the test data as we did to the training data.

Output:
-------
Generates a `submission.csv` file containing two columns:
-   `ID`: The unique identifier for the test sample.
-   `reod`: The predicted class (-1, 0, or 1).
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.stats

# --- Constants & Configuration ---
TRAIN_INPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\input_training.csv'
TRAIN_OUTPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\output_training_gmEd6Zt.csv'
TEST_INPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\input_test.csv'
SUBMISSION_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\submission_ensemble.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(self.dropout(out[:, -1, :]))
        return out

def load_data():
    """
    Loads both training and test datasets.
    """
    print("Loading valid training data...")
    X_train_raw = pd.read_csv(TRAIN_INPUT_PATH)
    y_train_raw = pd.read_csv(TRAIN_OUTPUT_PATH)
    train_data = pd.merge(X_train_raw, y_train_raw, on='ID')
    
    print("Loading test data...")
    test_data = pd.read_csv(TEST_INPUT_PATH)
    
    return train_data, test_data

def preprocess_features_lgb(data):
    return_cols = [f'r{i}' for i in range(53)]
    X = data[return_cols].fillna(0)
    arr = X.values
    
    half = 53 // 2
    X['mean_first_half'] = np.mean(arr[:, :half], axis=1)
    X['mean_second_half'] = np.mean(arr[:, half:], axis=1)
    X['std_first_half'] = np.std(arr[:, :half], axis=1, ddof=1)
    X['std_second_half'] = np.std(arr[:, half:], axis=1, ddof=1)
    
    X['momentum_shift'] = X['mean_second_half'] - X['mean_first_half']
    X['volatility_shift'] = X['std_second_half'] - X['std_first_half']
    
    X['mean_return'] = np.mean(arr, axis=1)
    X['std_return'] = np.std(arr, axis=1, ddof=1)
    X['min_return'] = np.min(arr, axis=1)
    X['max_return'] = np.max(arr, axis=1)
    X['sum_return'] = np.sum(arr, axis=1)
    X['last_return'] = arr[:, 52]
    
    X['momentum_30m'] = np.sum(arr[:, 47:53], axis=1)
    X['spread'] = X['max_return'] - X['min_return']
    
    chunk_size = 50000
    skew_vals = []
    kurt_vals = []
    for i in range(0, arr.shape[0], chunk_size):
        chunk = arr[i:i+chunk_size]
        skew_vals.append(scipy.stats.skew(chunk, axis=1))
        kurt_vals.append(scipy.stats.kurtosis(chunk, axis=1))
        
    X['skew'] = np.concatenate(skew_vals)
    X['kurtosis'] = np.concatenate(kurt_vals)
    X['q25'] = np.quantile(arr, 0.25, axis=1)
    X['q75'] = np.quantile(arr, 0.75, axis=1)
    X['iqr'] = X['q75'] - X['q25']
    X['median'] = np.median(arr, axis=1)
    X['pos_count'] = np.sum(arr > 0, axis=1)
    X['neg_count'] = np.sum(arr < 0, axis=1)
    X['zero_count'] = np.sum(arr == 0, axis=1)
    
    return X.astype(np.float32)

def preprocess_features_lstm(data):
    return_cols = [f'r{i}' for i in range(53)]
    return data[return_cols].fillna(0).values.astype(np.float32)

def generate_submission():
    train_df, test_df = load_data()
    y_train_full = train_df['reod']
    
    print("\n--- 1. Training LightGBM ---")
    X_lgb_train = preprocess_features_lgb(train_df)
    X_lgb_test = preprocess_features_lgb(test_df)
    
    X_t_lgb, X_v_lgb, y_t_lgb, y_v_lgb = train_test_split(X_lgb_train, y_train_full, test_size=0.05, random_state=42, shuffle=False)
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    lgb_model.fit(
        X_t_lgb, y_t_lgb,
        eval_set=[(X_v_lgb, y_v_lgb)],
        eval_metric='logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
    )
    
    print("\nPredicting probabilities from LightGBM...")
    probs_lgb = lgb_model.predict_proba(X_lgb_test)
    
    print("\n--- 2. Training LSTM ---")
    X_lstm_train = preprocess_features_lstm(train_df)
    X_lstm_test = preprocess_features_lstm(test_df)
    
    # Target mapped to 0, 1, 2
    y_lstm_train = (y_train_full + 1).values
    
    scaler = StandardScaler()
    X_lstm_train_scaled = scaler.fit_transform(X_lstm_train)
    X_lstm_test_scaled = scaler.transform(X_lstm_test)
    
    X_t_lstm, X_v_lstm, y_t_lstm, y_v_lstm = train_test_split(X_lstm_train_scaled, y_lstm_train, test_size=0.05, random_state=42, shuffle=False)
    
    X_t_lstm_t = torch.tensor(X_t_lstm.reshape(-1, 53, 1), dtype=torch.float32).to(DEVICE)
    y_t_lstm_t = torch.tensor(y_t_lstm, dtype=torch.long).to(DEVICE)
    X_v_lstm_t = torch.tensor(X_v_lstm.reshape(-1, 53, 1), dtype=torch.float32).to(DEVICE)
    y_v_lstm_t = torch.tensor(y_v_lstm, dtype=torch.long).to(DEVICE)
    
    train_loader = DataLoader(TensorDataset(X_t_lstm_t, y_t_lstm_t), batch_size=1024, shuffle=True)
    
    lstm_model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(15):
        lstm_model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = lstm_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        lstm_model.eval()
        val_loss = 0.0
        val_loader = DataLoader(TensorDataset(X_v_lstm_t, y_v_lstm_t), batch_size=1024, shuffle=False)
        with torch.no_grad():
            for v_inputs, v_labels in val_loader:
                v_outputs = lstm_model(v_inputs)
                val_loss += criterion(v_outputs, v_labels).item() * v_inputs.size(0)
            
            val_loss /= len(y_v_lstm_t)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = lstm_model.state_dict()
                
        print(f"  LSTM Epoch {epoch+1}/15 | Val Loss: {val_loss:.4f}")
        
    lstm_model.load_state_dict(best_model_state)
    lstm_model.eval()
    
    print("\nPredicting probabilities from LSTM...")
    X_test_lstm_t = torch.tensor(X_lstm_test_scaled.reshape(-1, 53, 1), dtype=torch.float32).to(DEVICE)
    test_loader_lstm = DataLoader(TensorDataset(X_test_lstm_t), batch_size=1024, shuffle=False)
    
    all_lstm_probs = []
    with torch.no_grad():
        for batch in test_loader_lstm:
            batch_inputs = batch[0]
            lstm_outputs = lstm_model(batch_inputs)
            batch_probs = torch.softmax(lstm_outputs, dim=1).cpu().numpy()
            all_lstm_probs.append(batch_probs)
            
    probs_lstm = np.vstack(all_lstm_probs)
        
    print("\n--- 3. Ensembling ---")
    # Average the probabilities (50/50 blend)
    final_probs = 0.5 * probs_lgb + 0.5 * probs_lstm
    
    # Argmax gives 0, 1, 2. Map back to -1, 0, 1
    final_preds_mapped = np.argmax(final_probs, axis=1)
    final_preds = final_preds_mapped - 1
    
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'reod': final_preds
    })
    
    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Done!")
    
    print("\nPrediction Distribution:")
    print(submission['reod'].value_counts(normalize=True))

if __name__ == "__main__":
    generate_submission()
