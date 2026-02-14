import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# File paths
TRAIN_INPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\input_training.csv'
TRAIN_OUTPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\output_training_gmEd6Zt.csv'

def load_data():
    print("Loading data...")
    X_train = pd.read_csv(TRAIN_INPUT_PATH)
    y_train = pd.read_csv(TRAIN_OUTPUT_PATH)
    data = pd.merge(X_train, y_train, on='ID')
    return data

def preprocess_features(data):
    print("Preprocessing features...")
    # Select return columns
    return_cols = [f'r{i}' for i in range(53)]
    
    # Check if all return columns exist
    missing_cols = [col for col in return_cols if col not in data.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
    
    # Fill missing values with 0 (assuming missing return means no change or just sparse data)
    # Using 0 is a neutral assumption for returns.
    X = data[return_cols].fillna(0)
    
    # Feature Engineering (Aggregate features)
    print("Generating aggregate features...")
    # Explicitly calculating row-wise stats to prevent leakage
    X['mean_return'] = X[return_cols].mean(axis=1)
    X['std_return'] = X[return_cols].std(axis=1)
    X['min_return'] = X[return_cols].min(axis=1)
    X['max_return'] = X[return_cols].max(axis=1)
    X['sum_return'] = X[return_cols].sum(axis=1) # Cumulative return proxy
    
    # Skewness requires a bit more work if pandas skew is slow, but standard skew is usually fine
    X['skew_return'] = X[return_cols].skew(axis=1).fillna(0)
    
    # Last return feature (momentum?)
    X['last_return'] = X['r52']
    
    return X, data['reod'], data['day'], data['equity']

def train_baseline_models(X, y, groups):
    print("\n--- Training Baseline Models ---")
    gkf = GroupKFold(n_splits=5)
    
    # Metrics storage
    scores = {
        'dummy_acc': [], 'dummy_f1': [],
        'logreg_acc': [], 'logreg_f1': []
    }
    
    fold = 1
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        print(f"Fold {fold} processing...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Scale features for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 1. Majority Class Baseline (Dummy)
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_val)
        
        scores['dummy_acc'].append(accuracy_score(y_val, y_pred_dummy))
        scores['dummy_f1'].append(f1_score(y_val, y_pred_dummy, average='macro'))
        
        # 2. Logistic Regression Baseline
        logreg = LogisticRegression(multi_class='multinomial', max_iter=1000, class_weight='balanced')
        logreg.fit(X_train_scaled, y_train)
        y_pred_logreg = logreg.predict(X_val_scaled)
        
        scores['logreg_acc'].append(accuracy_score(y_val, y_pred_logreg))
        scores['logreg_f1'].append(f1_score(y_val, y_pred_logreg, average='macro'))
        
        # Print fold report for LogReg
        if fold == 1:
            print("\n--- Fold 1 Classification Report (Logistic Regression) ---")
            print(classification_report(y_val, y_pred_logreg))
            print("Confusion Matrix:")
            print(confusion_matrix(y_val, y_pred_logreg))
            
        fold += 1
        
    print("\n--- Baseline Results (Average over 5 Folds) ---")
    print(f"Majority Class Accuracy: {np.mean(scores['dummy_acc']):.4f}")
    print(f"Majority Class Macro F1: {np.mean(scores['dummy_f1']):.4f}")
    print(f"Logistic Regression Accuracy: {np.mean(scores['logreg_acc']):.4f}")
    print(f"Logistic Regression Macro F1: {np.mean(scores['logreg_f1']):.4f}")
    
    print("\n--- Benchmark Targets ---")
    print("Random Guess (approx): 33%")
    print("Challenge Benchmark: 41.74%")

if __name__ == "__main__":
    df = load_data()
    X, y, days, equities = preprocess_features(df)
    train_baseline_models(X, y, days)
