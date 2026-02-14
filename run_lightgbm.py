import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# File paths
TRAIN_INPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\input_training.csv'
TRAIN_OUTPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\output_training_gmEd6Zt.csv'

def load_and_preprocess():
    print("Loading data...")
    X_train = pd.read_csv(TRAIN_INPUT_PATH)
    y_train = pd.read_csv(TRAIN_OUTPUT_PATH)
    data = pd.merge(X_train, y_train, on='ID')
    
    # Features
    return_cols = [f'r{i}' for i in range(53)]
    X = data[return_cols].fillna(0)
    
    # Feature Engineering (Row-wise only)
    X['mean_return'] = X[return_cols].mean(axis=1)
    X['std_return'] = X[return_cols].std(axis=1)
    X['min_return'] = X[return_cols].min(axis=1)
    X['max_return'] = X[return_cols].max(axis=1)
    X['sum_return'] = X[return_cols].sum(axis=1)
    X['last_return'] = X['r52']
    X['spread'] = X['max_return'] - X['min_return']
    
    # Target
    y = data['reod']
    
    return X, y, data['day']

def train_lightgbm(X, y, groups):
    print("\n--- Training LightGBM ---")
    gkf = GroupKFold(n_splits=5)
    
    metrics = {'acc': [], 'macro_f1': []}
    feature_importance_df = pd.DataFrame()
    feature_importance_df["feature"] = X.columns
    
    fold = 1
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        print(f"Fold {fold} processing...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Prepare LGB dataset
        # LightGBM handles class weights via parameter or sample_weight
        # We will use 'class_weight': 'balanced' explicitly in the model definition if using sklearn API
        # or calculate weights manually. Let's use sklearn API for simplicity.
        
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Callbacks for early stopping
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='logloss',
            callbacks=callbacks
        )
        
        y_pred = model.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        metrics['acc'].append(acc)
        metrics['macro_f1'].append(f1)
        
        # Accumulate importance
        feature_importance_df[f"fold_{fold}"] = model.feature_importances_
        
        if fold == 1:
            print("\n--- Fold 1 Classification Report (LightGBM) ---")
            print(classification_report(y_val, y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(y_val, y_pred))
            
        fold += 1
        
    print("\n--- LightGBM Results (Average 5 Folds) ---")
    print(f"Accuracy: {np.mean(metrics['acc']):.4f}")
    print(f"Macro F1: {np.mean(metrics['macro_f1']):.4f}")
    
    print("\n--- Feature Importance (Top 10) ---")
    feature_importance_df['average'] = feature_importance_df[[f"fold_{i}" for i in range(1, 6)]].mean(axis=1)
    print(feature_importance_df.sort_values(by="average", ascending=False).head(10)[['feature', 'average']])

if __name__ == "__main__":
    X, y, days = load_and_preprocess()
    train_lightgbm(X, y, days)
