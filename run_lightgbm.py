"""
Stock Trend Prediction - Gradient Boosting (LightGBM)
=====================================================

This script implements a Gradient Boosting Decision Tree (GBDT) model using LightGBM.
Gradient Boosting is a powerful ensemble technique that builds models sequentially, 
where each new model corrects the errors (residuals) of the previous ones.

Why LightGBM?
-------------
1.  **Handling Non-Linearity**: Unlike Logistic Regression, trees can naturally capture 
    non-linear relationships between returns and future price direction.
2.  **Efficiency**: LightGBM uses histogram-based algorithms, making it faster and 
    more memory-efficient than standard XGBoost or Random Forests.
3.  **Missing Values**: Tree-based models can handle missing values natively, often 
    treating them as a separate category or finding the best split direction for them.

Methodology details:
--------------------
-   **Feature Engineering**: We transform the raw 53-step time series into summary statistics.
    This "collapsing" of the time dimension assumes that the distribution of returns 
    (volatility, trend, skew) is more predictive than the exact sequence order for this specific horizon.
-   **Optimization**: We use `early_stopping` to prevent overfitting. The model stops training 
    if the validation score (logloss) stops improving.
-   **Class Imbalance**: We use `class_weight='balanced'` to punish the model more for 
    misclassifying minority classes (often the directional movements -1 and 1).

Usage:
------
Run this script to train the model, evaluate performance via 5-Fold CV, and inspect 
Feature Importance plots.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, recall_score
import matplotlib.pyplot as plt

# --- Constants & Configuration ---
TRAIN_INPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\input_training.csv'
TRAIN_OUTPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\output_training_gmEd6Zt.csv'

def load_and_preprocess():
    """
    Loads data and performs feature engineering.
    
    Returns:
        tuple: (X features, y target, groups for validation)
    """
    print("Loading data...")
    try:
        X_train = pd.read_csv(TRAIN_INPUT_PATH)
        y_train = pd.read_csv(TRAIN_OUTPUT_PATH)
        data = pd.merge(X_train, y_train, on='ID')
    except FileNotFoundError:
        print("Error: Files not found. Check paths.")
        exit()

    # --- Feature Engineering ---
    # We focus on aggregate statistics rather than raw sequences.
    return_cols = [f'r{i}' for i in range(53)]
    X = data[return_cols].fillna(0)
    
    # 1. Volatility Measures (std, spread)
    # High volatility might precede a breakout (up or down).
    X['std_return'] = X[return_cols].std(axis=1)
    X['min_return'] = X[return_cols].min(axis=1)
    X['max_return'] = X[return_cols].max(axis=1)
    X['spread'] = X['max_return'] - X['min_return']
    
    # 2. Trend Measures (mean, sum)
    # Captures the existing momentum of the stock over the 4.5h period.
    X['mean_return'] = X[return_cols].mean(axis=1)
    X['sum_return'] = X[return_cols].sum(axis=1)
    
    # 3. Recency Measures (last return)
    # The most recent price action (r52) might have more weight than r0.
    X['last_return'] = X['r52']
    
    # Momentum: Return of the last 30 minutes (last 6 intervals: r47 to r52)
    momentum_cols = [f'r{i}' for i in range(47, 53)]
    X['momentum_30m'] = X[momentum_cols].sum(axis=1)
    
    X['spread'] = X['max_return'] - X['min_return']
    
    y = data['reod']
    groups = data['day'] # For validation split
    
    return X, y, groups

def train_lightgbm(X, y, groups):
    """
    Trains LightGBM model with GroupKFold Cross-Validation.
    """
    print("\n--- Training LightGBM Model ---")
    
    # Use 5 folds to ensure robust evaluation
    gkf = GroupKFold(n_splits=5)
    
    gkf = GroupKFold(n_splits=5)
    
    metrics = {
        'acc': [], 
        'macro_f1': [],
        'recall_minus1': [],
        'recall_1': []
    }
    feature_importance_df = pd.DataFrame()
    feature_importance_df["feature"] = X.columns
    
    fold = 1
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        print(f"Fold {fold} processing...", end=" ")
        
        # Split data based on day groups
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # --- Model Configuration ---
        # n_estimators=1000: Maximum number of trees.
        # learning_rate=0.05: Step size shrinkage significantly prevents overfitting.
        # num_leaves=31: Controls the complexity of the tree model.
        # class_weight='balanced': Automatically adjusts weights inversely to class frequencies.
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,           # No limit on depth, controlled by leaves
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Early Stopping: Stop training if validation score doesn't improve for 50 rounds
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='logloss',
            callbacks=callbacks
        )
        
        # Inference
        y_pred = model.predict(X_val)
        
        # Evaluation
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        
        # Calculate per-class recall
        # labels=[-1, 1] ensures we get recall specifically for these classes
        # average=None returns an array [recall_-1, recall_1]
        recalls = recall_score(y_val, y_pred, labels=[-1, 1], average=None)
        
        metrics['acc'].append(acc)
        metrics['macro_f1'].append(f1)
        metrics['recall_minus1'].append(recalls[0])
        metrics['recall_1'].append(recalls[1])
        
        print(f"-> Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
        print(f"   Recall (-1): {recalls[0]:.4f}, Recall (1): {recalls[1]:.4f}")
        
        # Store Feature Importance
        feature_importance_df[f"fold_{fold}"] = model.feature_importances_
        
        if fold == 1:
            print("\n--- Fold 1 Classification Report ---")
            print(classification_report(y_val, y_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(y_val, y_pred))
            
        fold += 1
        
    # --- Final Results ---
    avg_acc = np.mean(metrics['acc'])
    avg_f1 = np.mean(metrics['macro_f1'])
    
    print("\n" + "="*40)
    print("     LIGHTGBM RESULTS (5-FOLD CV)")
    print("="*40)
    print(f"Average Accuracy:      {np.mean(metrics['acc']):.4f}")
    print(f"Average Macro F1:      {np.mean(metrics['macro_f1']):.4f}")
    print(f"Average Recall (-1):   {np.mean(metrics['recall_minus1']):.4f}")
    print(f"Average Recall (1):    {np.mean(metrics['recall_1']):.4f}")
    print("="*40)
    
    # --- Interpretability: Feature Importance ---
    print("\n--- Top 10 Feature Importance (averaged) ---")
    feature_importance_df['average'] = feature_importance_df[[f"fold_{i}" for i in range(1, 6)]].mean(axis=1)
    print(feature_importance_df.sort_values(by="average", ascending=False).head(10)[['feature', 'average']])
    print("\nNote: Importance is based on 'split', counting how many times a feature is used in trees.")

if __name__ == "__main__":
    X, y, days = load_and_preprocess()
    train_lightgbm(X, y, days)
