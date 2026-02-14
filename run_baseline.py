"""
Stock Trend Prediction - Baseline Models
========================================

This script establishes a performance baseline for the Stock Trend Prediction task.
Establishing a robust baseline is a critical first step in any machine learning project.
We implement two baseline models:

1.  **Majority Class Classifier (Dummy Classifier)**:
    -   **Principle**: Predicts the most frequent class observed in the training data.
    -   **Purpose**: Serves as the absolute lower bound. Any useful model must outperform this.
    -   **Interpretation**: If a complex model cannot beat this, it is learning nothing beyond the prior probability of the target.

2.  **Logistic Regression**:
    -   **Principle**: A linear model that estimates the probability of each class using a logistic function.
    -   **Purpose**: Provides a sanity check for linear separability. If the relationship between returns and future direction is purely linear, this model should perform well.
    -   **Configuration**: We use 'multinomial' logistic regression (Softmax Regression) since we have 3 classes (-1, 0, 1).

Methodology & Validation:
-------------------------
-   **Validation Strategy**: We use `GroupKFold` split by `day`.
    -   *Why?* Stock market data is time-dependent. Standard random splitting would leak information from the same day (e.g., general market sentiment) into the validation set, leading to overly optimistic estimates. Grouping by day ensures that the model is tested on unseen days, simulating real-world forecasting.
-   **Feature Scaling**: Logistic Regression is sensitive to the scale of features. We apply `StandardScaler` to normalize returns to unit variance.

Usage:
------
Run this script to observe the baseline performance metrics (Accuracy and Macro F1).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# --- Constants & Configuration ---
# Update these paths to your local directory structure
TRAIN_INPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\input_training.csv'
TRAIN_OUTPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\output_training_gmEd6Zt.csv'

def load_data():
    """
    Loads training input and output data and merges them.
    
    Returns:
        pd.DataFrame: Merged dataset containing features and target.
    """
    print("Loading data...")
    try:
        X_train = pd.read_csv(TRAIN_INPUT_PATH)
        y_train = pd.read_csv(TRAIN_OUTPUT_PATH)
        # Merge on ID to ensure the correct label is matched to the correct feature set
        data = pd.merge(X_train, y_train, on='ID')
        print(f"Data successfully loaded. Shape: {data.shape}")
        return data
    except FileNotFoundError as e:
        print(f"Error: Need to correctly set file paths. {e}")
        exit()

def preprocess_features(data):
    """
    Preprocesses raw return data into features for the baseline model.
    
    Data Processing Steps:
    1.  **Missing Values**: Filled with 0. In the context of returns, 0 implies 'no change', 
        which is a neutral and safe assumption compared to mean imputation.
    2.  **Feature Engineering**: 
        -   While Logistic Regression can use raw returns, we calculate summary statistics 
            (mean, std, sum, etc.) to capture the distribution of the price path.
        -   **Crucial Note**: Statistics are calculated row-wise (axis=1) for each sample. 
            This ensures NO information leakage across different samples or from the future.
            
    Args:
        data (pd.DataFrame): Raw merged dataframe.
        
    Returns:
        tuple: (X features, y target, groups)
    """
    print("Preprocessing features...")
    
    # Extract the 53 5-minute return columns (r0 to r52)
    return_cols = [f'r{i}' for i in range(53)]
    
    # Fill missing returns with 0 (neutral market assumption)
    X = data[return_cols].fillna(0)
    
    # Compute aggregate statistics
    # These capture the 'shape' of the trading session
    X['mean_return'] = X[return_cols].mean(axis=1)  # Trend
    X['std_return'] = X[return_cols].std(axis=1)    # Volatility
    X['min_return'] = X[return_cols].min(axis=1)    # Max Drawdown proxy
    X['max_return'] = X[return_cols].max(axis=1)    # Max Upside proxy
    X['sum_return'] = X[return_cols].sum(axis=1)    # Total return over the period
    
    # Extract administrative columns for validation splitting
    y = data['reod']
    groups = data['day'] # Used for GroupKFold
    
    return X, y, groups

def train_baseline_models(X, y, groups):
    """
    Trains and evaluates baseline models using Cross-Validation.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target labels.
        groups (pd.Series): Day identifiers for grouping.
    """
    print("\n--- Training Baseline Models ---")
    
    # GroupKFold ensures samples from the same day are consistently in either train or test,
    # never split between them. This prevents "looking ahead" within the same day.
    gkf = GroupKFold(n_splits=5)
    
    # Store metrics to average later
    scores = {
        'dummy_acc': [], 'dummy_f1': [],
        'logreg_acc': [], 'logreg_f1': []
    }
    
    fold = 1
    # Iterate over each fold
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        print(f"Fold {fold} processing...")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # --- Preprocessing Step: Scaling ---
        # Logistic Regression requires scaled data for convergence and valid coefficient interpretation.
        # We fit the scaler ONLY on the training set to avoid data leakage.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # --- Model 1: Majority Class (Dummy) ---
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_val)
        
        scores['dummy_acc'].append(accuracy_score(y_val, y_pred_dummy))
        scores['dummy_f1'].append(f1_score(y_val, y_pred_dummy, average='macro'))
        
        # --- Model 2: Logistic Regression ---
        # multi_class='multinomial': Use Softmax for 3 classes
        # class_weight='balanced': Adjust weights inversely proportional to class frequencies
        #                          This is critical because class '0' (no change) might be dominant.
        logreg = LogisticRegression(
            multi_class='multinomial', 
            max_iter=1000, 
            class_weight='balanced',
            random_state=42
        )
        logreg.fit(X_train_scaled, y_train)
        y_pred_logreg = logreg.predict(X_val_scaled)
        
        scores['logreg_acc'].append(accuracy_score(y_val, y_pred_logreg))
        scores['logreg_f1'].append(f1_score(y_val, y_pred_logreg, average='macro'))
        
        # Report for the first fold only to keep output clean
        if fold == 1:
            print("\n--- Fold 1 Detailed Report (Logistic Regression) ---")
            print(classification_report(y_val, y_pred_logreg))
            print("Confusion Matrix:")
            print(confusion_matrix(y_val, y_pred_logreg))
            
        fold += 1
        
    # --- Final Results ---
    print("\n" + "="*40)
    print("     BASELINE RESULTS (5-FOLD CV)")
    print("="*40)
    print(f"Majority Class Accuracy:     {np.mean(scores['dummy_acc']):.4f}")
    print(f"Majority Class Macro F1:     {np.mean(scores['dummy_f1']):.4f}")
    print("-" * 40)
    print(f"Logistic Regression Accuracy: {np.mean(scores['logreg_acc']):.4f}")
    print(f"Logistic Regression Macro F1: {np.mean(scores['logreg_f1']):.4f}")
    print("="*40)
    print("\nBenchmark Reference:")
    print("Challenge Benchmark: ~41.74% Accuracy")
    print("Random Guess:        ~33.33% Accuracy")

if __name__ == "__main__":
    df = load_data()
    X, y, days = preprocess_features(df)
    train_baseline_models(X, y, days)
