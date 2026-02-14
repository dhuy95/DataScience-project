"""
Stock Trend Prediction - Neural Network (MLP)
=============================================

This script implements a Multi-Layer Perceptron (MLP), a type of feedforward artificial neural network.
We use this as our second distinct modeling approach, following the Gradient Boosting model.

Why Neural Networks?
--------------------
1.  **Universal Approximation**: Neural networks can theoretically approximate any continuous function.
    They are capable of learning highly complex, non-linear interactions between input features.
2.  **Raw Feature Learning**: Unlike the LightGBM model where we manually engineered aggregate features
    (mean, std), here we feed the **raw sequence of returns** (r0 to r52). We trust the network's 
    hidden layers to automatically learn relevant representations (e.g., detecting a dip before a rise).

Architecture Details:
---------------------
-   **Input Layer**: 53 neurons (one for each 5-minute return interval).
-   **Hidden Layers**: Two dense layers with return sizes (100, 50).
    -   Layer 1 (100 units): Expands the feature space to capture interactions.
    -   Layer 2 (50 units): Compresses information before the final classification.
-   **Activation**: `ReLU` (Rectified Linear Unit) is used to introduce non-linearity while avoiding 
    the vanishing gradient problem common with Sigmoid/Tanh.
-   **Optimizer**: `Adam` (Adaptive Moment Estimation), which is generally robust and requires 
    less hyperparameter tuning than SGD.

Implementation Constraints:
---------------------------
-   **Scaling is Critical**: Unlike tree models, Neural Networks are extremely sensitive to the scale
    of inputs. We MUST use `StandardScaler` to normalize inputs to mean 0 and variance 1. 
    Without this, gradients can explode or vanish, preventing convergence.

Usage:
------
Run this script to train the neural network and evaluate its performance against the LightGBM model.
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

# --- Constants & Configuration ---
TRAIN_INPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\input_training.csv'
TRAIN_OUTPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\output_training_gmEd6Zt.csv'

def load_and_preprocess():
    """
    Loads data for the MLP model.
    
    Return Strategy:
    - We return the RAW price return columns (r0...r52) instead of aggregates.
    - This allows the Neural Network to "see" the entire sequence and learn 
      its own feature representation.
    """
    print("Loading data...")
    try:
        X_train = pd.read_csv(TRAIN_INPUT_PATH)
        y_train = pd.read_csv(TRAIN_OUTPUT_PATH)
        data = pd.merge(X_train, y_train, on='ID')
    except FileNotFoundError:
        print("Error: Files not found.")
        exit()
    
    # Select raw return columns
    return_cols = [f'r{i}' for i in range(53)]
    
    # Fill missing with 0 (neutral signal)
    X = data[return_cols].fillna(0)
    y = data['reod']
    days = data['day']
    
    return X, y, days

def train_mlp(X, y, groups):
    """
    Trains an MLP Classifier using GroupKFold Cross-Validation.
    """
    print("\n--- Training MLP (Neural Network) ---")
    
    gkf = GroupKFold(n_splits=5)
    metrics = {'acc': [], 'macro_f1': []}
    
    fold = 1
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        print(f"Fold {fold} processing...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # --- Preprocessing: Scaling ---
        # CRITICAL STEP: Neural Networks converge faster and better with scaled data.
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # --- Model Architecture ---
        # hidden_layer_sizes=(100, 50): Two hidden layers.
        # early_stopping=True: Automatically creates a 10% internal validation set 
        #                      from the training data to stop when validation score plateaus.
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,           # L2 Regularization term
            batch_size=256,         # Mini-batch size for gradient updates
            learning_rate_init=0.001,
            max_iter=50,            # Max epochs (iterations)
            early_stopping=True,    # Stop if no improvement
            validation_fraction=0.1,
            n_iter_no_change=3,     # Patience for early stopping
            random_state=42,
            verbose=True            # Print loss during training
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        
        # Evaluation
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        
        metrics['acc'].append(acc)
        metrics['macro_f1'].append(f1)
        
        if fold == 1:
            print("\n--- Fold 1 Classification Report (MLP) ---")
            print(classification_report(y_val, y_pred))
            
        fold += 1
        
    print("\n" + "="*40)
    print("     NEURAL NETWORK RESULTS (5-FOLD CV)")
    print("="*40)
    print(f"Average Accuracy: {np.mean(metrics['acc']):.4f}")
    print(f"Average Macro F1: {np.mean(metrics['macro_f1']):.4f}")
    print("="*40)

if __name__ == "__main__":
    X, y, days = load_and_preprocess()
    train_mlp(X, y, days)
