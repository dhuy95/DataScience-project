import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

# Constants
TRAIN_INPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\input_training.csv'
TRAIN_OUTPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\output_training_gmEd6Zt.csv'

def load_and_preprocess():
    print("Loading data...")
    X_train = pd.read_csv(TRAIN_INPUT_PATH)
    y_train = pd.read_csv(TRAIN_OUTPUT_PATH)
    data = pd.merge(X_train, y_train, on='ID')
    
    # Use raw returns for MLP to let it learn patterns
    return_cols = [f'r{i}' for i in range(53)]
    X = data[return_cols].fillna(0)
    y = data['reod']
    days = data['day']
    
    return X, y, days

def train_mlp(X, y, groups):
    print("\n--- Training MLP (Neural Network) ---")
    gkf = GroupKFold(n_splits=5)
    
    metrics = {'acc': [], 'macro_f1': []}
    
    fold = 1
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        print(f"Fold {fold} processing...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Scaling is critical for MLP
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # MLP Architecture: 2 hidden layers (100, 50)
        # Using Adam, Early Stopping
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=20, # Limited epochs for speed, relies on partial_fit/convergence
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=3,
            random_state=42,
            verbose=True
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        
        metrics['acc'].append(acc)
        metrics['macro_f1'].append(f1)
        
        if fold == 1:
            print("\n--- Fold 1 Classification Report (MLP) ---")
            print(classification_report(y_val, y_pred))
            
        fold += 1
        
    print("\n--- MLP Results (Average 5 Folds) ---")
    print(f"Accuracy: {np.mean(metrics['acc']):.4f}")
    print(f"Macro F1: {np.mean(metrics['macro_f1']):.4f}")

if __name__ == "__main__":
    X, y, days = load_and_preprocess()
    train_mlp(X, y, days)
