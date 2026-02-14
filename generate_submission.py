import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# File paths
TRAIN_INPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\input_training.csv'
TRAIN_OUTPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\output_training_gmEd6Zt.csv'
TEST_INPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\input_test.csv'
SUBMISSION_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\submission.csv'

def load_data():
    print("Loading valid training data...")
    X_train_raw = pd.read_csv(TRAIN_INPUT_PATH)
    y_train_raw = pd.read_csv(TRAIN_OUTPUT_PATH)
    train_data = pd.merge(X_train_raw, y_train_raw, on='ID')
    
    print("Loading test data...")
    test_data = pd.read_csv(TEST_INPUT_PATH)
    
    return train_data, test_data

def preprocess_features(data, is_train=True):
    # Features
    return_cols = [f'r{i}' for i in range(53)]
    X = data[return_cols].fillna(0)
    
    # Feature Engineering
    X['mean_return'] = X[return_cols].mean(axis=1)
    X['std_return'] = X[return_cols].std(axis=1)
    X['min_return'] = X[return_cols].min(axis=1)
    X['max_return'] = X[return_cols].max(axis=1)
    X['sum_return'] = X[return_cols].sum(axis=1)
    X['last_return'] = X['r52']
    X['spread'] = X['max_return'] - X['min_return']
    
    if is_train:
        return X, data['reod']
    else:
        return X

def generate_submission():
    train_df, test_df = load_data()
    
    print("Preprocessing...")
    X, y = preprocess_features(train_df, is_train=True)
    X_test = preprocess_features(test_df, is_train=False)
    
    # Split a small chunk for early stopping to avoid overfitting
    # Since we want to use as much data as possible, we keep the val set small (5%)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42, shuffle=True)
    
    print(f"Training LightGBM on {len(X_train)} samples...")
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=True)]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='logloss',
        callbacks=callbacks
    )
    
    print("Predicting on test set...")
    y_pred = model.predict(X_test)
    
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'reod': y_pred
    })
    
    print(f"Saving submission to {SUBMISSION_PATH}...")
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Done!")
    
    # Quick distribution check
    print("\nPrediction Distribution:")
    print(submission['reod'].value_counts(normalize=True))

if __name__ == "__main__":
    generate_submission()
