import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold

# File paths
TRAIN_INPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\input_training.csv'
TRAIN_OUTPUT_PATH = r'c:\Users\DANG HUY\OneDrive\Tài liệu\study\Master\DataScience\input_test\output_training_gmEd6Zt.csv'

def load_data():
    print("Loading data...")
    X_train = pd.read_csv(TRAIN_INPUT_PATH)
    y_train = pd.read_csv(TRAIN_OUTPUT_PATH)
    
    # Merge on ID to ensure alignment (though they should be aligned)
    data = pd.merge(X_train, y_train, on='ID')
    print(f"Data shape: {data.shape}")
    return data

def inspect_data(data):
    print("\n--- Missing Values ---")
    missing_counts = data.isnull().sum()
    print(missing_counts[missing_counts > 0])
    
    print("\n--- Class Distribution ---")
    class_counts = data['reod'].value_counts(normalize=True)
    print(class_counts)
    
    print("\n--- Unique Days and Equities ---")
    print(f"Unique days: {data['day'].nunique()}")
    print(f"Unique equities: {data['equity'].nunique()}")

def check_disjoint_validation(data, n_splits=5):
    print("\n--- Checking Validation Split ---")
    gkf = GroupKFold(n_splits=n_splits)
    
    fold = 1
    for train_idx, val_idx in gkf.split(data, groups=data['day']):
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        
        train_days = set(train_data['day'].unique())
        val_days = set(val_data['day'].unique())
        
        train_equities = set(train_data['equity'].unique())
        val_equities = set(val_data['equity'].unique())
        
        common_days = train_days.intersection(val_days)
        common_equities = train_equities.intersection(val_equities)
        
        print(f"Fold {fold}:")
        print(f"  Train samples: {len(train_data)}, Val samples: {len(val_data)}")
        print(f"  Common days: {len(common_days)} (Should be 0)")
        print(f"  Common equities: {len(common_equities)} / {len(val_equities)} val equities")
        
        if len(common_equities) > 0:
            print(f"  Warning: {len(common_equities)} equities appear in both train and validation.")
        else:
            print("  Success: Validation equities are fully disjoint from training.")
            
        fold += 1

if __name__ == "__main__":
    df = load_data()
    inspect_data(df)
    check_disjoint_validation(df)
