# Stock Trend Prediction

## 1. Project Overview
This project predicts the price direction (-1, 0, 1) for the last 2 hours of a trading session based on the returns of the first 4.5 hours. To capture the highest predictive power, two methodologically distinct models are used and ensembled:
1. **LightGBM (Tabular Approach)**: Learns from the overall distribution, volatility, and momentum shifts of the price path.
2. **PyTorch LSTM (Sequential Approach)**: Learns explicitly from the ordered time steps to isolate and exploit sequential temporal dependencies.

## 2. Environment Setup
To run this pipeline, ensure you have Python 3.10+ installed and install the following dependencies via standard pip:

```bash
pip install pandas numpy scikit-learn lightgbm torch matplotlib scipy
```

## 3. File Structure
*   `data_analysis.py`: Performs Exploratory Data Analysis and validates the logic of the GroupKFold structure.
*   `run_baseline.py`: Establishes the performance lower bounds using Majority Class prediction and a standard Logistic Regression.
*   `run_lightgbm.py`: Trains and validates the Gradient Boosting model using advanced engineered features and dynamic hyperparameter optimization.
*   `run_rnn.py`: Trains and validates the PyTorch LSTM sequence model with internal learning rate tuning and dropout regularization.
*   `generate_submission.py`: Retrains both LightGBM and LSTM on the full training dataset and ensembles their output probabilities (50/50 blend) to generate the final `submission_ensemble.csv`.

## 4. Execution Guide

**Step 1. Configure Data Paths:**
Ensure `input_training.csv`, `output_training_gmEd6Zt.csv`, and `input_test.csv` are in the directory. You may need to verify or update the `PATH` variables defined at the top of each Python script to match your local system.

**Step 2. Establish Baselines:**
```bash
python run_baseline.py
```

**Step 3. Validate the Core Models:**
Evaluate the theoretical out-of-sample performance of the models using a 5-fold cross-validation.
```bash
python run_lightgbm.py
python run_rnn.py
```

**Step 4. Generate Final Deployment Submission:**
```bash
python generate_submission.py
```
This script will produce `submission_ensemble.csv` in your root directory, ready for upload.

## 5. Methodology Summary
- **Strict Validation**: We utilize a strict `GroupKFold` (5 splits) based solely on `day`. This ensures no contextual/market data from the validation day leaks into the training set, correctly simulating a true forward-looking forecast. All feature scaling is strictly nested inside this loop.
- **Ensemble Magic**: LightGBM processes the macroscopic "shape" of the day (using summarized statistics and distribution metrics) while the LSTM processes the microscopic "flow" of the day (the exact step-by-step sequence). Averaging their predictions leverages the strengths of both tabular and deep learning paradigms, significantly solidifying the final accuracy.
