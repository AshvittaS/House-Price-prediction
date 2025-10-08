# House-Price-prediction

Predict house prices using a simple linear regression workflow built in a Jupyter notebook.

## Overview
This project explores and models the `Housing.csv` dataset to predict `price`.
It follows these steps in `house-Price.ipynb`:
- Load data and inspect shape/columns
- Exploratory Data Analysis (missing values, basic distributions, outliers)
- One-hot encode categorical features
- Outlier handling with IQR filtering
- Distribution and skewness checks; power transform and standardization
- Train/test split and Linear Regression modeling
- Evaluation and visualization of predictions

## Dataset
- File: `Housing.csv`
- Target: `price`
- Example features used:
  - Numeric: `area`
  - Categorical: `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `parking`, `prefarea`, `furnishingstatus`

## EDA Highlights
- No missing values were found via `df.isnull().sum()`.
- Train/test shapes before cleaning: `(436, 15)` and `(109, 15)`.
- Outliers (IQR method) observed notably in `area` and `price`:
  - Train: `area`=11, `price`=12
  - Test: `area`=1, `price`=2
- Skewness (after outlier removal, before power transform):
  - `area` train: ~0.80; test: ~0.44
  - `price` train: ~0.76; test: ~0.64

## Preprocessing
- One-hot encoding with `pd.get_dummies(..., drop_first=True)` for categorical columns.
- Outlier removal using IQR on `area` and `price` (applied on combined features-target frames, then split back).
- Yeo–Johnson `PowerTransformer` applied to `area` and `price` to reduce skew and standardize.
- `StandardScaler` applied to feature matrix.

## Modeling
- Model: `sklearn.linear_model.LinearRegression`
- Split: `train_test_split(test_size=0.2, random_state=42)`

## Results
- Test performance (on transformed scale):
  - R² ≈ 0.5744
  - MAE ≈ 0.4761
- Scatter plot of Actual vs Predicted shows moderate fit with dispersion around the diagonal.

## How to Run
1. Ensure Python environment with required packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
2. Open `house-Price.ipynb` in Jupyter (e.g., `jupyter notebook`).
3. Make sure `Housing.csv` is in the same directory.
4. Run cells top to bottom.

## Notes
- Evaluation metrics are computed on transformed targets (after Yeo–Johnson). If you need metrics in the original price scale, inverse-transform predictions and targets before evaluation.
