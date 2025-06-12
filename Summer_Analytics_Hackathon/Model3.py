import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from scipy.stats import linregress
from scipy.signal import savgol_filter  # For advanced denoising

# Step 1: Load the data
train_df = pd.read_csv('./hacktrain.csv')
test_df = pd.read_csv('./hacktest.csv')

# Step 2: Advanced Preprocessing
# Define NDVI columns
ndvi_cols = [col for col in train_df.columns if col.endswith('_N')]

# Replace empty strings and convert to numeric
train_df[ndvi_cols] = train_df[ndvi_cols].replace(r'^\s*$', np.nan, regex=True)
for col in ndvi_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

# Handle outliers: Cap NDVI values at 1st and 99th percentiles
for col in ndvi_cols:
    lower, upper = train_df[col].quantile([0.01, 0.99])
    train_df[col] = train_df[col].clip(lower, upper)

# Impute missing values with linear interpolation
train_df[ndvi_cols] = train_df[ndvi_cols].interpolate(method='linear', axis=1, limit_direction='both')
train_df[ndvi_cols] = train_df[ndvi_cols].fillna(0)

# Denoise using Savitzky-Golay filter (more robust than rolling mean)
ndvi_array = train_df[ndvi_cols].values
for i in range(ndvi_array.shape[0]):
    ndvi_array[i] = savgol_filter(ndvi_array[i], window_length=5, polyorder=2)
train_df[ndvi_cols] = pd.DataFrame(ndvi_array, columns=ndvi_cols)

# Preprocess test data similarly
test_df[ndvi_cols] = test_df[ndvi_cols].replace(r'^\s*$', np.nan, regex=True)
for col in ndvi_cols:
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    test_df[col] = test_df[col].clip(lower, upper)  # Use same bounds as training
test_df[ndvi_cols] = test_df[ndvi_cols].interpolate(method='linear', axis=1, limit_direction='both')
test_df[ndvi_cols] = test_df[ndvi_cols].fillna(0)

# Denoise test data
ndvi_array_test = test_df[ndvi_cols].values
for i in range(ndvi_array_test.shape[0]):
    ndvi_array_test[i] = savgol_filter(ndvi_array_test[i], window_length=5, polyorder=2)
test_df[ndvi_cols] = pd.DataFrame(ndvi_array_test, columns=ndvi_cols)

# Step 3: Advanced Feature Engineering
# 3.1 Basic Statistics
train_df['ndvi_mean'] = train_df[ndvi_cols].mean(axis=1)
train_df['ndvi_median'] = train_df[ndvi_cols].median(axis=1)
train_df['ndvi_std'] = train_df[ndvi_cols].std(axis=1)
train_df['ndvi_min'] = train_df[ndvi_cols].min(axis=1)
train_df['ndvi_max'] = train_df[ndvi_cols].max(axis=1)
train_df['ndvi_range'] = train_df['ndvi_max'] - train_df['ndvi_min']

test_df['ndvi_mean'] = test_df[ndvi_cols].mean(axis=1)
test_df['ndvi_median'] = test_df[ndvi_cols].median(axis=1)
test_df['ndvi_std'] = test_df[ndvi_cols].std(axis=1)
test_df['ndvi_min'] = test_df[ndvi_cols].min(axis=1)
test_df['ndvi_max'] = test_df[ndvi_cols].max(axis=1)
test_df['ndvi_range'] = test_df['ndvi_max'] - test_df['ndvi_min']

# 3.2 Trend Feature (slope of NDVI over time)
def compute_slope(row):
    x = np.arange(len(ndvi_cols))
    y = row[ndvi_cols].values
    y = np.array(y, dtype=float)
    mask = ~np.isnan(y)
    if np.sum(mask) > 1:
        slope, _, _, _, _ = linregress(x[mask], y[mask])
        return slope
    return 0

train_df['ndvi_slope'] = train_df.apply(compute_slope, axis=1)
test_df['ndvi_slope'] = test_df.apply(compute_slope, axis=1)

# 3.3 Seasonal Features
season_mapping = {
    '01': 'winter', '02': 'winter', '03': 'spring',
    '04': 'spring', '05': 'spring', '06': 'summer',
    '07': 'summer', '08': 'summer', '09': 'fall',
    '10': 'fall', '11': 'fall', '12': 'winter'
}
seasonal_ndvi = {}
for col in ndvi_cols:
    month = col[4:6]
    season = season_mapping[month]
    if season not in seasonal_ndvi:
        seasonal_ndvi[season] = []
    seasonal_ndvi[season].append(col)

for season, cols in seasonal_ndvi.items():
    train_df[f'ndvi_{season}'] = train_df[cols].mean(axis=1)
    test_df[f'ndvi_{season}'] = test_df[cols].mean(axis=1)

# 3.4 New Features: Differences between consecutive NDVI values
for i in range(len(ndvi_cols) - 1):
    train_df[f'ndvi_diff_{i}'] = train_df[ndvi_cols[i + 1]] - train_df[ndvi_cols[i]]
    test_df[f'ndvi_diff_{i}'] = test_df[ndvi_cols[i + 1]] - test_df[ndvi_cols[i]]

# 3.5 Interaction Features
train_df['slope_mean_interaction'] = train_df['ndvi_slope'] * train_df['ndvi_mean']
test_df['slope_mean_interaction'] = test_df['ndvi_slope'] * test_df['ndvi_mean']

# 3.6 Rolling Statistics
train_df['ndvi_rolling_max_5'] = train_df[ndvi_cols].rolling(window=5, axis=1, min_periods=1).max().mean(axis=1)
train_df['ndvi_rolling_min_5'] = train_df[ndvi_cols].rolling(window=5, axis=1, min_periods=1).min().mean(axis=1)
test_df['ndvi_rolling_max_5'] = test_df[ndvi_cols].rolling(window=5, axis=1, min_periods=1).max().mean(axis=1)
test_df['ndvi_rolling_min_5'] = test_df[ndvi_cols].rolling(window=5, axis=1, min_periods=1).min().mean(axis=1)

# 3.7 NDVI Rate of Change
train_df['ndvi_rate_of_change'] = (train_df[ndvi_cols].iloc[:, -1] - train_df[ndvi_cols].iloc[:, 0]) / len(ndvi_cols)
test_df['ndvi_rate_of_change'] = (test_df[ndvi_cols].iloc[:, -1] - test_df[ndvi_cols].iloc[:, 0]) / len(ndvi_cols)

# Step 4: Prepare features and target
feature_cols = ['ndvi_mean', 'ndvi_median', 'ndvi_std', 'ndvi_min', 'ndvi_max', 'ndvi_range', 'ndvi_slope',
                'ndvi_winter', 'ndvi_spring', 'ndvi_summer', 'ndvi_fall', 'slope_mean_interaction',
                'ndvi_rolling_max_5', 'ndvi_rolling_min_5', 'ndvi_rate_of_change'] + [f'ndvi_diff_{i}' for i in range(len(ndvi_cols) - 1)]
X_train = train_df[feature_cols].fillna(0)
X_test = test_df[feature_cols].fillna(0)

# Encode the target variable
le = LabelEncoder()
y_train = le.fit_transform(train_df['class'])

# Step 5: Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train XGBoost Classifier with Cross-Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        random_state=42,
        n_estimators=500,  # Increased for more boosting rounds
        max_depth=5,  # Slightly reduced to prevent overfitting
        learning_rate=0.05,  # Lowered for better convergence
        subsample=0.8,  # Subsampling to reduce overfitting
        colsample_bytree=0.8,  # Feature sampling
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        early_stopping_rounds=50  # Early stopping to prevent overfitting
    )
    model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=False
    )
    models.append(model)
    print(f"Fold {fold + 1} best iteration: {model.best_iteration}")

# Step 7: Predict on the test dataset (average predictions from all folds)
test_preds = np.zeros((X_test_scaled.shape[0], len(le.classes_)))
for model in models:
    test_preds += model.predict_proba(X_test_scaled) / n_splits
predictions = np.argmax(test_preds, axis=1)
predicted_classes = le.inverse_transform(predictions)

# Create submission file
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'class': predicted_classes
})
submission.to_csv('submission_advanced_xgboost.csv', index=False)
print("Submission file 'submission_advanced_xgboost.csv' created successfully!")