import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import linregress, skew, kurtosis
from scipy.signal import savgol_filter
from scipy.fft import fft

train_df = pd.read_csv('./hacktrain.csv')
test_df = pd.read_csv('./hacktest.csv')

ndvi_cols = [col for col in train_df.columns if col.endswith('_N')]
train_df[ndvi_cols] = train_df[ndvi_cols].replace(r'^\s*$', np.nan, regex=True)
for col in ndvi_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
for col in ndvi_cols:
    lower, upper = train_df[col].quantile([0.01, 0.99])
    train_df[col] = train_df[col].clip(lower, upper)
train_df[ndvi_cols] = train_df[ndvi_cols].interpolate(method='linear', axis=1, limit_direction='both')
train_df[ndvi_cols] = train_df[ndvi_cols].fillna(0)
ndvi_array = train_df[ndvi_cols].values
for i in range(ndvi_array.shape[0]):
    ndvi_array[i] = savgol_filter(ndvi_array[i], window_length=5, polyorder=2)
train_df[ndvi_cols] = pd.DataFrame(ndvi_array, columns=ndvi_cols)

test_df[ndvi_cols] = test_df[ndvi_cols].replace(r'^\s*$', np.nan, regex=True)
for col in ndvi_cols:
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    test_df[col] = test_df[col].clip(lower, upper)
test_df[ndvi_cols] = test_df[ndvi_cols].interpolate(method='linear', axis=1, limit_direction='both')
test_df[ndvi_cols] = test_df[ndvi_cols].fillna(0)
ndvi_array_test = test_df[ndvi_cols].values
for i in range(ndvi_array_test.shape[0]):
    ndvi_array_test[i] = savgol_filter(ndvi_array_test[i], window_length=5, polyorder=2)
test_df[ndvi_cols] = pd.DataFrame(ndvi_array_test, columns=ndvi_cols)

train_df['ndvi_mean'] = train_df[ndvi_cols].mean(axis=1)
train_df['ndvi_median'] = train_df[ndvi_cols].median(axis=1)
train_df['ndvi_std'] = train_df[ndvi_cols].std(axis=1)
train_df['ndvi_min'] = train_df[ndvi_cols].min(axis=1)
train_df['ndvi_max'] = train_df[ndvi_cols].max(axis=1)
train_df['ndvi_range'] = train_df['ndvi_max'] - train_df['ndvi_min']
train_df['ndvi_skew'] = train_df[ndvi_cols].apply(skew, axis=1)
train_df['ndvi_kurtosis'] = train_df[ndvi_cols].apply(kurtosis, axis=1)

test_df['ndvi_mean'] = test_df[ndvi_cols].mean(axis=1)
test_df['ndvi_median'] = test_df[ndvi_cols].median(axis=1)
test_df['ndvi_std'] = test_df[ndvi_cols].std(axis=1)
test_df['ndvi_min'] = test_df[ndvi_cols].min(axis=1)
test_df['ndvi_max'] = test_df[ndvi_cols].max(axis=1)
test_df['ndvi_range'] = test_df['ndvi_max'] - test_df['ndvi_min']
test_df['ndvi_skew'] = test_df[ndvi_cols].apply(skew, axis=1)
test_df['ndvi_kurtosis'] = test_df[ndvi_cols].apply(kurtosis, axis=1)

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

diff_cols_train = {}
diff_cols_test = {}
for i in range(len(ndvi_cols) - 1):
    diff_cols_train[f'ndvi_diff_{i}'] = train_df[ndvi_cols[i + 1]] - train_df[ndvi_cols[i]]
    diff_cols_test[f'ndvi_diff_{i}'] = test_df[ndvi_cols[i + 1]] - test_df[ndvi_cols[i]]
train_df = pd.concat([train_df, pd.DataFrame(diff_cols_train)], axis=1)
test_df = pd.concat([test_df, pd.DataFrame(diff_cols_test)], axis=1)

second_diff_cols_train = {}
second_diff_cols_test = {}
for i in range(len(ndvi_cols) - 2):
    second_diff_cols_train[f'ndvi_second_diff_{i}'] = diff_cols_train[f'ndvi_diff_{i + 1}'] - diff_cols_train[f'ndvi_diff_{i}']
    second_diff_cols_test[f'ndvi_second_diff_{i}'] = diff_cols_test[f'ndvi_diff_{i + 1}'] - diff_cols_test[f'ndvi_diff_{i}']
train_df = pd.concat([train_df, pd.DataFrame(second_diff_cols_train)], axis=1)
test_df = pd.concat([test_df, pd.DataFrame(second_diff_cols_test)], axis=1)

lag_cols_train = {}
lag_cols_test = {}
for i in range(1, 3):
    for col in ndvi_cols:
        lag_cols_train[f'{col}_lag_{i}'] = train_df[col].shift(i)
        lag_cols_test[f'{col}_lag_{i}'] = test_df[col].shift(i)
train_df = pd.concat([train_df, pd.DataFrame(lag_cols_train)], axis=1)
test_df = pd.concat([test_df, pd.DataFrame(lag_cols_test)], axis=1)
lag_cols = [f'{col}_lag_{i}' for col in ndvi_cols for i in range(1, 3)]
train_df[lag_cols] = train_df[lag_cols].fillna(0)
test_df[lag_cols] = test_df[lag_cols].fillna(0)

train_df['slope_mean_interaction'] = train_df['ndvi_slope'] * train_df['ndvi_mean']
test_df['slope_mean_interaction'] = test_df['ndvi_slope'] * test_df['ndvi_mean']

train_df['ndvi_rolling_max_5'] = train_df[ndvi_cols].T.rolling(window=5, min_periods=1).max().T.mean(axis=1)
train_df['ndvi_rolling_min_5'] = train_df[ndvi_cols].T.rolling(window=5, min_periods=1).min().T.mean(axis=1)
test_df['ndvi_rolling_max_5'] = test_df[ndvi_cols].T.rolling(window=5, min_periods=1).max().T.mean(axis=1)
test_df['ndvi_rolling_min_5'] = test_df[ndvi_cols].T.rolling(window=5, min_periods=1).min().T.mean(axis=1)

train_df['ndvi_rate_of_change'] = (train_df[ndvi_cols].iloc[:, -1] - train_df[ndvi_cols].iloc[:, 0]) / len(ndvi_cols)
test_df['ndvi_rate_of_change'] = (test_df[ndvi_cols].iloc[:, -1] - test_df[ndvi_cols].iloc[:, 0]) / len(ndvi_cols)

fft_vals = np.abs(fft(train_df[ndvi_cols].values, axis=1))
train_df['fft_mean'] = fft_vals.mean(axis=1)
train_df['fft_max'] = fft_vals.max(axis=1)
fft_vals_test = np.abs(fft(test_df[ndvi_cols].values, axis=1))
test_df['fft_mean'] = fft_vals_test.mean(axis=1)
test_df['fft_max'] = fft_vals_test.max(axis=1)

train_df = train_df.copy()
test_df = test_df.copy()


feature_cols = ['ndvi_mean', 'ndvi_median', 'ndvi_std', 'ndvi_min', 'ndvi_max', 'ndvi_range', 'ndvi_skew', 'ndvi_kurtosis',
                'ndvi_slope', 'ndvi_winter', 'ndvi_spring', 'ndvi_summer', 'ndvi_fall', 'slope_mean_interaction',
                'ndvi_rolling_max_5', 'ndvi_rolling_min_5', 'ndvi_rate_of_change', 'fft_mean', 'fft_max'] + \
               [f'ndvi_diff_{i}' for i in range(len(ndvi_cols) - 1)] + \
               [f'ndvi_second_diff_{i}' for i in range(len(ndvi_cols) - 2)] + lag_cols
X_train = pd.DataFrame(train_df[feature_cols].fillna(0), columns=feature_cols)  # Keep as DataFrame with feature names
X_test = pd.DataFrame(test_df[feature_cols].fillna(0), columns=feature_cols)

le = LabelEncoder()
y_train = le.fit_transform(train_df['class'])

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)  # Keep feature names
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

#Train Ensemble with XGBoost, LightGBM, and CatBoost
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
xgb_models = []
lgb_models = []
cat_models = []
xgb_train_preds = np.zeros((X_train_scaled.shape[0], len(le.classes_)))
lgb_train_preds = np.zeros((X_train_scaled.shape[0], len(le.classes_)))
cat_train_preds = np.zeros((X_train_scaled.shape[0], len(le.classes_)))
xgb_test_preds = np.zeros((X_test_scaled.shape[0], len(le.classes_)))
lgb_test_preds = np.zeros((X_test_scaled.shape[0], len(le.classes_)))
cat_test_preds = np.zeros((X_test_scaled.shape[0], len(le.classes_)))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    X_fold_train, X_fold_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    # XGBoost Model
    xgb_model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        random_state=42,
        n_estimators=1500,  # Increased
        max_depth=4,
        learning_rate=0.02,  # Lowered
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.5,
        early_stopping_rounds=150  # Adjusted
    )
    xgb_model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=False
    )
    xgb_models.append(xgb_model)
    xgb_train_preds[val_idx] = xgb_model.predict_proba(X_fold_val)
    xgb_test_preds += xgb_model.predict_proba(X_test_scaled) / n_splits
    print(f"Fold {fold + 1} XGBoost best iteration: {xgb_model.best_iteration}")

    #LightGBM Model
    lgb_model = LGBMClassifier(
        objective='multiclass',
        num_class=len(le.classes_),
        metric='multi_logloss',
        random_state=42,
        n_estimators=1500,  
        max_depth=4,
        learning_rate=0.02,  
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.5,
        verbosity=-1
    )
    lgb_model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        eval_metric='multi_logloss',
        callbacks=[early_stopping(stopping_rounds=150, verbose=False)]
    )
    lgb_models.append(lgb_model)
    lgb_train_preds[val_idx] = lgb_model.predict_proba(X_fold_val)
    lgb_test_preds += lgb_model.predict_proba(X_test_scaled) / n_splits
    print(f"Fold {fold + 1} LightGBM best iteration: {lgb_model.best_iteration_}")

    #CatBoost Model
    cat_model = CatBoostClassifier(
        iterations=1500,  
        depth=4,
        learning_rate=0.02, 
        random_seed=42,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        early_stopping_rounds=150,
        verbose=False
    )
    cat_model.fit(
        X_fold_train, y_fold_train,
        eval_set=(X_fold_val, y_fold_val),
        use_best_model=True
    )
    cat_models.append(cat_model)
    cat_train_preds[val_idx] = cat_model.predict_proba(X_fold_val)
    cat_test_preds += cat_model.predict_proba(X_test_scaled) / n_splits
    print(f"Fold {fold + 1} CatBoost best iteration: {cat_model.best_iteration_}")

#Feature Selection using Feature Importance
xgb_importance = np.mean([model.feature_importances_ for model in xgb_models], axis=0)
lgb_importance = np.mean([model.feature_importances_ for model in lgb_models], axis=0)
cat_importance = np.mean([model.feature_importances_ for model in cat_models], axis=0)
avg_importance = (xgb_importance + lgb_importance + cat_importance) / 3

top_features_idx = np.argsort(avg_importance)[-50:]
top_features = [feature_cols[i] for i in top_features_idx]
print("Top features selected:", top_features)

X_train_top = X_train_scaled[top_features]
X_test_top = X_test_scaled[top_features]

#Retrain models on top features
xgb_train_preds_top = np.zeros((X_train_top.shape[0], len(le.classes_)))
lgb_train_preds_top = np.zeros((X_train_top.shape[0], len(le.classes_)))
cat_train_preds_top = np.zeros((X_train_top.shape[0], len(le.classes_)))
xgb_test_preds_top = np.zeros((X_test_top.shape[0], len(le.classes_)))
lgb_test_preds_top = np.zeros((X_test_top.shape[0], len(le.classes_)))
cat_test_preds_top = np.zeros((X_test_top.shape[0], len(le.classes_)))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_top)):
    X_fold_train, X_fold_val = X_train_top.iloc[train_idx], X_train_top.iloc[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    xgb_model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        random_state=42,
        n_estimators=1500,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.5,
        early_stopping_rounds=150
    )
    xgb_model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=False
    )
    xgb_train_preds_top[val_idx] = xgb_model.predict_proba(X_fold_val)
    xgb_test_preds_top += xgb_model.predict_proba(X_test_top) / n_splits

    lgb_model = LGBMClassifier(
        objective='multiclass',
        num_class=len(le.classes_),
        metric='multi_logloss',
        random_state=42,
        n_estimators=1500,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.5,
        verbosity=-1
    )
    lgb_model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        eval_metric='multi_logloss',
        callbacks=[early_stopping(stopping_rounds=150, verbose=False)]
    )
    lgb_train_preds_top[val_idx] = lgb_model.predict_proba(X_fold_val)
    lgb_test_preds_top += lgb_model.predict_proba(X_test_top) / n_splits

    cat_model = CatBoostClassifier(
        iterations=1500,
        depth=4,
        learning_rate=0.02,
        random_seed=42,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        early_stopping_rounds=150,
        verbose=False
    )
    cat_model.fit(
        X_fold_train, y_fold_train,
        eval_set=(X_fold_val, y_fold_val),
        use_best_model=True
    )
    cat_train_preds_top[val_idx] = cat_model.predict_proba(X_fold_val)
    cat_test_preds_top += cat_model.predict_proba(X_test_top) / n_splits

# Stacking with Meta-Learner
stacked_train_preds = np.hstack([xgb_train_preds_top, lgb_train_preds_top, cat_train_preds_top])
stacked_test_preds = np.hstack([xgb_test_preds_top, lgb_test_preds_top, cat_test_preds_top])

meta_learner = LogisticRegression(random_state=42, C=0.5)  # Removed multi_class
meta_learner.fit(stacked_train_preds, y_train)

#Pseudo-Labeling
test_probs = meta_learner.predict_proba(stacked_test_preds)
confident_idx = test_probs.max(axis=1) > 0.95  # Lowered threshold
pseudo_labels = np.argmax(test_probs[confident_idx], axis=1)
X_pseudo = X_test_top[confident_idx]
y_pseudo = pseudo_labels

X_train_combined = pd.concat([X_train_top, X_pseudo.reset_index(drop=True)], axis=0, ignore_index=True)
y_train_combined = np.hstack([y_train, y_pseudo])

#Retrain models with combined data
final_xgb_train_preds = np.zeros((X_train_combined.shape[0], len(le.classes_)))
final_lgb_train_preds = np.zeros((X_train_combined.shape[0], len(le.classes_)))
final_cat_train_preds = np.zeros((X_train_combined.shape[0], len(le.classes_)))
final_test_preds = np.zeros((X_test_top.shape[0], len(le.classes_)))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_combined)):
    X_fold_train, X_fold_val = X_train_combined.iloc[train_idx], X_train_combined.iloc[val_idx]
    y_fold_train, y_fold_val = y_train_combined[train_idx], y_train_combined[val_idx]

    xgb_model = XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        eval_metric='mlogloss',
        random_state=42,
        n_estimators=1500,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.5,
        early_stopping_rounds=150
    )
    xgb_model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=False
    )
    final_xgb_train_preds[val_idx] = xgb_model.predict_proba(X_fold_val)
    final_test_preds += 0.4 * xgb_model.predict_proba(X_test_top) / n_splits  # Adjusted weight

    lgb_model = LGBMClassifier(
        objective='multiclass',
        num_class=len(le.classes_),
        metric='multi_logloss',
        random_state=42,
        n_estimators=1500,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.5,
        verbosity=-1
    )
    lgb_model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        eval_metric='multi_logloss',
        callbacks=[early_stopping(stopping_rounds=150, verbose=False)]
    )
    final_lgb_train_preds[val_idx] = lgb_model.predict_proba(X_fold_val)
    final_test_preds += 0.3 * lgb_model.predict_proba(X_test_top) / n_splits  # Adjusted weight

    cat_model = CatBoostClassifier(
        iterations=1500,
        depth=4,
        learning_rate=0.02,
        random_seed=42,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        early_stopping_rounds=150,
        verbose=False
    )
    cat_model.fit(
        X_fold_train, y_fold_train,
        eval_set=(X_fold_val, y_fold_val),
        use_best_model=True
    )
    final_cat_train_preds[val_idx] = cat_model.predict_proba(X_fold_val)
    final_test_preds += 0.3 * cat_model.predict_proba(X_test_top) / n_splits  # Adjusted weight

#Final stacking
final_stacked_train = np.hstack([final_xgb_train_preds, final_lgb_train_preds, final_cat_train_preds])
meta_learner.fit(final_stacked_train, y_train_combined)

#Final Predictions with Label Smoothing
final_test_preds = (final_test_preds + 0.03) / (1 + 0.03 * len(le.classes_))
predictions = np.argmax(final_test_preds, axis=1)
predicted_classes = le.inverse_transform(predictions)

#submission file
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'class': predicted_classes
})
submission.to_csv('submission_final_tuned.csv', index=False)
print("Submission file 'submission_final_tuned.csv' created successfully!")