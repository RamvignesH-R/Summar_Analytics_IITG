import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import linregress


train_df = pd.read_csv('./hacktrain.csv')
test_df = pd.read_csv('./hacktest.csv')


ndvi_cols = [col for col in train_df.columns if col.endswith('_N')]


train_df[ndvi_cols] = train_df[ndvi_cols].replace(r'^\s*$', np.nan, regex=True)
for col in ndvi_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    train_df[col] = train_df[col].fillna(0)  


train_df[ndvi_cols] = train_df[ndvi_cols].interpolate(method='linear', axis=1, limit_direction='both')


train_df[ndvi_cols] = train_df[ndvi_cols].fillna(0)


train_df[ndvi_cols] = train_df[ndvi_cols].T.rolling(window=3, min_periods=1).mean().T

train_df[ndvi_cols] = train_df[ndvi_cols].astype(float).fillna(0)


test_df[ndvi_cols] = test_df[ndvi_cols].replace(r'^\s*$', np.nan, regex=True)
for col in ndvi_cols:
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    test_df[col] = test_df[col].fillna(0)  
test_df[ndvi_cols] = test_df[ndvi_cols].interpolate(method='linear', axis=1, limit_direction='both')


test_df[ndvi_cols] = test_df[ndvi_cols].fillna(0)

test_df[ndvi_cols] = test_df[ndvi_cols].T.rolling(window=3, min_periods=1).mean().T

test_df[ndvi_cols] = test_df[ndvi_cols].astype(float).fillna(0)


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

def compute_slope(row, row_index):
    x = np.arange(len(ndvi_cols))
    y = row[ndvi_cols].values
    y = np.array(y, dtype=float)
    mask = ~np.isnan(y)
    if np.sum(mask) > 1:
        slope, _, _, _, _ = linregress(x[mask], y[mask])
        return slope
    return 0


train_df['ndvi_slope'] = [compute_slope(row, idx) for idx, row in train_df.iterrows()]

test_df['ndvi_slope'] = [compute_slope(row, idx) for idx, row in test_df.iterrows()]

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

feature_cols = ['ndvi_mean', 'ndvi_median', 'ndvi_std', 'ndvi_min', 'ndvi_max', 'ndvi_range', 'ndvi_slope',
                'ndvi_winter', 'ndvi_spring', 'ndvi_summer', 'ndvi_fall']
X_train = train_df[feature_cols].fillna(0)
X_test = test_df[feature_cols].fillna(0)

le = LabelEncoder()
y_train = le.fit_transform(train_df['class'])


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(multi_class='ovr', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)


predictions = model.predict(X_test_scaled)
predicted_classes = le.inverse_transform(predictions)


submission = pd.DataFrame({
    'ID': test_df['ID'],
    'class': predicted_classes
})
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")