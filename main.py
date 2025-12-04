import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

print("STEP 1: LOADING DATA")
df = pd.read_csv('data.csv')
print(df.head())

print("STEP 2: INITIAL DATASET INFO")
print(f"\nDataset Shape: {df.shape}")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\nBasic Statistics:")
print(df[['price', 'bedrooms', 'baths', 'Total_Area']].describe())

print("STEP 3: DATA CLEANING - REMOVING INVALID DATA")
df_clean = df.copy()

if 'Unnamed: 0' in df_clean.columns:
    df_clean.drop('Unnamed: 0', axis=1, inplace=True)
    print(" Dropped Unnamed: 0 column")

print(f"\nRows with price = 0: {(df_clean['price'] == 0).sum()}")
df_clean = df_clean[df_clean['price'] > 0]
print(f"After removing zero prices: {len(df_clean)} rows")

print(f"\nRows with Total_Area = 0: {(df_clean['Total_Area'] == 0).sum()}")
df_clean = df_clean[df_clean['Total_Area'] > 0]
print(f"After removing zero area: {len(df_clean)} rows")

print(f"\nRows with bedrooms = 0: {(df_clean['bedrooms'] == 0).sum()}")
df_clean = df_clean[df_clean['bedrooms'] > 0]
print(f"After removing zero bedrooms: {len(df_clean)} rows")

print(f"\nRows with baths = 0: {(df_clean['baths'] == 0).sum()}")
df_clean = df_clean[df_clean['baths'] > 0]
print(f"After removing zero baths: {len(df_clean)} rows")

print("STEP 4: REMOVING COLUMNS WITH TOO MANY MISSING VALUES")
missing_percent = (df_clean.isnull().sum() / len(df_clean)) * 100
print("\nMissing Values (%):")
print(missing_percent[missing_percent > 0])

drop_cols = []
for col in df_clean.columns:
    if missing_percent[col] > 30:
        drop_cols.append(col)
        print(f"  Dropping {col} ({missing_percent[col]:.1f}% missing)")

df_clean.drop(drop_cols, axis=1, inplace=True)

irrelevant = ['property_id', 'location_id', 'page_url', 'date_added', 'agency', 'agent', 'purpose', 'city']
for col in irrelevant:
    if col in df_clean.columns:
        df_clean.drop(col, axis=1, inplace=True)

print(f"\nRemaining columns: {df_clean.columns.tolist()}")
print(f"Remaining rows: {len(df_clean)}")

print("STEP 5: HANDLING MISSING VALUES")
print("\nMissing values before filling:")
print(df_clean.isnull().sum())

df_clean['bedrooms'].fillna(df_clean['bedrooms'].median(), inplace=True)
df_clean['baths'].fillna(df_clean['baths'].median(), inplace=True)
df_clean['Total_Area'].fillna(df_clean['Total_Area'].median(), inplace=True)

print("\n Filled missing numeric values with median")

print("\n" + "=" * 80)
print("STEP 6: REMOVING OUTLIERS")
print("=" * 80)

print(f"\nBefore outlier removal: {len(df_clean)} rows")

Q1 = df_clean['price'].quantile(0.25)
Q3 = df_clean['price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Price Q1: {Q1:,.0f}")
print(f"Price Q3: {Q3:,.0f}")
print(f"IQR: {IQR:,.0f}")
print(f"Lower Bound: {lower_bound:,.0f}")
print(f"Upper Bound: {upper_bound:,.0f}")

df_clean = df_clean[(df_clean['price'] >= lower_bound) & (df_clean['price'] <= upper_bound)]
print(f"\nAfter outlier removal: {len(df_clean)} rows")
print("STEP 7: FEATURE ENGINEERING")

print("\nCreating new features...")

df_clean['price_per_sqft'] = df_clean['price'] / df_clean['Total_Area']
df_clean['bedroom_bath_ratio'] = df_clean['bedrooms'] / df_clean['baths']
df_clean['total_rooms'] = df_clean['bedrooms'] + df_clean['baths']
df_clean['location_count'] = df_clean.groupby('location')['location'].transform('count')

print("  price_per_sqft = price / Total_Area")
print("  bedroom_bath_ratio = bedrooms / baths")
print("  total_rooms = bedrooms + baths")
print("  location_count = frequency of each location")

print("\nNew Features Statistics:")
print(df_clean[['price_per_sqft', 'bedroom_bath_ratio', 'total_rooms']].describe())

print("STEP 8: ENCODING CATEGORICAL VARIABLES")

le_property = LabelEncoder()
le_location = LabelEncoder()
le_province = LabelEncoder()

print("\nEncoding columns...")
df_clean['property_type_encoded'] = le_property.fit_transform(df_clean['property_type'])
df_clean['location_encoded'] = le_location.fit_transform(df_clean['location'])
df_clean['province_encoded'] = le_province.fit_transform(df_clean['province_name'])

print(f"   property_type encoded ({len(le_property.classes_)} classes)")
print(f"   location encoded ({len(le_location.classes_)} classes)")
print(f"   province_name encoded ({len(le_province.classes_)} classes)")

print("STEP 9: FEATURE SELECTION & SCALING")

features = ['property_type_encoded', 'location_encoded', 'province_encoded', 
            'latitude', 'longitude', 'baths', 'bedrooms', 'Total_Area',
            'price_per_sqft', 'bedroom_bath_ratio', 'total_rooms', 'location_count']

print(f"\nSelected {len(features)} features:")
for i, feat in enumerate(features, 1):
    print(f"  {i}. {feat}")

X = df_clean[features].copy()
y = df_clean['price'].copy()

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

print(f"\nInfinity values in X: {np.isinf(X.values).sum()}")
print(f"NaN values in X: {X.isnull().sum().sum()}")

X = X.replace([np.inf, -np.inf], np.nan)
valid_idx = X.notna().all(axis=1)
X = X[valid_idx]
y = y[valid_idx]

print(f"After removing inf/NaN: X shape {X.shape}, y shape {y.shape}")

print("\nScaling features using StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)
print("Features scaled successfully")

print("STEP 10: TRAIN-TEST SPLIT")

train_X, test_X, train_y, test_y = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

print(f"\nTraining set size: {train_X.shape[0]} rows")
print(f"Test set size: {test_X.shape[0]} rows")
print(f"Training ratio: {len(train_X) / len(X_scaled) * 100:.1f}%")
print(f"Test ratio: {len(test_X) / len(X_scaled) * 100:.1f}%")

print("STEP 11: LINEAR REGRESSION MODEL")

print("\nTraining Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(train_X, train_y)

lr_pred_train = lr_model.predict(train_X)
lr_pred_test = lr_model.predict(test_X)

lr_r2_train = r2_score(train_y, lr_pred_train)
lr_r2_test = r2_score(test_y, lr_pred_test)
lr_rmse_test = np.sqrt(mean_squared_error(test_y, lr_pred_test))
lr_mae_test = mean_absolute_error(test_y, lr_pred_test)

print(f"\nTraining R² Score: {round(lr_r2_train, 4)}")
print(f"Test R² Score: {round(lr_r2_test, 4)}")
print(f"Test RMSE: {lr_rmse_test:,.0f}")
print(f"Test MAE: {lr_mae_test:,.0f}")

print("STEP 12: RANDOM FOREST MODEL")

print("\nTraining Random Forest (this may take a moment)...")
rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=15)
rf_model.fit(train_X, train_y)

rf_pred_train = rf_model.predict(train_X)
rf_pred_test = rf_model.predict(test_X)

rf_r2_train = r2_score(train_y, rf_pred_train)
rf_r2_test = r2_score(test_y, rf_pred_test)
rf_rmse_test = np.sqrt(mean_squared_error(test_y, rf_pred_test))
rf_mae_test = mean_absolute_error(test_y, rf_pred_test)

print(f"\nTraining R² Score: {round(rf_r2_train, 4)}")
print(f"Test R² Score: {round(rf_r2_test, 4)}")
print(f"Test RMSE: {rf_rmse_test:,.0f}")
print(f"Test MAE: {rf_mae_test:,.0f}")

print("STEP 13: MODEL COMPARISON & SELECTION")

print("\nModel Performance Summary:")

print(f"{'Model':<20} {'Train R²':<15} {'Test R²':<15} {'RMSE':<20}")

print(f"{'Linear Regression':<20} {lr_r2_train:<15.4f} {lr_r2_test:<15.4f} {lr_rmse_test:<20,.0f}")
print(f"{'Random Forest':<20} {rf_r2_train:<15.4f} {rf_r2_test:<15.4f} {rf_rmse_test:<20,.0f}")

models_dict = {
    'Linear Regression': {'model': lr_model, 'r2': lr_r2_test, 'rmse': lr_rmse_test},
    'Random Forest': {'model': rf_model, 'r2': rf_r2_test, 'rmse': rf_rmse_test}
}

best_model_name = max(models_dict, key=lambda x: models_dict[x]['r2'])
best_model_obj = models_dict[best_model_name]['model']
best_r2 = models_dict[best_model_name]['r2']
best_rmse = models_dict[best_model_name]['rmse']

print(f"\n Best Model: {best_model_name}")
print(f"  R² Score: {best_r2:.4f}")
print(f"  RMSE: {best_rmse:,.0f}")

print("STEP 14: SAVING MODEL")
pickle.dump(best_model_obj, open('price_model.pkl', 'wb'))
print("Model saved to price_model.pkl")

pickle.dump(scaler, open('scaler.pkl', 'wb'))
print(" Scaler saved to scaler.pkl")

print("STEP 15: INVESTMENT SCORING")

df_investment = df_clean.copy()

def normalize_score(series):
    return ((series - series.min()) / (series.max() - series.min() + 1e-6)) * 100

df_investment['price_norm'] = normalize_score(df_investment['price'])
df_investment['area_norm'] = normalize_score(df_investment['Total_Area'])
df_investment['ppf_norm'] = normalize_score(df_investment['price_per_sqft'])

df_investment['investment_score'] = (
    0.45 * (100 - df_investment['price_norm']) +  
    0.30 * df_investment['area_norm'] +           
    0.25 * (100 - df_investment['ppf_norm'])     
)

def get_label(score):
    if score >= 75:
        return 'Good Investment'
    elif score >= 50:
        return 'Average'
    else:
        return 'Risky'

df_investment['recommendation'] = df_investment['investment_score'].apply(get_label)

print("\nInvestment Score Distribution:")
good = (df_investment['investment_score'] >= 75).sum()
avg = ((df_investment['investment_score'] >= 50) & (df_investment['investment_score'] < 75)).sum()
risky = (df_investment['investment_score'] < 50).sum()

print(f"  Good Investment (75-100): {good} properties ({good/len(df_investment)*100:.1f}%)")
print(f"  Average (50-75): {avg} properties ({avg/len(df_investment)*100:.1f}%)")
print(f"  Risky (<50): {risky} properties ({risky/len(df_investment)*100:.1f}%)")

print("STEP 16: TOP 10 INVESTMENT RECOMMENDATIONS")

recommendations = df_investment[[
    'location', 'property_type', 'bedrooms', 'baths', 'Total_Area',
    'price', 'price_per_sqft', 'investment_score', 'recommendation'
]].sort_values('investment_score', ascending=False).head(10)

print("\n")
print(recommendations.to_string(index=False))

print("STEP 17: SAVING RESULTS")

recommendations.to_csv('investment_recommendations.csv', index=False)
print(" Top recommendations saved to investment_recommendations.csv")

df_investment.to_csv('full_investment_analysis.csv', index=False)
print(" Full analysis saved to full_investment_analysis.csv")

print("STEP 19: GENERATING VISUALIZATIONS")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(df_investment['price'], bins=50, color='skyblue', edgecolor='black')
axes[0, 0].set_xlabel('Price (PKR)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Price Distribution')

axes[0, 1].hist(df_investment['investment_score'], bins=40, color='lightgreen', edgecolor='black')
axes[0, 1].set_xlabel('Investment Score')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Investment Score Distribution')

models_list = ['Linear Reg', 'Random Forest']
r2_scores = [lr_r2_test, rf_r2_test]
colors = ['steelblue', 'coral']
bars = axes[1, 0].bar(models_list, r2_scores, color=colors, edgecolor='black')
axes[1, 0].set_ylabel('R² Score')
axes[1, 0].set_title('Model Comparison (Test Set)')
axes[1, 0].set_ylim([0, 1])
for i, bar in enumerate(bars):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{r2_scores[i]:.3f}', ha='center', va='bottom')

rec_counts = df_investment['recommendation'].value_counts()
axes[1, 1].pie(rec_counts.values, labels=rec_counts.index, autopct='%1.1f%%',
              colors=['green', 'orange', 'red'], startangle=90)
axes[1, 1].set_title('Investment Recommendation Distribution')

plt.tight_layout()
plt.savefig('analysis_plots.png', dpi=300, bbox_inches='tight')
print(" Plots saved to analysis_plots.png")
plt.show()
print("PROJECT SUMMARY")

print(f"""
DATASET STATISTICS:
  Total Properties Analyzed: {len(df_investment):,}
  Average Property Price: PKR {df_investment['price'].mean():,.0f}
  Median Property Price: PKR {df_investment['price'].median():,.0f}
  Price Range: PKR {df_investment['price'].min():,.0f} - {df_investment['price'].max():,.0f}
  
  Average Area: {df_investment['Total_Area'].mean():,.0f} SqFt
  Average Bedrooms: {df_investment['bedrooms'].mean():.1f}
  Average Bathrooms: {df_investment['baths'].mean():.1f}

BEST ML MODEL:
  Model: {best_model_name}
  Test R² Score: {best_r2:.4f}
  Test RMSE: PKR {best_rmse:,.0f}

INVESTMENT SUMMARY:
  Good Investments: {good} properties
  Average Investments: {avg} properties
  Risky Investments: {risky} properties

OUTPUT FILES:
   price_model.pkl
   scaler.pkl
   investment_recommendations.csv
   full_investment_analysis.csv
   analysis_plots.png
""")

print("=" * 80)
print(" PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 80)