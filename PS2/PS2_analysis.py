import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor

# STAC WARS - PS2: Solar Radiation Analysis

# 1. LOAD DATA
df = pd.read_csv('solar_data.csv')

# Data Cleaning & Preprocessing
# Splitting the date string first and concatenating with time columns since they are separated in the raw data.
df['Datetime'] = pd.to_datetime(df['Data'].str.split().str[0] + ' ' + df['Time'])
df['TimeSunRise'] = pd.to_datetime(df['Data'].str.split().str[0] + ' ' + df['TimeSunRise'])
df['TimeSunSet'] = pd.to_datetime(df['Data'].str.split().str[0] + ' ' + df['TimeSunSet'])

# Extracting standalone date and hour for easier grouping later
df['Date'] = df['Datetime'].dt.date
df['Hour'] = df['Datetime'].dt.hour


# PART A: DATA VISUALIZATION

# Using dark background for better contrast in the presentation slides.
plt.style.use('dark_background')

# 1. Radiation vs Time of Day
plt.figure(figsize=(10,5))
sns.lineplot(data=df, x='Hour', y='Radiation', color='orange')
plt.title('Average Solar Radiation by Hour of Day')
plt.xlabel('Hour of Day (24H)')
plt.ylabel('Solar Radiation (W/m^2)')
plt.savefig('Radiation_vs_Time.png')
plt.close()

# 2. Radiation vs Temperature & Humidity
fig, axes = plt.subplots(1, 2, figsize=(15,5))
sns.scatterplot(data=df, x='Temperature', y='Radiation', alpha=0.3, ax=axes[0], color='red')
axes[0].set_title('Radiation vs Temperature')
sns.scatterplot(data=df, x='Humidity', y='Radiation', alpha=0.3, ax=axes[1], color='blue')
axes[1].set_title('Radiation vs Humidity')
plt.tight_layout()
plt.savefig('Radiation_vs_Weather.png')
plt.close()

# Bonus: Correlation Heatmap
# Checking feature correlations before building the regression model to see which variables impact radiation most.
plt.figure(figsize=(8,6))
corr = df[['Radiation', 'Temperature', 'Humidity', 'Pressure', 'Speed']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.savefig('Correlation_Heatmap.png')
plt.close()


# PART C: SOLAR DAY ANALYSIS

print("PART C: SOLAR DAY ANALYSIS")
# Grouping by date to find max and average radiation per day
daily_stats = df.groupby('Date').agg(
    Max_Radiation=('Radiation', 'max'),
    Avg_Radiation=('Radiation', 'mean')
).reset_index()

# Finding the exact row with the highest recorded radiation
max_rad_idx = df['Radiation'].idxmax()
max_row = df.loc[max_rad_idx]
print(f"1. Date with highest radiation: {max_row['Date']}")
print(f"2. Time of max radiation: {max_row['Time']}")
print(f"3. Conditions -> Temp: {max_row['Temperature']} F, Humidity: {max_row['Humidity']}%, Wind: {max_row['Speed']} mph")

# Time delta from sunrise
time_after_sunrise = max_row['Datetime'] - max_row['TimeSunRise']
print(f"4. Peak radiation occurred {time_after_sunrise} after sunrise.")

# Getting the top 5 days 
top_5_avg = daily_stats.sort_values(by='Avg_Radiation', ascending=False).head(5)
print("\n5. Top 5 Days with Highest Average Radiation:\n", top_5_avg.to_string(index=False))


# BONUS: MACHINE LEARNING PREDICTION

print("\nBONUS: ML PREDICTION")
# Dropping non-predictive features and setting up the target variable
features = ['Temperature', 'Humidity', 'Pressure', 'Speed', 'Hour']
X = df[features]
y = df['Radiation']

# Standard 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Swapping from Random Forest to CatBoost.CatBoost is used for tabular data tasks since it handles environmental datasets efficiently without needing heavy hyperparameter tuning.
model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_seed=42, verbose=0)
model.fit(X_train, y_train)

# Evaluating final model accuracy
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"Model RMSE: {rmse:.2f}")
print(f"Model R^2 Score: {r2:.2f}")

# Checking what features the model actually prioritized during training
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)