import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. LOAD DATA ---
print("Step 1: Loading data...")
sentiment = pd.read_csv(r'C:\Users\MSII\Desktop\Primetrade_Assignment\Data\fear_greed_index.csv')
trader = pd.read_csv(r'C:\Users\MSII\Desktop\Primetrade_Assignment\Data\historical_data.csv')

# Standardizing Dates
sent_date_col = [col for col in sentiment.columns if 'date' in col.lower()][0]
sentiment['Date'] = pd.to_datetime(sentiment[sent_date_col])

trader_time_col = [col for col in trader.columns if 'time' in col.lower() or 'date' in col.lower()][0]
trader['time'] = pd.to_datetime(trader[trader_time_col], dayfirst=True)
trader['Date'] = trader['time'].dt.normalize()

# --- 2. CLEANING & MERGING ---
print("Step 2: Cleaning and Aggregating...")
daily_data = trader.groupby(['Date', 'Account']).agg({
    'Closed PnL': 'sum',
    'Account': 'count' 
}).rename(columns={'Account': 'trade_frequency'}).reset_index()

sentiment_col = [col for col in sentiment.columns if 'class' in col.lower() or 'sentiment' in col.lower()][0]
sentiment = sentiment[['Date', sentiment_col]].rename(columns={sentiment_col: 'Classification'})

df = pd.merge(daily_data, sentiment, on='Date', how='inner')

# --- 3. CLUSTERING ---
print("Step 3: Creating Trader Segments...")
scaler = StandardScaler()
cluster_features = scaler.fit_transform(df[['trade_frequency', 'Closed PnL']])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Archetype_ID'] = kmeans.fit_predict(cluster_features)

archetype_map = {0: 'Steady Trader', 1: 'High-Volume Scalper', 2: 'Risky Gambler'}
df['Archetype_Name'] = df['Archetype_ID'].map(archetype_map)

# --- 4. PREDICTIVE MODEL ---
print("Step 4: Training Model...")
df['Tomorrow_Profitable'] = (df.groupby('Account')['Closed PnL'].shift(-1) > 0).astype(int)
df_model = df.dropna().copy()
df_model['Sent_Num'] = pd.factorize(df_model['Classification'])[0]

X = df_model[['trade_frequency', 'Closed PnL', 'Sent_Num']]
y = df_model['Tomorrow_Profitable']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"--- SUCCESS! Accuracy: {model.score(X_test, y_test)*100:.2f}% ---")
df.to_csv('enriched_data.csv', index=False)