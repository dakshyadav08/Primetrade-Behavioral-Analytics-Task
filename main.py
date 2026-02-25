import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. LOAD DATA ---
print("Step 1: Loading data...")
# Raw paths (Fixed with 'r')
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
    'Symbol': 'count' # Counting trades
}).rename(columns={'Symbol': 'Trade_Count'}).reset_index()

sentiment_col = [col for col in sentiment.columns if 'class' in col.lower() or 'sentiment' in col.lower()][0]
sentiment = sentiment[['Date', sentiment_col]].rename(columns={sentiment_col: 'Classification'})

df = pd.merge(daily_data, sentiment, on='Date', how='inner')

# --- 3. ADVANCED FEATURE ENGINEERING (For Accuracy) ---
print("Step 3: Engineering features for Accuracy...")
# Rolling PnL (Pichle 3 din ka avg)
df['Rolling_PnL'] = df.groupby('Account')['Closed PnL'].transform(lambda x: x.rolling(window=3).mean())
# Volatility (PnL kitna swing kar raha hai)
df['PnL_Volatility'] = df.groupby('Account')['Closed PnL'].transform(lambda x: x.rolling(window=3).std())
# Momentum
df['PnL_Change'] = df.groupby('Account')['Closed PnL'].diff()

df = df.fillna(0) # Filling empty values

# --- 4. CLUSTERING (Archetypes) ---
print("Step 4: Clustering traders...")
scaler = StandardScaler()
cluster_features = scaler.fit_transform(df[['Trade_Count', 'Closed PnL', 'PnL_Volatility']])
kmeans = KMeans(n_clusters=3, random_state=42)
df['Archetype_ID'] = kmeans.fit_predict(cluster_features)

archetype_map = {0: 'Steady Trader', 1: 'High-Volume Scalper', 2: 'Risky Gambler'}
df['Archetype_Name'] = df['Archetype_ID'].map(archetype_map)

# --- 5. PREDICTIVE MODEL ---
print("Step 5: Training Model...")
# Goal: Kal profit hoga ya nahi (Target)
df['Target'] = (df.groupby('Account')['Closed PnL'].shift(-1) > 0).astype(int)
df_model = df.dropna()

# Converting Sentiment to numbers
df_model['Sent_Num'] = pd.factorize(df_model['Classification'])[0]

features = ['Trade_Count', 'Closed PnL', 'Rolling_PnL', 'PnL_Volatility', 'Sent_Num']
X = df_model[features]
y = df_model['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

print(f"Model Training Done! Accuracy: {model.score(X_test, y_test)*100:.2f}%")

# Save the final file for Dashboard
df.to_csv('enriched_data.csv', index=False)
print("SUCCESS! 'enriched_data.csv' is ready.")

except FileNotFoundError:
    st.error("Error: 'enriched_data.csv' nahi mili. Pehle 'final_analysis.py'")