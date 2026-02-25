# Trader Behavior & Sentiment Analysis

An end-to-end Data Science pipeline to analyze how Bitcoin Market Sentiment (Fear & Greed Index) impacts trader performance on the Hyperliquid exchange.

## 🌟 Project Highlights
- **Trader Segmentation:** Used K-Means clustering to identify distinct behavioral archetypes.
- **Sentiment Correlation:** Analyzed PnL trends against market psychology.
- **Predictive Modeling:** Built a Random Forest Classifier to forecast daily profitability.
- **Live Dashboard:** Interactive Streamlit UI for real-time data exploration.

---

## 🛠️ Tech Stack
- **Languages:** Python 3.10+
- **Libraries:** Pandas, Scikit-Learn, Plotly, Streamlit
- **Environment:** VS Code / Jupyter Notebook

---
📂 Project Structure
```bash

├── Data/                   # Original CSV files (fear_greed, historical_data)
├── Notebook/
│   └── analysis.ipynb      # Step-by-step analysis (optional)
├── final_analysis.py       # Main processing script
├── app.py                  # Streamlit Dashboard
├── enriched_data.csv       # Final processed data
├── requirements.txt        # Libraries list
├── SUMMARY.md              # 1-page Write-up (Methodology & Strategy)
└── README.md               # Setup & How to run

```
---

## 🧠 Methodology & Write-up

### 1. Data Integration & Engineering
- **Aggregation:** Combined historical trade data with daily Fear & Greed indices.
- **Feature Creation:** Engineered `trade_frequency` (activity level) and `rolling_pnl` to capture momentum and over-trading tendencies.
- **Normalization:** Used `StandardScaler` to prepare features for unbiased clustering.

### 2. Behavioral Clustering (K-Means)
Identified 3 key trader segments:
- **Steady Traders:** Low frequency, consistent small gains, high emotional control.
- **High-Volume Scalpers:** Extreme activity, small margins, highly sensitive to volatility.
- **Risky Gamblers:** Large PnL swings, often making irrational moves during market extremes.

### 3. Predictive Insights
Developed a Random Forest model (Accuracy: ~60%) to predict if a trader will be profitable the next day based on their current behavior and market mood.



---

## 📈 Key Insights & Strategic Recommendations

### Core Observations
* **Revenge Trading:** Traders in the "Risky" segment increase activity by **25%** during "Extreme Fear" days to recover losses, leading to higher liquidations.
* **Late Entry Trap:** "High-Volume Scalpers" show diminishing returns during "Extreme Greed" as they tend to enter near local price tops.
* **Volume vs. Value:** High trade frequency does not correlate with high PnL; "Steady Traders" actually have the best risk-adjusted returns.

### Strategy Recommendations
* **Dynamic Risk Gates:** Implement auto-caps on leverage for "Risky" accounts when Market Sentiment drops below 20 (Extreme Fear).
* **Emotional Nudges:** Launch "Cool-off" alerts for users whose `trade_frequency` spikes beyond 2 standard deviations during high volatility.
* **Incentive Tiers:** Provide fee discounts to "Steady Traders" to maintain platform liquidity during bearish cycles.

---

## 🚀 Setup & Execution

### 1. Installation
Clone the repository and install the required libraries:
```bash
git clone [https://github.com/YOUR_USERNAME/Hyperliquid-Trader-Behavior-Analysis.git](https://github.com/YOUR_USERNAME/Hyperliquid-Trader-Behavior-Analysis.git)
cd Hyperliquid-Trader-Behavior-Analysis
pip install -r requirements.txt
```
---

# Project Summary: Behavioral Analysis & Strategic Insights

### 1. Methodology
- **Data Integration:** Merged transaction-level data with daily BTC Sentiment (Fear & Greed Index). Normalized timestamps to align market mood with PnL.
- **Feature Engineering:** Created `trade_frequency` to detect over-trading and rolling PnL averages to observe momentum.
- **Clustering:** Applied **K-Means Clustering** on frequency and profitability to segment users into 3 archetypes: *Steady Traders*, *High-Volume Scalpers*, and *Risky Gamblers*.
- **Machine Learning:** Implemented a **Random Forest Classifier** to predict daily profitability based on prior performance and sentiment.

### 2. Core Insights
- **The Fear Factor:** "Risky Gamblers" show a 25% increase in trade count during "Extreme Fear," suggesting emotional revenge trading.
- **Greed Stagnation:** Profitability for "Scalpers" actually dips during "Extreme Greed" due to over-leveraging on local price tops.
- **Stability:** "Steady Traders" maintain consistent win rates regardless of sentiment, proving that low-frequency trading is more sustainable.

### 3. Strategy Recommendations
- **Dynamic Risk Gates:** For "Risky Gambler" segments, the platform could implement auto-leverage caps when the Fear & Greed index drops below 20.
- **Behavioral Nudges:** Send "Cool-off" notifications to users who exceed their average `trade_frequency` by 2 standard deviations during high-volatility sentiment.
- **Tiered Incentives:** Offer fee rebates to "Steady Traders" to encourage them to provide more liquidity during market extremes.
