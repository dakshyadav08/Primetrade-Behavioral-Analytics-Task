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
