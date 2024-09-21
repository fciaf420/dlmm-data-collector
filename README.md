# 🧠 DLMM Data Collector

## 🚀 Overview
The **DLMM Data Collector** is an advanced script designed to gather, analyze, and optimize settings for the **SOL Decoder DLMM Bot**. It collects data from trending pools on the **Solana network** using the **GeckoTerminal API** and leverages powerful mathematical and machine learning models to generate actionable insights.

## ✨ Key Features
- 📊 **Real-Time Data Collection:** Gather live data from trending Solana pools.
- 📈 **Advanced Analysis:** Perform metrics calculations and time series analysis.
- 🤖 **Machine Learning Integration:** Utilizes Random Forest for predictive analytics.
- 🎯 **Recommendations:** Provides risk-adjusted entry criteria & adaptive stop loss.
- ⚙️ **Dynamic Optimization:** Adjusts timeframe and risk profile automatically.
- 🔄 **Continuous Updates:** Auto-refreshes data every 30 minutes in the background.
- 🎨 **Enhanced UI:** Cyberpunk-style ASCII art and colored text output.

## 🧩 Functional Components
1. **📡 Data Collection**
   - API interaction with **GeckoTerminal**
   - Collects prices, market caps, volumes, price changes
   - Robust error handling & rate limiting
2. **🔍 Data Processing & Analysis**
   - Data cleaning and preprocessing
   - Calculate key metrics (volatility, volume ratios, price changes)
   - Machine learning-based feature importance analysis
3. **💡 Recommendation Engine**
   - ML-powered timeframe selection
   - Provides risk-adjusted recommendations (Degen, Moderate, Conservative)
   - Calculates dynamic stop losses based on market volatility
4. **🖥️ User Interface**
   - Interactive console with real-time data updates
   - On-demand recommendations and data pulls

## 📐 Mathematical and ML Models
- **Volatility Calculation:** GARCH Model
- **Multifactor Scoring:** Combines price momentum, volume trends, market cap, and liquidity
- **Random Forest Regression:** For predicting optimal trading parameters
- **Dynamic Stop Loss:** Adaptive based on recent market volatility

## ⚙️ Advanced Features
- **Background Data Updates:** Automatically refreshes data every 30 minutes
- **ML-based Timeframe Optimization:** Evaluates and recommends optimal timeframes
- **Enhanced Market Cap Analysis:** Improved filtering and distribution analysis
- **Consistency Checks:** Verifies recommendations against explanations

## 📦 Requirements
Refer to `requirements.txt` for a full list of Python dependencies, including:
- requests
- pandas & numpy
- scikit-learn
- arch
- rich
- colorama

## ⚠️ Disclaimer
This script is for informational purposes only. Always conduct your own research and consider your risk tolerance before making investment decisions.

## 🛠️ Usage
To run the script:
```bash
python dlmm_data_collector.py

Follow the on-screen prompts to:

Select risk profile 🎲
Choose a timeframe ⏳
View recommendations 👀
Get optimized settings 🎯

🔄 Continuous Operation

Automated Updates: Data refreshes every 30 minutes in the background 🕒
Commands:

r: Get recommendations 🎯
t: Get recommended timeframe and ML predictions 🕰️
p: Manually trigger a data pull 🔄
q: Quit program 🚪



✅ Best Practices for Long-Running Scripts

Use a terminal multiplexer (e.g., tmux) for persistent sessions
