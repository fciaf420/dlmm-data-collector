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

## 🚀 Getting Started

### Prerequisites

1. **Python Installation**
   - Ensure you have Python 3.8 or newer installed on your system.
   - To check your Python version, open a terminal and run:
     ```
     python3 --version
     ```
   - If Python is not installed or you need to update, download it from [python.org](https://www.python.org/downloads/).

2. **Git Installation** (Optional, for cloning the repository)
   - Install Git from [git-scm.com](https://git-scm.com/downloads) if you haven't already.

### Installation

1. **Clone the Repository** (If you're using Git)
git clone https://github.com/yourusername/dlmm-data-collector.git
cd dlmm-data-collector
CopyOr download and extract the ZIP file from the GitHub repository.

2. **Set Up a Virtual Environment** (Recommended)
- Create a virtual environment:
  ```
  python3 -m venv venv
  ```
- Activate the virtual environment:
  - On macOS and Linux:
    ```
    source venv/bin/activate
    ```
  - On Windows:
    ```
    venv\Scripts\activate
    ```

3. **Install Required Packages**
- Install all required packages using pip:
  ```
  pip install -r requirements.txt
  ```

### Pip Requirements

Here's a list of the main Python packages required for this project:
requests==2.26.0
pandas==1.3.3
numpy==1.21.2
scikit-learn==0.24.2
arch==5.0.1
pykalman==0.9.5
dtaidistance==2.3.6
rich==10.12.0
colorama==0.4.4
ratelimit==2.2.1
Copy
You can install these packages by running:
pip install -r requirements.txt
Copy
### Cross-Platform Compatibility

The script was initially designed for Windows systems and contains some Windows-specific components. To run it on Linux or macOS, you'll need to make the following modifications:

1. **Remove Windows-specific imports:**
   Open `dlmm_data_collector.py` in a text editor and remove or comment out the following line:
   ```python
   # import msvcrt  # Comment out or remove this line

Replace msvcrt.kbhit() function:
Find any occurrences of msvcrt.kbhit() in the script and replace them with a cross-platform alternative. For example:
pythonCopyimport sys
import select

def kbhit():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

# Then replace msvcrt.kbhit() with kbhit()

Adjust any other Windows-specific code:
Review the script for any other Windows-specific functions or libraries and replace them with cross-platform alternatives as needed.

Running the Script

Execute the Script
Copypython3 dlmm_data_collector.py

Navigate the Interface

Follow the on-screen prompts to:

Select risk profile (Degen, Moderate, Conservative)
Choose a timeframe
View recommendations
Get optimized settings




Available Commands

r: Get recommendations
t: Get recommended timeframe and ML predictions
p: Manually trigger a data pull
q: Quit program



Troubleshooting

If you encounter platform-specific errors, ensure you've made the necessary modifications as described in the Cross-Platform Compatibility section.
On Linux or macOS, if you get a "Permission denied" error, make the script executable:
Copychmod +x dlmm_data_collector.py

If you encounter any package-related errors, ensure all dependencies are correctly installed:
Copypip install -r requirements.txt --upgrade

For API-related issues, check your internet connection and verify your API key.
Consult the error messages in the console for specific issues and their potential resolutions.

Continuous Operation
For long-term operation:

Use a terminal multiplexer like tmux or screen.
Run the script in the background:
Copynohup python3 dlmm_data_collector.py &


Updating
To get the latest version of the script:

If you used Git, pull the latest changes:
Copygit pull origin main

Reinstall requirements in case of updates:
Copypip install -r requirements.txt --upgrade


📈 Monitoring and Maintenance

Regularly check the console output for warnings or errors.
Review the generated recommendations and data pulls periodically to ensure accuracy.
Keep your Python environment and dependencies up to date.

For any persistent issues or feature requests, please open an issue on the GitHub repository.
🧩 Functional Components

📡 Data Collection

API interaction with GeckoTerminal
Collects prices, market caps, volumes, price changes
Robust error handling & rate limiting


🔍 Data Processing & Analysis

Data cleaning and preprocessing
Calculate key metrics (volatility, volume ratios, price changes)
Machine learning-based feature importance analysis


💡 Recommendation Engine

ML-powered timeframe selection
Provides risk-adjusted recommendations (Degen, Moderate, Conservative)
Calculates dynamic stop losses based on market volatility


🖥️ User Interface

Interactive console with real-time data updates
On-demand recommendations and data pulls



📐 Mathematical and ML Models

Volatility Calculation: GARCH Model
Multifactor Scoring: Combines price momentum, volume trends, market cap, and liquidity
Random Forest Regression: For predicting optimal trading parameters
Dynamic Stop Loss: Adaptive based on recent market volatility

⚙️ Advanced Features

Background Data Updates: Automatically refreshes data every 30 minutes
ML-based Timeframe Optimization: Evaluates and recommends optimal timeframes
Enhanced Market Cap Analysis: Improved filtering and distribution analysis
Consistency Checks: Verifies recommendations against explanations

🛠 Contributing
If you've successfully modified the script to work on Linux or macOS, consider submitting a pull request with your changes to help improve cross-platform compatibility for all users.
⚠️ Disclaimer
This script is for informational purposes only. Always conduct your own research and consider your risk tolerance before making investment decisions. The creators of this script are not responsible for any financial losses incurred.
