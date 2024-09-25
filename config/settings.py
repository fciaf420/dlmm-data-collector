# config/settings.py

BASE_URL = "https://api.geckoterminal.com/api/v2"
DATA_FILE = 'meme_coin_data.csv'
RECOMMENDATIONS_FILE = 'dlmm_recommendations.json'
COLLECTION_INTERVAL = 30  # minutes

# API rate limiting
API_CALLS_LIMIT = 20
API_CALLS_PERIOD = 60  # seconds
MAX_BACKOFF_TIME = 120  # Maximum backoff time in seconds
INITIAL_BACKOFF_TIME = 5  # Initial backoff time in seconds

# Sleep times for API calls (in seconds)
SLEEP_BETWEEN_OHLCV_CALLS = 5
SLEEP_BETWEEN_POOL_INFO_AND_OHLCV = 5
SLEEP_BETWEEN_POOLS = 10

# Risk factors
RISK_FACTORS = {
    'degen': 1.5,
    'moderate': 1.2,
    'conservative': 1.0
}

# Timeframes for analysis (in minutes)
TIMEFRAMES = [30, 60, 120, 180, 360, 720, 1440]

# Minimum thresholds
MIN_MARKET_CAP = 1000
MIN_VOLUME = 50000
MIN_TOKEN_AGE = 2  # hours