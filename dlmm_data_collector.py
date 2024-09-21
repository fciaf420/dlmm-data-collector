import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone, timedelta
import time
import logging
import os
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.table import Table
from rich.box import ROUNDED
from ratelimit import limits, sleep_and_retry
from requests.exceptions import RequestException
import random
import sys
import traceback
import msvcrt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from dtaidistance import dtw
from pykalman import KalmanFilter
from arch import arch_model
from json import JSONEncoder
import threading
from colorama import init, Fore, Back, Style

# Initialize colorama
init(autoreset=True)

# Set up rich console and logging
console = Console()
logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", 
                    handlers=[RichHandler(rich_tracebacks=True, console=console)])
log = logging.getLogger("rich")

BASE_URL = "https://api.geckoterminal.com/api/v2"

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class DataManager:
    def __init__(self):
        self.df = None
        self.last_update = None

    def load_data(self):
        if self.df is None or (datetime.now() - self.last_update) > timedelta(minutes=30):
            console.print("Loading data from CSV file...")
            self.df = pd.read_csv('meme_coin_data.csv')
            console.print(f"Loaded {len(self.df)} rows of data")
            
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
            
            invalid_timestamps = self.df['timestamp'].isnull()
            if invalid_timestamps.any():
                console.print(f"[yellow]Dropped {invalid_timestamps.sum()} rows with invalid timestamps[/yellow]")
                self.df = self.df.dropna(subset=['timestamp'])
            
            self.last_update = datetime.now()
            console.print("Data loading complete")

    def get_recent_data(self):
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        return self.df[self.df['timestamp'] > seven_days_ago]

data_manager = DataManager()

class GlobalState:
    def __init__(self):
        self.last_pull_time = datetime.now()

global_state = GlobalState()

@sleep_and_retry
@limits(calls=20, period=60)
def call_api_with_retry(url, params=None, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers={'accept': 'application/json'}, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            if response.status_code == 429 or (500 <= response.status_code < 600):
                sleep_time = min(30, (2 ** attempt) + random.uniform(0, 1))
                log.warning(f"Rate limit hit or server error. Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            elif response.status_code == 400:
                log.error(f"Bad request for URL: {url}")
                return None
            else:
                raise
    raise Exception("Max retries reached")

def get_trending_pools(network='solana', page=1):
    url = f"{BASE_URL}/networks/{network}/trending_pools"
    return call_api_with_retry(url, params={'page': page})

def get_pool_info(network, pool_address):
    url = f"{BASE_URL}/networks/{network}/pools/{pool_address}"
    return call_api_with_retry(url, params={'include': 'base_token'})

def get_ohlcv_data(network, pool_address):
    ohlcv_data = {}
    timeframes = [
        ('day', 1, 30), ('hour', 4, 42), ('hour', 1, 24)
    ]
    for timeframe, aggregate, limit in timeframes:
        url = f"{BASE_URL}/networks/{network}/pools/{pool_address}/ohlcv/{timeframe}"
        params = {'aggregate': aggregate, 'limit': limit, 'currency': 'usd', 'token': 'base'}
        try:
            response = call_api_with_retry(url, params)
            ohlcv_data[timeframe] = response
            time.sleep(2)
        except Exception as e:
            log.error(f"Failed to get OHLCV data for {pool_address} ({timeframe}): {str(e)}")
    return ohlcv_data

def calculate_ohlcv_metrics(ohlcv_data):
    metrics = {}
    for timeframe, data in ohlcv_data.items():
        if 'data' not in data or 'attributes' not in data['data'] or 'ohlcv_list' not in data['data']['attributes']:
            continue
        
        ohlcv_list = data['data']['attributes']['ohlcv_list']
        if not ohlcv_list:
            continue
        
        closes = [item[4] for item in ohlcv_list]
        volumes = [item[5] for item in ohlcv_list]
        
        avg_volume = np.mean(volumes)
        volatility = np.std(np.log(np.array(closes[1:]) / np.array(closes[:-1])))
        
        period = '30d' if timeframe == 'day' else '7d' if timeframe == 'hour' and len(ohlcv_list) > 24 else '24h'
        
        metrics.update({
            f'avg_{timeframe}_volume': avg_volume,
            f'{period}_volatility': volatility,
            f'price_{period}_ago': closes[-1] if closes else None,
            f'price_change_{period}': ((closes[0] / closes[-1]) - 1) * 100 if len(closes) > 1 else None
        })
    return metrics

def collect_data():
    console.print(Panel(Text("Starting data collection...", style="bold green")))
    all_data = []
    skipped_tokens = []

    try:
        trending_pools = get_trending_pools(network='solana', page=1)
        if 'data' not in trending_pools:
            console.print(f"[bold red]Unexpected response structure:[/bold red] {trending_pools}")
            return
    except Exception as e:
        console.print(f"[bold red]Error fetching trending pools:[/bold red] {str(e)}")
        return

    with console.status("[cyan]Processing pools...", spinner="dots") as status:
        for index, pool in enumerate(trending_pools['data'], 1):
            try:
                pool_attributes = pool['attributes']
                token_name = pool_attributes.get('name', 'Unknown')
                network = 'solana'
                pool_address = pool_attributes.get('address', 'Unknown')
                status.update(f"[cyan]Processing pool {index}: {token_name}")
                
                pool_info = get_pool_info(network, pool_address)
                ohlcv_data = get_ohlcv_data(network, pool_address)
                metrics = calculate_ohlcv_metrics(ohlcv_data)
                
                current_time = datetime.now(timezone.utc)
                pool_created_at = datetime.fromisoformat(pool_attributes.get('pool_created_at', current_time.isoformat()))
                pool_age_hours = (current_time - pool_created_at).total_seconds() / 3600

                pool_data = {
                    'timestamp': current_time.isoformat(),
                    'token_name': token_name,
                    'network': network,
                    'token_price': float(pool_attributes.get('base_token_price_usd', 0)),
                    'market_cap': float(pool_attributes.get('market_cap_usd', 0) or 0),
                    'fdv': float(pool_attributes.get('fdv_usd', 0) or 0),
                    'liquidity': float(pool_attributes.get('reserve_in_usd', 0) or 0),
                    'volume_5m': float(pool_attributes.get('volume_usd', {}).get('m5', 0) or 0),
                    'volume_1h': float(pool_attributes.get('volume_usd', {}).get('h1', 0) or 0),
                    'volume_6h': float(pool_attributes.get('volume_usd', {}).get('h6', 0) or 0),
                    'volume_24h': float(pool_attributes.get('volume_usd', {}).get('h24', 0) or 0),
                    'price_change_5m': float(pool_attributes.get('price_change_percentage', {}).get('m5', 0) or 0),
                    'price_change_1h': float(pool_attributes.get('price_change_percentage', {}).get('h1', 0) or 0),
                    'price_change_6h': float(pool_attributes.get('price_change_percentage', {}).get('h6', 0) or 0),
                    'price_change_24h': float(pool_attributes.get('price_change_percentage', {}).get('h24', 0) or 0),
                    'transactions_5m_buys': pool_attributes.get('transactions', {}).get('m5', {}).get('buys', 0),
                    'transactions_5m_sells': pool_attributes.get('transactions', {}).get('m5', {}).get('sells', 0),
                    'transactions_1h_buys': pool_attributes.get('transactions', {}).get('h1', {}).get('buys', 0),
                    'transactions_1h_sells': pool_attributes.get('transactions', {}).get('h1', {}).get('sells', 0),
                    'transactions_6h_buys': pool_attributes.get('transactions', {}).get('h6', {}).get('buys', 0),
                    'transactions_6h_sells': pool_attributes.get('transactions', {}).get('h6', {}).get('sells', 0),
                    'transactions_24h_buys': pool_attributes.get('transactions', {}).get('h24', {}).get('buys', 0),
                    'transactions_24h_sells': pool_attributes.get('transactions', {}).get('h24', {}).get('sells', 0),
                    'pool_created_at': pool_created_at.isoformat(),
                    'pool_age_hours': pool_age_hours,
                }
                
                pool_data['market_cap'] = pool_data['market_cap'] or pool_data['fdv']
                
                pool_data.update(metrics)

                all_data.append(pool_data)
                console.print(f"Processed {token_name} / {network.upper()}: Price ${pool_data['token_price']:.6f}, Market Cap/FDV ${pool_data['market_cap']:,.2f}")
                
                time.sleep(5)
            except KeyError as e:
                log.warning(f"Skipping token {token_name} due to missing data: {str(e)}")
                skipped_tokens.append(token_name)
                console.print(f"[yellow]Skipping token {token_name} due to missing data: {str(e)}[/yellow]")
            except Exception as e:
                log.error(f"Error processing pool {pool_attributes.get('address', 'Unknown')}: {str(e)}")
                console.print(f"[red]Error processing pool {pool_attributes.get('address', 'Unknown')}: {str(e)}[/red]")
                console.print(f"[yellow]Pool attributes: {json.dumps(pool_attributes, indent=2)}[/yellow]")

    df = pd.DataFrame(all_data)
    
    file_exists = os.path.isfile('meme_coin_data.csv')
    
    if not file_exists:
        df.to_csv('meme_coin_data.csv', mode='w', header=True, index=False)
        console.print(f"[green]Created new meme_coin_data.csv with {len(df)} records[/green]")
    else:
        df.to_csv('meme_coin_data.csv', mode='a', header=False, index=False)
        console.print(f"[green]Appended {len(df)} new records to meme_coin_data.csv[/green]")

    console.print(f"[yellow]Skipped {len(skipped_tokens)} tokens due to missing data: {', '.join(skipped_tokens)}[/yellow]")
    
    data_manager.load_data()

def multifactor_score(data):
    factors = {
        'price_momentum': data['price_change_1h'],
        'volume_trend': np.where((data['volume_24h'] > 0) & (data['volume_1h'] > 0), data['volume_1h'] / data['volume_24h'], 0),
        'market_cap_rank': data['effective_market_cap'].rank(ascending=False),
        'volatility': implement_garch(data, 60),  # 1-hour GARCH volatility
        'liquidity_ratio': np.where((data['effective_market_cap'] > 0) & (data['volume_24h'] > 0), data['volume_24h'] / data['effective_market_cap'], 0)
    }
    weights = {k: 1/len(factors) for k in factors.keys()}  # Equal weights to start
    return sum(factors[k] * weights[k] for k in factors)

def implement_garch(df, timeframe_minutes):
    price_col = 'price_change_1h'  # Use 1-hour price changes
    
    price_data = df[price_col].dropna()
    returns = np.log(price_data + 1).diff().dropna()
    
    # Remove inf and NaN values
    returns = returns[np.isfinite(returns)]
    
    if len(returns) < 100:
        console.print(f"[yellow]Warning: Insufficient data for GARCH modeling. Using standard deviation.[/yellow]")
        return returns.std() if len(returns) > 1 else 0
    
    try:
        model = arch_model(returns, vol='GARCH', p=1, q=1)
        results = model.fit(disp='off')
        forecast = results.forecast(horizon=1)
        volatility_forecast = np.sqrt(forecast.variance.values[-1, :])[0]
        
        console.print(f"[green]Successfully calculated GARCH volatility: {volatility_forecast:.4f}[/green]")
        return volatility_forecast
    except Exception as e:
        console.print(f"[yellow]Warning: GARCH modeling failed. Using standard deviation. Error: {str(e)}[/yellow]")
        return returns.std() if len(returns) > 1 else 0

def kalman_trend(prices):
    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1], 
                      initial_state_mean=0, initial_state_covariance=1, 
                      observation_covariance=1, transition_covariance=.01)
    return kf.filter(prices)[0].flatten()

def estimate_tail_risk(returns, threshold=0.05):
    try:
        valid_returns = returns[np.isfinite(returns)]
        
        if len(valid_returns) < 10:
            console.print("[yellow]Warning: Not enough valid data for tail risk estimation.[/yellow]")
            return None, None

        tail_returns = valid_returns[valid_returns < np.quantile(valid_returns, threshold)]
        
        if len(tail_returns) < 5:
            console.print("[yellow]Warning: Not enough tail data for estimation.[/yellow]")
            return None, None

        shape, _, scale = stats.genpareto.fit(tail_returns)
        return shape, scale
    except Exception as e:
        console.print(f"[yellow]Warning: Error in tail risk estimation - {str(e)}[/yellow]")
        return None, None

def prepare_data_for_timeframe(df, timeframe_minutes):
    base_columns = ['price_change_5m', 'price_change_1h', 'volume_1h', 'volume_24h', 'pool_age_hours', 'market_cap', 'fdv']
    
    price_change_col = f'price_change_{timeframe_minutes}m'
    
    required_columns = base_columns + [price_change_col]
    
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    
    if price_change_col not in df.columns:
        df.loc[:, price_change_col] = df['price_change_24h'] * (timeframe_minutes / 1440)
    
    df.loc[:, 'effective_market_cap'] = df['market_cap'].fillna(df['fdv'])
    df.loc[:, 'effective_market_cap'] = df['effective_market_cap'].fillna(df['liquidity'] * 2)
    df.loc[:, 'effective_market_cap'] = df['effective_market_cap'].clip(lower=1000)
    
    df.loc[:, 'volume_1h'] = df['volume_1h'].clip(lower=0)
    df.loc[:, 'volume_24h'] = df['volume_24h'].clip(lower=0)
    
    df_prepared = df[required_columns + ['effective_market_cap']].dropna(subset=['effective_market_cap'])
    
    df_prepared = df_prepared[df_prepared['effective_market_cap'] >= 1000]
    
    df_prepared.loc[:, 'pool_age_hours'] = df_prepared['pool_age_hours'].clip(lower=0)
    
    if df_prepared.empty:
        log.warning(f"No valid data for {timeframe_minutes} minute timeframe after removing NaN values")
        return None
    
    return df_prepared

def get_current_recommendations(timeframe_minutes, mode, df_recent):
    log.info(f"Generating recommendations for {timeframe_minutes} minute timeframe in {mode} mode")
    
    try:
        df_prepared = prepare_data_for_timeframe(df_recent, timeframe_minutes)
        if df_prepared is None or df_prepared.empty:
            raise ValueError(f"Insufficient data for {timeframe_minutes} minute timeframe")
        
        risk_factors = {'degen': 1.5, 'moderate': 1.2, 'conservative': 1.0}
        risk_factor = risk_factors[mode]
        log.info(f"Applied risk factor: {risk_factor}")
        
        recommendations = {}
        explanations = {}
        
        # Market Cap calculation (fixed)
        valid_market_caps = df_prepared['effective_market_cap'][df_prepared['effective_market_cap'] < 100000000]
        raw_mcap = valid_market_caps.quantile(0.25)
        adjusted_mcap = raw_mcap * risk_factor
        final_mcap = max(round(adjusted_mcap, -3), 1000000)
        final_mcap = min(final_mcap, 30000000)
        recommendations['Min Token Market Cap'] = final_mcap
        explanations['Min Token Market Cap'] = f"""
        Raw 25th percentile: ${raw_mcap:,.0f}
        Risk-adjusted value: ${adjusted_mcap:,.0f}
        Minimum threshold: $1,000,000
        Maximum threshold: $30,000,000
        Final value: ${final_mcap:,.0f}
        """
        
        # Price change calculations
        price_change_col = f'price_change_{timeframe_minutes}m'
        if price_change_col not in df_prepared.columns:
            df_prepared[price_change_col] = df_prepared['price_change_24h'] * (timeframe_minutes / 1440)
        
        # 5M Price Change
        raw_value_5m = df_prepared[price_change_col].quantile(0.10) * (5 / timeframe_minutes)
        adjusted_value_5m = round(raw_value_5m * risk_factor, 2)
        recommendations['Min Token 5M Price Change (%)'] = adjusted_value_5m
        explanations['Min Token 5M Price Change (%)'] = f"""
        Raw 10th percentile (scaled to 5M): {raw_value_5m:.2f}%
        Risk adjustment: {risk_factor:.2f}x
        Final value: {adjusted_value_5m:.2f}%
        """
        
        # 1H Price Change
        raw_value_1h = df_prepared[price_change_col].quantile(0.10) * (60 / timeframe_minutes)
        adjusted_value_1h = round(raw_value_1h * risk_factor, 2)
        recommendations['Min Token 1H Price Change (%)'] = adjusted_value_1h
        explanations['Min Token 1H Price Change (%)'] = f"""
        Raw 10th percentile (scaled to 1H): {raw_value_1h:.2f}%
        Risk adjustment: {risk_factor:.2f}x
        Final value: {adjusted_value_1h:.2f}%
        """
        
        # Max 1H Price Change
        max_change = df_prepared[price_change_col].quantile(0.95) * (60 / timeframe_minutes)
        adjusted_max_change = min(round(max_change * risk_factor, 2), 500)
        recommendations['Max Token 1H Price Change (%)'] = adjusted_max_change
        explanations['Max Token 1H Price Change (%)'] = f"""
        Raw 95th percentile (scaled to 1H): {max_change:.2f}%
        Risk adjustment: {risk_factor:.2f}x
        Capped at 500%
        Final value: {adjusted_max_change:.2f}%
        """
        
        # Volume calculations
        volume_col = 'volume_1h' if timeframe_minutes <= 60 else 'volume_24h'
        raw_volume = df_prepared[volume_col].quantile(0.25)
        
        # 1H Volume
        scaled_volume_1h = raw_volume * (60 / (60 if volume_col == 'volume_1h' else 1440)) * (timeframe_minutes / 60)
        adjusted_volume_1h = round(scaled_volume_1h * risk_factor, -3)
        final_volume_1h = max(adjusted_volume_1h, 50000)
        recommendations['Min Token 1H Volume'] = final_volume_1h
        explanations['Min Token 1H Volume'] = f"""
        Raw 25th percentile (scaled to 1H): ${scaled_volume_1h:,.0f}
        Risk-adjusted value: ${adjusted_volume_1h:,.0f}
        Minimum threshold: $50,000
        Final value: ${final_volume_1h:,.0f}
        """
        
        # 24H Volume
        scaled_volume_24h = raw_volume * (1440 / (60 if volume_col == 'volume_1h' else 1440)) * (timeframe_minutes / 1440)
        adjusted_volume_24h = round(scaled_volume_24h * risk_factor, -3)
        final_volume_24h = max(adjusted_volume_24h, 1000000)
        recommendations['Min Token 24H Volume'] = final_volume_24h
        explanations['Min Token 24H Volume'] = f"""
        Raw 25th percentile (scaled to 24H): ${scaled_volume_24h:,.0f}
        Risk-adjusted value: ${adjusted_volume_24h:,.0f}
        Minimum threshold: $1,000,000
        Final value: ${final_volume_24h:,.0f}
        """
        
        # Token Age
        raw_age = df_prepared['pool_age_hours'].quantile(0.10)
        adjusted_age = round(max(raw_age * risk_factor, 0.1), 1)
        final_age = max(adjusted_age, timeframe_minutes / 60)
        recommendations['Min Token Age (hrs)'] = final_age
        explanations['Min Token Age (hrs)'] = f"""
        Raw 10th percentile: {raw_age:.1f} hours
        Risk-adjusted value: {adjusted_age:.1f} hours
        Minimum threshold: {timeframe_minutes / 60:.1f} hours
        Final value: {final_age:.1f} hours
        """
        
        # Dynamic Stop Loss
        volatility = df_prepared[price_change_col].std() * np.sqrt(1440 / timeframe_minutes)
        base_stop_loss = volatility * 2
        adjusted_stop_loss = base_stop_loss * (1 / risk_factor)  # Inverse risk factor for stop loss
        final_stop_loss = max(min(adjusted_stop_loss, 20), 5)
        recommendations['Dynamic Stop Loss (%)'] = -round(final_stop_loss, 2)
        explanations['Dynamic Stop Loss (%)'] = f"""
        Volatility (24h equivalent): {volatility:.2f}%
        Base stop loss (2x volatility): {base_stop_loss:.2f}%
        Risk-adjusted stop loss: {adjusted_stop_loss:.2f}%
        Final stop loss (capped between 5% and 20%): {final_stop_loss:.2f}%
        """
        
        return recommendations, explanations
    
    except Exception as e:
        log.error(f"An error occurred while generating recommendations: {str(e)}")
        log.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def prepare_data_for_ml():
    data_manager.load_data()
    df = data_manager.df.copy()
    
    features = ['market_cap', 'volume_24h', 'price_change_24h', 'transactions_24h_buys', 'transactions_24h_sells', 'pool_age_hours']
    
    for col in features:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=features)
    
    # Log market cap distribution before filtering
    market_cap_percentiles_before = df['market_cap'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    console.print(f"[cyan]Market Cap Percentiles (Before Filtering):[/cyan]")
    for percentile, value in market_cap_percentiles_before.items():
        console.print(f"[cyan]{percentile*100}%: ${value:,.0f}[/cyan]")
    
    # Filter out high market cap tokens
    df = df[df['market_cap'] < 100000000]
    
    # Log market cap distribution after filtering
    market_cap_percentiles_after = df['market_cap'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    console.print(f"[cyan]Market Cap Percentiles (After Filtering):[/cyan]")
    for percentile, value in market_cap_percentiles_after.items():
        console.print(f"[cyan]{percentile*100}%: ${value:,.0f}[/cyan]")
    
    # Calculate volatility (using 24h price change as a simple proxy)
    df['volatility'] = df['price_change_24h'].rolling(window=7).std()
    
    # Calculate dynamic stop loss (between -10% and -50%)
    df['dynamic_stop_loss'] = df['volatility'].clip(lower=10, upper=50) * -1
    
    df['volume_ratio'] = df['volume_24h'] / df['volume_24h'].rolling(window=24).mean()
    df['best_timeframe'] = np.clip(df['volume_ratio'] * 720, 30, 1440)
    df['min_token_5m_price_change'] = df['price_change_24h'] / (24 * 12)
    df['min_token_1h_price_change'] = df['price_change_24h'] / 24
    df['max_token_1h_price_change'] = df['price_change_24h'].rolling(window=24).max()
    df['min_token_1h_volume'] = df['volume_24h'] / 24
    df['min_token_24h_volume'] = df['volume_24h']
    df['min_token_market_cap'] = df['market_cap'].clip(upper=50000000)
    df['min_token_age'] = df['pool_age_hours']
    
    targets = ['best_timeframe', 'min_token_5m_price_change', 'min_token_1h_price_change', 
               'max_token_1h_price_change', 'min_token_1h_volume', 'min_token_24h_volume', 
               'min_token_market_cap', 'min_token_age', 'dynamic_stop_loss']
    
    df = df.dropna(subset=targets)
    
    for col in features + targets:
        console.print(f"[cyan]{col} - min: {df[col].min():.2f}, max: {df[col].max():.2f}, mean: {df[col].mean():.2f}[/cyan]")
    
    X = df[features].values
    y = df[targets].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return (X_train, X_test, y_train, y_test), targets, scaler

def train_random_forest():
    try:
        (X_train, X_test, y_train, y_test), target_names, scaler = prepare_data_for_ml()
        
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
        r2 = r2_score(y_test, y_pred, multioutput='raw_values')
        
        console.print(f"[green]Random Forest Model Trained:[/green]")
        for i, target in enumerate(target_names):
            console.print(f"{target} - MSE: {mse[i]:.4f}, R-squared: {r2[i]:.4f}")
        
        return model, target_names, scaler
    except Exception as e:
        console.print(f"[bold red]Error in training Random Forest: {str(e)}[/bold red]")
        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        return None, None, None

def generate_ml_recommendations(model, target_names, scaler):
    try:
        recent_data = data_manager.get_recent_data().iloc[-1]
        
        features = ['market_cap', 'volume_24h', 'price_change_24h', 'transactions_24h_buys', 'transactions_24h_sells', 'pool_age_hours']
        X = recent_data[features].values.reshape(1, -1)
        
        X_scaled = scaler.transform(X)
        
        predictions = model.predict(X_scaled)[0]
        
        recommendations = {target: max(0, pred) if 'stop_loss' not in target else pred for target, pred in zip(target_names, predictions)}
        
        # Ensure stop loss is within -10% to -50% range
        if 'dynamic_stop_loss' in recommendations:
            recommendations['dynamic_stop_loss'] = max(-50, min(-10, recommendations['dynamic_stop_loss']))
        
        return recommendations
    except Exception as e:
        console.print(f"[bold red]Error in generating ML recommendations: {str(e)}[/bold red]")
        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        return None

def print_logo():
    logo = r"""
__| |____________________________________________________| |__
__   ____________________________________________________   __
  | |                                                    | |  
  | |____ ____ _       ___  ____ ____ ____ ___  ____ ____| |  
  | |[__  |  | |       |  \ |___ |    |  | |  \ |___ |__/| |  
  | |___] |__| |___    |__/ |___ |___ |__| |__/ |___ |  \| |  
__| |____________________________________________________| |__
__   ____________________________________________________   __
  | |                                                    | |  
"""
    print(logo)

def print_cyberpunk_header():
    print_logo()
    print(f"\n{Fore.GREEN}{'=' * 80}")
    print(f"{Fore.YELLOW}                         DLMM Settings Script v1.0")
    print(f"{Fore.CYAN}                     made with <3 by @mininghelium")
    print(f"{Fore.GREEN}{'=' * 80}\n")

def print_menu():
    print(f"{Fore.CYAN}[R] {Fore.MAGENTA}Get Recommendations")
    print(f"{Fore.CYAN}[T] {Fore.MAGENTA}Get Timeframe")
    print(f"{Fore.CYAN}[P] {Fore.MAGENTA}Pull Data")
    print(f"{Fore.CYAN}[Q] {Fore.MAGENTA}Quit")
    print(f"\n{Fore.GREEN}{'=' * 80}")

def r_command():
    print(f"\n{Fore.YELLOW}[SYS] Executing Recommendation Algorithm...")
    data_manager.load_data()
    df_recent = data_manager.get_recent_data()
    timeframe_minutes = get_user_input("Select your timeframe for LP pool:", 
                                       {'1': 30, '2': 60, '3': 120, '4': 180, '5': 360, '6': 720, '7': 1440})
    mode = get_user_input("Choose your preferred mode:", 
                          {'1': 'degen', '2': 'moderate', '3': 'conservative'})
    try:
        recommendations, explanations = get_current_recommendations(timeframe_minutes, mode, df_recent)
        if recommendations and explanations:
            display_recommendations(recommendations, explanations)
            save_recommendations(recommendations, explanations)
    except Exception as e:
        console.print(f"[red]An error occurred: {str(e)}[/red]")
        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
    print(f"{Fore.GREEN}[SYS] Recommendations generated successfully!")

def t_command():
    print(f"\n{Fore.YELLOW}[SYS] Analyzing Optimal Timeframes...")
    console.print("[bold cyan]Training Random Forest model...[/bold cyan]")
    ml_model, target_names, scaler = train_random_forest()
    if ml_model is not None:
        recommendations = generate_ml_recommendations(ml_model, target_names, scaler)
        if recommendations:
            timeframe_options = [30, 60, 120, 180, 360, 720, 1440]
            best_timeframe = recommendations['best_timeframe']
            closest_timeframe = min(timeframe_options, key=lambda x: abs(x - best_timeframe))
            
            console.print("\n[bold green]Timeframe Analysis Results:[/bold green]")
            console.print(f"[cyan]Recommended Timeframe:[/cyan] {closest_timeframe} minutes")
            console.print(f"[yellow](ML suggestion: {best_timeframe:.0f} minutes)[/yellow]")
            
            console.print("\n[bold green]Predicted Metrics for Recommended Timeframe:[/bold green]")
            metrics_to_show = [
                ('Min Token 5M Price Change (%)', 'min_token_5m_price_change', '%'),
                ('Min Token 1H Price Change (%)', 'min_token_1h_price_change', '%'),
                ('Max Token 1H Price Change (%)', 'max_token_1h_price_change', '%'),
                ('Min Token 1H Volume', 'min_token_1h_volume', '$'),
                ('Min Token 24H Volume', 'min_token_24h_volume', '$'),
                ('Min Token Effective Market Cap', 'min_token_market_cap', '$'),
                ('Min Token Age (hrs)', 'min_token_age', 'hours'),
                ('Dynamic Stop Loss (%)', 'dynamic_stop_loss', '%')
            ]
            
            for label, metric, unit in metrics_to_show:
                if metric in recommendations:
                    value = recommendations[metric]
                    if unit == '$':
                        formatted_value = f"${value:,.0f}"
                    elif unit == 'hours':
                        formatted_value = f"{value:.1f} hours"
                    elif unit == '%':
                        formatted_value = f"{value:.2f}%"
                    else:
                        formatted_value = f"{value:.4f}"
                    console.print(f"[cyan]{label}:[/cyan] {formatted_value}")
                else:
                    console.print(f"[cyan]{label}:[/cyan] Not available")
            
            console.print("\n[bold yellow]Note:[/bold yellow] These are ML-based predictions for the recommended timeframe.")
            console.print("[bold yellow]The Dynamic Stop Loss is based on recent market volatility and should be used as a guideline.[/bold yellow]")
            console.print("[bold yellow]Use the 'R' command with your chosen timeframe for detailed entry criteria and explanations.[/bold yellow]")
    
    print(f"{Fore.GREEN}[SYS] Timeframe analysis complete!")

def pull_data():
    global global_state
    print(f"\n{Fore.YELLOW}[SYS] Initiating Data Pull Sequence...")
    try:
        collect_data()
        global_state.last_pull_time = datetime.now()
        print(f"{Fore.GREEN}[SYS] Data successfully extracted and processed!")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Data pull failed: {str(e)}")
        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
    print(f"{Fore.CYAN}[INFO] Last pull time: {global_state.last_pull_time.strftime('%Y-%m-%d %H:%M:%S')}")

def background_data_pull():
    global global_state
    while True:
        try:
            current_time = datetime.now()
            time_since_last_pull = current_time - global_state.last_pull_time
            
            if time_since_last_pull >= timedelta(minutes=30):
                print(f"\n{Fore.MAGENTA}[BACKGROUND] {Fore.YELLOW}Executing scheduled data pull...")
                pull_data()
            
            time.sleep(60)  # Check every minute
        except Exception as e:
            print(f"{Fore.RED}[ERROR] Background data pull error: {str(e)}")
            console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
            time.sleep(60)  # Wait a minute before trying again

def display_last_pull_time():
    global global_state
    time_since_last_pull = datetime.now() - global_state.last_pull_time
    minutes, seconds = divmod(time_since_last_pull.seconds, 60)
    print(f"{Fore.CYAN}[INFO] Last data pull: {minutes} minutes and {seconds} seconds ago")
    print(f"{Fore.CYAN}[INFO] Next scheduled pull in: {max(0, 30 - minutes)} minutes")

def get_user_input(prompt, options):
    while True:
        console.print(f"\n[bold cyan]{prompt}[/bold cyan]")
        for key, value in options.items():
            console.print(f"{key}. {value}")
        
        choice = console.input("Enter your choice: ")
        if choice in options:
            return options[choice]
        else:
            console.print("[red]Invalid choice. Please try again.[/red]")

def display_recommendations(recommendations, explanations):
    console.print(f"\n[bold green]Optimized DLMM Entry Criteria:[/bold green]")
    for setting, value in recommendations.items():
        if 'Volume' in setting or 'Market Cap' in setting:
            formatted_value = f"${value:,.0f}"
        elif 'Age' in setting:
            formatted_value = f"{value:.1f} hours"
        elif 'Tail Risk' in setting or 'Pattern Index' in setting:
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = f"{value:.2f}%"
        console.print(f"[cyan]{setting}:[/cyan] {formatted_value}")
        console.print(f"[yellow]{explanations[setting]}[/yellow]\n")

def save_recommendations(recommendations, explanations):
    if isinstance(recommendations, dict) and isinstance(explanations, dict):
        data_to_save = {
            "timestamp": datetime.now().isoformat(),
            "recommendations": recommendations,
            "explanations": explanations
        }
        
        filename = 'dlmm_recommendations.json'
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=4, cls=NumpyEncoder)
        console.print(f"\n[green]Recommendations and explanations saved to {filename}[/green]")
    else:
        console.print("\n[yellow]No recommendations or explanations to save.[/yellow]")

def main():
    global global_state
    
    # Start background data pull
    bg_thread = threading.Thread(target=background_data_pull, daemon=True)
    bg_thread.start()

    while True:
        try:
            print_cyberpunk_header()
            print_menu()
            display_last_pull_time()
            choice = input(f"{Fore.CYAN}[USER] Enter your choice: {Fore.WHITE}").lower()

            if choice == 'r':
                r_command()
            elif choice == 't':
                t_command()
            elif choice == 'p':
                pull_data()
            elif choice == 'q':
                print(f"\n{Fore.YELLOW}[SYS] Exiting SOL DECODER. Stay rad in cyberspace!")
                break
            else:
                print(f"\n{Fore.RED}[ERROR] Invalid choice. Please try again.")
        except Exception as e:
            console.print(f"[bold red]An error occurred: {str(e)}[/bold red]")
            console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        
        input(f"\n{Fore.GREEN}[SYS] Press Enter to continue...")

if __name__ == "__main__":
    console.print(Panel(Text("DLMM Data Collector", style="bold magenta")))
    log.info("Script started. Data collection will begin shortly.")
    try:
        main()
    except KeyboardInterrupt:
        console.print(Panel(Text("Script terminated by user.", style="bold red")))
        log.info("Script terminated by user.")
    except Exception as e:
        console.print(Panel(Text(f"An unexpected error occurred: {str(e)}", style="bold red")))
        log.error(f"An unexpected error occurred: {str(e)}")
        log.error(traceback.format_exc())
