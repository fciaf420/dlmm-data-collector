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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.align import Align
from rich.table import Table
from rich.box import ROUNDED
from ratelimit import limits, sleep_and_retry
from requests.exceptions import RequestException
import random
import sys
import csv
from multiprocessing import Pool
from arch import arch_model
import select
import platform
import traceback
import msvcrt

# Set up rich console
console = Console()

# Set up logging with rich
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)

log = logging.getLogger("rich")

BASE_URL = "https://api.geckoterminal.com/api/v2"

def exponential_backoff(attempt):
    return min(30, (2 ** attempt) + random.uniform(0, 1))

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
                sleep_time = exponential_backoff(attempt)
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
        ('day', 1, 30),  # 30 days of daily data
        ('hour', 4, 42),  # 7 days of 4-hourly data
        ('hour', 1, 24)  # 24 hours of hourly data
    ]
    for timeframe, aggregate, limit in timeframes:
        url = f"{BASE_URL}/networks/{network}/pools/{pool_address}/ohlcv/{timeframe}"
        params = {
            'aggregate': aggregate,
            'limit': limit,
            'currency': 'usd',
            'token': 'base'
        }
        try:
            response = call_api_with_retry(url, params)
            ohlcv_data[timeframe] = response
            time.sleep(2)  # Add a 2-second delay between OHLCV requests
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
        
        if timeframe == 'day':
            period = '30d'
        elif timeframe == 'hour' and len(ohlcv_list) > 24:
            period = '7d'
        else:
            period = '24h'
        
        metrics.update({
            f'avg_{timeframe}_volume': avg_volume,
            f'{period}_volatility': volatility,
            f'price_{period}_ago': closes[-1] if closes else None,
            f'price_change_{period}': ((closes[0] / closes[-1]) - 1) * 100 if len(closes) > 1 else None
        })
    return metrics

def get_volume_column(timeframe_minutes):
    timeframe_minutes = int(timeframe_minutes)
    if timeframe_minutes <= 5:
        return 'volume_5m'
    elif timeframe_minutes <= 60:
        return 'volume_1h'
    elif timeframe_minutes <= 360:
        return 'volume_6h'
    else:
        return 'volume_24h'

def get_price_change_column(timeframe_minutes):
    timeframe_minutes = int(timeframe_minutes)
    if timeframe_minutes <= 5:
        return 'price_change_5m'
    elif timeframe_minutes <= 60:
        return 'price_change_1h'
    elif timeframe_minutes <= 360:
        return 'price_change_6h'
    else:
        return 'price_change_24h'

def calculate_metrics(timeframe, data):
    price_col = get_price_change_column(timeframe)
    volume_col = get_volume_column(timeframe)
    
    if isinstance(data, pd.DataFrame):
        avg_price_change = data[price_col].mean()
        avg_volume = data[volume_col].mean()
        volatility = data[price_col].std()
    else:  # Single row
        avg_price_change = data[price_col]
        avg_volume = data[volume_col]
        volatility = 0  # Cannot calculate volatility for a single point
    
    # Avoid division by zero
    if volatility != 0:
        sharpe_ratio = avg_price_change / volatility
    else:
        sharpe_ratio = 0
    
    volume_adjusted_change = avg_price_change * (avg_volume ** 0.5)
    score = (sharpe_ratio * 0.4) + (volume_adjusted_change * 0.6)
    
    return {'timeframe': timeframe, 'score': score, 'sharpe_ratio': sharpe_ratio, 'avg_price_change': avg_price_change, 'volatility': volatility, 'avg_volume': avg_volume}

class DataManager:
    def __init__(self):
        self.df = None
        self.last_update = None

    def load_data(self):
        if self.df is None or (datetime.now() - self.last_update) > timedelta(minutes=30):
            console.print("Loading data from CSV file...")
            self.df = pd.read_csv('meme_coin_data.csv', on_bad_lines='skip')
            console.print(f"Loaded {len(self.df)} rows of data")
            
            # Convert timestamp column to datetime, handling errors
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
            
            # Drop rows with invalid timestamps
            invalid_timestamps = self.df['timestamp'].isnull()
            if invalid_timestamps.any():
                console.print(f"[yellow]Dropped {invalid_timestamps.sum()} rows with invalid timestamps[/yellow]")
                self.df = self.df.dropna(subset=['timestamp'])
            
            self.last_update = datetime.now()
            console.print("Data loading complete")

    def get_recent_data(self):
        # Return data from the last 7 days
        seven_days_ago = datetime.now() - timedelta(days=7)
        return self.df[self.df['timestamp'] > seven_days_ago]

data_manager = DataManager()

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
                
                # Get detailed pool info
                pool_info = get_pool_info(network, pool_address)
                
                # Get OHLCV data
                ohlcv_data = get_ohlcv_data(network, pool_address)
                
                # Calculate additional metrics
                metrics = calculate_ohlcv_metrics(ohlcv_data)
                
                pool_data = {
                    'timestamp': datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
                    'token_name': token_name,
                    'network': network,
                    'token_price': float(pool_attributes.get('base_token_price_usd', 0)),
                    'market_cap': float(pool_attributes.get('market_cap_usd', 0) or 0),
                    'fdv_usd': float(pool_attributes.get('fdv_usd', 0) or 0),
                    'volume_5m': float(pool_attributes.get('volume_usd', {}).get('m5', 0) or 0),
                    'volume_1h': float(pool_attributes.get('volume_usd', {}).get('h1', 0) or 0),
                    'volume_6h': float(pool_attributes.get('volume_usd', {}).get('h6', 0) or 0),
                    'volume_24h': float(pool_attributes.get('volume_usd', {}).get('h24', 0) or 0),
                    'price_change_5m': float(pool_attributes.get('price_change_percentage', {}).get('m5', 0) or 0),
                    'price_change_1h': float(pool_attributes.get('price_change_percentage', {}).get('h1', 0) or 0),
                    'price_change_6h': float(pool_attributes.get('price_change_percentage', {}).get('h6', 0) or 0),
                    'price_change_24h': float(pool_attributes.get('price_change_percentage', {}).get('h24', 0) or 0),
                    'pool_created_at': pool_attributes.get('pool_created_at'),
                    'pool_age_hours': (datetime.now(timezone.utc) - datetime.fromisoformat(pool_attributes.get('pool_created_at', datetime.now(timezone.utc).isoformat()))).total_seconds() / 3600,
                    'reserve_in_usd': float(pool_attributes.get('reserve_in_usd', 0) or 0),
                    'transactions_5m_buys': pool_attributes.get('transactions', {}).get('m5', {}).get('buys', 0),
                    'transactions_5m_sells': pool_attributes.get('transactions', {}).get('m5', {}).get('sells', 0),
                    'transactions_1h_buys': pool_attributes.get('transactions', {}).get('h1', {}).get('buys', 0),
                    'transactions_1h_sells': pool_attributes.get('transactions', {}).get('h1', {}).get('sells', 0),
                    'transactions_24h_buys': pool_attributes.get('transactions', {}).get('h24', {}).get('buys', 0),
                    'transactions_24h_sells': pool_attributes.get('transactions', {}).get('h24', {}).get('sells', 0),
                }
                
                # Add additional metrics from OHLCV data
                pool_data.update(metrics)
                
                all_data.append(pool_data)
                console.print(f"[green]Processed {token_name}: Price ${pool_data['token_price']:.6f}, Market Cap ${pool_data['market_cap']:,.2f}[/green]")
                
                time.sleep(5)  # Add a 5-second delay between processing each pool
            except KeyError as e:
                log.warning(f"Skipping token {token_name} due to missing data: {str(e)}")
                skipped_tokens.append(token_name)
                console.print(f"[yellow]Skipping token {token_name} due to missing data: {str(e)}[/yellow]")
            except Exception as e:
                log.error(f"Error processing pool {pool_attributes.get('address', 'Unknown')}: {str(e)}")
                console.print(f"[red]Error processing pool {pool_attributes.get('address', 'Unknown')}: {str(e)}[/red]")
                console.print(f"[yellow]Pool attributes: {json.dumps(pool_attributes, indent=2)}[/yellow]")

    # After collecting all data
    df = pd.DataFrame(all_data)
    
    # Check if the CSV file already exists
    file_exists = os.path.isfile('meme_coin_data.csv')
    
    # If the file doesn't exist, write with header. If it exists, append without header
    if not file_exists:
        df.to_csv('meme_coin_data.csv', mode='w', header=True, index=False)
        console.print(f"[green]Created new meme_coin_data.csv with {len(df)} records[/green]")
    else:
        # Read the existing CSV to get the current columns
        existing_df = pd.read_csv('meme_coin_data.csv', nrows=0)
        existing_columns = set(existing_df.columns)
        
        # Check if there are any new columns in the current data
        new_columns = set(df.columns) - existing_columns
        
        if new_columns:
            # If there are new columns, we need to rewrite the entire file
            old_df = pd.read_csv('meme_coin_data.csv')
            combined_df = pd.concat([old_df, df], ignore_index=True)
            combined_df.to_csv('meme_coin_data.csv', mode='w', header=True, index=False)
            console.print(f"[green]Updated meme_coin_data.csv with new columns and added {len(df)} new records[/green]")
        else:
            # If no new columns, simply append the new data
            df.to_csv('meme_coin_data.csv', mode='a', header=False, index=False)
            console.print(f"[green]Appended {len(df)} new records to meme_coin_data.csv[/green]")

    console.print(f"[yellow]Skipped {len(skipped_tokens)} tokens due to missing data: {', '.join(skipped_tokens)}[/yellow]")

def get_timeframe_input():
    timeframe_options = {
        '1': 30, '2': 60, '3': 120, '4': 180, '5': 360, '6': 720, '7': 1440
    }
    console.print("\n[bold cyan]Select your timeframe for LP pool:[/bold cyan]")
    console.print("1. 30 minutes")
    console.print("2. 60 minutes")
    console.print("3. 2 hours")
    console.print("4. 3 hours")
    console.print("5. 6 hours")
    console.print("6. 12 hours")
    console.print("7. 24 hours")
    
    while True:
        choice = console.input("Enter your choice (1-7): ")
        if choice in timeframe_options:
            return timeframe_options[choice]
        else:
            console.print("[red]Invalid choice. Please enter a number between 1 and 7.[/red]")

def get_risk_mode_input():
    console.print("\n[bold cyan]Choose your preferred mode:[/bold cyan]")
    console.print("1. Degen (Higher risk, more opportunities)")
    console.print("2. Moderate (Balanced risk and opportunities)")
    console.print("3. Conservative (Lower risk, fewer opportunities)")
    
    while True:
        mode_choice = console.input("Enter your choice (1-3): ")
        if mode_choice in ['1', '2', '3']:
            return ['degen', 'moderate', 'conservative'][int(mode_choice) - 1]
        else:
            console.print("[red]Invalid choice. Please enter a number between 1 and 3.[/red]")

def calculate_constrained_stop_loss(token_data, mode):
    price_changes = [
        abs(token_data.get('price_change_5m', 0)),
        abs(token_data.get('price_change_1h', 0)),
        abs(token_data.get('price_change_6h', 0)),
        abs(token_data.get('price_change_24h', 0))
    ]
    max_price_change = max(price_changes)
    
    # Adjust risk factors to spread across the 10% to 50% range
    risk_factors = {'degen': 0.8, 'moderate': 1.0, 'conservative': 1.2}
    risk_factor = risk_factors[mode]
    
    # Scale the max_price_change to fit within our desired range
    scaled_change = (max_price_change / 100) * 40  # Scale to 0-40% range
    base_stop_loss = 10 + scaled_change  # Shift to 10-50% range
    
    # Apply risk factor and constrain to 10-50% range
    stop_loss_percentage = max(10, min(base_stop_loss * risk_factor, 50))
    
    return -stop_loss_percentage  # Return as a negative percentage

def explain_recommendations(recommendations, mode, df_recent):
    explanations = {}
    risk_factors = {'degen': 1.2, 'moderate': 1.0, 'conservative': 0.8}
    risk_factor = risk_factors[mode]

    for criterion, value in recommendations.items():
        if criterion == 'Min Token 5 Min Price Change (%)':
            explanations[criterion] = f"""
            Mathematical basis: 10th percentile of 5-minute price changes, adjusted for risk.
            Raw 10th percentile: {df_recent['price_change_5m'].quantile(0.1):.2f}%
            Risk adjustment: {value / (df_recent['price_change_5m'].quantile(0.1) * risk_factor):.2f}x
            This threshold helps filter out tokens in rapid short-term decline.
            """
        elif criterion == 'Min Token 1 HR Price Change (%)':
            explanations[criterion] = f"""
            Mathematical basis: 10th percentile of 1-hour price changes, adjusted for risk.
            Raw 10th percentile: {df_recent['price_change_1h'].quantile(0.1):.2f}%
            Risk adjustment: {value / (df_recent['price_change_1h'].quantile(0.1) * risk_factor):.2f}x
            This threshold identifies potential entry points while avoiding significant short-term declines.
            """
        elif criterion == 'Max Token 1 HR Price Change (%)':
            explanations[criterion] = f"""
            Mathematical basis: 90th percentile of 1-hour price changes, adjusted for risk.
            Raw 90th percentile: {df_recent['price_change_1h'].quantile(0.9):.2f}%
            Risk adjustment: {value / (df_recent['price_change_1h'].quantile(0.9) * risk_factor):.2f}x
            This upper limit helps avoid entering during potential unsustainable price spikes.
            """
        elif criterion == 'Min Token 1 HR Volume':
            explanations[criterion] = f"""
            Mathematical basis: 25th percentile of 1-hour volumes, with a minimum threshold.
            Raw 25th percentile: ${df_recent['volume_1h'].quantile(0.25):,.0f}
            Adjusted value: max(25th percentile * risk factor, $50,000)
            This ensures sufficient short-term liquidity for safer entry and exit.
            """
        elif criterion == 'Min Token 24 Hrs Volume':
            explanations[criterion] = f"""
            Mathematical basis: 25th percentile of 24-hour volumes, with a minimum threshold.
            Raw 25th percentile: ${df_recent['volume_24h'].quantile(0.25):,.0f}
            Adjusted value: max(25th percentile * risk factor, $975,000)
            This ensures sustained market interest and overall liquidity.
            """
        elif criterion == 'Min Token Age (hrs)':
            explanations[criterion] = f"""
            Mathematical basis: 10th percentile of token ages, adjusted for risk.
            Raw 10th percentile: {df_recent['pool_age_hours'].quantile(0.1):.1f} hours
            Risk adjustment: {value / (df_recent['pool_age_hours'].quantile(0.1) * risk_factor):.2f}x
            This helps avoid very new, potentially unstable tokens.
            """
        elif criterion == 'Min Token Market Cap':
            valid_market_caps = df_recent['market_cap'][(df_recent['market_cap'] > 100000) & (df_recent['market_cap'] < 1e9)]
            market_cap_10th_percentile = valid_market_caps.quantile(0.10)
            explanations[criterion] = f"""
            Mathematical basis: 10th percentile of market caps between $100K and $1B, adjusted for risk.
            Raw 10th percentile: ${market_cap_10th_percentile:,.0f}
            Adjusted value: max(10th percentile * risk factor, $200,000)
            This balances opportunity with stability across the strategy's price bins.
            """
        elif criterion == 'Stop Loss (%)':
            explanations[criterion] = f"""
            Mathematical basis: Average of individual token stop losses based on historical volatility.
            Calculation: For each token, max(10%, min((max_price_change * 0.4 + 10) * risk factor, 50%))
            This adaptive stop loss balances protection against excessive losses with room for normal price fluctuations.
            """

    return explanations

def get_current_recommendations(timeframe_minutes, mode, df_recent):
    console.print(Panel(Text(f"Generating optimized recommendations for DLMM entry criteria (Timeframe: {timeframe_minutes} minutes)...", style="bold blue")))
    
    try:
        # Validate data
        required_columns = ['price_change_5m', 'price_change_1h', 'price_change_6h', 'price_change_24h', 
                            'volume_5m', 'volume_1h', 'volume_6h', 'volume_24h', 'pool_age_hours', 'market_cap']
        missing_columns = [col for col in required_columns if col not in df_recent.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Remove rows with NaN values in required columns
        df_recent = df_recent.dropna(subset=required_columns)
        
        if df_recent.empty:
            raise ValueError("No valid data remaining after removing NaN values")
        
        # Sort by timestamp and keep only the most recent entry for each token
        df_recent = df_recent.sort_values('timestamp', ascending=False).groupby('token_name').first().reset_index()
        
        # Calculate scores for the entire dataframe
        metrics = calculate_metrics(timeframe_minutes, df_recent)
        df_recent['score'] = metrics['score']
        
        # Sort tokens by score and get the top performers (adjust the number as needed)
        top_performers = df_recent.sort_values('score', ascending=False).head(30)
        
        # Calculate recommendations based on top performers
        recommendations = {
            'Min Token 5 Min Price Change (%)': top_performers['price_change_5m'].quantile(0.10),
            'Min Token 1 HR Price Change (%)': top_performers['price_change_1h'].quantile(0.10),
            'Max Token 1 HR Price Change (%)': top_performers['price_change_1h'].quantile(0.90),
            'Min Token 1 HR Volume': top_performers['volume_1h'].quantile(0.25),
            'Min Token 24 Hrs Volume': top_performers['volume_24h'].quantile(0.25),
            'Min Token Age (hrs)': top_performers['pool_age_hours'].quantile(0.10)
        }

        # Adjust Market Cap calculation
        valid_market_caps = top_performers['market_cap'][(top_performers['market_cap'] > 100000) & (top_performers['market_cap'] < 1e9)]
        market_cap_10th_percentile = valid_market_caps.quantile(0.10)
        
        risk_factors = {'degen': 1.2, 'moderate': 1.0, 'conservative': 0.8}
        risk_factor = risk_factors[mode]
        
        min_market_cap = max(market_cap_10th_percentile * risk_factor, 200000)
        
        recommendations['Min Token Market Cap'] = round(min_market_cap, -3)  # Round to nearest thousand

        # Adjust based on risk mode
        for key in recommendations:
            if key != 'Min Token Market Cap':  # Skip market cap as it's already adjusted
                recommendations[key] *= risk_factor

        # Round values for better readability
        recommendations['Min Token 5 Min Price Change (%)'] = max(round(recommendations['Min Token 5 Min Price Change (%)'], 2), -10)
        recommendations['Min Token 1 HR Price Change (%)'] = max(round(recommendations['Min Token 1 HR Price Change (%)'], 2), -30)
        recommendations['Max Token 1 HR Price Change (%)'] = min(round(recommendations['Max Token 1 HR Price Change (%)'], 2), 500)
        recommendations['Min Token 1 HR Volume'] = max(round(recommendations['Min Token 1 HR Volume'], -3), 50000)
        recommendations['Min Token 24 Hrs Volume'] = max(round(recommendations['Min Token 24 Hrs Volume'], -3), 975000)
        recommendations['Min Token Age (hrs)'] = max(round(recommendations['Min Token Age (hrs)'], 1), 2)

        # Calculate stop loss
        stop_losses = []
        for _, token in top_performers.iterrows():
            stop_loss = calculate_constrained_stop_loss(token, mode)
            stop_losses.append(stop_loss)

        # Calculate average stop loss
        avg_stop_loss = np.mean(stop_losses)
        recommendations['Stop Loss (%)'] = round(avg_stop_loss, 2)

        # Generate explanations
        explanations = explain_recommendations(recommendations, mode, df_recent)

        # Display recommendations with explanations
        console.print(f"\n[bold green]Optimized DLMM Entry Criteria ({mode} mode):[/bold green]")
        for setting, value in recommendations.items():
            if 'Volume' in setting or 'Market Cap' in setting:
                formatted_value = f"${value:,.0f}"
            elif 'Age' in setting:
                formatted_value = f"{value:.1f}"
            else:
                formatted_value = f"{value:.2f}%"
            console.print(f"[cyan]{setting}:[/cyan] {formatted_value}")
            console.print(f"[yellow]{explanations[setting]}[/yellow]\n")

        return recommendations

    except Exception as e:
        console.print(f"[red]An error occurred while generating recommendations: {str(e)}[/red]")
        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        return None

def save_recommendations(recommendations):
    if isinstance(recommendations, dict):
        recommendations_dict = {
            "timestamp": datetime.now().isoformat(),
            "recommendations": recommendations
        }
        
        filename = 'dlmm_recommendations.json'
        with open(filename, 'w') as f:
            json.dump(recommendations_dict, f, indent=4)
        console.print(f"\n[green]Recommendations saved to {filename}[/green]")
    else:
        console.print("\n[yellow]No recommendations to save.[/yellow]")

def implement_garch(df, timeframes):
    volatilities = {}
    for timeframe in map(int, timeframes):
        price_col = get_price_change_column(timeframe)
        
        # Ensure we're working with a copy of the data
        price_data = df[[price_col]].copy()
        
        # Remove any NaN or inf values
        price_data = price_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Calculate returns, removing any remaining NaN or inf values
        epsilon = 1e-8  # Small constant to avoid divide by zero
        returns = np.log(price_data + epsilon) - np.log(price_data.shift(1) + epsilon)
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Check if we have enough data points after cleaning
        if len(returns) < 100:
            console.print(f"[yellow]Warning: Insufficient data for GARCH modeling for {timeframe} minutes timeframe after removing NaN/inf values. Using standard deviation instead.[/yellow]")
            volatilities[timeframe] = returns[price_col].std()
            continue
        
        try:
            # Fit GARCH(1,1) model
            model = arch_model(returns, vol='GARCH', p=1, q=1)
            results = model.fit(disp='off')

            # Forecast volatility
            forecast = results.forecast(horizon=1)
            
            # Extract the volatility forecast
            volatility_forecast = np.sqrt(forecast.variance.values[-1, :])[0]
            volatilities[timeframe] = volatility_forecast
            console.print(f"[green]Successfully calculated GARCH volatility for {timeframe} minutes timeframe: {volatility_forecast:.4f}[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: GARCH modeling failed for {timeframe} minutes timeframe. Using standard deviation instead. Error: {str(e)}[/yellow]")
            volatilities[timeframe] = returns[price_col].std()

    return volatilities

def calculate_market_condition_factor():
    # Placeholder function - implement actual market condition analysis
    return 1.0

def calculate_age_volatility_factor(volatility):
    # Placeholder function - implement relationship between age and volatility
    return max(1 - (volatility / 100), 0.5)

def calculate_market_correlation(token_data):
    # Placeholder function - implement correlation calculation
    return 1.0

def calculate_historical_volatility(price_data, window=30):
    if len(price_data) < window:
        return np.nan  # Return NaN if there's not enough data
    returns = price_data.pct_change().dropna()
    if len(returns) == 0:
        return np.nan  # Return NaN if there are no valid returns
    volatility = returns.rolling(window=window).std().mean()  # Use mean instead of last value
    return volatility * np.sqrt(365)  # Annualize the volatility

def calculate_price_range(current_price, bin_count=69, bin_step=0.01):
    lower_bound = current_price * (1 - bin_step) ** (bin_count // 2)
    upper_bound = current_price * (1 + bin_step) ** (bin_count // 2)
    return lower_bound, upper_bound

def generate_explanation(setting, value, mode, timeframe_minutes, df_top, market_cap_10th_percentile=None, valid_market_caps=None):
    risk_factors = {'degen': 1.2, 'moderate': 1.0, 'conservative': 0.8}
    risk_factor = risk_factors[mode]

    if setting == 'Min Token Market Cap':
        return f"""\
        Minimum market capitalization recommended: ${value:,}
        
        Calculation method:
        1. We analyzed the market caps of the top 30 tokens based on a custom score, excluding extremely high and low values.
        2. 10th percentile of valid market caps: ${market_cap_10th_percentile:,.2f}
        3. This value was adjusted by the risk factor ({risk_factor} for {mode} mode).
        4. The result was compared with a minimum threshold of $200,000.
        5. The higher value was chosen to ensure sufficient liquidity.
        
        Number of tokens analyzed: {len(valid_market_caps)}
        Minimum market cap in data: ${valid_market_caps.min():,.2f}
        Maximum market cap in data: ${valid_market_caps.max():,.2f}
        
        This setting balances opportunity with stability, ensuring sufficient liquidity across the 69 bins in the Spot-Wide strategy.
        It helps filter out extremely small or potentially manipulated tokens while still allowing entry into promising new projects.
        """
    elif setting == 'Min Token 5 Min Price Change (%)':
        price_changes_5m = df_top['price_change_5m'].dropna()
        mean_5m = price_changes_5m.mean()
        std_5m = price_changes_5m.std()
        percentile_10 = price_changes_5m.quantile(0.10)
        
        return f"""\
        Minimum 5-minute price change threshold for entry or re-entry, based on analysis of historical data.
        This helps avoid entering or re-entering a pool that is experiencing a rapid price decline. The recommended value is {value}%.

        Calculation method:
        1. We analyzed all 5-minute price changes in the recent data of top-performing tokens.
        2. Key statistics:
           - Mean: {mean_5m:.2f}%
           - Standard Deviation: {std_5m:.2f}%
           - 10th Percentile: {percentile_10:.2f}%
        3. We used the 10th percentile as the base threshold.
        4. This base threshold was then adjusted by the risk factor ({risk_factor} for {mode} mode).

        How to use this setting to prevent entering dumping pools:
        - Monitor the live 5-minute price change of the token.
        - Only enter the pool if the 5-minute price change is above this threshold.
        - This setting helps prevent entering during rapid, short-term price declines, which could indicate a dumping scenario.
        """
    elif setting == 'Min Token 1 HR Price Change (%)':
        return f"""\
        Minimum 1-hour price change for entry, based on the 10th percentile of recent data from top-performing tokens.
        Value: {value}%
        This helps identify potential entry points while avoiding tokens in significant short-term decline.
        Adjusted for {mode} risk profile (factor: {risk_factor}).
        """
    elif setting == 'Max Token 1 HR Price Change (%)':
        return f"""\
        Maximum 1-hour price change for entry, based on the 90th percentile of recent data from top-performing tokens.
        Value: {value}%
        This helps avoid entering tokens that might be experiencing unsustainable short-term growth.
        Adjusted for {mode} risk profile (factor: {risk_factor}).
        """
    elif setting == 'Min Token 1 HR Volume':
        return f"""\
        Minimum 1-hour trading volume required for entry, ensuring sufficient liquidity.
        Value: ${value:,.2f}
        Based on the 25th percentile of volumes from top-performing tokens.
        Adjusted for {mode} risk profile (factor: {risk_factor}).
        """
    elif setting == 'Min Token 24 Hrs Volume':
        return f"""\
        Minimum 24-hour trading volume, indicating sustained market interest.
        Value: ${value:,.2f}
        Based on the 25th percentile of 24-hour volumes from top-performing tokens.
        Adjusted for {mode} risk profile (factor: {risk_factor}).
        """
    elif setting == 'Min Token Age (hrs)':
        return f"""\
        Minimum age of the token pool, helping to avoid brand new, untested tokens.
        Value: {value:.1f} hours
        Based on the 10th percentile of ages from top-performing tokens.
        Adjusted for {mode} risk profile (factor: {risk_factor}).
        """
    
    return "No explanation available for this setting."

def validate_data(df):
    required_columns = ['timestamp', 'token_name', 'token_price', 'volume_5m', 'volume_1h', 'volume_6h', 'volume_24h', 'price_change_5m', 'price_change_1h', 'price_change_6h', 'price_change_24h']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        console.print(f"[bold red]Missing required columns: {', '.join(missing_columns)}[/bold red]")
        return False
    return True

def fallback_recommendation():
    console.print("[yellow]Using fallback recommendation due to calculation issues.[/yellow]")
    return 60, {'timeframe': 60, 'score': 0, 'sharpe_ratio': 0, 'avg_price_change': 0, 'volatility': 0, 'avg_volume': 0}

def recommend_timeframe():
    try:
        data_manager.load_data()
        df_recent = data_manager.get_recent_data()
        
        if not validate_data(df_recent):
            return fallback_recommendation()
        
        if df_recent.empty or (datetime.now() - df_recent['timestamp'].max()).total_seconds() / 3600 > 24:
            console.print("[bold yellow]Data is too old or empty. Please run a new data collection cycle.[/bold yellow]")
            return None, None

        console.print("[cyan]Calculating metrics for multiple timeframes...[/cyan]")
        
        timeframes = [30, 60, 120, 180, 360, 720, 1440]
        results = [calculate_metrics(tf, df_recent) for tf in timeframes]

        valid_results = [r for r in results if r is not None and isinstance(r, dict) and 'score' in r]

        if not valid_results:
            console.print("[bold red]No valid results were returned from calculate_metrics.[/bold red]")
            return fallback_recommendation()

        best_result = max(valid_results, key=lambda x: x['score'])
        best_timeframe = best_result['timeframe']
        
        return best_timeframe, best_result
    
    except Exception as e:
        console.print(f"[bold red]Error in recommend_timeframe: {str(e)}[/bold red]")
        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        return fallback_recommendation()

def display_command_list():
    console.print(Panel(Text("Available Commands:", style="bold cyan")))
    console.print("r - Get recommendations for a specific timeframe and risk mode")
    console.print("t - Get recommended timeframe and its settings")
    console.print("q - Quit the program")
    console.print("\nPress any key to continue...")

def create_status_table(time_to_next_run):
    table = Table(show_header=False, border_style="bold", box=ROUNDED)
    table.add_row("Next data pull in:", f"{time_to_next_run.seconds // 60:02d}:{time_to_next_run.seconds % 60:02d}")
    return table

def check_for_input():
    if os.name == 'nt':  # Windows
        return msvcrt.kbhit()
    else:  # Unix-like systems
        import select
        return select.select([sys.stdin], [], [], 0)[0]

def get_input():
    if os.name == 'nt':  # Windows
        return msvcrt.getch().decode('utf-8').lower()
    else:  # Unix-like systems
        return sys.stdin.readline().strip().lower()

def get_timeframe_minutes(timeframe):
    if timeframe == 'day':
        return 1440
    elif timeframe == 'hour':
        return 60
    else:
        return 5  # Default to 5 minutes if unknown

def main():
    collection_interval = timedelta(minutes=30)
    last_run_time = datetime.now() - collection_interval

    with Live(console=console, refresh_per_second=1) as live:
        while True:
            current_time = datetime.now()
            if current_time - last_run_time >= collection_interval:
                live.stop()
                collect_data()
                last_run_time = current_time
                display_command_list()
                live.start()
            
            next_run_time = last_run_time + collection_interval
            time_to_next_run = next_run_time - current_time
            
            status_table = create_status_table(time_to_next_run)
            live.update(status_table)
            
            if check_for_input():
                live.stop()
                command = get_input()
                
                if command == 'r':
                    data_manager.load_data()
                    df_recent = data_manager.get_recent_data()
                    timeframe_minutes = get_timeframe_input()
                    mode = get_risk_mode_input()
                    try:
                        recommendations = get_current_recommendations(timeframe_minutes, mode, df_recent)
                        if recommendations:
                            save_recommendations(recommendations)
                    except Exception as e:
                        console.print(f"[red]An error occurred: {str(e)}[/red]")
                        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
                elif command == 't':
                    best_timeframe, best_result = recommend_timeframe()
                    if best_timeframe and best_result:
                        console.print(f"\n[bold green]Recommended Timeframe: {best_timeframe} minutes[/bold green]")
                        console.print(f"Score: {best_result['score']:.4f}")
                        
                        # Get and display settings for the recommended timeframe
                        data_manager.load_data()
                        df_recent = data_manager.get_recent_data()
                        recommendations = get_current_recommendations(best_timeframe, 'moderate', df_recent)
                        if recommendations:
                            console.print("\n[bold green]Recommended settings for this timeframe:[/bold green]")
                            for setting, value in recommendations.items():
                                if 'Volume' in setting or 'Market Cap' in setting:
                                    formatted_value = f"${value:,.0f}"
                                elif 'Age' in setting:
                                    formatted_value = f"{value:.1f}"
                                else:
                                    formatted_value = f"{value:.2f}%"
                                console.print(f"[cyan]{setting}:[/cyan] {formatted_value}")
                    else:
                        console.print("[bold red]Unable to determine the best timeframe.[/bold red]")
                elif command == 'q':
                    console.print(Panel(Text("Exiting the program.", style="bold red")))
                    return
                else:
                    console.print(Panel(Text("Invalid input. Please enter 'r', 't', or 'q'.", style="bold yellow")))
                
                display_command_list()
                live.start()
            
            time.sleep(0.1)

    console.print("Program ended.")

if __name__ == "__main__":
    console.print(Panel(Text("DLMM Data Collector", style="bold magenta")))
    log.info("Script started. Data collection will begin shortly.")
    try:
        main()
    except KeyboardInterrupt:
        console.print(Panel(Text("Script terminated by user.", style="bold red")))
        log.info("Script terminated by user.")