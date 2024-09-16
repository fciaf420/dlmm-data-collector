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
from dtaidistance import dtw
from pykalman import KalmanFilter
from arch import arch_model
from json import JSONEncoder

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
    
    if len(returns) < 100:
        console.print(f"[yellow]Warning: Insufficient data for GARCH modeling. Using standard deviation.[/yellow]")
        return returns.std() if len(returns) > 1 else 0  # Return 0 if there's only one or no data point
    
    try:
        model = arch_model(returns, vol='GARCH', p=1, q=1)
        results = model.fit(disp='off')
        forecast = results.forecast(horizon=1)
        volatility_forecast = np.sqrt(forecast.variance.values[-1, :])[0]
        
        console.print(f"[green]Successfully calculated GARCH volatility: {volatility_forecast:.4f}[/green]")
        return volatility_forecast
    except Exception as e:
        console.print(f"[yellow]Warning: GARCH modeling failed. Using standard deviation. Error: {str(e)}[/yellow]")
        return returns.std() if len(returns) > 1 else 0  # Return 0 if there's only one or no data point

def find_similar_patterns(current_data, historical_data):
    distances = [dtw.distance(current_data, hist) for hist in historical_data]
    return np.argmin(distances)

def kalman_trend(prices):
    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1], 
                      initial_state_mean=0, initial_state_covariance=1, 
                      observation_covariance=1, transition_covariance=.01)
    return kf.filter(prices)[0].flatten()

def estimate_tail_risk(returns, threshold=0.05):
    try:
        # Remove non-finite values
        valid_returns = returns[np.isfinite(returns)]
        
        if len(valid_returns) < 10:  # Arbitrary minimum number of samples
            console.print("[yellow]Warning: Not enough valid data for tail risk estimation.[/yellow]")
            return None, None

        tail_returns = valid_returns[valid_returns < np.quantile(valid_returns, threshold)]
        
        if len(tail_returns) < 5:  # Another arbitrary minimum
            console.print("[yellow]Warning: Not enough tail data for estimation.[/yellow]")
            return None, None

        shape, _, scale = stats.genpareto.fit(tail_returns)
        return shape, scale
    except Exception as e:
        console.print(f"[yellow]Warning: Error in tail risk estimation - {str(e)}[/yellow]")
        return None, None

def feature_importance(X, y):
    try:
        # Remove rows with infinite or NaN values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 10:  # Arbitrary minimum number of samples
            console.print("[yellow]Warning: Not enough valid data for feature importance calculation.[/yellow]")
            return None

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_clean, y_clean)
        return rf.feature_importances_
    except Exception as e:
        console.print(f"[yellow]Warning: Error in feature importance calculation - {str(e)}[/yellow]")
        return None

def prepare_data_for_timeframe(df, timeframe_minutes):
    base_columns = ['price_change_5m', 'price_change_1h', 'volume_1h', 'volume_24h', 'pool_age_hours', 'market_cap', 'fdv']
    
    price_change_col = f'price_change_{timeframe_minutes}m' if timeframe_minutes < 1440 else 'price_change_24h'
    
    required_columns = base_columns + [price_change_col]
    
    if price_change_col not in df.columns:
        df[price_change_col] = df['price_change_24h'] * (timeframe_minutes / 1440)
    
    # Use FDV as fallback for market cap, and set a minimum value
    df['effective_market_cap'] = df['market_cap'].fillna(df['fdv']).fillna(1000)
    df['effective_market_cap'] = df['effective_market_cap'].clip(lower=1000)
    
    # Ensure volume columns are non-negative
    df['volume_1h'] = df['volume_1h'].clip(lower=0)
    df['volume_24h'] = df['volume_24h'].clip(lower=0)
    
    df_prepared = df[required_columns + ['effective_market_cap']].dropna(subset=['effective_market_cap'])
    
    # Filter out tokens with market cap less than $1000
    df_prepared = df_prepared[df_prepared['effective_market_cap'] >= 1000]
    
    # Instead of removing, let's set a minimum value for pool age
    df_prepared['pool_age_hours'] = df_prepared['pool_age_hours'].clip(lower=0)
    
    if df_prepared.empty:
        console.print(f"[yellow]Warning: No valid data for {timeframe_minutes} minute timeframe after removing NaN values[/yellow]")
        return None
    
    return df_prepared

def get_current_recommendations(timeframe_minutes, mode, df_recent):
    log.info(f"Generating recommendations for {timeframe_minutes} minute timeframe in {mode} mode")
    
    try:
        df_prepared = prepare_data_for_timeframe(df_recent, timeframe_minutes)
        if df_prepared is None:
            raise ValueError(f"Insufficient data for {timeframe_minutes} minute timeframe")
        
        df_prepared = df_prepared.reset_index(drop=True)
        
        risk_factors = {'degen': 1.5, 'moderate': 1.2, 'conservative': 1.0}
        risk_factor = risk_factors[mode]
        log.info(f"Applied risk factor: {risk_factor}")
        
        # Calculate metrics
        df_prepared['multifactor_score'] = multifactor_score(df_prepared)
        df_prepared['kalman_trend'] = kalman_trend(df_prepared['price_change_1h'])
        
        returns = df_prepared['price_change_1h'].pct_change().dropna()
        shape, scale = estimate_tail_risk(returns)
        
        top_performers = df_prepared.sort_values('multifactor_score', ascending=False).head(30)
        
        recommendations = {}
        explanations = {}
        
        # Price change calculations
        for col, percentile in [('price_change_5m', 0.10), ('price_change_1h', 0.10)]:
            raw_value = df_prepared[col].quantile(percentile)
            adjusted_value = round(raw_value * risk_factor, 2)
            recommendations[f'Min Token {col.split("_")[2].upper()} Price Change (%)'] = adjusted_value
            explanations[f'Min Token {col.split("_")[2].upper()} Price Change (%)'] = f"""
            Raw {percentile*100}th percentile: {raw_value:.2f}%
            Risk adjustment: {risk_factor:.2f}x
            Final value: {adjusted_value:.2f}%
            """
            log.info(f"Calculated {col} change: raw={raw_value:.2f}, adjusted={adjusted_value:.2f}")
        
        # Max 1 HR Price Change
        max_price_change = min(round(df_prepared['price_change_1h'].quantile(0.95) * risk_factor, 2), 500)
        recommendations['Max Token 1 HR Price Change (%)'] = max_price_change
        explanations['Max Token 1 HR Price Change (%)'] = f"""
        Raw 95th percentile: {df_prepared['price_change_1h'].quantile(0.95):.2f}%
        Risk adjustment: {risk_factor:.2f}x
        Capped at 500%
        Final value: {max_price_change:.2f}%
        """
        log.info(f"Calculated max 1 HR price change: {max_price_change:.2f}")
        
        # Volume calculations
        for col, threshold in [('volume_1h', 50000), ('volume_24h', 50000)]:
            raw_value = top_performers[col].quantile(0.25)
            risk_adjusted = round(raw_value * risk_factor, -3)
            final_value = max(risk_adjusted, threshold)
            recommendations[f'Min Token {col.split("_")[1].upper()} Volume'] = final_value
            explanations[f'Min Token {col.split("_")[1].upper()} Volume'] = f"""
            Raw 25th percentile: ${raw_value:,.0f}
            Risk-adjusted value: ${risk_adjusted:,.0f}
            Minimum threshold: ${threshold:,.0f}
            Final value: ${final_value:,.0f}
            """
            log.info(f"Calculated {col}: raw={raw_value:.0f}, adjusted={risk_adjusted:.0f}, final={final_value:.0f}")
        
        # Token Age
        raw_age = top_performers['pool_age_hours'].quantile(0.10)
        adjusted_age = round(raw_age * risk_factor, 1)
        final_age = max(adjusted_age, 2)
        recommendations['Min Token Age (hrs)'] = final_age
        explanations['Min Token Age (hrs)'] = f"""
        Raw 10th percentile: {raw_age:.1f} hours
        Risk-adjusted value: {adjusted_age:.1f} hours
        Minimum threshold: 2 hours
        Final value: {final_age:.1f} hours
        """
        log.info(f"Calculated token age: raw={raw_age:.1f}, adjusted={adjusted_age:.1f}, final={final_age:.1f}")
        
        # Diagnostic logging for market cap
        log.info(f"Market cap statistics:")
        log.info(f"Min: ${df_prepared['effective_market_cap'].min():,.0f}")
        log.info(f"Max: ${df_prepared['effective_market_cap'].max():,.0f}")
        log.info(f"Mean: ${df_prepared['effective_market_cap'].mean():,.0f}")
        log.info(f"Median: ${df_prepared['effective_market_cap'].median():,.0f}")
        
        # Market Cap calculation
        raw_mcap = top_performers['effective_market_cap'].quantile(0.10)
        log.info(f"Raw 10th percentile market cap: ${raw_mcap:,.0f}")
        raw_mcap = max(raw_mcap, 1000)  # Ensure minimum of $1000
        log.info(f"Raw market cap after minimum applied: ${raw_mcap:,.0f}")
        adjusted_mcap = raw_mcap * risk_factor
        log.info(f"Risk-adjusted market cap: ${adjusted_mcap:,.0f}")
        adjusted_mcap = round(max(adjusted_mcap, 1000), -3)  # Ensure minimum of $1000 after risk adjustment
        log.info(f"Risk-adjusted market cap after rounding: ${adjusted_mcap:,.0f}")
        final_mcap = max(adjusted_mcap, 200000)
        log.info(f"Final market cap after applying $200,000 minimum: ${final_mcap:,.0f}")

        recommendations['Min Token Market Cap'] = final_mcap
        explanations['Min Token Market Cap'] = f"""
        Raw 10th percentile: ${raw_mcap:,.0f}
        Risk-adjusted value: ${adjusted_mcap:,.0f}
        Minimum threshold: $200,000
        Final value: ${final_mcap:,.0f}
        Note: {"Using minimum threshold due to low market cap values" if final_mcap == 200000 else ""}
        """
        
        # Additional metrics
        if shape is not None and scale is not None:
            recommendations['Tail Risk Shape'] = shape
            recommendations['Tail Risk Scale'] = scale
            explanations['Tail Risk Shape'] = f"Estimated shape parameter: {shape:.4f}"
            explanations['Tail Risk Scale'] = f"Estimated scale parameter: {scale:.4f}"
            log.info(f"Calculated tail risk: shape={shape:.4f}, scale={scale:.4f}")
        
        # Consistency check
        inconsistencies = check_consistency(recommendations, explanations)
        if inconsistencies:
            log.warning("Inconsistencies found between recommendations and explanations:")
            for key, (rec_value, exp_value) in inconsistencies.items():
                log.warning(f"{key}: Recommendation={rec_value}, Explanation={exp_value}")
        
        return recommendations, explanations
    
    except Exception as e:
        log.error(f"An error occurred while generating recommendations: {str(e)}")
        log.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def check_consistency(recommendations, explanations):
    inconsistencies = {}
    for key in recommendations:
        if key in explanations:
            rec_value = recommendations[key]
            exp_lines = explanations[key].split('\n')
            exp_value = None
            for line in exp_lines:
                if 'Final value:' in line:
                    value_str = line.split(':')[-1].strip()
                    if '$' in value_str:
                        # Handle dollar amounts
                        exp_value = float(value_str.replace('$', '').replace(',', ''))
                    elif '%' in value_str:
                        # Handle percentages
                        exp_value = float(value_str.replace('%', ''))
                    elif 'hours' in value_str:
                        # Handle time durations
                        exp_value = float(value_str.split()[0])
                    else:
                        # Handle other numeric values
                        exp_value = float(value_str)
                    break
            
            if exp_value is not None:
                # Compare values based on their type
                if isinstance(rec_value, (int, float)) and isinstance(exp_value, (int, float)):
                    if abs(rec_value - exp_value) > 0.01:  # Allow small float discrepancies
                        inconsistencies[key] = (rec_value, exp_value)
                elif rec_value != exp_value:
                    inconsistencies[key] = (rec_value, exp_value)
    
    return inconsistencies

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

def recommend_timeframe():
    try:
        data_manager.load_data()
        df_recent = data_manager.get_recent_data()
        
        if df_recent.empty:
            console.print("[bold yellow]No recent data available. Please run a new data collection cycle.[/bold yellow]")
            return None, None, None

        console.print("[cyan]Calculating metrics for multiple timeframes...[/cyan]")
        
        timeframes = [30, 60, 120, 180, 360, 720, 1440]
        results = []
        for tf in timeframes:
            df_prepared = prepare_data_for_timeframe(df_recent, tf)
            if df_prepared is not None:
                df_prepared['multifactor_score'] = multifactor_score(df_prepared)
                
                result = {
                    'timeframe': tf,
                    'score': df_prepared['multifactor_score'].mean()
                }
                results.append(result)
            else:
                console.print(f"[yellow]Skipping timeframe {tf} due to insufficient data[/yellow]")

        if not results:
            console.print("[bold red]No valid results were returned from calculations.[/bold red]")
            return None, None, None

        best_result = max(results, key=lambda x: x['score'])
        best_timeframe = best_result['timeframe']
        
        recommendations, _ = get_current_recommendations(best_timeframe, 'moderate', df_recent)
        
        return best_timeframe, best_result, recommendations
    
    except Exception as e:
        console.print(f"[bold red]Error in recommend_timeframe: {str(e)}[/bold red]")
        console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        return None, None, None

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
    return msvcrt.kbhit() if os.name == 'nt' else select.select([sys.stdin], [], [], 0)[0]

def get_input():
    return msvcrt.getch().decode('utf-8').lower() if os.name == 'nt' else sys.stdin.readline().strip().lower()

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
                elif command == 't':
                    best_timeframe, best_result, recommendations = recommend_timeframe()
                    if best_timeframe and best_result and recommendations:
                        console.print(f"\n[bold green]Recommended Timeframe: {best_timeframe} minutes[/bold green]")
                        console.print(f"Score: {best_result['score']:.4f}")
                        
                        console.print("\n[bold green]Recommended settings for this timeframe:[/bold green]")
                        for setting, value in recommendations.items():
                            formatted_value = f"${value:,.0f}" if 'Volume' in setting or 'Market Cap' in setting else f"{value:.2f}%"
                            console.print(f"[cyan]{setting}:[/cyan] {formatted_value}")
                    else:
                        console.print("[bold red]Unable to determine the best timeframe or generate recommendations.[/bold red]")
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
