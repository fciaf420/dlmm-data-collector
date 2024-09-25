# src/analysis/metrics.py

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from dtaidistance import dtw
from pykalman import KalmanFilter
from arch import arch_model
from rich.console import Console
from rich.table import Table

console = Console()

def calculate_multifactor_score(data):
    factors = {}
    
    if 'price_change_1h' in data.columns:
        factors['price_momentum'] = data['price_change_1h']
    
    if 'volume_1h' in data.columns and 'volume_24h' in data.columns:
        factors['volume_trend'] = np.where((data['volume_24h'] > 0) & (data['volume_1h'] > 0), 
                                           data['volume_1h'] / data['volume_24h'], 0)
    
    if 'effective_market_cap' in data.columns:
        factors['market_cap_rank'] = data['effective_market_cap'].rank(ascending=False)
    
    if 'price_change_1h' in data.columns:
        factors['volatility'] = implement_garch(data, 60)  # 1-hour GARCH volatility
    
    if 'effective_market_cap' in data.columns and 'volume_24h' in data.columns:
        factors['liquidity_ratio'] = np.where((data['effective_market_cap'] > 0) & (data['volume_24h'] > 0), 
                                              data['volume_24h'] / data['effective_market_cap'], 0)
    
    weights = {k: 1/len(factors) for k in factors.keys()}  # Equal weights to start
    return sum(factors[k] * weights[k] for k in factors)

def get_top_tokens_by_multifactor_score(df, top_n=5):
    df['multifactor_score'] = calculate_multifactor_score(df)
    top_tokens = df.nlargest(top_n, 'multifactor_score')
    return top_tokens[['token_name', 'multifactor_score', 'token_price', 'market_cap', 'volume_24h']]

def display_top_tokens_table(top_tokens):
    table = Table(title="Top 5 Tokens by Multifactor Score")
    table.add_column("Token Name", style="cyan")
    table.add_column("Multifactor Score", style="magenta")
    table.add_column("Price (USD)", style="green")
    table.add_column("Market Cap (USD)", style="yellow")
    table.add_column("24h Volume (USD)", style="blue")

    for _, row in top_tokens.iterrows():
        table.add_row(
            row['token_name'],
            f"{row['multifactor_score']:.4f}",
            f"${row['token_price']:.6f}",
            f"${row['market_cap']:,.0f}",
            f"${row['volume_24h']:,.0f}"
        )

    console.print(table)

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