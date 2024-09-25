# src/analysis/recommendations.py

import numpy as np
import pandas as pd
from rich.console import Console
import json
from datetime import datetime
import logging
import traceback

from config.settings import RISK_FACTORS, MIN_MARKET_CAP, MIN_VOLUME, MIN_TOKEN_AGE, RECOMMENDATIONS_FILE
from src.analysis.metrics import calculate_multifactor_score, implement_garch, kalman_trend, estimate_tail_risk
from src.utils.math_utils import NumpyEncoder

console = Console()
log = logging.getLogger("rich")

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
        
        risk_factor = RISK_FACTORS[mode]
        log.info(f"Applied risk factor: {risk_factor}")
        
        # Calculate metrics
        df_prepared['multifactor_score'] = calculate_multifactor_score(df_prepared)
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
        for col, threshold in [('volume_1h', MIN_VOLUME), ('volume_24h', MIN_VOLUME)]:
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
        final_age = max(adjusted_age, MIN_TOKEN_AGE)
        recommendations['Min Token Age (hrs)'] = final_age
        explanations['Min Token Age (hrs)'] = f"""
        Raw 10th percentile: {raw_age:.1f} hours
        Risk-adjusted value: {adjusted_age:.1f} hours
        Minimum threshold: {MIN_TOKEN_AGE} hours
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
        raw_mcap = max(raw_mcap, MIN_MARKET_CAP)
        log.info(f"Raw market cap after minimum applied: ${raw_mcap:,.0f}")
        adjusted_mcap = raw_mcap * risk_factor
        log.info(f"Risk-adjusted market cap: ${adjusted_mcap:,.0f}")
        adjusted_mcap = round(max(adjusted_mcap, MIN_MARKET_CAP), -3)
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
        
        with open(RECOMMENDATIONS_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=4, cls=NumpyEncoder)
        console.print(f"\n[green]Recommendations and explanations saved to {RECOMMENDATIONS_FILE}[/green]")
    else:
        console.print("\n[yellow]No recommendations or explanations to save.[/yellow]")

def recommend_timeframe(data_manager):
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