# src/data/collector.py

import pandas as pd
from datetime import datetime, timezone
import time
import logging
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import json

from config.settings import (
    BASE_URL, 
    SLEEP_BETWEEN_OHLCV_CALLS, 
    SLEEP_BETWEEN_POOL_INFO_AND_OHLCV, 
    SLEEP_BETWEEN_POOLS
)
from src.utils.api_utils import call_api_with_retry

console = Console()
log = logging.getLogger("rich")

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
            time.sleep(SLEEP_BETWEEN_OHLCV_CALLS)
        except Exception as e:
            log.error(f"Failed to get OHLCV data for {pool_address} ({timeframe}): {str(e)}")
    return ohlcv_data

def collect_data():
    console.print(Panel(Text("Starting data collection...", style="bold green")))
    all_data = []
    skipped_tokens = []

    try:
        trending_pools = get_trending_pools(network='solana', page=1)
        if 'data' not in trending_pools:
            console.print(f"[bold red]Unexpected response structure:[/bold red] {trending_pools}")
            return []
    except Exception as e:
        console.print(f"[bold red]Error fetching trending pools:[/bold red] {str(e)}")
        return []

    for index, pool in enumerate(trending_pools['data'], 1):
        try:
            pool_attributes = pool['attributes']
            token_name = pool_attributes.get('name', 'Unknown')
            network = 'solana'
            pool_address = pool_attributes.get('address', 'Unknown')
            console.print(f"[cyan]Processing pool {index}: {token_name}")
            
            pool_info = get_pool_info(network, pool_address)
            time.sleep(SLEEP_BETWEEN_POOL_INFO_AND_OHLCV)
            ohlcv_data = get_ohlcv_data(network, pool_address)
            
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
            
            all_data.append(pool_data)
            console.print(f"Processed {token_name} / {network.upper()}: Price ${pool_data['token_price']:.6f}, Market Cap/FDV ${pool_data['market_cap']:,.2f}")
            
            time.sleep(SLEEP_BETWEEN_POOLS)
        except KeyError as e:
            log.warning(f"Skipping token {token_name} due to missing data: {str(e)}")
            skipped_tokens.append(token_name)
            console.print(f"[yellow]Skipping token {token_name} due to missing data: {str(e)}[/yellow]")
        except Exception as e:
            log.error(f"Error processing pool {pool_attributes.get('address', 'Unknown')}: {str(e)}")
            console.print(f"[red]Error processing pool {pool_attributes.get('address', 'Unknown')}: {str(e)}[/red]")
            console.print(f"[yellow]Pool attributes: {json.dumps(pool_attributes, indent=2)}[/yellow]")

    console.print(f"[yellow]Skipped {len(skipped_tokens)} tokens due to missing data: {', '.join(skipped_tokens)}[/yellow]")
    
    return all_data