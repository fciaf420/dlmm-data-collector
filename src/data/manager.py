# src/data/manager.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from rich.console import Console

from config.settings import DATA_FILE

console = Console()

class DataManager:
    def __init__(self):
        self.df = None
        self.last_update = None

    def load_data(self):
        current_time = datetime.now()
        if self.df is None or self.last_update is None or (current_time - self.last_update) > timedelta(minutes=30):
            console.print("Loading data from CSV file...")
            try:
                self.df = pd.read_csv(DATA_FILE)
                console.print(f"Loaded {len(self.df)} rows of data")
                
                # Convert timestamp to datetime
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce', utc=True)
                
                # Calculate effective_market_cap
                self.df['effective_market_cap'] = self.df['market_cap'].fillna(self.df['fdv']).fillna(1000)
                self.df['effective_market_cap'] = self.df['effective_market_cap'].clip(lower=1000)
                
                invalid_timestamps = self.df['timestamp'].isnull()
                if invalid_timestamps.any():
                    console.print(f"[yellow]Dropped {invalid_timestamps.sum()} rows with invalid timestamps[/yellow]")
                    self.df = self.df.dropna(subset=['timestamp'])
                
                self.last_update = current_time
                console.print("Data loading complete")
            except FileNotFoundError:
                console.print(f"[yellow]Warning: {DATA_FILE} not found. Creating an empty DataFrame.[/yellow]")
                self.df = pd.DataFrame()
                self.last_update = current_time

    def get_recent_data(self):
        self.load_data()  # Ensure data is loaded before filtering
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        return self.df[self.df['timestamp'] > seven_days_ago]

    def append_data(self, new_data):
        new_df = pd.DataFrame(new_data)
        
        # Convert timestamp to datetime in new data
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], errors='coerce', utc=True)
        
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        
        # Ensure all timestamps are in datetime format before saving
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce', utc=True)
        
        self.df.to_csv(DATA_FILE, index=False)
        console.print(f"[green]Appended {len(new_df)} new records to {DATA_FILE}[/green]")
        self.last_update = datetime.now()