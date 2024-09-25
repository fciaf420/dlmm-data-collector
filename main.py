# src/main.py

import time
from datetime import datetime, timedelta
from rich.live import Live
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from config.settings import COLLECTION_INTERVAL
from src.data.collector import collect_data
from src.data.manager import DataManager
from src.analysis.recommendations import get_current_recommendations, recommend_timeframe, display_recommendations, save_recommendations
from src.analysis.metrics import get_top_tokens_by_multifactor_score, display_top_tokens_table
from src.utils.console_utils import display_command_list, create_status_table, check_for_input, get_input, get_user_input

console = Console()
data_manager = DataManager()

def display_main_menu():
    console.print(Panel(Text("DLMM Data Collector", style="bold magenta")))
    console.print("\nMain Menu:")
    console.print("1. Start data collection")
    console.print("2. View existing results")
    console.print("3. Exit")
    return console.input("\nEnter your choice (1-3): ")

def view_results():
    data_manager.load_data()
    if data_manager.df.empty:
        console.print("[yellow]No data available. Please run data collection first.[/yellow]")
        return

    while True:
        console.print("\n[bold cyan]Results Menu:[/bold cyan]")
        console.print("1. Get recommendations for a specific timeframe and risk mode")
        console.print("2. Get recommended timeframe and its settings")
        console.print("3. View top 5 tokens by Multifactor Score")
        console.print("4. Return to main menu")
        
        choice = console.input("\nEnter your choice (1-4): ")
        
        if choice == '1':
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
        elif choice == '2':
            best_timeframe, best_result, recommendations = recommend_timeframe(data_manager)
            if best_timeframe and best_result and recommendations:
                console.print(f"\n[bold green]Recommended Timeframe: {best_timeframe} minutes[/bold green]")
                console.print(f"Score: {best_result['score']:.4f}")
                
                console.print("\n[bold green]Recommended settings for this timeframe:[/bold green]")
                for setting, value in recommendations.items():
                    formatted_value = f"${value:,.0f}" if 'Volume' in setting or 'Market Cap' in setting else f"{value:.2f}%"
                    console.print(f"[cyan]{setting}:[/cyan] {formatted_value}")
            else:
                console.print("[bold red]Unable to determine the best timeframe or generate recommendations.[/bold red]")
        elif choice == '3':
            df_recent = data_manager.get_recent_data()
            top_tokens = get_top_tokens_by_multifactor_score(df_recent)
            display_top_tokens_table(top_tokens)
        elif choice == '4':
            break
        else:
            console.print("[red]Invalid choice. Please try again.[/red]")

def main():
    while True:
        choice = display_main_menu()
        
        if choice == '1':
            collection_interval = timedelta(minutes=COLLECTION_INTERVAL)
            last_run_time = datetime.now() - collection_interval

            with Live(console=console, refresh_per_second=1) as live:
                while True:
                    current_time = datetime.now()
                    if current_time - last_run_time >= collection_interval:
                        live.stop()
                        new_data = collect_data()
                        data_manager.append_data(new_data)
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
                            view_results()
                        elif command == 't':
                            best_timeframe, best_result, recommendations = recommend_timeframe(data_manager)
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
        elif choice == '2':
            view_results()
        elif choice == '3':
            console.print(Panel(Text("Exiting the program.", style="bold red")))
            break
        else:
            console.print("[red]Invalid choice. Please try again.[/red]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print(Panel(Text("Script terminated by user.", style="bold red")))