# src/utils/console_utils.py

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.box import ROUNDED
import os
import sys
import select

console = Console()

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
    if os.name == 'nt':
        import msvcrt
        return msvcrt.kbhit()
    else:
        return select.select([sys.stdin], [], [], 0)[0]

def get_input():
    if os.name == 'nt':
        import msvcrt
        return msvcrt.getch().decode('utf-8').lower()
    else:
        return sys.stdin.readline().strip().lower()

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