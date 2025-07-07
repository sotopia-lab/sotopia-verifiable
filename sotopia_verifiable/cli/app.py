"""Command-line interface for Sotopia Verifiable."""

import sys
from typing import List, Optional

from rich.console import Console

console = Console()


def app(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        argv: Command line arguments
        
    Returns:
        Exit code
    """
    if argv is None:
        argv = sys.argv[1:]
    
    console.print("[bold green]Sotopia Verifiable[/bold green]")
    console.print("A collection of verifiable games in Sotopia format")
    
    if not argv:
        console.print("\n[bold]Available commands:[/bold]")
        console.print("  verify    - Verify a game outcome")
        console.print("  list      - List available verification rules")
        console.print("\nRun with --help for more information.")
        return 0
    
    command = argv[0]
    
    if command == "verify":
        return _handle_verify(argv[1:])
    elif command == "list":
        return _handle_list(argv[1:])
    elif command in ["--help", "-h", "help"]:
        _print_help()
        return 0
    else:
        console.print(f"[bold red]Unknown command:[/bold red] {command}")
        console.print("Run with --help for available commands.")
        return 1


def _handle_verify(args: List[str]) -> int:
    """Handle the verify command.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    console.print("[bold]Verify command[/bold] (not implemented yet)")
    return 0


def _handle_list(args: List[str]) -> int:
    """Handle the list command.
    
    Args:
        args: Command arguments
        
    Returns:
        Exit code
    """
    console.print("[bold]List command[/bold] (not implemented yet)")
    return 0


def _print_help() -> None:
    """Print help information."""
    console.print("[bold]Sotopia Verifiable CLI[/bold]")
    console.print("\n[bold]Usage:[/bold]")
    console.print("  sotopia-verifiable [command] [options]")
    
    console.print("\n[bold]Commands:[/bold]")
    console.print("  verify    - Verify a game outcome")
    console.print("  list      - List available verification rules")
    
    console.print("\n[bold]Options:[/bold]")
    console.print("  --help, -h    - Show this help message")


if __name__ == "__main__":
    sys.exit(app())