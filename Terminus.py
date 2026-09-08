#!/usr/bin/env python3
"""TerminusECE - Command-Driven Terminal Workspace for Electrical and Computer Engineering."""

import argparse
import sys
import os

from UI.app import TerminusApp, TerminusEngineBridge
from CORE.ipc_router import IPCRouter
from CORE.common_math import split_smart_statements


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


from rich.console import Console

console = Console()


def strip_or_render_markup(text: str) -> None:
    """Renders text with rich ANSI colors and box drawings in standard terminal."""
    if text:
        console.print(text)


def run_cli_commands(commands: str) -> None:
    bridge = TerminusEngineBridge()
    for line in split_smart_statements(commands, ";"):
        line = line.strip()
        if not line:
            continue
        try:
            res = bridge.execute_command(line)
            if res:
                strip_or_render_markup(res)
        except Exception as e:
            print(f"Error executing '{line}': {e}", file=sys.stderr)


def run_cli_repl() -> None:
    """Interactive Command-by-Command CLI REPL Shell in standard terminal."""
    bridge = TerminusEngineBridge()
    console.print("[bold green]=== TerminusECE Interactive Command Line Shell ===[/bold green]")
    console.print("Unified EDA command workspace. Type [bold cyan]'help'[/bold cyan] for commands, [bold cyan]'exit'[/bold cyan] or [bold cyan]'quit'[/bold cyan] to exit.")
    console.print("Quick Start: Type [bold yellow]'mode dynamic'[/bold yellow] to enter Simulink Dynamic Systems mode.\n")

    while True:
        try:
            prompt_str = f"Terminus [{bridge.mode}] > "
            line = input(prompt_str).strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit", "q"):
                console.print("[yellow]Exiting TerminusECE CLI. Goodbye![/yellow]")
                break
            if line.lower() == "clear" and bridge.mode != "DYNAMIC":
                os.system("cls" if os.name == "nt" else "clear")
                continue

            res = bridge.execute_command(line)
            if res:
                strip_or_render_markup(res)
                print()  # Empty line separator
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session interrupted. Exiting.[/yellow]")
            break
        except Exception as err:
            console.print(f"[bold red]Error:[/bold red] {err}\n")


def run_script_file(file_path: str) -> None:
    if not os.path.exists(file_path):
        print(f"Error: Script file '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    run_cli_commands(content.replace("\n", ";"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TerminusECE - Terminal-based unified workspace for ECE"
    )
    parser.add_argument(
        "--tui", "--gui",
        action="store_true",
        help="Start in optional Textual full-screen TUI mode"
    )
    parser.add_argument(
        "--cli", "-i", "--repl",
        action="store_true",
        help="Start in interactive command-by-command CLI REPL mode"
    )
    parser.add_argument(
        "--cmd", "-c",
        type=str,
        help="Execute semicolon-separated commands in CLI batch mode and exit"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Execute commands from a script file and exit"
    )
    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Start the background IPC synchronization server daemon"
    )

    args = parser.parse_args()

    if args.daemon:
        print("Starting TerminusECE IPC Daemon on 127.0.0.1:8765...")
        router = IPCRouter()
        router.start_background()
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            router.stop()
            print("\nDaemon stopped.")
        return

    if args.cmd:
        run_cli_commands(args.cmd)
        return

    if args.file:
        run_script_file(args.file)
        return

    if args.tui:
        app = TerminusApp()
        app.run()
        return

    # Default: clean interactive CLI REPL
    run_cli_repl()


if __name__ == "__main__":
    main()