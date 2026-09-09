#!/usr/bin/env python3
"""TerminusECE - Command-Driven Terminal Workspace for Electrical and Computer Engineering.
Supports interactive CLI REPL, SPICE/LTspice circuit modeling, Simulink dynamic systems,
and High-Capacity Multi-Terminal Session Networking (500+ terminals across LAN/devices).
"""

import argparse
import sys
import os
import socket

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from UI.app import TerminusApp, TerminusEngineBridge
from CORE.ipc_router import IPCRouter, ipc_router_instance, ipc_client_instance
from CORE.common_math import split_smart_statements

console = Console()


def strip_or_render_markup(text: str) -> None:
    """Renders text with rich ANSI colors and box drawings in standard terminal."""
    if text:
        console.print(text)


def run_cli_commands(commands: str, mode: str = "CIRCUIT", session_id: int = None, connect_addr: str = None) -> None:
    bridge = TerminusEngineBridge()
    if mode:
        bridge.switch_mode(mode)
    if connect_addr:
        bridge.execute_command(f"session connect {connect_addr} {session_id or ''}")

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


def run_cli_repl(
    initial_mode: str = "CIRCUIT",
    session_id: int = None,
    connect_addr: str = None,
    auto_server: bool = False
) -> None:
    """Interactive Command-by-Command CLI REPL Shell in standard terminal."""
    bridge = TerminusEngineBridge()
    if initial_mode:
        try:
            bridge.switch_mode(initial_mode)
        except Exception:
            pass

    if auto_server:
        bridge.execute_command("session server start")
    elif connect_addr:
        try:
            bridge.execute_command(f"session connect {connect_addr} {session_id or ''}")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not auto-connect to router {connect_addr}: {e}[/yellow]")
    else:
        # Attempt auto-connection to local router if running
        try:
            bridge.ipc_client.set_command_executor(bridge.execute_command)
            bridge.ipc_client.connect(mode=bridge.mode, active_file="untitled", requested_session_id=session_id)
        except Exception:
            pass

    console.print("[bold green]=== TerminusECE Interactive Command Line Shell ===[/bold green]")
    session_badge = f" [bold magenta][Session #{bridge.ipc_client.session_id}][/bold magenta]" if bridge.ipc_client.is_connected else ""
    console.print(f"Unified EDA command workspace.{session_badge} Type [bold cyan]'help'[/bold cyan] for commands, [bold cyan]'exit'[/bold cyan] to exit.")
    console.print("Quick Start: 'mode circuit' (LTspice), 'mode dynamic' (Simulink), 'mode kicad' (KiCad EDA), 'mode numerical' (MATLAB), 'mode digital' (Xilinx), 'mode embedded' (Arduino).\n")

    while True:
        try:
            sid_prompt = f"#{bridge.ipc_client.session_id} " if bridge.ipc_client.is_connected else ""
            prompt_str = f"Terminus [{bridge.mode}] {sid_prompt}> "
            line = input(prompt_str).strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit", "q"):
                console.print("[yellow]Exiting TerminusECE CLI. Goodbye![/yellow]")
                break
            if line.lower() == "clear" and bridge.mode not in ("DYNAMIC", "CIRCUIT", "KICAD"):
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


def run_script_file(file_path: str, mode: str = "CIRCUIT") -> None:
    if not os.path.exists(file_path):
        print(f"Error: Script file '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    run_cli_commands(content.replace("\n", ";"), mode=mode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TerminusECE - Terminal-based unified workspace for ECE"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="CIRCUIT",
        help="Initial subsystem mode (circuit, dynamic, numerical, digital, embedded, kicad)"
    )
    parser.add_argument(
        "--session", "-s",
        type=int,
        help="Designate specific Session ID number for this terminal instance"
    )
    parser.add_argument(
        "--connect",
        type=str,
        help="Auto-connect this terminal session to a network router (e.g. 192.168.1.50:8765)"
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the high-capacity IPC router network server on this computer"
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

    if args.daemon or (args.server and not args.cli and not args.cmd and not args.file):
        print("Starting TerminusECE High-Capacity IPC Router Server on 0.0.0.0:8765 (Ready for 500+ connections)...")
        router = IPCRouter()
        router.start()
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            router.stop()
            print("\nIPC Router Server stopped.")
        return

    if args.cmd:
        run_cli_commands(args.cmd, mode=args.mode, session_id=args.session, connect_addr=args.connect)
        return

    if args.file:
        run_script_file(args.file, mode=args.mode)
        return

    if args.tui:
        app = TerminusApp()
        app.run()
        return

    # Default: clean interactive CLI REPL
    run_cli_repl(
        initial_mode=args.mode,
        session_id=args.session,
        connect_addr=args.connect,
        auto_server=args.server
    )


if __name__ == "__main__":
    main()