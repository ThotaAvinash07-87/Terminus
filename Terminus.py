#!/usr/bin/env python3
"""TerminusECE - Command-Driven Terminal Workspace for Electrical and Computer Engineering.
Supports interactive CLI REPL, SPICE/LTspice circuit modeling, Simulink dynamic systems,
and High-Capacity Multi-Terminal Session Networking (500+ terminals across LAN/devices).
"""

import argparse
import sys
import os
import socket
import re

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


import shutil

def prompt_welcome_mode_selection() -> str:
    """Renders authentic Terminus welcome hub and prompts user for their starting engineering mode."""
    os.system("cls" if os.name == "nt" else "clear")
    raw_size = shutil.get_terminal_size(fallback=(105, 30))
    w = max(80, raw_size.columns - 1)

    title_box = [
        "╔" + "═" * (w - 2) + "╗",
        "║" + "  ████████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗██╗   ██╗███████╗  ECE Engineering Studio v1.0".ljust(w - 2) + "║",
        "║" + "  ╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██║   ██║██╔════╝  Fixed Interactive Workspace".ljust(w - 2) + "║",
        "║" + "     ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║██║   ██║███████╗  High-Capacity Session Engine".ljust(w - 2) + "║",
        "║" + "     ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██║   ██║╚════██║  [Ready for 500+ Connections]".ljust(w - 2) + "║",
        "║" + "     ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║╚██████╔╝███████║".ljust(w - 2) + "║",
        "║" + "     ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝".ljust(w - 2) + "║",
        "╠" + "═" * (w - 2) + "╣",
        "║" + "  Welcome to Terminus! Select an Engineering Subsystem Mode to Launch:".ljust(w - 2) + "║",
        "║" + " ".ljust(w - 2) + "║",
        "║" + "    [1] CIRCUIT   (LTspice)     - LTspice IV/XVII Schematic & Waveform Studio".ljust(w - 2) + "║",
        "║" + "    [2] KICAD     (PCB / EDA)   - KiCad 8.0/10.0 PCB & Schematic Layout Studio".ljust(w - 2) + "║",
        "║" + "    [3] DYNAMIC   (Simulink)    - Simulink Multi-Domain Dynamic Modeling Engine".ljust(w - 2) + "║",
        "║" + "    [4] NUMERICAL (MATLAB / DSP)- MATLAB Signal Processing & Script Workspace".ljust(w - 2) + "║",
        "║" + "    [5] DIGITAL   (Xilinx)      - Xilinx Vivado HDL Logic & Timing Simulator".ljust(w - 2) + "║",
        "║" + "    [6] EMBEDDED  (Arduino IDE) - Arduino IDE Firmware Studio & Real Serial Telemetry".ljust(w - 2) + "║",
        "║" + " ".ljust(w - 2) + "║",
        "║" + "  Tip: You can switch modes anytime inside the workspace by entering 'mode <name>'.".ljust(w - 2) + "║",
        "╚" + "═" * (w - 2) + "╝",
    ]
    console.print("\n".join(f"[bold cyan]{l}[/bold cyan]" for l in title_box))

    try:
        choice = input("\nSelect Subsystem Mode [1-6 or name] (default: 1 - Circuit): ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        sys.exit(0)

    mode_map = {
        "1": "CIRCUIT", "circuit": "CIRCUIT", "ltspice": "CIRCUIT", "spice": "CIRCUIT",
        "2": "KICAD", "kicad": "KICAD", "pcb": "KICAD", "eda": "KICAD",
        "3": "DYNAMIC", "dynamic": "DYNAMIC", "simulink": "DYNAMIC",
        "4": "NUMERICAL", "numerical": "NUMERICAL", "matlab": "NUMERICAL", "dsp": "NUMERICAL",
        "5": "DIGITAL", "digital": "DIGITAL", "xilinx": "DIGITAL", "verilog": "DIGITAL",
        "6": "EMBEDDED", "embedded": "EMBEDDED", "arduino": "EMBEDDED", "arduino_ide": "EMBEDDED",
    }
    return mode_map.get(choice, "CIRCUIT")


def run_cli_repl(
    initial_mode: str = None,
    session_id: int = None,
    connect_addr: str = None,
    auto_server: bool = False
) -> None:
    """Interactive Command-by-Command In-Place Workspace Shell in standard terminal."""
    if not initial_mode:
        initial_mode = prompt_welcome_mode_selection()

    bridge = TerminusEngineBridge()
    try:
        bridge.switch_mode(initial_mode)
    except Exception:
        bridge.switch_mode("CIRCUIT")

    if auto_server:
        bridge.execute_command("session server start")
    elif connect_addr:
        try:
            bridge.execute_command(f"session connect {connect_addr} {session_id or ''}")
        except Exception as e:
            pass
    else:
        # Attempt auto-connection to local router if running
        try:
            bridge.ipc_client.set_command_executor(bridge.execute_command)
            bridge.ipc_client.connect(mode=bridge.mode, active_file="untitled", requested_session_id=session_id)
        except Exception:
            pass

    status_msg = f"Ready. Operating Mode: {bridge.mode}"

    while True:
        try:
            # Clear terminal screen and redraw the full fixed interactive workspace in-place
            os.system("cls" if os.name == "nt" else "clear")
            workspace_art = bridge.render_split_workspace(status_msg)
            strip_or_render_markup(workspace_art)

            sid_prompt = f"#{bridge.ipc_client.session_id} " if bridge.ipc_client.is_connected else ""
            prompt_str = f"Terminus [{bridge.mode}] {sid_prompt}> "
            line = input(prompt_str).strip()

            if not line:
                continue

            # Check exit
            if line.lower() in ("exit", "quit", "q"):
                if bridge.has_unsaved_work():
                    save_ans = input(f"\nSave current {bridge.mode} workspace before exiting? (y/n): ").strip().lower()
                    if save_ans in ("y", "yes"):
                        save_out = bridge.execute_command("file save")
                        print(save_out)
                print("\nExiting TerminusECE CLI. Goodbye!")
                break

            # Check mode switch
            tokens = line.split()
            if tokens[0].lower() == "mode" and len(tokens) > 1:
                target_mode = tokens[1].upper()
                if bridge.has_unsaved_work():
                    save_ans = input(f"\nSave current {bridge.mode} workspace before switching to {target_mode}? (y/n): ").strip().lower()
                    if save_ans in ("y", "yes"):
                        save_out = bridge.execute_command("file save")
                        status_msg = f"Saved {bridge.mode} workspace. "
                try:
                    new_m = bridge.switch_mode(tokens[1])
                    status_msg = f"Switched context to {new_m} Studio"
                    continue
                except Exception as e:
                    status_msg = f"Error switching mode: {e}"
                    continue

            # Execute command
            res = bridge.execute_command(line)
            first_l = res.splitlines()[0] if res else "OK"
            status_msg = cls_first_l = re.sub(r'\[/?[a-zA-Z0-9_#\s,-]+\]', '', first_l)

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.")
            break
        except Exception as err:
            status_msg = f"Error: {err}"


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
        default=None,
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