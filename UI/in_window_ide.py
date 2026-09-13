"""In-Window Interactive IDE for TerminusECE 60% Split Screen.

Provides authentic in-place cursor navigation, character-by-character typing,
mouse click line jumping, arrow keys, backspace, enter, syntax shortcuts
(:w, :run, :mode, :q, :help), and live split-screen redrawing inside the 60% Left Window.
"""

from __future__ import annotations
import os
import sys
import time
import re
import ctypes
from typing import Optional, List, Tuple, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
console = Console(highlight=False)

# Windows Console Constants
ENABLE_PROCESSED_INPUT = 0x0001
ENABLE_LINE_INPUT = 0x0002
ENABLE_ECHO_INPUT = 0x0004
ENABLE_WINDOW_INPUT = 0x0008
ENABLE_MOUSE_INPUT = 0x0010
ENABLE_INSERT_MODE = 0x0020
ENABLE_QUICK_EDIT_MODE = 0x0040
ENABLE_EXTENDED_FLAGS = 0x0080
ENABLE_AUTO_POSITION = 0x0100
ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200

ENABLE_PROCESSED_OUTPUT = 0x0001
ENABLE_WRAP_AT_EOL_OUTPUT = 0x0002
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004


class WindowsConsoleController:
    """Manages Windows Terminal VT input, mouse capture, and cursor processing."""

    def __init__(self):
        self.is_windows = (os.name == "nt")
        self.h_in = None
        self.h_out = None
        self.orig_in_mode = ctypes.c_ulong()
        self.orig_out_mode = ctypes.c_ulong()
        self._configured = False

    def enable(self) -> bool:
        if not self.is_windows:
            return True
        try:
            kernel32 = ctypes.windll.kernel32
            GENERIC_READ = 0x80000000
            GENERIC_WRITE = 0x40000000
            FILE_SHARE_READ = 1
            FILE_SHARE_WRITE = 2
            OPEN_EXISTING = 3

            self.h_in = kernel32.CreateFileW("CONIN$", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING, 0, None)
            self.h_out = kernel32.CreateFileW("CONOUT$", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING, 0, None)

            if kernel32.GetConsoleMode(self.h_in, ctypes.byref(self.orig_in_mode)) and kernel32.GetConsoleMode(self.h_out, ctypes.byref(self.orig_out_mode)):
                # Enable VT Input; Disable QuickEdit, binary mouse input and window input
                new_in = (self.orig_in_mode.value | ENABLE_VIRTUAL_TERMINAL_INPUT | ENABLE_EXTENDED_FLAGS) & ~ENABLE_QUICK_EDIT_MODE & ~ENABLE_MOUSE_INPUT & ~ENABLE_WINDOW_INPUT
                new_out = self.orig_out_mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING | ENABLE_PROCESSED_OUTPUT
                kernel32.SetConsoleMode(self.h_in, new_in)
                kernel32.SetConsoleMode(self.h_out, new_out)
                self._configured = True
        except Exception:
            self._configured = False

        # Explicitly disable any mouse reporting sequences to keep stdin clean and prevent CPU spinning
        try:
            sys.stdout.write("\033[?1000l\033[?1002l\033[?1003l\033[?1006l\033[?1015l\033[?25h")
            sys.stdout.flush()
        except Exception:
            pass

        return self._configured

    def restore(self) -> None:
        try:
            sys.stdout.write("\033[?1000l\033[?1002l\033[?1003l\033[?1006l\033[?1015l\033[?25h")
            sys.stdout.flush()
        except Exception:
            pass

        if self.is_windows and self._configured:
            try:
                kernel32 = ctypes.windll.kernel32
                if self.h_in:
                    kernel32.SetConsoleMode(self.h_in, self.orig_in_mode)
                if self.h_out:
                    kernel32.SetConsoleMode(self.h_out, self.orig_out_mode)
            except Exception:
                pass


class InWindowIDE:
    """Manages full interactive in-window text editing inside the 60% Left Screen."""

    @classmethod
    def run(cls, bridge: Any) -> str:
        """Launches the in-window interactive editing loop with mouse and cursor support."""
        mode = getattr(bridge, "mode", "NUMERICAL").upper()
        if mode not in ("NUMERICAL", "MATLAB", "DIGITAL", "XILINX", "EMBEDDED", "ARDUINO_IDE"):
            return f"[yellow]Interactive In-Window IDE is available for NUMERICAL, DIGITAL, and EMBEDDED modes (Current: {mode}).[/yellow]"

        # Initial cursor coordinates (1-indexed line, 0-indexed column)
        cursor_line = 1
        cursor_col = 0
        status = f"In-Window IDE: {mode}. Click line or type. [F5 / :run = Run | Ctrl+S / :w = Save | ESC = Exit]"

        # Check Windows msvcrt
        has_msvcrt = False
        try:
            import msvcrt
            has_msvcrt = True
        except ImportError:
            has_msvcrt = False

        if not has_msvcrt:
            return cls._run_line_fallback(bridge)

        # Setup Windows Console for VT Input & Cursor
        console_ctl = WindowsConsoleController()
        console_ctl.enable()

        # Clear console once on startup
        os.system("cls" if os.name == "nt" else "clear")

        try:
            while True:
                current_mode = getattr(bridge, "mode", "NUMERICAL").upper()
                if current_mode not in ("NUMERICAL", "MATLAB", "DIGITAL", "XILINX", "EMBEDDED", "ARDUINO_IDE"):
                    break

                # Get code lines from active subsystem
                code_text = cls._get_code(bridge, current_mode)
                lines = code_text.splitlines()
                if not lines:
                    lines = [""]

                # Constrain cursor within valid bounds
                cursor_line = max(1, min(len(lines), cursor_line))
                current_line_text = lines[cursor_line - 1] if cursor_line <= len(lines) else ""
                cursor_col = max(0, min(len(current_line_text), cursor_col))

                # Set bridge cursor properties for renderer
                bridge.ide_cursor_line = cursor_line
                bridge.ide_cursor_col = cursor_col
                bridge.ide_is_active = True

                # Redraw full screen in-place and position physical terminal cursor directly on the line
                cls._redraw_screen(bridge, status, cursor_line, cursor_col, len(lines))

                # Read keystroke / mouse event
                try:
                    ch = msvcrt.getwch()
                except (KeyboardInterrupt, EOFError):
                    break

                # 1. Escape Sequences (\x1b...)
                if ch == '\x1b':
                    seq = ""
                    time.sleep(0.015)
                    while msvcrt.kbhit():
                        c = msvcrt.getwch()
                        seq += c
                        if c in ('~', 'A', 'B', 'C', 'D', 'H', 'F', 'P', 'Q', 'R', 'S'):
                            break

                    if seq in ("[A", "OA"):  # UP Arrow
                        if cursor_line > 1:
                            cursor_line -= 1
                            cursor_col = min(cursor_col, len(lines[cursor_line - 1]))
                        continue
                    elif seq in ("[B", "OB"):  # DOWN Arrow
                        if cursor_line < len(lines):
                            cursor_line += 1
                            cursor_col = min(cursor_col, len(lines[cursor_line - 1]))
                        continue
                    elif seq in ("[C", "OC"):  # RIGHT Arrow
                        if cursor_col < len(current_line_text):
                            cursor_col += 1
                        elif cursor_line < len(lines):
                            cursor_line += 1
                            cursor_col = 0
                        continue
                    elif seq in ("[D", "OD"):  # LEFT Arrow
                        if cursor_col > 0:
                            cursor_col -= 1
                        elif cursor_line > 1:
                            cursor_line -= 1
                            cursor_col = len(lines[cursor_line - 1])
                        continue
                    elif seq in ("[H", "[1~"):  # HOME
                        cursor_col = 0
                        continue
                    elif seq in ("[F", "[4~"):  # END
                        cursor_col = len(current_line_text)
                        continue
                    elif seq in ("[3~",):  # DELETE
                        if cursor_col < len(current_line_text):
                            new_line = current_line_text[:cursor_col] + current_line_text[cursor_col + 1:]
                            lines[cursor_line - 1] = new_line
                            cls._set_code(bridge, current_mode, "\n".join(lines) + "\n")
                        elif cursor_line < len(lines):
                            next_line = lines.pop(cursor_line)
                            lines[cursor_line - 1] += next_line
                            cls._set_code(bridge, current_mode, "\n".join(lines) + "\n")
                        continue
                    elif seq in ("[15~", "[17~", "[18~", "[19~"):  # F5-F8
                        status = cls._execute_run(bridge, current_mode)
                        continue
                    elif seq == "":
                        # Single ESC key: prompt in-window command bar
                        cmd_res = cls._prompt_in_window_command(bridge, current_mode, len(lines))
                        if cmd_res == "QUIT":
                            break
                        elif cmd_res:
                            status = cmd_res
                        continue
                    continue

                # 2. Windows Extended Function Keys (\x00 or \xe0)
                if ch in ('\x00', '\xe0'):
                    try:
                        ext = msvcrt.getwch()
                    except Exception:
                        continue

                    if ext == 'H':  # UP Arrow
                        if cursor_line > 1:
                            cursor_line -= 1
                            cursor_col = min(cursor_col, len(lines[cursor_line - 1]))
                    elif ext == 'P':  # DOWN Arrow
                        if cursor_line < len(lines):
                            cursor_line += 1
                            cursor_col = min(cursor_col, len(lines[cursor_line - 1]))
                    elif ext == 'K':  # LEFT Arrow
                        if cursor_col > 0:
                            cursor_col -= 1
                        elif cursor_line > 1:
                            cursor_line -= 1
                            cursor_col = len(lines[cursor_line - 1])
                    elif ext == 'M':  # RIGHT Arrow
                        if cursor_col < len(current_line_text):
                            cursor_col += 1
                        elif cursor_line < len(lines):
                            cursor_line += 1
                            cursor_col = 0
                    elif ext == 'G':  # HOME
                        cursor_col = 0
                    elif ext == 'O':  # END
                        cursor_col = len(current_line_text)
                    elif ext == 'S':  # DELETE key
                        if cursor_col < len(current_line_text):
                            new_line = current_line_text[:cursor_col] + current_line_text[cursor_col + 1:]
                            lines[cursor_line - 1] = new_line
                            cls._set_code(bridge, current_mode, "\n".join(lines) + "\n")
                        elif cursor_line < len(lines):
                            next_line = lines.pop(cursor_line)
                            lines[cursor_line - 1] += next_line
                            cls._set_code(bridge, current_mode, "\n".join(lines) + "\n")
                    elif ext in ('?', ';', '<'):  # F5 / F6 key -> Compile and Run
                        status = cls._execute_run(bridge, current_mode)
                    elif ext == 'I':  # Page UP
                        cursor_line = max(1, cursor_line - 10)
                        cursor_col = min(cursor_col, len(lines[cursor_line - 1]))
                    elif ext == 'Q':  # Page DOWN
                        cursor_line = min(len(lines), cursor_line + 10)
                        cursor_col = min(cursor_col, len(lines[cursor_line - 1]))
                    continue

                # 3. Control Hotkeys
                if ch == '\x13':  # Ctrl+S -> Save
                    cls._save_code(bridge, current_mode)
                    status = f"Saved {current_mode} script to disk."
                    continue
                elif ch == '\x12':  # Ctrl+R -> Run
                    status = cls._execute_run(bridge, current_mode)
                    continue
                elif ch == '\x11':  # Ctrl+Q -> Quit
                    break
                elif ch == '\r':  # ENTER key -> Split line and insert new line
                    left_part = current_line_text[:cursor_col]
                    right_part = current_line_text[cursor_col:]
                    lines[cursor_line - 1] = left_part
                    lines.insert(cursor_line, right_part)
                    cursor_line += 1
                    cursor_col = 0
                    cls._set_code(bridge, current_mode, "\n".join(lines) + "\n")
                    status = f"Ln {cursor_line}, Col {cursor_col + 1}"
                    continue
                elif ch == '\b':  # BACKSPACE key
                    if cursor_col > 0:
                        new_line = current_line_text[:cursor_col - 1] + current_line_text[cursor_col:]
                        lines[cursor_line - 1] = new_line
                        cursor_col -= 1
                        cls._set_code(bridge, current_mode, "\n".join(lines) + "\n")
                    elif cursor_line > 1:
                        prev_line = lines[cursor_line - 2]
                        cursor_col = len(prev_line)
                        lines[cursor_line - 2] = prev_line + current_line_text
                        lines.pop(cursor_line - 1)
                        cursor_line -= 1
                        cls._set_code(bridge, current_mode, "\n".join(lines) + "\n")
                    status = f"Ln {cursor_line}, Col {cursor_col + 1}"
                    continue
                elif ch == '\t':  # TAB key -> 4 spaces
                    insert_str = "    "
                    new_line = current_line_text[:cursor_col] + insert_str + current_line_text[cursor_col:]
                    lines[cursor_line - 1] = new_line
                    cursor_col += len(insert_str)
                    cls._set_code(bridge, current_mode, "\n".join(lines) + "\n")
                    continue

                # 4. In-Window Command Bar Trigger (:run, :mode <name>, :w, :q, :help)
                if ch == ':' and cursor_col == 0 and not current_line_text.strip():
                    cmd_res = cls._prompt_in_window_command(bridge, current_mode, len(lines))
                    if cmd_res == "QUIT":
                        break
                    elif cmd_res:
                        status = cmd_res
                    continue

                # 5. Normal Character Typing directly into the 60% Left Window
                if ord(ch) >= 32:
                    new_line = current_line_text[:cursor_col] + ch + current_line_text[cursor_col:]
                    lines[cursor_line - 1] = new_line
                    cursor_col += 1
                    cls._set_code(bridge, current_mode, "\n".join(lines) + "\n")
                    status = f"Ln {cursor_line}, Col {cursor_col + 1} | Editing {current_mode}"

        finally:
            console_ctl.restore()

        bridge.ide_cursor_line = None
        bridge.ide_cursor_col = None
        bridge.ide_is_active = False
        return f"Exited In-Window IDE."

    @classmethod
    def _prompt_in_window_command(cls, bridge: Any, mode: str, total_lines: int) -> str:
        """Prompts for an in-window colon command right inside the bottom frame."""
        import shutil
        size = shutil.get_terminal_size(fallback=(105, 30))
        bottom_row = max(18, size.lines - 1)
        sys.stdout.write(f"\033[{bottom_row};1H\033[K[In-Window Command (:run, :w, :mode <name>, :help, :q)] : ")
        sys.stdout.flush()

        import msvcrt
        cmd = ""
        try:
            while True:
                ch = msvcrt.getwch()
                if ch in ('\r', '\n'):
                    break
                elif ch == '\x1b':  # ESC to cancel
                    return ""
                elif ch == '\b':  # Backspace
                    if cmd:
                        cmd = cmd[:-1]
                        sys.stdout.write("\b \b")
                        sys.stdout.flush()
                elif ord(ch) >= 32:
                    cmd += ch
                    sys.stdout.write(ch)
                    sys.stdout.flush()
        except Exception:
            return ""

        cmd = cmd.strip()
        if not cmd:
            return ""

        low = cmd.lower().lstrip(":")

        if low in ("q", "quit", "exit"):
            return "QUIT"
        elif low in ("w", "save"):
            cls._save_code(bridge, mode)
            return f"Saved {mode} workspace to disk."
        elif low in ("run", "sim", "compile", "exec"):
            return cls._execute_run(bridge, mode)
        elif low in ("help", "man", "?"):
            bridge.show_help_manual = not getattr(bridge, "show_help_manual", False)
            return "Toggled Help Manual in Right Window."
        elif low in ("cls", "clear"):
            cls._set_code(bridge, mode, "")
            return "Cleared script buffer."
        elif low.startswith("mode "):
            target_mode = low.split()[1]
            try:
                bridge.switch_mode(target_mode)
                return f"Switched context to {bridge.mode} Studio"
            except Exception as e:
                return f"Error switching mode: {e}"
        elif low.startswith("edit "):
            ed_target = low.split(maxsplit=1)[1]
            if mode in ("NUMERICAL", "MATLAB"):
                ok, msg = bridge.matlab_proj.launch_external_editor(preferred_editor=ed_target)
            elif mode in ("DIGITAL", "XILINX"):
                ok, msg = bridge.logic_circuit.launch_external_editor(preferred_editor=ed_target)
            else:
                ok, msg = bridge.sketch_proj.launch_external_editor(preferred_editor=ed_target)
            return msg
        else:
            # Execute general bridge command
            clean_cmd = cmd.lstrip(":")
            res = bridge.execute_command(clean_cmd)
            first_l = res.splitlines()[0] if res else "OK"
            return re.sub(r'\[/?[a-zA-Z0-9_#\s,-]+\]', '', first_l)

    @classmethod
    def _redraw_screen(cls, bridge: Any, status: str, cursor_line: int, cursor_col: int, total_lines: int) -> None:
        """Renders split workspace in-place without screen clearing or flickering."""
        status_bar = f"{status} [Ln {cursor_line}/{max(1, total_lines)}, Col {cursor_col + 1}]"
        workspace_art = bridge.render_split_workspace(status_bar)

        # Move cursor to home and render smoothly in-place
        sys.stdout.write("\033[H")
        console.print(workspace_art)

        # Calculate exact hardware terminal cursor row & col
        term_row = 2 + cursor_line
        term_col = 9 + cursor_col
        sys.stdout.write(f"\033[?25h\033[{term_row};{term_col}H")
        sys.stdout.flush()

    @classmethod
    def _get_code(cls, bridge: Any, mode: str) -> str:
        if mode in ("NUMERICAL", "MATLAB"):
            bridge.matlab_proj.sync_from_disk()
            return bridge.matlab_proj.source_code
        elif mode in ("DIGITAL", "XILINX"):
            bridge.logic_circuit.sync_from_disk()
            return bridge.logic_circuit.source_code
        else:
            bridge.sketch_proj.sync_from_disk()
            return bridge.sketch_proj.source_code

    @classmethod
    def _set_code(cls, bridge: Any, mode: str, code: str) -> None:
        if mode in ("NUMERICAL", "MATLAB"):
            bridge.matlab_proj.set_content(code)
        elif mode in ("DIGITAL", "XILINX"):
            bridge.logic_circuit.source_code = code
            bridge.logic_circuit.save_to_disk()
        else:
            bridge.sketch_proj.set_content(code)

    @classmethod
    def _save_code(cls, bridge: Any, mode: str) -> None:
        if mode in ("NUMERICAL", "MATLAB"):
            bridge.matlab_proj.save_project()
        elif mode in ("DIGITAL", "XILINX"):
            bridge.logic_circuit.save_to_disk()
        else:
            bridge.sketch_proj.save_project()

    @classmethod
    def _execute_run(cls, bridge: Any, mode: str) -> str:
        cls._save_code(bridge, mode)
        if mode in ("NUMERICAL", "MATLAB"):
            res = bridge.execute_command("run")
        elif mode in ("DIGITAL", "XILINX"):
            res = bridge.execute_command("sim 100ns")
        elif mode in ("DYNAMIC", "SIMULINK"):
            res = bridge.execute_command("sim")
        elif mode in ("CIRCUIT", "LTSPICE"):
            res = bridge.execute_command("run")
        else:
            res = bridge.execute_command("compile")
        first_l = res.splitlines()[0] if res else "Executed"
        return re.sub(r'\[/?[a-zA-Z0-9_#\s,-]+\]', '', first_l)

    @classmethod
    def _run_line_fallback(cls, bridge: Any) -> str:
        """Line-by-line fallback when raw character reading is not available."""
        mode = getattr(bridge, "mode", "NUMERICAL").upper()
        cursor_line = 1
        bridge.ide_cursor_line = cursor_line
        status = f"Line IDE: {mode}. Type lines directly. Shortcuts: :q, :run, :w, :del <N>"

        while True:
            os.system("cls" if os.name == "nt" else "clear")
            workspace_art = bridge.render_split_workspace(status)
            from rich.console import Console
            Console().print(workspace_art)

            code = cls._get_code(bridge, mode)
            total_l = len(code.splitlines())

            prompt = f"[{mode} IDE Line {cursor_line:02d}/{max(1, total_l):02d}] (:q to exit)> "
            try:
                line_in = input(prompt)
            except (KeyboardInterrupt, EOFError):
                break

            stripped = line_in.strip()
            if not stripped and line_in == "":
                cursor_line = min(total_l + 1, cursor_line + 1)
                bridge.ide_cursor_line = cursor_line
                continue

            if stripped.lower() in (":q", ":quit", ":exit", "exit", "quit"):
                break

            if stripped.lower() in (":run", ":sim"):
                status = cls._execute_run(bridge, mode)
                continue

            if stripped.lower() in (":w", ":save"):
                cls._save_code(bridge, mode)
                status = "Saved to disk."
                continue

            # Normal line edit/append
            if mode in ("NUMERICAL", "MATLAB"):
                bridge.matlab_proj.edit_line(cursor_line, line_in)
            elif mode in ("DIGITAL", "XILINX"):
                bridge.logic_circuit.edit_line(cursor_line, line_in)
            else:
                bridge.sketch_proj.edit_line(cursor_line, line_in)

            cursor_line += 1
            bridge.ide_cursor_line = cursor_line
            status = f"Updated Line #{cursor_line - 1}."

        bridge.ide_cursor_line = None
        return "Exited Interactive IDE."
