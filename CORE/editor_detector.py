"""CORE/editor_detector.py - Host Code Editor Detector & Launcher for TerminusECE.

Discovers installed text and code editors on Windows / Linux / macOS (Notepad, VS Code,
Cursor, Notepad++, Antigravity, Sublime Text, nano, vim), launches them asynchronously,
and tracks file timestamps for seamless live-reload into Terminus workspaces.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import shutil
import sys
import os


@dataclass
class EditorInfo:
    id: str
    name: str
    command: str
    args: List[str]
    is_available: bool = False
    path: Optional[str] = None


class HostEditorManager:
    """Detects available code editors on the user system and manages launch & file watching."""

    @classmethod
    def get_candidate_editors(cls) -> List[EditorInfo]:
        """Returns list of popular editor definitions with platform-specific candidate paths."""
        candidates = []

        if sys.platform.startswith("win"):
            # 1. Notepad (Always available on Windows)
            candidates.append(EditorInfo(id="notepad", name="Notepad (Default Fast)", command="notepad.exe", args=[], is_available=True))

            # 2. Visual Studio Code
            local_appdata = os.environ.get("LOCALAPPDATA", "")
            prog_files = os.environ.get("ProgramFiles", "")
            code_paths = [
                shutil.which("code.cmd"),
                shutil.which("code"),
                shutil.which("code.exe"),
                os.path.join(local_appdata, "Programs", "Microsoft VS Code", "bin", "code.cmd"),
                os.path.join(local_appdata, "Programs", "Microsoft VS Code", "Code.exe"),
                os.path.join(prog_files, "Microsoft VS Code", "bin", "code.cmd"),
                os.path.join(prog_files, "Microsoft VS Code", "Code.exe"),
            ]
            found_code = next((p for p in code_paths if p and os.path.exists(p)), None)
            candidates.append(EditorInfo(
                id="vscode",
                name="Visual Studio Code",
                command=found_code or "code",
                args=[] if found_code else [],
                is_available=bool(found_code or shutil.which("code")),
                path=found_code
            ))

            # 3. Cursor AI Editor
            cursor_paths = [
                shutil.which("cursor.cmd"),
                shutil.which("cursor"),
                shutil.which("cursor.exe"),
                os.path.join(local_appdata, "Programs", "cursor", "Cursor.exe"),
                os.path.join(local_appdata, "Programs", "cursor", "resources", "app", "bin", "cursor.cmd"),
                os.path.join(local_appdata, "Programs", "Cursor", "bin", "cursor.cmd"),
            ]
            found_cursor = next((p for p in cursor_paths if p and os.path.exists(p)), None)
            candidates.append(EditorInfo(
                id="cursor",
                name="Cursor AI Code Editor",
                command=found_cursor or "cursor",
                args=[],
                is_available=bool(found_cursor or shutil.which("cursor")),
                path=found_cursor
            ))

            # 4. Notepad++
            npp_paths = [
                shutil.which("notepad++.exe"),
                shutil.which("notepad++"),
                os.path.join(prog_files, "Notepad++", "notepad++.exe"),
                os.path.join(os.environ.get("ProgramFiles(x86)", ""), "Notepad++", "notepad++.exe"),
            ]
            found_npp = next((p for p in npp_paths if p and os.path.exists(p)), None)
            candidates.append(EditorInfo(
                id="notepad++",
                name="Notepad++",
                command=found_npp or "notepad++",
                args=[],
                is_available=bool(found_npp or shutil.which("notepad++")),
                path=found_npp
            ))

            # 5. Antigravity IDE / CLI
            agy_paths = [
                shutil.which("antigravity.cmd"),
                shutil.which("antigravity.exe"),
                shutil.which("agy.cmd"),
                shutil.which("agy.exe"),
                os.path.expanduser(r"~\.gemini\antigravity-ide\antigravity.cmd"),
                os.path.expanduser(r"~\.gemini\antigravity-ide\antigravity.exe"),
            ]
            found_agy = next((p for p in agy_paths if p and os.path.exists(p)), None)
            candidates.append(EditorInfo(
                id="antigravity",
                name="Antigravity IDE",
                command=found_agy or "antigravity",
                args=[],
                is_available=bool(found_agy or shutil.which("antigravity")),
                path=found_agy
            ))

            # 6. Sublime Text
            subl_paths = [
                shutil.which("subl.exe"),
                shutil.which("subl"),
                os.path.join(prog_files, "Sublime Text", "sublime_text.exe"),
                os.path.join(prog_files, "Sublime Text 3", "sublime_text.exe"),
            ]
            found_subl = next((p for p in subl_paths if p and os.path.exists(p)), None)
            candidates.append(EditorInfo(
                id="sublime",
                name="Sublime Text",
                command=found_subl or "subl",
                args=[],
                is_available=bool(found_subl or shutil.which("subl")),
                path=found_subl
            ))
        else:
            # Linux / macOS
            candidates.append(EditorInfo(id="nano", name="GNU nano", command="nano", args=[], is_available=bool(shutil.which("nano"))))
            candidates.append(EditorInfo(id="vim", name="Vim Editor", command="vim", args=[], is_available=bool(shutil.which("vim"))))
            candidates.append(EditorInfo(id="vscode", name="Visual Studio Code", command="code", args=[], is_available=bool(shutil.which("code"))))
            candidates.append(EditorInfo(id="cursor", name="Cursor AI Editor", command="cursor", args=[], is_available=bool(shutil.which("cursor"))))
            candidates.append(EditorInfo(id="gedit", name="GNOME Text Editor (gedit)", command="gedit", args=[], is_available=bool(shutil.which("gedit"))))

        return candidates

    @classmethod
    def get_available_editors(cls) -> List[EditorInfo]:
        """Returns only editors currently installed and available on system."""
        candidates = cls.get_candidate_editors()
        available = []
        for cand in candidates:
            if cand.id == "notepad" and sys.platform.startswith("win"):
                cand.is_available = True
                available.append(cand)
            elif cand.is_available:
                available.append(cand)
            elif shutil.which(cand.command):
                cand.is_available = True
                available.append(cand)
        return available

    @classmethod
    def format_editor_menu(cls) -> str:
        """Formats an interactive selection menu of available editors."""
        editors = cls.get_available_editors()
        lines = [
            "[bold cyan]=== Select Code Editor to Open Project ===[/bold cyan]",
            "Detected Installed Editors on Your System:"
        ]
        for idx, ed in enumerate(editors, 1):
            tag = "[bold green](Installed)[/bold green]" if ed.is_available else "[dim](Not detected)[/dim]"
            lines.append(f"  [{idx}] [bold white]{ed.name:28s}[/bold white] {tag}")
        lines.append(f"\nEnter number [1-{len(editors)}] or editor name (e.g. 'code', 'cursor', 'notepad'):")
        return "\n".join(lines)

    @classmethod
    def launch(cls, filepath: Path, preferred_editor: Optional[str] = None) -> Tuple[bool, str]:
        """Launches the preferred or detected editor asynchronously on target file."""
        if not filepath.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.touch()

        editors = cls.get_available_editors()
        target_ed = None

        if preferred_editor:
            pref = preferred_editor.lower().strip()
            # Support numeric selection e.g. "1", "2"
            if pref.isdigit():
                idx = int(pref) - 1
                if 0 <= idx < len(editors):
                    target_ed = editors[idx]

            if not target_ed:
                for ed in editors:
                    if pref in (ed.id.lower(), ed.name.lower(), ed.command.lower()):
                        target_ed = ed
                        break

        if not target_ed:
            # Use environment variable or first available editor (VS Code / Cursor / Notepad)
            env_ed = os.environ.get("EDITOR", "").lower()
            if env_ed:
                for ed in editors:
                    if env_ed in ed.command.lower() or env_ed in ed.id:
                        target_ed = ed
                        break

        if not target_ed:
            # Default to first available
            target_ed = editors[0] if editors else EditorInfo(id="notepad", name="Notepad", command="notepad.exe", args=[], is_available=True)

        cmd = [target_ed.command] + target_ed.args + [str(filepath)]
        try:
            subprocess.Popen(cmd, shell=False)
            return True, f"Launched {target_ed.name} on '{filepath.name}'. Save changes in your editor (Ctrl+S) for live in-workspace update."
        except Exception as e:
            # Fallback to notepad / default
            if sys.platform.startswith("win") and target_ed.id != "notepad":
                try:
                    subprocess.Popen(["notepad.exe", str(filepath)])
                    return True, f"Launched Notepad on '{filepath.name}' (fallback). Save file for live update."
                except Exception as ex:
                    return False, f"Could not launch editor: {ex}"
            return False, f"Could not launch editor '{target_ed.command}': {e}"
