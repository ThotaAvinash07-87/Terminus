"""
Engines/Numerical/matlab_manager.py - MATLAB Project & Script Manager for TerminusECE.

Manages multi-file MATLAB-style scripts and projects:
Documents/Terminus Files/Numerical/<ProjectName>/<ProjectName>.m
Provides script loading, saving, multi-line editing, syntax formatting, and execution.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import os
import sys

from CORE.storage_manager import StorageManager


@dataclass
class MatlabProjectFile:
    filename: str
    content: str
    is_main: bool = False


class MatlabProjectManager:
    """Manages MATLAB project folders, scripts (.m), and the interactive editor buffer."""

    def __init__(self, project_name: str = "SignalWorkspace"):
        self.project_name = project_name
        self.storage = StorageManager.get_instance()
        self.files: Dict[str, MatlabProjectFile] = {}
        self.active_filename = f"{project_name}.m"
        self._init_default_script()

    def _init_default_script(self):
        main_content = (
            "% ==========================================\n"
            "% Terminus MATLAB / Numerical Script\n"
            "% Signal Processing & Dynamic Analysis\n"
            "% ==========================================\n\n"
            "% 1. Time Vector & Signal Generation\n"
            "Fs = 1000;\n"
            "t = 0:1/Fs:0.1;\n"
            "f1 = 50;\n"
            "f2 = 120;\n"
            "x = 2.5 * sin(2*pi*f1*t) + 1.2 * sin(2*pi*f2*t);\n\n"
            "% 2. Visualization\n"
            "plot(t, x);\n"
        )
        self.files[self.active_filename] = MatlabProjectFile(
            filename=self.active_filename,
            content=main_content,
            is_main=True
        )

    @property
    def main_file(self) -> Optional[MatlabProjectFile]:
        for f in self.files.values():
            if f.is_main:
                return f
        if self.files:
            return next(iter(self.files.values()))
        return None

    @property
    def source_code(self) -> str:
        mf = self.main_file
        return mf.content if mf else ""

    @source_code.setter
    def source_code(self, val: str):
        mf = self.main_file
        if mf:
            mf.content = val
        else:
            fname = f"{self.project_name}.m"
            self.files[fname] = MatlabProjectFile(filename=fname, content=val, is_main=True)
            self.active_filename = fname

    def new_project(self, project_name: str) -> None:
        self.project_name = project_name.strip() or "Untitled"
        self.files.clear()
        self.active_filename = f"{self.project_name}.m"
        self._init_default_script()

    def get_project_dir(self) -> Path:
        base = self.storage.get_mode_dir("NUMERICAL")
        proj_dir = base / self.project_name
        proj_dir.mkdir(parents=True, exist_ok=True)
        return proj_dir

    def get_main_filepath(self) -> Path:
        return self.get_project_dir() / self.active_filename

    def save_project(self, custom_name: Optional[str] = None) -> Tuple[bool, str]:
        if custom_name:
            self.project_name = Path(custom_name).stem
            self.active_filename = f"{self.project_name}.m"
            if self.main_file:
                self.main_file.filename = self.active_filename

        proj_dir = self.get_project_dir()
        saved_files = []
        for fname, pfile in self.files.items():
            fpath = proj_dir / fname
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(pfile.content)
            saved_files.append(fpath.name)

        return True, f"MATLAB Project '{self.project_name}' saved into: {proj_dir} ({', '.join(saved_files)})"

    def load_project(self, target_path_or_name: str) -> Tuple[bool, str]:
        mode_dir = self.storage.get_mode_dir("NUMERICAL")
        candidate = mode_dir / target_path_or_name

        if candidate.is_dir():
            proj_dir = candidate
        elif candidate.with_suffix(".m").is_file():
            # Old single file fallback
            with open(candidate.with_suffix(".m"), "r", encoding="utf-8") as f:
                c = f.read()
            self.new_project(candidate.stem)
            self.set_content(c)
            return True, f"Loaded MATLAB script from {candidate.with_suffix('.m')}"
        elif (mode_dir / target_path_or_name).is_dir():
            proj_dir = mode_dir / target_path_or_name
        else:
            p_obj = Path(target_path_or_name)
            if p_obj.is_dir():
                proj_dir = p_obj
            elif p_obj.is_file():
                self.new_project(p_obj.stem)
                with open(p_obj, "r", encoding="utf-8") as f:
                    self.set_content(f.read())
                return True, f"Loaded MATLAB script '{p_obj.name}'."
            else:
                return False, f"MATLAB Project folder '{target_path_or_name}' not found in {mode_dir}."

        self.project_name = proj_dir.name
        self.files.clear()
        m_files = list(proj_dir.glob("*.m"))
        if not m_files:
            self._init_default_script()
            return True, f"Loaded empty project folder '{self.project_name}'."

        main_target = f"{self.project_name}.m"
        has_named_main = any(f.name == main_target for f in m_files)

        for mf in m_files:
            with open(mf, "r", encoding="utf-8") as f:
                content = f.read()
            is_m = (mf.name == main_target) if has_named_main else (mf == m_files[0])
            self.files[mf.name] = MatlabProjectFile(filename=mf.name, content=content, is_main=is_m)
            if is_m:
                self.active_filename = mf.name

        return True, f"Loaded MATLAB project '{self.project_name}' with {len(self.files)} file(s) from {proj_dir}."

    def add_file(self, filename: str, content: str = "") -> None:
        self.files[filename] = MatlabProjectFile(filename=filename, content=content, is_main=False)

    def set_content(self, content: str, filename: Optional[str] = None):
        target = filename or self.active_filename
        if target not in self.files:
            self.files[target] = MatlabProjectFile(filename=target, content=content, is_main=(target == self.active_filename))
        else:
            self.files[target].content = content

    def append_content(self, line_text: str, filename: Optional[str] = None):
        target = filename or self.active_filename
        if target not in self.files:
            self.set_content(line_text + "\n", target)
        else:
            self.files[target].content += ("\n" if not self.files[target].content.endswith("\n") else "") + line_text + "\n"

    def view_code(self, filename: Optional[str] = None) -> str:
        target = filename or self.active_filename
        pfile = self.files.get(target)
        if not pfile or not pfile.content:
            return f"[yellow]File '{target}' in project '{self.project_name}' is empty.[/yellow]"

        lines = pfile.content.splitlines()
        header = f"[bold cyan]=== Project: {self.project_name} | File: {target} ({len(lines)} lines) ===[/bold cyan]"
        formatted = [header]
        for i, l in enumerate(lines, 1):
            formatted.append(f"  [dim]{i:>4} |[/dim] {l}")
        return "\n".join(formatted)

    def launch_external_editor(self, filename: Optional[str] = None) -> Tuple[bool, str]:
        self.save_project()
        target = filename or self.active_filename
        target_path = self.get_project_dir() / target

        if sys.platform.startswith("win"):
            editor = os.environ.get("EDITOR", "notepad.exe")
        else:
            editor = os.environ.get("EDITOR", "nano")

        try:
            subprocess.Popen([editor, str(target_path)])
            return True, f"Launched external editor '{editor}' on {target_path}. Re-run 'run' after editing."
        except Exception as e:
            return False, f"Could not launch editor '{editor}': {e}"
