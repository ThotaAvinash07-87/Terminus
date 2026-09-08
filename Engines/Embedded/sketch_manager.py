"""
Engines/Embedded/sketch_manager.py - Project Sketch Manager & Full Terminal Code Editor.
Manages multi-file Arduino & Embedded project directories:
Documents/Terminus Files/Embedded/<ProjectName>/<ProjectName>.ino, config.h, etc.
Provides full interactive multi-line terminal typing/pasting, direct buffer editing, and external editor integration.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import shutil
import os
import sys

from CORE.storage_manager import StorageManager


@dataclass
class ProjectFile:
    filename: str
    content: str
    is_main: bool = False


class ProjectSketchManager:
    """Manages multi-file embedded projects and the interactive terminal code editor buffer."""

    def __init__(self, project_name: str = "MyProject"):
        self.project_name = project_name
        self.storage = StorageManager.get_instance()
        self.files: Dict[str, ProjectFile] = {}
        self.active_filename = f"{project_name}.ino"
        self._init_default_sketch()

    def _init_default_sketch(self):
        main_content = (
            "// ==========================================\n"
            "// Terminus Embedded IDE - Arduino Sketch\n"
            "// ==========================================\n\n"
            "const int ledPin = 13;\n\n"
            "void setup() {\n"
            "  pinMode(ledPin, OUTPUT);\n"
            "  Serial.begin(115200);\n"
            "  Serial.println(\"System Initialized. Ready.\");\n"
            "}\n\n"
            "void loop() {\n"
            "  digitalWrite(ledPin, HIGH);\n"
            "  Serial.println(\"LED: HIGH\");\n"
            "  delay(500);\n"
            "  digitalWrite(ledPin, LOW);\n"
            "  Serial.println(\"LED: LOW\");\n"
            "  delay(500);\n"
            "}\n"
        )
        self.files[self.active_filename] = ProjectFile(
            filename=self.active_filename,
            content=main_content,
            is_main=True
        )

    @property
    def main_file(self) -> Optional[ProjectFile]:
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
            fname = f"{self.project_name}.ino"
            self.files[fname] = ProjectFile(filename=fname, content=val, is_main=True)
            self.active_filename = fname

    def new_project(self, project_name: str) -> None:
        self.project_name = project_name.strip() or "Untitled"
        self.files.clear()
        self.active_filename = f"{self.project_name}.ino"
        self._init_default_sketch()

    def get_project_dir(self) -> Path:
        base = self.storage.get_mode_dir("EMBEDDED")
        proj_dir = base / self.project_name
        proj_dir.mkdir(parents=True, exist_ok=True)
        return proj_dir

    def get_main_filepath(self) -> Path:
        return self.get_project_dir() / self.active_filename

    def save_project(self, custom_name: Optional[str] = None) -> Tuple[bool, str]:
        if custom_name:
            self.project_name = Path(custom_name).stem
            self.active_filename = f"{self.project_name}.ino"
            if self.main_file:
                self.main_file.filename = self.active_filename

        proj_dir = self.get_project_dir()
        saved_files = []
        for fname, pfile in self.files.items():
            fpath = proj_dir / fname
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(pfile.content)
            saved_files.append(fpath.name)

        return True, f"Project '{self.project_name}' saved into: {proj_dir} ({', '.join(saved_files)})"

    def load_project(self, target_path_or_name: str) -> Tuple[bool, str]:
        mode_dir = self.storage.get_mode_dir("EMBEDDED")
        p = Path(target_path_or_name)
        
        # Check if project folder exists
        candidate_dir = mode_dir / p.stem
        if candidate_dir.is_dir():
            self.project_name = candidate_dir.name
            self.files.clear()
            for fpath in candidate_dir.glob("*"):
                if fpath.is_file() and fpath.suffix.lower() in (".ino", ".cpp", ".c", ".h", ".py", ".asm"):
                    with open(fpath, "r", encoding="utf-8") as f:
                        is_main = (fpath.stem == self.project_name or fpath.suffix == ".ino")
                        self.files[fpath.name] = ProjectFile(filename=fpath.name, content=f.read(), is_main=is_main)
            if self.files:
                self.active_filename = next((k for k, v in self.files.items() if v.is_main), next(iter(self.files.keys())))
                return True, f"Loaded project '{self.project_name}' with {len(self.files)} source file(s) from {candidate_dir}."

        # Check single file
        resolved = self.storage.resolve_file_path("EMBEDDED", target_path_or_name)
        if resolved.is_file():
            self.project_name = resolved.parent.name if resolved.parent != mode_dir else resolved.stem
            self.active_filename = resolved.name
            with open(resolved, "r", encoding="utf-8") as f:
                self.files = {resolved.name: ProjectFile(filename=resolved.name, content=f.read(), is_main=True)}
            return True, f"Loaded sketch '{resolved.name}' into project '{self.project_name}'."

        return False, f"Project or file '{target_path_or_name}' not found in {mode_dir}."

    def add_file(self, filename: str, content: str = "") -> None:
        self.files[filename] = ProjectFile(filename=filename, content=content, is_main=False)

    def set_active_file(self, filename: str) -> bool:
        if filename in self.files:
            self.active_filename = filename
            return True
        return False

    def view_code(self) -> str:
        """Returns the formatted code buffer with line numbers and project header."""
        mf = self.files.get(self.active_filename) or self.main_file
        if not mf or not mf.content.strip():
            return f"[dim]Project '{self.project_name}' ({self.active_filename}) is empty. Type 'edit' or write code.[/dim]"

        lines = [
            f"[bold cyan]=== Project: {self.project_name} | File: {mf.filename} ({len(self.files)} files in project) ===[/bold cyan]",
            "  +-----+-------------------------------------------------------------------------+"
        ]
        for idx, line_text in enumerate(mf.content.splitlines(), start=1):
            lines.append(f"  | {idx:3d} | {line_text}")
        lines.append("  +-----+-------------------------------------------------------------------------+")
        return "\n".join(lines)

    def set_content(self, new_text: str) -> None:
        """Replaces current active file content entirely (for full pasting or typing)."""
        mf = self.files.get(self.active_filename) or self.main_file
        if mf:
            mf.content = new_text

    def append_content(self, text: str) -> None:
        mf = self.files.get(self.active_filename) or self.main_file
        if mf:
            mf.content = (mf.content + "\n" if mf.content else "") + text

    def launch_external_editor(self) -> Tuple[bool, str]:
        """Launches host GUI editor (Notepad / VS Code / nano) on the project file."""
        self.save_project()
        main_path = self.get_main_filepath()

        # Find available editor
        editor = os.environ.get("EDITOR")
        if not editor:
            if sys.platform.startswith("win"):
                editor = "notepad.exe"
            else:
                editor = "nano"

        try:
            subprocess.Popen([editor, str(main_path)])
            return True, f"Opened '{main_path.name}' in external editor ({editor}). When finished, save file and run 'compile'."
        except Exception as e:
            return False, f"Failed to open editor '{editor}': {str(e)}"
