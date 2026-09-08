"""Structured File Storage and Project Workspace Manager for TerminusECE.

Automatically initializes and maintains organized storage in the user's Documents folder:
C:\\Users\\<Username>\\Documents\\Terminus Files\\
├── Circuit/          (.cir, .net, .sp)
├── Numerical/        (.m, .mat, .num)
├── Dynamic_System/   (.tmdl, .json)
├── Digital_Logic/    (.v, .hdl)
├── Embedded/         (.asm, .c, .hex)
└── Exports/          (.csv, .dat, .txt)
"""

from __future__ import annotations
import os
import shutil
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class StorageManager:
    """Manages mode-specific file storage, structured naming, and directory lifecycle."""

    MODE_DIRECTORIES = {
        "CIRCUIT": ("Circuit", ".cir"),
        "NUMERICAL": ("Numerical", ".m"),
        "DYNAMIC": ("Dynamic_System", ".tmdl"),
        "DIGITAL": ("Digital_Logic", ".v"),
        "EMBEDDED": ("Embedded", ".ino"),
        "EXPORTS": ("Exports", ".csv"),
    }

    _instance: Optional[StorageManager] = None

    def __init__(self, root_override: Optional[str] = None):
        if root_override:
            self.root_dir = Path(root_override).resolve()
        else:
            self.root_dir = self._detect_documents_folder()

        self._ensure_directory_tree()

    @classmethod
    def get_instance(cls) -> StorageManager:
        if cls._instance is None:
            cls._instance = StorageManager()
        return cls._instance

    def _detect_documents_folder(self) -> Path:
        """Locates the Windows/OS Documents directory and initializes 'Terminus Files'."""
        try:
            if os.name == "nt":
                import ctypes.wintypes
                buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
                # CSIDL_PERSONAL = 5 (My Documents folder)
                ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf)
                if buf.value:
                    win_docs = Path(buf.value) / "Terminus Files"
                    win_docs.mkdir(parents=True, exist_ok=True)
                    return win_docs
            user_docs = Path.home() / "Documents" / "Terminus Files"
            user_docs.mkdir(parents=True, exist_ok=True)
            return user_docs
        except Exception:
            fallback = Path.home() / "Terminus Files"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    def _ensure_directory_tree(self) -> None:
        """Creates mode-specific subdirectories if they do not exist."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        for folder_name, _ in self.MODE_DIRECTORIES.values():
            folder_path = self.root_dir / folder_name
            folder_path.mkdir(parents=True, exist_ok=True)

    def get_mode_dir(self, mode: str) -> Path:
        m = mode.upper().strip()
        folder_name, _ = self.MODE_DIRECTORIES.get(m, ("Dynamic_System", ".tmdl"))
        target = self.root_dir / folder_name
        target.mkdir(parents=True, exist_ok=True)
        return target

    def get_default_ext(self, mode: str) -> str:
        m = mode.upper().strip()
        _, ext = self.MODE_DIRECTORIES.get(m, ("Dynamic_System", ".tmdl"))
        return ext

    def resolve_file_path(self, mode: str, filename: str) -> Path:
        """Resolves filename to absolute path, applying mode folder and default extension if needed."""
        p = Path(filename)
        if p.is_absolute():
            return p

        mode_dir = self.get_mode_dir(mode)
        default_ext = self.get_default_ext(mode)
        m_upper = mode.upper().strip()

        # Handle Embedded Arduino-style project directories: Embedded/<ProjectName>/<ProjectName>.ino
        if m_upper == "EMBEDDED":
            clean_stem = p.stem
            # Check if user specified a sub-path like "MyProject/MyProject.ino" or "MyProject/config.h"
            if len(p.parts) > 1:
                return mode_dir / filename
            # Check if project folder exists: Embedded/MyProject/MyProject.ino
            proj_dir = mode_dir / clean_stem
            proj_main_file = proj_dir / f"{clean_stem}{p.suffix or default_ext}"
            if proj_main_file.exists():
                return proj_main_file
            # Check flat file: Embedded/filename.ino
            if (mode_dir / filename).exists():
                return mode_dir / filename
            if not filename.endswith(default_ext) and (mode_dir / f"{filename}{default_ext}").exists():
                return mode_dir / f"{filename}{default_ext}"
            # By default for new files in EMBEDDED: create project folder Embedded/<ProjectName>/<ProjectName>.ino
            return proj_dir / f"{clean_stem}{p.suffix or default_ext}"

        # Standard resolution for other modes
        if (mode_dir / filename).exists():
            return mode_dir / filename

        if not filename.endswith(default_ext):
            with_ext = mode_dir / f"{filename}{default_ext}"
            if with_ext.exists():
                return with_ext

        if not p.suffix:
            return mode_dir / f"{filename}{default_ext}"
        return mode_dir / filename

    def list_files(self, mode: str) -> List[Dict[str, Any]]:
        """Lists all files and project directories stored in the mode's directory with metadata."""
        mode_dir = self.get_mode_dir(mode)
        results = []
        if not mode_dir.exists():
            return results

        # For EMBEDDED: also inspect project folders
        m_upper = mode.upper().strip()
        if m_upper == "EMBEDDED":
            for item in sorted(mode_dir.glob("**/*")):
                if item.is_file():
                    stat = item.stat()
                    mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
                    size_kb = stat.st_size / 1024.0
                    rel_name = str(item.relative_to(mode_dir)).replace("\\", "/")
                    results.append({
                        "name": rel_name,
                        "stem": item.stem,
                        "project": item.parent.name if item.parent != mode_dir else item.stem,
                        "size_bytes": stat.st_size,
                        "size_formatted": f"{size_kb:.1f} KB" if size_kb >= 1.0 else f"{stat.st_size} B",
                        "modified": mod_time,
                        "path": str(item),
                    })
            return results

        for item in sorted(mode_dir.glob("*")):
            if item.is_file():
                stat = item.stat()
                mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
                size_kb = stat.st_size / 1024.0
                results.append({
                    "name": item.name,
                    "stem": item.stem,
                    "size_bytes": stat.st_size,
                    "size_formatted": f"{size_kb:.1f} KB" if size_kb >= 1.0 else f"{stat.st_size} B",
                    "modified": mod_time,
                    "path": str(item),
                })
        return results

    def save_file(self, mode: str, filename: str, content: Union[str, bytes, dict]) -> Path:
        """Saves content into the mode's designated storage directory."""
        target_path = self.resolve_file_path(mode, filename)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, dict):
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2)
        elif isinstance(content, bytes):
            with open(target_path, "wb") as f:
                f.write(content)
        else:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(str(content))

        return target_path

    def load_file(self, mode: str, filename: str) -> Tuple[Path, Union[str, dict]]:
        """Loads file content from mode's directory or absolute path."""
        target_path = self.resolve_file_path(mode, filename)
        if not target_path.exists():
            raise FileNotFoundError(f"File '{filename}' not found in {self.get_mode_dir(mode)} or current path.")

        if target_path.suffix.lower() in (".json", ".tmdl"):
            with open(target_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    return target_path, data
                except Exception:
                    f.seek(0)
                    return target_path, f.read()
        else:
            with open(target_path, "r", encoding="utf-8") as f:
                return target_path, f.read()

    def rename_file(self, mode: str, old_name: str, new_name: str) -> Path:
        """Renames a file in the mode's storage folder."""
        old_path = self.resolve_file_path(mode, old_name)
        if not old_path.exists():
            raise FileNotFoundError(f"Source file '{old_name}' does not exist.")

        default_ext = self.get_default_ext(mode)
        if not Path(new_name).suffix:
            new_name = f"{new_name}{old_path.suffix or default_ext}"

        new_path = old_path.parent / new_name
        old_path.rename(new_path)
        return new_path

    def delete_file(self, mode: str, filename: str) -> bool:
        """Deletes a file from the mode's storage folder."""
        target_path = self.resolve_file_path(mode, filename)
        if target_path.exists() and target_path.is_file():
            target_path.unlink()
            return True
        return False

    def get_summary_table_data(self) -> List[Tuple[str, str, int, str]]:
        """Returns summary of all mode storage directories and file counts."""
        summary = []
        for mode_key, (folder_name, ext) in self.MODE_DIRECTORIES.items():
            folder_path = self.root_dir / folder_name
            file_count = len(list(folder_path.glob("*"))) if folder_path.exists() else 0
            summary.append((mode_key, folder_name, file_count, ext))
        return summary
