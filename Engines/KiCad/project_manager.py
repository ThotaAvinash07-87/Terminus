"""KiCad Project Manager for TerminusECE.

Manages project directories in Documents/Terminus Files/KiCad/<ProjectName>/
containing .kicad_pro, .kicad_sch, .kicad_pcb, Gerber layers, and BOM.
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from CORE.storage_manager import StorageManager
from Engines.KiCad.schematic import KiCadSchematic
from Engines.KiCad.pcb_engine import KiCadPCB
from Engines.KiCad.gerber_exporter import GerberExporter


class KiCadProject:
    """Represents a complete Terminus KiCad EDA PCB design project."""

    def __init__(self, name: str = "MyPCBProject", storage: Optional[StorageManager] = None):
        self.name = name
        self.storage = storage or StorageManager.get_instance()
        self.schematic = KiCadSchematic(name=name)
        self.pcb = KiCadPCB(name=name)
        self.project_settings: Dict[str, Any] = {
            "meta": {
                "project_name": name,
                "version": "1.0",
                "generator": "TerminusECE_KiCad_EDA_Engine",
                "extensions": {
                    "project": ".tkcad",
                    "schematic": ".tsch",
                    "pcb": ".tpcb",
                    "bom": ".tbom",
                }
            },
            "board": {
                "design_settings": {
                    "rules": {
                        "min_clearance_mm": 0.2,
                        "min_track_width_mm": 0.15,
                        "min_via_diameter_mm": 0.6,
                        "min_via_drill_mm": 0.3
                    }
                }
            },
            "schematic": {
                "page_layout": "A4",
                "title": name
            }
        }

    def get_project_dir(self) -> Path:
        """Returns the unified project directory: Documents/Terminus Files/KiCad/<name>/"""
        mode_dir = self.storage.get_mode_dir("KICAD")
        proj_dir = mode_dir / self.name
        proj_dir.mkdir(parents=True, exist_ok=True)
        return proj_dir

    def save_project(self, project_name: Optional[str] = None) -> Tuple[bool, str]:
        """Saves all project files (.tkcad, .tsch, .tpcb, .tbom, gerbers) into unified project directory."""
        if project_name:
            self.name = project_name
            self.schematic.name = project_name
            self.pcb.name = project_name
            self.project_settings["meta"]["project_name"] = project_name

        proj_dir = self.get_project_dir()

        # 1. Save .tkcad Project Manifest
        tkcad_path = proj_dir / f"{self.name}.tkcad"
        with open(tkcad_path, "w", encoding="utf-8") as f:
            json.dump(self.project_settings, f, indent=2)

        # 2. Save .tsch Schematic
        tsch_path = proj_dir / f"{self.name}.tsch"
        with open(tsch_path, "w", encoding="utf-8") as f:
            f.write(self.schematic.export_kicad_sch_content())

        # 3. Save .tpcb PCB Layout
        tpcb_path = proj_dir / f"{self.name}.tpcb"
        with open(tpcb_path, "w", encoding="utf-8") as f:
            f.write(self.pcb.export_kicad_pcb_content())

        # 4. Save .tbom Bill of Materials
        tbom_path = proj_dir / f"{self.name}.tbom"
        bom_lines = ["Reference,Value,Footprint,Category,Quantity"]
        bom_counts: Dict[Tuple[str, str, str, str], List[str]] = {}
        for c in self.schematic.components.values():
            key = (c.value, c.footprint_name, c.comp_type, c.description)
            if key not in bom_counts:
                bom_counts[key] = []
            bom_counts[key].append(c.ref)
        for (val, fp, cat, desc), refs in bom_counts.items():
            bom_lines.append(f'"{"; ".join(refs)}","{val}","{fp}","{cat}",{len(refs)}')
        with open(tbom_path, "w", encoding="utf-8") as f:
            f.write("\n".join(bom_lines))

        # 5. Save Gerber layers
        gerber_dir = proj_dir / f"{self.name}-gerbers"
        GerberExporter.export_all_layers(self.pcb, self.schematic, gerber_dir)

        return True, f"Saved Terminus KiCad EDA project '{self.name}' to {proj_dir}\n  Files: {self.name}.tkcad, {self.name}.tsch, {self.name}.tpcb, {self.name}.tbom, {self.name}-gerbers/"

    def load_project(self, project_name: str) -> Tuple[bool, str]:
        """Loads a Terminus KiCad project from Documents/Terminus Files/KiCad/<name>/"""
        self.name = project_name
        proj_dir = self.get_project_dir()
        
        # Check .tkcad or legacy .kicad_pro
        tkcad_path = proj_dir / f"{project_name}.tkcad"
        if not tkcad_path.exists():
            tkcad_path = proj_dir / f"{project_name}.kicad_pro"
            if not tkcad_path.exists():
                return False, f"Project '{project_name}' not found at {proj_dir}"

        try:
            with open(tkcad_path, "r", encoding="utf-8") as f:
                self.project_settings = json.load(f)
        except Exception:
            pass

        self.schematic.name = project_name
        self.pcb.name = project_name
        return True, f"Loaded Terminus KiCad EDA project '{project_name}' ({len(self.schematic.components)} components, {len(self.pcb.tracks)} tracks)"

    def export_gerbers(self) -> Dict[str, Path]:
        """Generates all RS-274X Gerber layers, drill file, and BOM in project directory."""
        proj_dir = self.get_project_dir()
        gerber_dir = proj_dir / f"{self.name}-gerbers"
        return GerberExporter.export_all_layers(self.pcb, self.schematic, gerber_dir)
