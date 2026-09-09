"""KiCad RS-274X Gerber & Excellon Drill Generator for TerminusECE.

Generates standard manufacturing files (.gbr, .drl) and Bill of Materials (BOM)
compatible with major PCB fabrication houses (JLCPCB, PCBWay, OSH Park).
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Engines.KiCad.schematic import KiCadSchematic
from Engines.KiCad.pcb_engine import KiCadPCB


class GerberExporter:
    """RS-274X Gerber & Excellon Drill Generator."""

    @classmethod
    def export_all_layers(
        cls,
        pcb: KiCadPCB,
        schematic: Optional[KiCadSchematic],
        output_dir: Path
    ) -> Dict[str, Path]:
        """Generates all standard manufacturing Gerber layers, Drill file, and BOM."""
        output_dir.mkdir(parents=True, exist_ok=True)
        files = {}

        # 1. Top Copper (F_Cu.gbr)
        f_cu_path = output_dir / f"{pcb.name}-F_Cu.gbr"
        with open(f_cu_path, "w", encoding="utf-8") as f:
            f.write(cls._generate_copper_gerber(pcb, schematic, layer="F.Cu"))
        files["F.Cu"] = f_cu_path

        # 2. Bottom Copper (B_Cu.gbr)
        b_cu_path = output_dir / f"{pcb.name}-B_Cu.gbr"
        with open(b_cu_path, "w", encoding="utf-8") as f:
            f.write(cls._generate_copper_gerber(pcb, schematic, layer="B.Cu"))
        files["B.Cu"] = b_cu_path

        # 3. Board Outline (Edge_Cuts.gbr)
        edge_path = output_dir / f"{pcb.name}-Edge_Cuts.gbr"
        with open(edge_path, "w", encoding="utf-8") as f:
            f.write(cls._generate_edge_cuts_gerber(pcb))
        files["Edge_Cuts"] = edge_path

        # 4. Top Silkscreen (F_Silk.gbr)
        silk_path = output_dir / f"{pcb.name}-F_Silk.gbr"
        with open(silk_path, "w", encoding="utf-8") as f:
            f.write(cls._generate_silkscreen_gerber(pcb))
        files["F.SilkS"] = silk_path

        # 5. Excellon Drill File (.drl)
        drill_path = output_dir / f"{pcb.name}-Drill.drl"
        with open(drill_path, "w", encoding="utf-8") as f:
            f.write(cls._generate_excellon_drill(pcb, schematic))
        files["Drill"] = drill_path

        # 6. Bill of Materials (BOM .csv)
        if schematic:
            bom_path = output_dir.parent / f"{pcb.name}_BOM.csv"
            with open(bom_path, "w", encoding="utf-8") as f:
                f.write(cls.generate_bom_csv(schematic))
            files["BOM"] = bom_path

        return files

    @classmethod
    def _generate_copper_gerber(cls, pcb: KiCadPCB, schematic: Optional[KiCadSchematic], layer: str) -> str:
        lines = [
            f"%TF.GenerationSoftware,TerminusECE,KiCad_Engine,1.0*%",
            f"%TF.FileFunction,Copper,L1,Top*%" if layer == "F.Cu" else f"%TF.FileFunction,Copper,L2,Bot*%",
            f"%FSLAX46Y46*%",
            f"%MOMM*%",
            f"%LPD*%",
            # Standard Apertures: D10=Track, D11=Pad
            f"%ADD10C,0.254*%",
            f"%ADD11R,1.200X1.400*%",
            f"%ADD12C,1.600*%",
            f"G04 === Copper Layer: {layer} ===*",
            f"G01*",
        ]

        # Draw pads on this layer
        for comp in pcb.components.values():
            if comp.layer == layer or True:  # THT pads appear on all layers
                pads = comp.get_absolute_pad_locations(schematic)
                for p in pads:
                    if p.layer == layer or p.is_tht:
                        x_coord = int(p.x_mm * 1000000)
                        y_coord = int(p.y_mm * 1000000)
                        aperture = "D11" if not p.is_tht else "D12"
                        lines.append(f"{aperture}*")
                        lines.append(f"X{x_coord:010d}Y{y_coord:010d}D03*")

        # Draw routed copper tracks on this layer
        lines.append("D10*")
        for t in pcb.tracks:
            if t.layer == layer:
                x1 = int(t.start_x_mm * 1000000)
                y1 = int(t.start_y_mm * 1000000)
                x2 = int(t.end_x_mm * 1000000)
                y2 = int(t.end_y_mm * 1000000)
                lines.append(f"X{x1:010d}Y{y1:010d}D02*")
                lines.append(f"X{x2:010d}Y{y2:010d}D01*")

        lines.append("M02*")
        return "\n".join(lines)

    @classmethod
    def _generate_edge_cuts_gerber(cls, pcb: KiCadPCB) -> str:
        w_int = int(pcb.width_mm * 1000000)
        h_int = int(pcb.height_mm * 1000000)
        return (
            "%TF.GenerationSoftware,TerminusECE,KiCad_Engine,1.0*%\n"
            "%TF.FileFunction,Profile,NP*%\n"
            "%FSLAX46Y46*%\n"
            "%MOMM*%\n"
            "%LPD*%\n"
            "%ADD10C,0.100*%\n"
            "D10*\n"
            "G01*\n"
            "X0000000000Y0000000000D02*\n"
            f"X{w_int:010d}Y0000000000D01*\n"
            f"X{w_int:010d}Y{h_int:010d}D01*\n"
            f"X0000000000Y{h_int:010d}D01*\n"
            "X0000000000Y0000000000D01*\n"
            "M02*\n"
        )

    @classmethod
    def _generate_silkscreen_gerber(cls, pcb: KiCadPCB) -> str:
        lines = [
            f"%TF.GenerationSoftware,TerminusECE,KiCad_Engine,1.0*%",
            f"%TF.FileFunction,Legend,Top*%",
            f"%FSLAX46Y46*%",
            f"%MOMM*%",
            f"%LPD*%",
            f"%ADD10C,0.150*%",
            f"D10*",
            f"G01*",
        ]
        # Component outline box on silkscreen
        for comp in pcb.components.values():
            fp = comp.footprint_spec
            w = fp.width_mm if fp else 2.0
            h = fp.height_mm if fp else 2.0
            x1 = int((comp.x_mm - w/2) * 1000000)
            y1 = int((comp.y_mm - h/2) * 1000000)
            x2 = int((comp.x_mm + w/2) * 1000000)
            y2 = int((comp.y_mm + h/2) * 1000000)
            lines.append(f"X{x1:010d}Y{y1:010d}D02*")
            lines.append(f"X{x2:010d}Y{y1:010d}D01*")
            lines.append(f"X{x2:010d}Y{y2:010d}D01*")
            lines.append(f"X{x1:010d}Y{y2:010d}D01*")
            lines.append(f"X{x1:010d}Y{y1:010d}D01*")

        lines.append("M02*")
        return "\n".join(lines)

    @classmethod
    def _generate_excellon_drill(cls, pcb: KiCadPCB, schematic: Optional[KiCadSchematic]) -> str:
        lines = [
            "; TerminusECE KiCad Excellon Drill File",
            "M48",
            "METRIC,LZ",
            "T1C0.800",
            "T2C1.000",
            "T3C0.300",
            "%",
            "G90",
            "G05",
        ]

        # THT component pin holes
        lines.append("T1")
        for comp in pcb.components.values():
            pads = comp.get_absolute_pad_locations(schematic)
            for p in pads:
                if p.is_tht and p.drill_mm <= 0.8:
                    lines.append(f"X{p.x_mm:.3f}Y{p.y_mm:.3f}")

        # Larger power pin holes
        lines.append("T2")
        for comp in pcb.components.values():
            pads = comp.get_absolute_pad_locations(schematic)
            for p in pads:
                if p.is_tht and p.drill_mm > 0.8:
                    lines.append(f"X{p.x_mm:.3f}Y{p.y_mm:.3f}")

        # Vias
        if pcb.vias:
            lines.append("T3")
            for v in pcb.vias:
                lines.append(f"X{v.x_mm:.3f}Y{v.y_mm:.3f}")

        lines.append("M30")
        return "\n".join(lines)

    @classmethod
    def generate_bom_csv(cls, schematic: KiCadSchematic) -> str:
        """Generates structured Bill of Materials (BOM) CSV."""
        # Group components by (Value, Footprint)
        groups: Dict[Tuple[str, str], List[str]] = {}
        for ref, comp in schematic.components.items():
            key = (comp.value, comp.footprint_name)
            groups.setdefault(key, []).append(ref)

        lines = [
            "Item,Designators,Quantity,Value,Footprint,Description",
        ]
        item_idx = 1
        for (val, fp), refs in sorted(groups.items()):
            refs_str = " ".join(sorted(refs))
            lines.append(f'{item_idx},"{refs_str}",{len(refs)},"{val}","{fp}","KiCad Component"')
            item_idx += 1

        return "\n".join(lines)
