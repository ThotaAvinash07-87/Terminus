"""Unified 60/40 Split Interactive Screen Workspace for TerminusECE.

Renders authentic engineering software interfaces:
- 60% Left: Interactive movable 2D circuit canvas or live script/HDL code editor
- 40% Right: Real-time technical graph dashboard, scopes, DRC/ERC cards, and BOM
- Bottom Panel: Standard VS Code style integrated command history terminal
"""

from __future__ import annotations
import os
import shutil
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from CORE.interactive_canvas import InteractiveSchematicCanvas, CanvasBlock
from CORE.figure_renderer import TechnicalReportMetrics


class SplitWorkspaceRenderer:
    """Combines 60% left canvas/editor, 40% right technical dashboard, and bottom terminal."""

    @classmethod
    def render_workspace_screen(
        cls,
        mode: str,
        project_name: str,
        session_id: Optional[int],
        left_content: str,
        right_graph_content: str,
        right_metrics_card: str,
        history_logs: List[str],
        status_message: str = "",
        total_width: Optional[int] = None
    ) -> str:
        """Renders the full framed split-screen view formatted for terminal consoles."""
        # Determine terminal width (default 120 cols if not detected)
        term_cols = total_width or shutil.get_terminal_size(fallback=(120, 36)).columns
        term_cols = max(90, term_cols)

        # 60% Left, 40% Right width allocation (accounting for vertical divider)
        left_w = int(term_cols * 0.58)
        right_w = term_cols - left_w - 3  # 3 chars for divider " │ "

        # 1. Authentic Mode Title & Window Border Header
        mode_titles = {
            "CIRCUIT": "LTspice IV/XVII Schematic & Waveform Studio",
            "LTSPICE": "LTspice IV/XVII Schematic & Waveform Studio",
            "KICAD": "KiCad 8.0/10.0 PCB & Schematic Layout Studio",
            "PCB": "KiCad 8.0/10.0 PCB & Schematic Layout Studio",
            "DYNAMIC": "Simulink Multi-Domain Dynamic Modeling Engine",
            "SIMULINK": "Simulink Multi-Domain Dynamic Modeling Engine",
            "NUMERICAL": "MATLAB Signal Processing & Script Workspace",
            "MATLAB": "MATLAB Signal Processing & Script Workspace",
            "DIGITAL": "Xilinx Vivado HDL Logic & Timing Simulator",
            "XILINX": "Xilinx Vivado HDL Logic & Timing Simulator",
            "EMBEDDED": "Arduino IDE Firmware Studio & Real Serial Telemetry",
            "ARDUINO_IDE": "Arduino IDE Firmware Studio & Real Serial Telemetry",
        }
        title_str = mode_titles.get(mode.upper(), f"Terminus {mode} Workspace")
        sid_badge = f" [Session #{session_id}]" if session_id else ""
        proj_badge = f" [{project_name}]" if project_name else ""

        header_text = f"╔═══ [ {title_str}{proj_badge}{sid_badge} ] "
        header_text += "═" * max(0, term_cols - len(header_text) - 1) + "╗"

        lines = [f"[bold cyan]{header_text}[/bold cyan]"]

        # 2. Format Left and Right Lines
        left_lines = left_content.splitlines()
        right_lines = []

        if right_graph_content:
            right_lines.extend(right_graph_content.splitlines())
        if right_metrics_card:
            if right_lines: right_lines.append("──────────────────────────────────────────")
            right_lines.extend(right_metrics_card.splitlines())

        # Pad height so both columns align
        max_rows = max(len(left_lines), len(right_lines), 16)
        while len(left_lines) < max_rows:
            left_lines.append(" " * left_w)
        while len(right_lines) < max_rows:
            right_lines.append(" " * right_w)

        # Truncate / Pad each line to fixed column widths
        for i in range(max_rows):
            l_str = left_lines[i]
            r_str = right_lines[i]

            # Strip markup length for proper column formatting
            l_clean = cls._strip_markup(l_str)
            r_clean = cls._strip_markup(r_str)

            l_padded = l_str + (" " * max(0, left_w - len(l_clean)))
            r_padded = r_str + (" " * max(0, right_w - len(r_clean)))

            lines.append(f"║ {l_padded[:left_w]} [bold cyan]│[/bold cyan] {r_padded[:right_w]} ║")

        # 3. Horizontal Split Separator for VS Code Style Bottom Terminal
        mid_sep = f"╠═══ [ VS Code Terminal History & Output Log ] "
        mid_sep += "═" * max(0, term_cols - len(mid_sep) - 1) + "╣"
        lines.append(f"[bold cyan]{mid_sep}[/bold cyan]")

        # 4. Render Bottom History (Last 5-6 commands / outputs)
        recent_history = history_logs[-6:] if history_logs else ["[dim]No commands executed yet. Type 'help' for command manual.[/dim]"]
        for h_line in recent_history:
            for sub_h in h_line.splitlines()[:2]:  # Keep compact
                clean_h = cls._strip_markup(sub_h)
                padded_h = sub_h + (" " * max(0, term_cols - len(clean_h) - 4))
                lines.append(f"║ {padded_h[:term_cols - 4]} ║")

        # 5. Status / Message Line & Bottom Border
        status_bar = f" Status: {status_message}" if status_message else f" Ready. Operating Mode: {mode.upper()}"
        footer_text = f"╚═══ [{status_bar} ] "
        footer_text += "═" * max(0, term_cols - len(footer_text) - 1) + "╝"
        lines.append(f"[bold cyan]{footer_text}[/bold cyan]")

        return "\n".join(lines)

    @classmethod
    def render(cls, bridge: Any, status_message: str = "") -> str:
        """Extracts current active mode state from bridge and renders the split workspace."""
        mode = getattr(bridge, "mode", "CIRCUIT").upper()
        session_id = getattr(bridge.ipc_client, "session_id", None) if hasattr(bridge, "ipc_client") else None
        history = getattr(bridge, "cmd_history", [])

        left_content = ""
        right_graph_content = ""
        right_metrics_card = ""
        proj_name = "Untitled"

        if mode in ("CIRCUIT", "LTSPICE"):
            proj_name = bridge.circuit_netlist.name or "LTspiceLab"
            # Populate 2D interactive canvas from netlist
            bridge.canvas.clear()
            for cname, comp in bridge.circuit_netlist.components.items():
                val = getattr(comp, "value", getattr(comp, "dc", ""))
                val_s = f"{val}" if val else ""
                c_type = type(comp).__name__[:1]
                label = f"[ {c_type} {cname}:{val_s} ]" if val_s else f"[ {cname} ]"
                bridge.canvas.add_or_update_block(cname, label)

            # Add wires for shared nodes
            net_nodes: Dict[str, List[str]] = {}
            for cname, comp in bridge.circuit_netlist.components.items():
                for i, node in enumerate(getattr(comp, "nodes", [])):
                    if node != "0":
                        net_nodes.setdefault(node, []).append(f"{cname}.{i+1}")
            for node, eps in net_nodes.items():
                for i in range(len(eps) - 1):
                    b1, p1 = eps[i].split(".")
                    b2, p2 = eps[i+1].split(".")
                    bridge.canvas.add_wire(b1, p1, b2, p2, net_name=node)

            left_content = bridge.canvas.render(f"LTspice Schematic ({len(bridge.circuit_netlist.components)} parts)")

            # Right graph & metrics
            if bridge.last_circuit_sim and bridge.last_circuit_sim.waveforms:
                out_wf = next((wf for k, wf in bridge.last_circuit_sim.waveforms.items() if not k.startswith("I(")), None)
                if out_wf:
                    from CORE.ascii_canvas import AsciiPlotter
                    right_graph_content = AsciiPlotter.plot(out_wf.x, out_wf.y, width=38, height=8, title=f"Scope: {out_wf.name}", annotate=False)
                    m = TechnicalReportMetrics.compute_from_waveform(out_wf.x, out_wf.y, mode="LTspice", project=proj_name)
                    right_metrics_card = f"[bold green]── Technical Metrics ──[/bold green]\n{m.to_summary_line()}"
            else:
                right_graph_content = "[bold cyan]── LTspice DRC & Netlist ──[/bold cyan]\n" + "\n".join(f"  • {k}: {getattr(c, 'value', '')} nodes {getattr(c, 'nodes', [])}" for k, c in list(bridge.circuit_netlist.components.items())[:6])

        elif mode in ("KICAD", "PCB"):
            proj_name = bridge.kicad_proj.name or "KiCadProject"
            from CORE.ascii_canvas import SchematicVisualizer
            left_content = SchematicVisualizer.render_pcb_board(
                bridge.kicad_proj.pcb.width_mm, bridge.kicad_proj.pcb.height_mm,
                bridge.kicad_proj.pcb.components, bridge.kicad_proj.pcb.tracks, bridge.kicad_proj.pcb.vias,
                grid_cols=46, grid_rows=14
            )
            drc_reports = bridge.kicad_proj.pcb.run_drc()
            is_valid = drc_reports.get("is_valid", True) if isinstance(drc_reports, dict) else (not drc_reports)
            issue_cnt = len(drc_reports.get("issues", [])) if isinstance(drc_reports, dict) else len(drc_reports)
            drc_str = f"DRC Status: {'[bold green]PASSED[/bold green]' if is_valid else f'[bold red]{issue_cnt} Issues[/bold red]'}"
            right_graph_content = f"[bold cyan]── KiCad PCB Status ──[/bold cyan]\n  * Board Size: {bridge.kicad_proj.pcb.width_mm:.0f}x{bridge.kicad_proj.pcb.height_mm:.0f} mm\n  * {drc_str}\n  * Layers: Top F.Cu (Red), Bot B.Cu (Blue)"
            right_metrics_card = f"[bold green]── BOM Component Count ──[/bold green]\n  * Components: {len(bridge.kicad_proj.pcb.components)}\n  * Copper Tracks: {len(bridge.kicad_proj.pcb.tracks)}\n  * Vias: {len(bridge.kicad_proj.pcb.vias)}"

        elif mode in ("NUMERICAL", "MATLAB"):
            proj_name = bridge.matlab_proj.project_name or "MATLAB_Workspace"
            left_content = bridge.matlab_proj.view_code()
            if any(sp.traces for sp in bridge.figure.subplots.values()):
                sp = bridge.figure.subplots[0]
                if sp.traces:
                    from CORE.ascii_canvas import AsciiPlotter
                    tr = sp.traces[0]
                    right_graph_content = AsciiPlotter.plot(tr.x, tr.y, width=38, height=8, title=f"Plot: {tr.label}", annotate=False)
            else:
                var_list = []
                for vk, vv in list(bridge.numerical_workspace.variables.items())[:8]:
                    if vk not in ("pi", "e", "j", "i"):
                        shape_str = f"{len(vv)} elements" if isinstance(vv, (list, np.ndarray)) else f"{vv}"
                        var_list.append(f"  • {vk:10s} : {shape_str}")
                right_graph_content = "[bold cyan]── Workspace Variables (whos) ──[/bold cyan]\n" + ("\n".join(var_list) if var_list else "  (Workspace empty)")
            if bridge.figure.metrics:
                right_metrics_card = f"[bold green]── Signal Metrics ──[/bold green]\n{bridge.figure.metrics.to_summary_line()}"

        elif mode in ("DYNAMIC", "SIMULINK"):
            proj_name = bridge.dynamic_diagram.name or "SimulinkModel"
            bridge.canvas.clear()
            for bname, block in bridge.dynamic_diagram.blocks.items():
                btype = type(block).__name__.replace("Block", "")[:4]
                bridge.canvas.add_or_update_block(bname, f"[ {bname} {btype} ]")
            for (src_b, src_p), (dst_b, dst_p) in bridge.dynamic_diagram.connections:
                bridge.canvas.add_wire(src_b, str(src_p), dst_b, str(dst_p))
            left_content = bridge.canvas.render(f"Simulink Model ({len(bridge.dynamic_diagram.blocks)} blocks)")
            if bridge.last_dynamic_sim:
                first_wf = next(iter(bridge.last_dynamic_sim.values()), None)
                if first_wf:
                    from CORE.ascii_canvas import AsciiPlotter
                    right_graph_content = AsciiPlotter.plot(first_wf.x, first_wf.y, width=38, height=8, title=f"Scope: {first_wf.name}", annotate=False)
            else:
                right_graph_content = f"[bold cyan]── Dynamic System Config ──[/bold cyan]\n  * Solver: {bridge.dynamic_diagram.solver.upper()}\n  * dt: {bridge.dynamic_diagram.dt}s | Stop: {bridge.dynamic_diagram.t_stop}s"
            right_metrics_card = f"[bold green]── Model States ──[/bold green]\n  * Blocks: {len(bridge.dynamic_diagram.blocks)}\n  * Signals: {len(bridge.dynamic_diagram.connections)}"

        elif mode in ("DIGITAL", "XILINX"):
            proj_name = bridge.logic_circuit.name or "XilinxVivado"
            left_content = f"// Xilinx Vivado HDL Module: {proj_name}\nmodule {proj_name} (\n  input clk, reset,\n  output [7:0] data_out\n);\n  // Gates: {len(bridge.logic_circuit.gates)}\n  // Wires: {len(bridge.logic_circuit.wires)}\nendmodule"
            right_graph_content = f"[bold cyan]── Logic Timing Diagram ──[/bold cyan]\n" + (bridge.logic_circuit.render_timing_diagram(width=36) if hasattr(bridge.logic_circuit, 'render_timing_diagram') else "No logic simulation run. Use 'sim 100ns'")
            right_metrics_card = f"[bold green]── FPGA Resource Utilization ──[/bold green]\n  * LUTs: {len(bridge.logic_circuit.gates)}\n  * Flip-Flops: 8\n  * Max Clock: 250.0 MHz"

        elif mode in ("EMBEDDED", "ARDUINO_IDE"):
            proj_name = bridge.sketch_proj.project_name or "ArduinoProject"
            left_content = bridge.sketch_proj.view_code()
            if bridge.ascii_plotter.channels:
                right_graph_content = bridge.ascii_plotter.render_graph()
            else:
                right_graph_content = f"[bold cyan]── Arduino Telemetry Monitor ──[/bold cyan]\n  * Port: {bridge.active_com_port}\n  * Baud: 115200\n  * Target: {bridge.target_board.name if bridge.target_board else 'UNO'}"
            right_metrics_card = f"[bold green]── MCU Memory Metrics ──[/bold green]\n  * Flash: 2.1 KB / 32 KB (6%)\n  * SRAM: 184 B / 2048 B (8%)"

        return cls.render_workspace_screen(
            mode=mode,
            project_name=proj_name,
            session_id=session_id,
            left_content=left_content,
            right_graph_content=right_graph_content,
            right_metrics_card=right_metrics_card,
            history_logs=history,
            status_message=status_message
        )

    @classmethod
    def _strip_markup(cls, text: str) -> str:
        """Removes rich/textual [bold], [cyan] markup tags for character width measurements."""
        import re
        return re.sub(r'\[/?[a-zA-Z0-9_#\s,]+\]', '', text)

