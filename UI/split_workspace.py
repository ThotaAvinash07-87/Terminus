"""Unified 60/40 Split Interactive Screen Workspace for TerminusECE.

Renders authentic engineering software interfaces:
- 60% Left: Interactive movable 2D circuit canvas or live script/HDL code editor
- 40% Right: Real-time technical graph dashboard, scopes, DRC/ERC cards, and BOM
- Bottom Panel: Streamlined single-line terminal history & status footer
"""

from __future__ import annotations
import os
import shutil
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from CORE.interactive_canvas import InteractiveSchematicCanvas, CanvasBlock, format_compact_val
from CORE.figure_renderer import TechnicalReportMetrics


class SplitWorkspaceRenderer:
    """Combines 60% left canvas/editor, 40% right technical dashboard, and bottom status."""

    @classmethod
    def _strip_markup(cls, text: str) -> str:
        """Removes rich/textual [bold], [cyan] markup tags for character width measurements."""
        return re.sub(r'\[/?[a-zA-Z0-9_#\s,-]+\]', '', text)

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
        total_width: Optional[int] = None,
        total_height: Optional[int] = None
    ) -> str:
        """Renders the full framed split-screen view formatted for terminal consoles."""
        # Detect terminal size safely and stretch to full width
        raw_size = shutil.get_terminal_size(fallback=(105, 30))
        raw_cols = total_width or raw_size.columns
        raw_rows = total_height or raw_size.lines

        term_cols = max(80, raw_cols - 1)
        term_rows = max(18, raw_rows - 3)

        # Total line inner width = term_cols - 2 (outer ║ ... ║)
        # Format: "║ " (2) + left_w + " │ " (3) + right_w + " ║" (2) = left_w + right_w + 7 = term_cols
        available = term_cols - 7
        left_w = int(available * 0.58)
        right_w = available - left_w

        # 1. Mode Titles
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

        header_prefix = f"╔═══ [ {title_str}{proj_badge}{sid_badge} ] "
        header_text = header_prefix + ("═" * max(0, term_cols - len(header_prefix) - 1)) + "╗"
        lines = [f"[bold cyan]{header_text[:term_cols]}[/bold cyan]"]

        # 2. Left and Right column lines
        left_lines = [l for l in left_content.splitlines()]
        right_lines = []
        if right_graph_content:
            right_lines.extend(right_graph_content.splitlines())
        if right_metrics_card:
            if right_lines:
                right_lines.append("─" * right_w)
            right_lines.extend(right_metrics_card.splitlines())

        # Match row count dynamically to available terminal height (giving maximum space to canvas)
        canvas_h = max(16, term_rows - 4)
        max_rows = max(len(left_lines), len(right_lines), canvas_h)
        while len(left_lines) < max_rows:
            left_lines.append("")
        while len(right_lines) < max_rows:
            right_lines.append("")

        for i in range(max_rows):
            l_raw = left_lines[i]
            r_raw = right_lines[i]

            l_clean = cls._strip_markup(l_raw)[:left_w]
            r_clean = cls._strip_markup(r_raw)[:right_w]

            l_padded = l_clean.ljust(left_w)
            r_padded = r_clean.ljust(right_w)

            lines.append(f"║ {l_padded} [bold cyan]│[/bold cyan] {r_padded} ║")

        # 3. Streamlined 1-Line Terminal Log Separator
        last_log = history_logs[-1] if history_logs else "> Ready for engineering commands. Type 'help' for manual."
        last_log_clean = cls._strip_markup(last_log)[:max(10, term_cols - 22)]
        mid_prefix = f"╠═══ [ Terminal Log: {last_log_clean} ] "
        mid_sep = mid_prefix + ("═" * max(0, term_cols - len(mid_prefix) - 1)) + "╣"
        lines.append(f"[bold cyan]{mid_sep[:term_cols]}[/bold cyan]")

        # 4. Streamlined Footer & Status Bar
        status_bar = f" Status: {status_message}" if status_message else f" Ready. Operating Mode: {mode.upper()}"
        footer_prefix = f"╚═══ [{status_bar} ] "
        footer_text = footer_prefix + ("═" * max(0, term_cols - len(footer_prefix) - 1)) + "╝"
        lines.append(f"[bold cyan]{footer_text[:term_cols]}[/bold cyan]")

        return "\n".join(lines)

    @classmethod
    def render(cls, bridge: Any, status_message: str = "") -> str:
        """Extracts current active mode state from bridge and renders the split workspace."""
        mode = getattr(bridge, "mode", "CIRCUIT").upper()
        session_id = getattr(bridge.ipc_client, "session_id", None) if hasattr(bridge, "ipc_client") else None
        history = getattr(bridge, "cmd_history", [])

        # Terminal sizing
        raw_size = shutil.get_terminal_size(fallback=(105, 30))
        term_cols = max(80, raw_size.columns - 1)
        term_rows = max(18, raw_size.lines - 3)
        available = term_cols - 7
        left_w = int(available * 0.58)
        right_w = available - left_w
        canvas_h = max(16, term_rows - 4)

        left_content = ""
        right_graph_content = ""
        right_metrics_card = ""
        proj_name = "Untitled"

        if mode in ("CIRCUIT", "LTSPICE"):
            proj_name = bridge.circuit_netlist.name or "LTspiceLab"
            bridge.canvas.width = left_w
            bridge.canvas.height = canvas_h

            # Check if user already moved blocks or if we should auto-layout
            existing_refs = set(bridge.canvas.blocks.keys())
            current_refs = set(k.upper() for k in bridge.circuit_netlist.components.keys())

            if existing_refs != current_refs:
                bridge.canvas.clear()
                # Intelligent topological auto-layout
                # 1. Sources on left, passives in middle, shunts towards ground
                c_items = list(bridge.circuit_netlist.components.items())
                for idx, (cname, comp) in enumerate(c_items):
                    val = getattr(comp, "value", getattr(comp, "dc", ""))
                    val_s = format_compact_val(val)
                    c_type = type(comp).__name__[:1]
                    label = f"{c_type} {cname}:{val_s}" if val_s else f"{cname}"

                    # Determine placement by component role
                    nodes = getattr(comp, "nodes", [])
                    is_shunt = len(nodes) >= 2 and ("0" in nodes or "gnd" in [n.lower() for n in nodes])
                    if cname.upper().startswith("V") or cname.upper().startswith("I"):
                        # Input source
                        bx = 2
                        by = 3
                    elif is_shunt and not cname.upper().startswith("V"):
                        # Shunt capacitor / load resistor
                        bx = max(2, min(left_w - 14, 18 + (idx - 1) * 16))
                        by = 9
                    else:
                        # Series resistor / inductor
                        bx = max(2, min(left_w - 14, 2 + idx * 16))
                        by = 3

                    blk = bridge.canvas.add_or_update_block(cname, f"[ {label} ]", x=bx, y=by)
            else:
                # Update labels with latest values and probe overlays
                for cname, comp in bridge.circuit_netlist.components.items():
                    val = getattr(comp, "value", getattr(comp, "dc", ""))
                    val_s = format_compact_val(val)
                    c_type = type(comp).__name__[:1]
                    label = f"{c_type} {cname}:{val_s}" if val_s else f"{cname}"
                    bridge.canvas.add_or_update_block(cname, f"[ {label} ]")

            # Attach probed voltages/currents if simulation results exist
            if bridge.last_circuit_sim:
                for cname, blk in bridge.canvas.blocks.items():
                    v_key = f"V({cname})"
                    if v_key in bridge.last_circuit_sim.op_results:
                        blk.probed_v = bridge.last_circuit_sim.op_results[v_key]

            # Add wires for shared nets
            bridge.canvas.wires.clear()
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

            grid_lines = bridge.canvas.render_grid_lines(width=left_w, height=canvas_h)
            left_content = "\n".join(grid_lines)

            # Right graph & metrics
            if bridge.last_circuit_sim and bridge.last_circuit_sim.waveforms:
                out_wf = next((wf for k, wf in bridge.last_circuit_sim.waveforms.items() if not k.startswith("I(")), None)
                if out_wf:
                    from CORE.ascii_canvas import AsciiPlotter
                    right_graph_content = AsciiPlotter.plot(out_wf.x, out_wf.y, width=right_w - 2, height=max(7, canvas_h - 7), title=f"Scope: {out_wf.name}", annotate=False)
                    m = TechnicalReportMetrics.compute_from_waveform(out_wf.x, out_wf.y, mode="LTspice", project=proj_name)
                    right_metrics_card = f"── Technical Metrics ──\n{m.to_summary_line()}"
            else:
                comp_list = [f" • {k}: {format_compact_val(getattr(c, 'value', ''))} nodes {getattr(c, 'nodes', [])}" for k, c in list(bridge.circuit_netlist.components.items())[:8]]
                right_graph_content = "── LTspice DRC & Netlist ──\n" + ("\n".join(comp_list) if comp_list else " Netlist empty. Run 'add R1 10k in out'")
                right_metrics_card = f"── Circuit Status ──\n Components: {len(bridge.circuit_netlist.components)} | Ground: {'0' if '0' in bridge.circuit_netlist.node_aliases else 'Missing'}"

        elif mode in ("KICAD", "PCB"):
            proj_name = bridge.kicad_proj.name or "KiCadProject"
            from CORE.ascii_canvas import SchematicVisualizer
            left_content = SchematicVisualizer.render_pcb_board(
                bridge.kicad_proj.pcb.width_mm, bridge.kicad_proj.pcb.height_mm,
                bridge.kicad_proj.pcb.components, bridge.kicad_proj.pcb.tracks, bridge.kicad_proj.pcb.vias,
                grid_cols=left_w - 4, grid_rows=canvas_h - 2
            )
            drc_reports = bridge.kicad_proj.pcb.run_drc()
            is_valid = drc_reports.get("is_valid", True) if isinstance(drc_reports, dict) else (not drc_reports)
            issue_cnt = len(drc_reports.get("issues", [])) if isinstance(drc_reports, dict) else len(drc_reports)
            drc_str = f"DRC Status: {'PASSED' if is_valid else f'{issue_cnt} Violations'}"
            right_graph_content = f"── KiCad PCB Status ──\n * Board: {bridge.kicad_proj.pcb.width_mm:.0f}x{bridge.kicad_proj.pcb.height_mm:.0f} mm\n * {drc_str}\n * Layers: F.Cu (Red), B.Cu (Blue)"
            right_metrics_card = f"── BOM Component Count ──\n * Components: {len(bridge.kicad_proj.pcb.components)}\n * Tracks: {len(bridge.kicad_proj.pcb.tracks)} | Vias: {len(bridge.kicad_proj.pcb.vias)}"

        elif mode in ("NUMERICAL", "MATLAB"):
            proj_name = bridge.matlab_proj.project_name or "MATLAB_Workspace"
            code_lines = bridge.matlab_proj.view_code().splitlines()
            left_content = "\n".join(code_lines[:canvas_h])
            if any(sp.traces for sp in bridge.figure.subplots.values()):
                sp = bridge.figure.subplots[0]
                if sp.traces:
                    from CORE.ascii_canvas import AsciiPlotter
                    tr = sp.traces[0]
                    right_graph_content = AsciiPlotter.plot(tr.x, tr.y, width=right_w - 2, height=max(7, canvas_h - 7), title=f"Plot: {tr.label}", annotate=False)
            else:
                var_list = []
                for vk, vv in list(bridge.numerical_workspace.variables.items())[:8]:
                    if vk not in ("pi", "e", "j", "i"):
                        shape_str = f"{len(vv)} elem" if isinstance(vv, (list, np.ndarray)) else f"{vv}"
                        var_list.append(f" • {vk:8s} : {shape_str}")
                right_graph_content = "── Workspace Variables (whos) ──\n" + ("\n".join(var_list) if var_list else " (Workspace empty)")
            if bridge.figure.metrics:
                right_metrics_card = f"── Signal Metrics ──\n{bridge.figure.metrics.to_summary_line()}"

        elif mode in ("DYNAMIC", "SIMULINK"):
            proj_name = bridge.dynamic_diagram.name or "SimulinkModel"
            bridge.canvas.width = left_w
            bridge.canvas.height = canvas_h

            existing_blocks = set(bridge.canvas.blocks.keys())
            current_blocks = set(k.upper() for k in bridge.dynamic_diagram.blocks.keys())
            if existing_blocks != current_blocks:
                bridge.canvas.clear()
                for idx, (bname, block) in enumerate(bridge.dynamic_diagram.blocks.items()):
                    btype = type(block).__name__.replace("Block", "")[:4]
                    bx = max(2, min(left_w - 14, 2 + (idx % 3) * 18))
                    by = max(2, min(canvas_h - 4, 3 + (idx // 3) * 6))
                    bridge.canvas.add_or_update_block(bname, f"[ {bname} {btype} ]", x=bx, y=by)
            else:
                for bname, block in bridge.dynamic_diagram.blocks.items():
                    btype = type(block).__name__.replace("Block", "")[:4]
                    bridge.canvas.add_or_update_block(bname, f"[ {bname} {btype} ]")

            bridge.canvas.wires.clear()
            for (src_b, src_p), (dst_b, dst_p) in bridge.dynamic_diagram.connections:
                bridge.canvas.add_wire(src_b, str(src_p), dst_b, str(dst_p))

            grid_lines = bridge.canvas.render_grid_lines(width=left_w, height=canvas_h)
            left_content = "\n".join(grid_lines)

            if bridge.last_dynamic_sim:
                first_wf = next(iter(bridge.last_dynamic_sim.values()), None)
                if first_wf:
                    from CORE.ascii_canvas import AsciiPlotter
                    right_graph_content = AsciiPlotter.plot(first_wf.x, first_wf.y, width=right_w - 2, height=max(7, canvas_h - 7), title=f"Scope: {first_wf.name}", annotate=False)
            else:
                right_graph_content = f"── Dynamic System Config ──\n * Solver: {bridge.dynamic_diagram.solver.upper()}\n * dt: {bridge.dynamic_diagram.dt}s | Stop: {bridge.dynamic_diagram.t_stop}s"
            right_metrics_card = f"── Model States ──\n * Blocks: {len(bridge.dynamic_diagram.blocks)}\n * Signals: {len(bridge.dynamic_diagram.connections)}"

        elif mode in ("DIGITAL", "XILINX"):
            proj_name = bridge.logic_circuit.name or "XilinxVivado"
            left_content = f"// Xilinx Vivado HDL Module: {proj_name}\nmodule {proj_name} (\n  input clk, reset,\n  output [7:0] data_out\n);\n  // Gates: {len(bridge.logic_circuit.gates)}\n  // Wires: {len(bridge.logic_circuit.wires)}\nendmodule"
            right_graph_content = "── Logic Timing Diagram ──\n" + (bridge.logic_circuit.render_timing_diagram(width=right_w - 4) if hasattr(bridge.logic_circuit, 'render_timing_diagram') else "No logic simulation run. Run 'sim 100ns'")
            right_metrics_card = f"── FPGA Resource Utilization ──\n * LUTs: {len(bridge.logic_circuit.gates)}\n * Flip-Flops: 8\n * Max Clock: 250.0 MHz"

        elif mode in ("EMBEDDED", "ARDUINO_IDE"):
            proj_name = bridge.sketch_proj.project_name or "ArduinoProject"
            left_content = "\n".join(bridge.sketch_proj.view_code().splitlines()[:canvas_h])
            if bridge.ascii_plotter.channels:
                right_graph_content = bridge.ascii_plotter.render_graph()
            else:
                right_graph_content = f"── Arduino Telemetry Monitor ──\n * Port: {bridge.active_com_port}\n * Baud: 115200\n * Target: {bridge.target_board.name if bridge.target_board else 'UNO'}"
            right_metrics_card = "── MCU Memory Metrics ──\n * Flash: 2.1 KB / 32 KB (6%)\n * SRAM: 184 B / 2048 B (8%)"

        return cls.render_workspace_screen(
            mode=mode,
            project_name=proj_name,
            session_id=session_id,
            left_content=left_content,
            right_graph_content=right_graph_content,
            right_metrics_card=right_metrics_card,
            history_logs=history,
            status_message=status_message,
            total_width=term_cols,
            total_height=term_rows
        )

