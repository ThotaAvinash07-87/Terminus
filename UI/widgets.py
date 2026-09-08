"""Custom Textual widgets for plots, block diagrams, library browser, inspector, and diagnostics."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from textual.widgets import Static, Tree
from textual.app import ComposeResult
from rich.text import Text
from rich.panel import Panel
from rich.table import Table

from CORE.ascii_canvas import AsciiCanvas, AsciiPlotter, AsciiBodePlotter, SchematicVisualizer
from CORE.common_math import Waveform
from Engines.Dynamic_System.catalog import DynamicBlockCatalog, BlockDescriptor
from Engines.Dynamic_System.diagnostics import DiagnosticReport, DiagnosticSeverity
from Engines.Dynamic_System.base import Block


class AsciiPlotWidget(Static):
    """Widget that renders high-contrast terminal waveform and frequency response plots."""

    def __init__(self, **kwargs):
        super().__init__("[italic dim]No plot data available.[/italic dim]", **kwargs)

    def set_waveform(self, wf: Waveform, title: str = "") -> None:
        plot_title = title or wf.name
        if wf.is_complex or wf.domain == "frequency":
            bode_str = AsciiBodePlotter.plot_bode(
                wf.x,
                wf.magnitude_db,
                wf.phase_deg,
                width=68,
                height_each=6,
                title=plot_title
            )
            self.update(Panel(Text.from_markup(bode_str), title="Frequency Response"))
        else:
            plot_str = AsciiPlotter.plot(
                wf.x,
                wf.y,
                width=68,
                height=12,
                title=plot_title,
                x_label=f"Time ({wf.x_unit})",
                y_label=f"Signal ({wf.y_unit})"
            )
            self.update(Panel(Text.from_markup(plot_str), title="Scope Waveform Viewer"))


class LibraryBrowserWidget(Static):
    """Widget that displays the classified library catalog and block specifications."""

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self.active_category: Optional[str] = None
        self.update_content()

    def set_category(self, category: str) -> None:
        self.active_category = category
        self.update_content()

    def update_content(self, search_query: str = "") -> None:
        table = Table(title="Simulink Block Library Catalog", border_style="cyan", show_header=True)
        table.add_column("Category", style="magenta", width=18)
        table.add_column("Block Type", style="bold green", width=16)
        table.add_column("Display Name", style="white", width=22)
        table.add_column("In/Out", style="yellow", width=8)
        table.add_column("Description", style="dim", width=36)

        if search_query:
            descriptors = DynamicBlockCatalog.search_blocks(search_query)
        elif self.active_category:
            descriptors = DynamicBlockCatalog.get_blocks_by_category(self.active_category)
        else:
            # Show sample from top categories
            descriptors = []
            for cat in DynamicBlockCatalog.list_categories()[:6]:
                descriptors.extend(DynamicBlockCatalog.get_blocks_by_category(cat)[:3])

        for desc in descriptors:
            in_count = len(desc.input_ports)
            out_count = len(desc.output_ports)
            table.add_row(
                desc.category,
                desc.type_name,
                desc.display_name,
                f"{in_count} / {out_count}",
                desc.description[:35] + "..." if len(desc.description) > 35 else desc.description
            )

        self.update(Panel(table, title="[bold cyan]Library Browser[/bold cyan]"))


class BlockInspectorWidget(Static):
    """Widget displaying detailed property sheets, tuned parameters, ports, and state values."""

    def __init__(self, **kwargs):
        super().__init__("[italic dim]Select a block to inspect properties.[/italic dim]", **kwargs)

    def inspect_block(
        self,
        block: Block,
        block_type: str = "",
        incoming_conns: Optional[List[Tuple[str, int, int]]] = None,
        outgoing_conns: Optional[List[Tuple[int, str, int]]] = None
    ) -> None:
        desc = DynamicBlockCatalog.get_descriptor(block_type or block.__class__.__name__)
        display_title = desc.display_name if desc else block.__class__.__name__
        cat_name = desc.category if desc else "Custom"

        table = Table(title=f"Block Properties: {block.name.upper()} ({display_title})", border_style="blue")
        table.add_column("Property / Parameter", style="cyan", width=22)
        table.add_column("Current Value", style="bold yellow", width=24)
        table.add_column("Details", style="dim", width=26)

        table.add_row("Category", cat_name, "Library category")
        table.add_row("Class Type", block.__class__.__name__, "Python class")
        table.add_row("Sample Time", f"{getattr(block, 'sample_time', 0.0)} s", "0 = Continuous")
        table.add_row("Feedthrough", str(getattr(block, "direct_feedthrough", True)), "Direct input-to-output")
        table.add_row("Continuous States", str(block.num_states), f"States: {block.states.tolist()}")

        # Parameters
        table.add_section()
        params = getattr(block, "parameters", {})
        if params:
            for k, v in params.items():
                p_desc = desc.parameters.get(k) if desc else None
                p_unit = f" ({p_desc.unit})" if (p_desc and p_desc.unit) else ""
                p_info = p_desc.description if p_desc else ""
                table.add_row(f"[bold]{k}[/bold]{p_unit}", str(v), p_info)
        else:
            table.add_row("Parameters", "(None)", "No configurable parameters")

        # Input Ports
        table.add_section()
        inc_map = {dst_p: (src_b, src_p) for src_b, src_p, dst_p in (incoming_conns or [])}
        for i in range(block.num_inputs):
            p_name = desc.input_ports[i].name if (desc and i < len(desc.input_ports)) else f"in{i}"
            if i in inc_map:
                src_b, src_p = inc_map[i]
                conn_status = f"[green]Connected from {src_b}.{src_p}[/green]"
            else:
                conn_status = "[bold red][UNCONNECTED][/bold red]"
            table.add_row(f"Input Port {i} ({p_name})", conn_status, f"Input Val: {block.inputs[i] if i < len(block.inputs) else 0.0}")

        # Output Ports
        for i in range(block.num_outputs):
            p_name = desc.output_ports[i].name if (desc and i < len(desc.output_ports)) else f"out{i}"
            targets = [f"{dst_b}.{dst_p}" for src_p, dst_b, dst_p in (outgoing_conns or []) if src_p == i]
            targets_str = ", ".join(targets) if targets else "[dim]Unconnected[/dim]"
            table.add_row(f"Output Port {i} ({p_name})", targets_str, f"Output Val: {block.outputs[i] if i < len(block.outputs) else 0.0}")

        self.update(Panel(table, title="[bold blue]Block Inspector & Tuner[/bold blue]"))


class DiagnosticReportWidget(Static):
    """Widget displaying the Model Advisor pre-flight diagnostic report and health status."""

    def __init__(self, **kwargs):
        super().__init__("[italic dim]No diagnostic check performed yet. Type 'check' or press F7.[/italic dim]", **kwargs)

    def set_report(self, report: DiagnosticReport) -> None:
        status_color = "green" if report.is_valid else "red"
        status_text = "PASSED (0 Errors)" if report.is_valid else f"FAILED ({report.num_errors} Errors)"

        table = Table(title=f"Model Diagnostic Status: [{status_color}]{status_text}[/{status_color}]", border_style=status_color)
        table.add_column("Severity", width=12)
        table.add_column("Category", style="cyan", width=18)
        table.add_column("Target Block", style="bold", width=14)
        table.add_column("Issue & Remediation", width=46)

        if not report.issues:
            table.add_row("[green]OK[/green]", "Model Advisor", "All", "[green]All blocks and signal wires are 100% verified.[/green]")
        else:
            for issue in report.issues:
                if issue.severity == DiagnosticSeverity.ERROR:
                    sev = "[bold red]ERROR[/bold red]"
                elif issue.severity == DiagnosticSeverity.WARNING:
                    sev = "[bold yellow]WARNING[/bold yellow]"
                else:
                    sev = "[bold cyan]INFO[/bold cyan]"

                port_str = f" (Port {issue.port_index})" if issue.port_index is not None else ""
                desc_text = f"{issue.message}\n[dim]Fix: {issue.remediation}[/dim]"
                table.add_row(sev, issue.category, f"{issue.block_name}{port_str}", desc_text)

        self.update(Panel(table, title="[bold yellow]Model Advisor & Diagnostics[/bold yellow]"))


class SchematicCanvasWidget(Static):
    """Widget that renders ASCII 2D circuit schematics and dynamic block diagrams."""

    def __init__(self, **kwargs):
        super().__init__("[italic dim]Canvas empty.[/italic dim]", **kwargs)

    def render_circuit(self, netlist_components: Dict[str, Any], pin_map: Dict[str, List[str]]) -> None:
        canvas = AsciiCanvas(width=72, height=18)
        canvas.draw_box(0, 0, 72, 18, title="Circuit Schematic Canvas")

        col = 3
        row = 3
        for i, (cname, comp) in enumerate(netlist_components.items()):
            pins = pin_map.get(cname, comp.nodes)
            pins_str = ",".join(pins)
            val_str = str(getattr(comp, "value", getattr(comp, "dc", "")))
            
            # Place block
            canvas.draw_box(col, row, 18, 4, title=cname)
            canvas.draw_text(col + 2, row + 1, f"Val: {val_str[:8]}")
            canvas.draw_text(col + 2, row + 2, f"Nets: {pins_str[:8]}")

            col += 22
            if col > 50:
                col = 3
                row += 5

        self.update(Panel(Text(canvas.render()), title="Schematic Canvas"))

    def render_dynamic_diagram(self, blocks: Dict[str, Block], connections: List[Any]) -> None:
        rendered = SchematicVisualizer.render_dynamic_block_diagram(blocks, connections)
        self.update(Panel(Text.from_markup(rendered), title="Dynamic Block Diagram Canvas"))


class McuStateWidget(Static):
    """Widget displaying register contents, flags, and memory dump of the embedded MCU."""

    def __init__(self, **kwargs):
        super().__init__("[italic dim]MCU not loaded.[/italic dim]", **kwargs)

    def update_state(self, state_text: str) -> None:
        self.update(Panel(Text.from_markup(state_text), title="MCU Core Debugger"))


class LogicTimingWidget(Static):
    """Widget displaying multi-channel digital logic trace timing diagrams."""

    def __init__(self, **kwargs):
        super().__init__("[italic dim]No logic simulation trace recorded.[/italic dim]", **kwargs)

    def set_timing_diagram(self, timing_str: str) -> None:
        self.update(Panel(Text.from_markup(timing_str), title="Logic Timing Diagram"))
