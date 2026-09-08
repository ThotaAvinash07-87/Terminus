"""Subsystem layout screens for TerminusECE."""

from textual.screen import Screen
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Input, RichLog, Static

from .widgets import (
    AsciiPlotWidget,
    SchematicCanvasWidget,
    LibraryBrowserWidget,
    BlockInspectorWidget,
    DiagnosticReportWidget,
    McuStateWidget,
    LogicTimingWidget,
)


class BaseSubsystemScreen(Screen):
    """Base layout screen containing split terminal console and visualization panel."""

    def __init__(self, mode_name: str, **kwargs):
        super().__init__(**kwargs)
        self.mode_name = mode_name.upper()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main_container"):
            with Vertical(id="left_pane"):
                yield RichLog(id="console", highlight=True, markup=True)
            with Vertical(id="right_pane"):
                yield Static(f"[bold cyan]=== {self.mode_name} WORKSPACE ===[/bold cyan]", id="panel_header")
        yield Input(placeholder=f"Terminus [{self.mode_name}] > Enter command...", id="command_input")
        yield Footer()


class WorkbenchScreen(BaseSubsystemScreen):
    def __init__(self, **kwargs):
        super().__init__("UNIFIED WORKBENCH", **kwargs)


class CircuitScreen(BaseSubsystemScreen):
    def __init__(self, **kwargs):
        super().__init__("CIRCUIT (LTSPICE)", **kwargs)


class NumericalScreen(BaseSubsystemScreen):
    def __init__(self, **kwargs):
        super().__init__("NUMERICAL (MATLAB)", **kwargs)


class DynamicSystemScreen(Screen):
    """Rich multi-pane Simulink-like workspace screen for Dynamic Systems."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="simulink_workspace"):
            # Left pane: Library Browser
            with Vertical(id="simulink_left", classes="sidebar"):
                yield LibraryBrowserWidget(id="library_browser")
            
            # Center pane: Model Canvas & Console Log / Plotter
            with Vertical(id="simulink_center"):
                yield SchematicCanvasWidget(id="block_canvas")
                yield AsciiPlotWidget(id="scope_viewer")
                yield RichLog(id="console", highlight=True, markup=True)
            
            # Right pane: Block Inspector & Diagnostics
            with Vertical(id="simulink_right", classes="sidebar"):
                yield BlockInspectorWidget(id="block_inspector")
                yield DiagnosticReportWidget(id="diagnostic_panel")

        yield Input(
            placeholder="Terminus [DYNAMIC] > Enter command (e.g. 'library', 'add step Step1', 'connect Step1.0 Plant.0', 'check', 'sim 10')...",
            id="command_input"
        )
        yield Footer()


class DigitalLogicScreen(BaseSubsystemScreen):
    def __init__(self, **kwargs):
        super().__init__("DIGITAL LOGIC (XILINX)", **kwargs)


class EmbeddedScreen(BaseSubsystemScreen):
    def __init__(self, **kwargs):
        super().__init__("EMBEDDED (MCU/DSP)", **kwargs)
