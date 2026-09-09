"""
CORE/figure_renderer.py - Universal High-Resolution PNG Figure Renderer,
Multi-Subplot Manager, Technical Metrics Banner, and In-Terminal Truecolor Previewer.

Supports all Terminus subsystems:
1. Numerical / MATLAB & DSP Mode
2. Circuit (LTspice) Simulation Mode
3. Dynamic Systems (Simulink) Scope Mode
4. Digital Logic Timing Mode
5. Embedded USB Serial Telemetry Mode
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import math
import os
import sys
import time
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive headless backend for fast PNG rendering
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class TraceData:
    x: np.ndarray
    y: np.ndarray
    label: str = "Signal"
    color: str = "#00e5ff"  # Neon cyan
    style: str = "-"       # '-', '--', ':', 'stem', 'step', 'scatter'
    unit_x: str = "s"
    unit_y: str = "V"


@dataclass
class SubplotData:
    title: str = "Signal Analysis"
    xlabel: str = "Time (s)"
    ylabel: str = "Amplitude (V)"
    xscale: str = "linear"  # 'linear', 'log'
    yscale: str = "linear"  # 'linear', 'log', 'db'
    grid: bool = True
    traces: List[TraceData] = field(default_factory=list)


@dataclass
class TechnicalReportMetrics:
    mode: str = "MATLAB / Numerical"
    project_name: str = "Workspace"
    timestamp: str = ""
    sampling_freq: Optional[float] = None
    v_peak: Optional[float] = None
    v_rms: Optional[float] = None
    v_pp: Optional[float] = None
    dominant_freq: Optional[float] = None
    thd_pct: Optional[float] = None
    snr_db: Optional[float] = None
    notes: str = ""

    @classmethod
    def compute_from_waveform(cls, x: np.ndarray, y: np.ndarray, mode: str = "MATLAB", project: str = "Project") -> TechnicalReportMetrics:
        """Automatically extracts signal processing & electrical metrics from data."""
        metrics = cls(
            mode=mode,
            project_name=project,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        if len(y) == 0:
            return metrics

        y_clean = np.asarray(y, dtype=float)
        metrics.v_peak = float(np.max(np.abs(y_clean)))
        metrics.v_rms = float(np.sqrt(np.mean(y_clean**2)))
        metrics.v_pp = float(np.max(y_clean) - np.min(y_clean))

        # Compute Fs and dominant frequency if x is uniform time array
        if len(x) > 1 and len(x) == len(y):
            dx = np.diff(x)
            if np.all(dx > 0):
                avg_dt = float(np.mean(dx))
                if avg_dt > 0:
                    fs = 1.0 / avg_dt
                    metrics.sampling_freq = fs
                    # Compute FFT peak
                    n = len(y_clean)
                    fft_vals = np.abs(np.fft.rfft(y_clean - np.mean(y_clean)))
                    freqs = np.fft.rfftfreq(n, d=avg_dt)
                    if len(fft_vals) > 1:
                        peak_idx = int(np.argmax(fft_vals[1:])) + 1
                        metrics.dominant_freq = float(freqs[peak_idx])

                        # Approximate THD & SNR
                        fundamental_mag = fft_vals[peak_idx]
                        harmonics_mag = np.sqrt(np.sum(fft_vals[1:]**2) - fundamental_mag**2)
                        if fundamental_mag > 1e-12:
                            metrics.thd_pct = float((harmonics_mag / fundamental_mag) * 100.0)
                            metrics.snr_db = float(20.0 * np.log10(fundamental_mag / max(harmonics_mag, 1e-12)))

        return metrics


class TerminusFigure:
    """Manages multi-subplot figures, technical metadata, and universal rendering."""

    def __init__(self, title: str = "Terminus Engineering Figure", rows: int = 1, cols: int = 1):
        self.title = title
        self.rows = rows
        self.cols = cols
        self.current_index = 0
        self.subplots: Dict[int, SubplotData] = {}
        self.metrics: Optional[TechnicalReportMetrics] = None
        self._ensure_subplot(0)

    def _ensure_subplot(self, idx: int) -> SubplotData:
        if idx not in self.subplots:
            self.subplots[idx] = SubplotData()
        return self.subplots[idx]

    def set_subplot_grid(self, rows: int, cols: int, index: int = 1):
        """Sets MATLAB-style subplot(rows, cols, index) - 1-indexed."""
        self.rows = max(1, rows)
        self.cols = max(1, cols)
        self.current_index = max(0, index - 1)
        self._ensure_subplot(self.current_index)

    def add_trace(self, x: Union[List, np.ndarray], y: Union[List, np.ndarray],
                  label: str = "Signal", color: Optional[str] = None,
                  style: str = "-", subplot_index: Optional[int] = None):
        """Adds a waveform trace to the active or specified subplot."""
        idx = self.current_index if subplot_index is None else subplot_index
        sp = self._ensure_subplot(idx)

        # Palette colors
        palette = ["#00e5ff", "#ffab00", "#ff4081", "#76ff03", "#d500f9", "#00e676", "#ff6d00", "#40c4ff"]
        trace_color = color or palette[len(sp.traces) % len(palette)]

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        sp.traces.append(TraceData(x=x_arr, y=y_arr, label=label, color=trace_color, style=style))

    def set_labels(self, title: Optional[str] = None, xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None, subplot_index: Optional[int] = None):
        idx = self.current_index if subplot_index is None else subplot_index
        sp = self._ensure_subplot(idx)
        if title is not None: sp.title = title
        if xlabel is not None: sp.xlabel = xlabel
        if ylabel is not None: sp.ylabel = ylabel

    def clear(self):
        self.subplots.clear()
        self.current_index = 0
        self._ensure_subplot(0)
        self.metrics = None

    def export_to_png(self, output_path: Union[str, Path], dpi: int = 200,
                      dark_mode: bool = True) -> Tuple[bool, str]:
        """Renders the multi-subplot figure to a high-resolution PNG image on disk."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if not HAS_MATPLOTLIB:
            return self._export_fallback_png(out_p)

        try:
            # Color schemes
            bg_canvas = "#0b0e14" if dark_mode else "#ffffff"
            bg_axes = "#151922" if dark_mode else "#f8f9fa"
            text_color = "#f0f6fc" if dark_mode else "#1f2328"
            grid_color = "#2d333b" if dark_mode else "#d0d7de"
            border_color = "#30363d" if dark_mode else "#d0d7de"

            fig = plt.figure(figsize=(12, 4 * self.rows + 1.2), facecolor=bg_canvas, dpi=dpi)
            
            # Layout with space at top for technical info banner
            total_slots = self.rows * self.cols
            gs = gridspec.GridSpec(self.rows, self.cols, figure=fig, top=0.88, bottom=0.08, left=0.08, right=0.95, hspace=0.35, wspace=0.25)

            # 1. Render Technical Information Header Banner
            header_text = f"TERMINUS ECE ENGINE REPORT | {self.title.upper()}"
            fig.text(0.08, 0.94, header_text, fontsize=12, fontweight="bold", color="#58a6ff", family="sans-serif")

            if self.metrics:
                m = self.metrics
                metric_parts = [f"Mode: {m.mode}", f"Project: {m.project_name}"]
                if m.sampling_freq: metric_parts.append(f"Fs: {m.sampling_freq/1e3:.1f} kHz" if m.sampling_freq >= 1e3 else f"Fs: {m.sampling_freq:.1f} Hz")
                if m.v_peak is not None: metric_parts.append(f"Vpk: {m.v_peak:.3f}V")
                if m.v_rms is not None: metric_parts.append(f"Vrms: {m.v_rms:.3f}V")
                if m.dominant_freq is not None: metric_parts.append(f"Peak Freq: {m.dominant_freq:.1f} Hz")
                if m.thd_pct is not None: metric_parts.append(f"THD: {m.thd_pct:.2f}%")
                if m.snr_db is not None: metric_parts.append(f"SNR: {m.snr_db:.1f} dB")
                
                sub_text = " | ".join(metric_parts) + f" | Generated: {m.timestamp}"
                fig.text(0.08, 0.905, sub_text, fontsize=8.5, color="#8b949e", family="monospace")

            # 2. Render Subplots
            for r in range(self.rows):
                for c in range(self.cols):
                    slot_idx = r * self.cols + c
                    ax = fig.add_subplot(gs[r, c])
                    ax.set_facecolor(bg_axes)

                    # Styling
                    for spine in ax.spines.values():
                        spine.set_color(border_color)
                        spine.set_linewidth(1.0)
                    ax.tick_params(colors=text_color, labelsize=8)
                    ax.xaxis.label.set_color(text_color)
                    ax.yaxis.label.set_color(text_color)
                    ax.title.set_color(text_color)

                    if slot_idx in self.subplots and self.subplots[slot_idx].traces:
                        sp_data = self.subplots[slot_idx]
                        ax.set_title(sp_data.title, fontsize=10, fontweight="bold", pad=8)
                        ax.set_xlabel(sp_data.xlabel, fontsize=9)
                        ax.set_ylabel(sp_data.ylabel, fontsize=9)
                        if sp_data.grid:
                            ax.grid(True, linestyle="--", alpha=0.5, color=grid_color)

                        if sp_data.xscale == "log": ax.set_xscale("log")
                        if sp_data.yscale == "log": ax.set_yscale("log")

                        for trace in sp_data.traces:
                            if trace.style == "stem":
                                markerline, stemlines, baseline = ax.stem(trace.x, trace.y, linefmt=trace.color, markerfmt="o", basefmt="r-")
                                plt.setp(stemlines, 'color', trace.color, 'linewidth', 1.5)
                                plt.setp(markerline, 'color', trace.color, 'markersize', 4)
                            elif trace.style == "step":
                                ax.step(trace.x, trace.y, color=trace.color, label=trace.label, linewidth=1.8, where='post')
                            elif trace.style == "scatter":
                                ax.scatter(trace.x, trace.y, color=trace.color, label=trace.label, s=15)
                            else:
                                ax.plot(trace.x, trace.y, color=trace.color, label=trace.label, linewidth=1.8, linestyle=trace.style)

                        if any(t.label and t.label != "Signal" for t in sp_data.traces) or len(sp_data.traces) > 1:
                            leg = ax.legend(loc="upper right", fontsize=8, facecolor=bg_axes, edgecolor=border_color)
                            for text in leg.get_texts():
                                text.set_color(text_color)
                    else:
                        ax.set_title(f"Plot ({r+1}, {c+1}) - Empty", fontsize=9, color="#8b949e")
                        ax.grid(True, linestyle=":", alpha=0.3, color=grid_color)

            plt.savefig(str(out_p), dpi=dpi, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)
            return True, f"Rendered high-resolution figure ({dpi} DPI) -> {out_p}"

        except Exception as e:
            return False, f"Failed to export PNG: {str(e)}"

    def _export_fallback_png(self, out_p: Path) -> Tuple[bool, str]:
        """Fallback pure-Python bitmap exporter when matplotlib is not available."""
        if HAS_PIL:
            img = Image.new("RGB", (800, 450), color=(15, 18, 24))
            img.save(str(out_p))
            return True, f"Saved basic PNG placeholder (Matplotlib recommended) -> {out_p}"
    @staticmethod
    def export_pcb_to_png(pcb: Any, schematic: Any, output_path: Union[str, Path], dpi: int = 200) -> Tuple[bool, str]:
        """Renders high-resolution 2D PCB board layout with authentic layer colors."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if not HAS_MATPLOTLIB:
            if HAS_PIL:
                img = Image.new("RGB", (1000, 800), color=(10, 30, 20))
                img.save(str(out_p))
                return True, f"Saved basic PCB image -> {out_p}"
            return False, "Matplotlib required for high-res PCB artwork rendering."

        try:
            fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0a1a14", dpi=dpi)
            ax.set_facecolor("#0e2b1f")  # Solder mask green

            # Board Outline (Edge.Cuts)
            rect = plt.Rectangle((0, 0), pcb.width_mm, pcb.height_mm, fill=True, color="#0e2b1f", edgecolor="#e6db74", linewidth=2.5)
            ax.add_patch(rect)

            # Copper Tracks
            for t in pcb.tracks:
                t_color = "#e63946" if t.layer == "F.Cu" else "#457b9d"  # F.Cu Red, B.Cu Blue
                ax.plot([t.start_x_mm, t.end_x_mm], [t.start_y_mm, t.end_y_mm], color=t_color, linewidth=max(1.0, t.width_mm * 3.0), solid_capstyle='round')

            # Vias
            for v in pcb.vias:
                via_pad = plt.Circle((v.x_mm, v.y_mm), v.pad_dia_mm/2, color="#ffd166")
                via_hole = plt.Circle((v.x_mm, v.y_mm), v.drill_mm/2, color="#0a1a14")
                ax.add_patch(via_pad)
                ax.add_patch(via_hole)

            # Footprints and Pads
            for ref, comp in pcb.components.items():
                pads = comp.get_absolute_pad_locations(schematic)
                for p in pads:
                    pad_color = "#ffd166" if p.is_tht else ("#ef476f" if p.layer == "F.Cu" else "#118ab2")
                    p_rect = plt.Rectangle((p.x_mm - p.width_mm/2, p.y_mm - p.height_mm/2), p.width_mm, p.height_mm, color=pad_color)
                    ax.add_patch(p_rect)
                    if p.is_tht and p.drill_mm > 0:
                        hole = plt.Circle((p.x_mm, p.y_mm), p.drill_mm/2, color="#0a1a14")
                        ax.add_patch(hole)

                # Silkscreen reference text
                ax.text(comp.x_mm, comp.y_mm, ref, color="#f8f9fa", fontsize=8, fontweight="bold", ha="center", va="center")

            ax.set_xlim(-5, pcb.width_mm + 5)
            ax.set_ylim(-5, pcb.height_mm + 5)
            ax.set_aspect('equal')
            ax.set_title(f"KiCad PCB Layout Artwork: {pcb.name} ({pcb.width_mm:.0f}x{pcb.height_mm:.0f} mm)", color="#58a6ff", fontsize=11, fontweight="bold", pad=12)
            ax.tick_params(colors="#8b949e", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#30363d")

            plt.savefig(str(out_p), dpi=dpi, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
            plt.close(fig)
            return True, f"Rendered high-resolution KiCad PCB graphic ({dpi} DPI) -> {out_p}"
        except Exception as e:
            return False, f"Failed to export PCB PNG: {str(e)}"


class TerminalImagePreviewer:
    """Renders 24-bit Truecolor ANSI Half-Block image previews directly in the CLI terminal."""

    @staticmethod
    def preview_image(image_path_or_array: Union[str, Path, np.ndarray], max_cols: int = 70, max_rows: int = 24) -> str:
        """Converts an image file or numpy RGB array into Truecolor ANSI unicode half-block characters."""
        if isinstance(image_path_or_array, (str, Path)):
            p = Path(image_path_or_array)
            if not p.exists():
                return f"[red]Image file not found: {p}[/red]"
            if not HAS_PIL:
                return f"[yellow]PIL/Pillow required for in-terminal image preview. File saved at: {p}[/yellow]"
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                return f"[red]Cannot load image: {e}[/red]"
        elif isinstance(image_path_or_array, np.ndarray):
            if not HAS_PIL:
                return "[yellow]Pillow required for array preview.[/yellow]"
            img = Image.fromarray(image_path_or_array.astype('uint8')).convert("RGB")
        else:
            return "[red]Invalid image source.[/red]"

        # Calculate scaled dimensions maintaining aspect ratio
        orig_w, orig_h = img.size
        # Each character cell has 2 vertical pixels (top and bottom half blocks)
        target_w = min(max_cols, orig_w)
        target_h = int((orig_h / orig_w) * target_w * 0.5) * 2  # Must be even number of pixels
        target_h = max(4, min(max_rows * 2, target_h))

        img_resized = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        pixels = np.array(img_resized)

        lines = [
            f"[bold cyan]=== In-Terminal Truecolor Graphic Preview ({target_w}x{target_h//2} cells) ===[/bold cyan]",
            "┌" + "─" * target_w + "┐"
        ]

        # Render rows pairwise using upper half block '▀' (U+2580)
        # Foreground color = top pixel RGB, Background color = bottom pixel RGB
        for y in range(0, target_h, 2):
            row_str = "│"
            for x in range(target_w):
                r_top, g_top, b_top = pixels[y, x][:3]
                if y + 1 < target_h:
                    r_bot, g_bot, b_bot = pixels[y + 1, x][:3]
                else:
                    r_bot, g_bot, b_bot = 0, 0, 0
                # ANSI 24-bit Truecolor sequence: \033[38;2;R;G;Bm (FG) \033[48;2;R;G;Bm (BG)
                row_str += f"\033[38;2;{r_top};{g_top};{b_top}m\033[48;2;{r_bot};{g_bot};{b_bot}m▀"
            row_str += "\033[0m│"
            lines.append(row_str)

        lines.append("└" + "─" * target_w + "┘")
        return "\n".join(lines)


# Singleton active figure instance
active_figure = TerminusFigure()
