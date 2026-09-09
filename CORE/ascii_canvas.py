"""2D ASCII/Unicode Canvas rendering, A* wire routing, continuous high-res Braille plotting, and Bode/Schematic visualizers."""

from __future__ import annotations
import math
import heapq
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from .common_math import format_eng_unit, sanitize_array, SignalMetrics

BOX_CHARS = {
    "h": "─",
    "v": "│",
    "tl": "┌",
    "tr": "┐",
    "bl": "└",
    "br": "┘",
    "cross": "┼",
    "t_down": "┬",
    "t_up": "┴",
    "t_right": "├",
    "t_left": "┤",
    "arrow_r": "►",
    "arrow_l": "◄",
    "arrow_u": "▲",
    "arrow_d": "▼",
    "dot": "•",
    "circle": "○",
    "block": "█",
    "sub_block": "▀",
}

# Braille sub-pixel offsets (2 columns x 4 rows per character cell)
# Standard Unicode Braille pattern offset table
BRAILLE_DOTS = [
    [0x01, 0x08],  # Row 0 (top): left, right
    [0x02, 0x10],  # Row 1: left, right
    [0x04, 0x20],  # Row 2: left, right
    [0x40, 0x80],  # Row 3 (bottom): left, right
]


class RoutePath:
    """Represents a routed connection between 2D points on a canvas."""
    def __init__(self, points: List[Tuple[int, int]], label: str = ""):
        self.points = points
        self.label = label


class AsciiCanvas:
    """2D character grid with drawing primitives, wire routing, and component blocks."""
    
    def __init__(self, width: int = 80, height: int = 24, bg_char: str = " "):
        self.width = width
        self.height = height
        self.bg_char = bg_char
        self.grid = [[bg_char for _ in range(width)] for _ in range(height)]
        self.obstacles: Set[Tuple[int, int]] = set()
        self.junctions: Set[Tuple[int, int]] = set()

    def clear(self) -> None:
        self.grid = [[self.bg_char for _ in range(self.width)] for _ in range(self.height)]
        self.obstacles.clear()
        self.junctions.clear()

    def set_char(self, x: int, y: int, char: str) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = char[0] if char else " "

    def get_char(self, x: int, y: int) -> str:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return " "

    def draw_text(self, x: int, y: int, text: str) -> None:
        """Writes text horizontally starting at (x, y)."""
        if not (0 <= y < self.height):
            return
        for i, ch in enumerate(text):
            px = x + i
            if 0 <= px < self.width:
                self.grid[y][px] = ch

    def draw_line(self, x1: int, y1: int, x2: int, y2: int, char: str = "─") -> None:
        """Draws a line between (x1, y1) and (x2, y2) using Bresenham's line algorithm."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        cx, cy = x1, y1
        while True:
            self.set_char(cx, cy, char)
            if cx == x2 and cy == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

    def draw_box(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        title: str = "",
        as_obstacle: bool = True
    ) -> None:
        """Draws a rectangular box with corners and title."""
        if w < 2 or h < 2:
            return
        
        # Corners
        self.set_char(x, y, BOX_CHARS["tl"])
        self.set_char(x + w - 1, y, BOX_CHARS["tr"])
        self.set_char(x, y + h - 1, BOX_CHARS["bl"])
        self.set_char(x + w - 1, y + h - 1, BOX_CHARS["br"])

        # Horizontal borders
        for ix in range(x + 1, x + w - 1):
            self.set_char(ix, y, BOX_CHARS["h"])
            self.set_char(ix, y + h - 1, BOX_CHARS["h"])

        # Vertical borders
        for iy in range(y + 1, y + h - 1):
            self.set_char(x, iy, BOX_CHARS["v"])
            self.set_char(x + w - 1, iy, BOX_CHARS["v"])

        # Optional Title
        if title:
            max_len = w - 4
            disp_title = f" {title[:max_len]} " if max_len > 0 else title[:w-2]
            self.draw_text(x + 2, y, disp_title)

        if as_obstacle:
            for iy in range(y, y + h):
                for ix in range(x, x + w):
                    self.obstacles.add((ix, iy))

    def route_wire(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        arrow_end: bool = False
    ) -> List[Tuple[int, int]]:
        """A* / Rectilinear router finding collision-free orthogonal path between points."""
        start_x, start_y = start
        end_x, end_y = end

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        cost_so_far: Dict[Tuple[int, int], int] = {start: 0}

        found = False
        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current == end:
                found = True
                break

            cx, cy = current
            neighbors = [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)]
            for nx, ny in neighbors:
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if (nx, ny) in self.obstacles and (nx, ny) != end and (nx, ny) != start:
                    continue

                prev = came_from[current]
                turn_penalty = 0
                if prev is not None:
                    dx1, dy1 = cx - prev[0], cy - prev[1]
                    dx2, dy2 = nx - cx, ny - cy
                    if (dx1, dy1) != (dx2, dy2):
                        turn_penalty = 2

                new_cost = cost_so_far[current] + 1 + turn_penalty
                neighbor = (nx, ny)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, end)
                    heapq.heappush(open_set, (priority, new_cost, neighbor))
                    came_from[neighbor] = current

        path: List[Tuple[int, int]] = []
        if found:
            curr: Optional[Tuple[int, int]] = end
            while curr is not None:
                path.append(curr)
                curr = came_from.get(curr)
            path.reverse()
        else:
            path = []
            mid_x = (start_x + end_x) // 2
            step_x = 1 if end_x >= start_x else -1
            step_y = 1 if end_y >= start_y else -1
            for ix in range(start_x, mid_x + step_x, step_x):
                path.append((ix, start_y))
            for iy in range(start_y, end_y + step_y, step_y):
                path.append((mid_x, iy))
            for ix in range(mid_x, end_x + step_x, step_x):
                path.append((ix, end_y))

        for i in range(len(path)):
            px, py = path[i]
            prev_p = path[i - 1] if i > 0 else None
            next_p = path[i + 1] if i < len(path) - 1 else None

            if prev_p is None and next_p is None:
                char = BOX_CHARS["dot"]
            elif prev_p is None:
                char = BOX_CHARS["h"] if next_p[0] != px else BOX_CHARS["v"]
            elif next_p is None:
                if arrow_end:
                    if prev_p[0] < px:
                        char = BOX_CHARS["arrow_r"]
                    elif prev_p[0] > px:
                        char = BOX_CHARS["arrow_l"]
                    elif prev_p[1] < py:
                        char = BOX_CHARS["arrow_d"]
                    else:
                        char = BOX_CHARS["arrow_u"]
                else:
                    char = BOX_CHARS["h"] if prev_p[0] != px else BOX_CHARS["v"]
            else:
                dx1, dy1 = prev_p[0] - px, prev_p[1] - py
                dx2, dy2 = next_p[0] - px, next_p[1] - py
                if dx1 != 0 and dx2 != 0:
                    char = BOX_CHARS["h"]
                elif dy1 != 0 and dy2 != 0:
                    char = BOX_CHARS["v"]
                else:
                    dirs = {(dx1, dy1), (dx2, dy2)}
                    if {(0, -1), (1, 0)} == dirs:
                        char = BOX_CHARS["bl"]
                    elif {(0, -1), (-1, 0)} == dirs:
                        char = BOX_CHARS["br"]
                    elif {(0, 1), (1, 0)} == dirs:
                        char = BOX_CHARS["tl"]
                    elif {(0, 1), (-1, 0)} == dirs:
                        char = BOX_CHARS["tr"]
                    else:
                        char = BOX_CHARS["cross"]

            current_ch = self.get_char(px, py)
            if current_ch in (BOX_CHARS["h"], BOX_CHARS["v"], BOX_CHARS["cross"]) and current_ch != char:
                char = BOX_CHARS["cross"]

            self.set_char(px, py, char)

        return path

    def render(self) -> str:
        return "\n".join("".join(row) for row in self.grid)


class AsciiPlotter:
    """High-resolution continuous Braille curve plotter with sub-pixel interpolation and feature annotation."""

    @classmethod
    def plot(
        cls,
        x: Sequence[float],
        y: Sequence[float],
        width: int = 72,
        height: int = 14,
        title: str = "",
        x_label: str = "Time (s)",
        y_label: str = "Signal",
        annotate: bool = True
    ) -> str:
        """Generates a high-contrast continuous Braille wave plot with 8x sub-pixel resolution."""
        x_arr = sanitize_array(x)
        y_arr = sanitize_array(y)

        if len(x_arr) == 0 or len(y_arr) == 0:
            return "[Empty Plot Data]"

        x_min, x_max = float(np.min(x_arr)), float(np.max(x_arr))
        y_min, y_max = float(np.min(y_arr)), float(np.max(y_arr))

        if x_max == x_min:
            x_max += 1.0
        if y_max == y_min:
            y_max += 1.0
            y_min -= 1.0

        # Add small vertical margin
        y_margin = 0.05 * (y_max - y_min)
        plot_y_min = y_min - y_margin
        plot_y_max = y_max + y_margin

        # Grid dimensions in character cells
        char_w = max(20, width - 14)
        char_h = max(6, height - 3)

        # Sub-pixel dimensions (Braille is 2 horizontal dots x 4 vertical dots per cell)
        sub_w = char_w * 2
        sub_h = char_h * 4

        # Initialize Braille bitmasks: grid of char_h x char_w (each cell contains 8-bit mask)
        braille_grid = np.zeros((char_h, char_w), dtype=int)

        # Resample and draw continuous connected line segments
        interp_x = np.linspace(x_min, x_max, sub_w * 4)
        interp_y = np.interp(interp_x, x_arr, y_arr)

        def set_subpixel(sx: int, sy: int) -> None:
            if 0 <= sx < sub_w and 0 <= sy < sub_h:
                cell_x = sx // 2
                cell_y = sy // 4
                dot_x = sx % 2
                dot_y = sy % 4
                braille_grid[cell_y, cell_x] |= BRAILLE_DOTS[dot_y][dot_x]

        def draw_line(x0: int, y0: int, x1: int, y1: int) -> None:
            """Bresenham line rasterization on sub-pixel grid for gapless continuous curves."""
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            cx, cy = x0, y0
            while True:
                set_subpixel(cx, cy)
                if cx == x1 and cy == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    cx += sx
                if e2 < dx:
                    err += dx
                    cy += sy

        prev_sub_x: Optional[int] = None
        prev_sub_y: Optional[int] = None

        for px_val, py_val in zip(interp_x, interp_y):
            sub_x = int((px_val - x_min) / (x_max - x_min) * (sub_w - 1))
            # Invert Y so max is top
            sub_y = int((plot_y_max - py_val) / (plot_y_max - plot_y_min) * (sub_h - 1))
            sub_y = max(0, min(sub_h - 1, sub_y))

            if prev_sub_x is not None and prev_sub_y is not None:
                draw_line(prev_sub_x, prev_sub_y, sub_x, sub_y)
            else:
                set_subpixel(sub_x, sub_y)

            prev_sub_x = sub_x
            prev_sub_y = sub_y

        # Build output strings
        lines: List[str] = []
        if title:
            lines.append(f"  [bold cyan]{title}[/bold cyan]".center(width))

        for r in range(char_h):
            y_val = plot_y_max - (r / (char_h - 1)) * (plot_y_max - plot_y_min)
            row_chars = []
            for c in range(char_w):
                mask = braille_grid[r, c]
                if mask == 0:
                    row_chars.append(" ")
                else:
                    # Unicode Braille base is 0x2800
                    row_chars.append(chr(0x2800 + mask))

            row_str = "".join(row_chars)
            if r == 0:
                y_label_str = f"{y_val:8.2e} ┌"
            elif r == char_h - 1:
                y_label_str = f"{y_val:8.2e} └"
            elif r == char_h // 2:
                y_label_str = f"{y_val:8.2e} ┤"
            else:
                y_label_str = f"{y_val:8.2e} │"
            lines.append(f"{y_label_str}{row_str}")

        # X-axis line and ticks
        x_axis_line = " " * 10 + "└" + "─" * (char_w - 1)
        lines.append(x_axis_line)
        x_ticks = f"{' ' * 10}{x_min:8.2e}{' ' * max(2, char_w - 20)}{x_max:8.2e}"
        lines.append(x_ticks)
        if x_label:
            lines.append(f"{' ' * (10 + char_w // 2 - len(x_label) // 2)}{x_label}")

        # Feature annotations
        if annotate:
            metrics = SignalMetrics.measure_transient_metrics(x_arr, y_arr)
            anno_lines = []
            if "peak_value" in metrics:
                anno_lines.append(f"• Peak: {metrics['peak_value']:.4g} at t = {metrics.get('peak_time_s', 0):.4e}s")
            if "overshoot_pct" in metrics and metrics["overshoot_pct"] > 0.1:
                anno_lines.append(f"• Overshoot (%OS): {metrics['overshoot_pct']:.2f}%")
            if "rise_time_s" in metrics:
                anno_lines.append(f"• Rise Time (10%-90% tr): {metrics['rise_time_s']:.4e}s")
            if "settling_time_s" in metrics:
                anno_lines.append(f"• Settling Time (2% ts): {metrics['settling_time_s']:.4e}s")
            if "detected_glitches" in metrics:
                g_count = len(metrics["detected_glitches"])
                first_g = metrics["detected_glitches"][0]
                anno_lines.append(f"• [bold yellow]Notice:[/bold yellow] {g_count} sudden slope/glitch transitions detected (first at t = {first_g['time_s']:.4e}s, slew={first_g['slew_rate']:.2e})")

            if anno_lines:
                lines.append("\n  [bold green]── Analysis & Key Metrics ──[/bold green]")
                for al in anno_lines:
                    lines.append(f"  {al}")

        return "\n".join(lines)


class AsciiBodePlotter:
    """Renders continuous Frequency Response (Magnitude dB and Phase deg) with full engineering stability analysis."""

    @classmethod
    def plot_bode(
        cls,
        frequencies: Sequence[float],
        magnitude_db: Sequence[float],
        phase_deg: Sequence[float],
        width: int = 74,
        height_each: int = 10,
        title: str = "Bode Plot"
    ) -> str:
        freq_arr = sanitize_array(frequencies)
        mag_arr = sanitize_array(magnitude_db)
        ph_arr = sanitize_array(phase_deg)

        if len(freq_arr) == 0:
            return "[Empty Frequency Data]"

        log_freq = np.log10(np.where(freq_arr <= 0, 1e-12, freq_arr))

        mag_plot = AsciiPlotter.plot(
            log_freq,
            mag_arr,
            width=width,
            height=height_each,
            title=f"{title} - Magnitude Response",
            x_label="log10(Frequency [Hz])",
            y_label="Gain (dB)",
            annotate=False
        )

        phase_plot = AsciiPlotter.plot(
            log_freq,
            ph_arr,
            width=width,
            height=height_each,
            title=f"{title} - Phase Response",
            x_label="log10(Frequency [Hz])",
            y_label="Phase (°)",
            annotate=False
        )

        # Compute comprehensive frequency response engineering metrics
        metrics = SignalMetrics.measure_bode_metrics(freq_arr, mag_arr, ph_arr)
        metric_lines = ["[bold green]── Bode Frequency Response Metrics ──[/bold green]"]

        if "cutoff_fc_hz" in metrics:
            metric_lines.append(f"  • -3dB Cutoff Bandwidth (fc): [bold yellow]{format_eng_unit(metrics['cutoff_fc_hz'], 'Hz')}[/bold yellow]")
        if "resonance_freq_hz" in metrics:
            metric_lines.append(f"  • Resonance Peak (Mp): [bold yellow]+{metrics['resonance_peak_db']:.2f} dB[/bold yellow] at [bold yellow]{format_eng_unit(metrics['resonance_freq_hz'], 'Hz')}[/bold yellow] (Q ≈ {metrics.get('q_factor', 1.0):.2f})")
        if "gain_crossover_hz" in metrics:
            metric_lines.append(f"  • Gain Crossover (0dB): {format_eng_unit(metrics['gain_crossover_hz'], 'Hz')} | Phase Margin (PM): [bold cyan]{metrics.get('phase_margin_deg', 0):.1f}°[/bold cyan]")
        if "phase_crossover_hz" in metrics:
            metric_lines.append(f"  • Phase Crossover (-180°): {format_eng_unit(metrics['phase_crossover_hz'], 'Hz')} | Gain Margin (GM): [bold cyan]{metrics.get('gain_margin_db', 0):.1f} dB[/bold cyan]")
        if "rolloff_db_per_decade" in metrics:
            metric_lines.append(f"  • High-Frequency Roll-off: {metrics['rolloff_db_per_decade']} dB/decade")

        analysis_str = "\n".join(metric_lines)
        return f"{mag_plot}\n\n{phase_plot}\n\n{analysis_str}"


class SchematicVisualizer:
    """Visualizes circuit schematics and PCB layouts with compact paper-style electronic blocks and calculations."""

    @classmethod
    def get_component_short_block(cls, name: str, comp_type: str, val_str: str, probed_info: Optional[str] = None) -> str:
        """Generates a compact paper-style notation block e.g. '[ R1 10k ]' or '[ U1 NE555 ]'."""
        c_clean = comp_type.upper().replace("COMPONENT", "").replace("MODEL", "").strip()
        
        # Determine clean single-letter/short prefix
        if name.startswith(("R", "r")) or c_clean.startswith("R"):
            prefix = "R"
        elif name.startswith(("C", "c")) or c_clean.startswith("C"):
            prefix = "C"
        elif name.startswith(("L", "l")) or c_clean.startswith("L"):
            prefix = "L"
        elif name.startswith(("D", "d")) or c_clean.startswith("D") or "LED" in c_clean:
            prefix = "D"
        elif name.startswith(("Q", "q")) or c_clean.startswith("BJT") or "NPN" in c_clean or "PNP" in c_clean:
            prefix = "Q"
        elif name.startswith(("M", "m")) or c_clean.startswith("MOS") or "FET" in c_clean:
            prefix = "M"
        elif name.startswith(("U", "u")) or "OPAMP" in c_clean or "OP-AMP" in c_clean or "IC" in c_clean:
            prefix = "U"
        elif name.startswith(("V", "v")) or c_clean.startswith("VOLT") or c_clean.startswith("V"):
            prefix = "V"
        elif name.startswith(("I", "i")) or c_clean.startswith("CURR") or c_clean.startswith("I"):
            prefix = "I"
        elif name.startswith(("J", "j")) or c_clean.startswith("CONN") or c_clean.startswith("HEADER"):
            prefix = "J"
        else:
            prefix = name[:2]

        val_display = val_str.replace(" ", "") if val_str else ""
        label = f"[ {name}:{val_display} ]" if val_display else f"[ {name} ]"
        if probed_info:
            label += f" ({probed_info})"
        return label

    @classmethod
    def render_circuit_topology(
        cls,
        components: Dict[str, Any],
        pin_map: Dict[str, List[str]],
        probed_values: Optional[Dict[str, Dict[str, float]]] = None
    ) -> str:
        """Renders an ultra-clean, industry-standard paper-style circuit schematic topology with short blocks and calculations."""
        if not components:
            return "[Empty Circuit Netlist]"

        lines: List[str] = [
            "[bold cyan]┌────────────────────────────────────────────────────────┐[/bold cyan]",
            "[bold cyan]│ CIRCUIT SCHEMATIC TOPOLOGY BLOCK - PAPER STANDARD ART  │[/bold cyan]",
            "[bold cyan]└────────────────────────────────────────────────────────┘[/bold cyan]"
        ]

        net_connections: Dict[str, List[str]] = {}
        sources: List[Tuple[str, Any]] = []
        passives_and_ics: List[Tuple[str, Any]] = []
        grounded_comps: List[Tuple[str, Any]] = []

        for cname, comp in components.items():
            pins = pin_map.get(cname, getattr(comp, "nodes", []))
            for i, p in enumerate(pins):
                net = p if p != "0" else "GND"
                if net not in net_connections:
                    net_connections[net] = []
                net_connections[net].append(f"{cname}.p{i+1}")

            c_upper = cname.upper()
            if c_upper.startswith("V") or c_upper.startswith("I") or type(comp).__name__.startswith("Voltage") or type(comp).__name__.startswith("Current"):
                sources.append((cname, comp))
            elif "0" in pins or "GND" in [str(p).upper() for p in pins]:
                grounded_comps.append((cname, comp))
            else:
                passives_and_ics.append((cname, comp))

        # 1. Paper-Style Schematic ASCII Circuit Art
        lines.append("  [bold yellow]Standard Circuit Schematic (Left: In/Src ─► Middle: Core ─► Right: Out | Bottom: GND ⏚)[/bold yellow]")
        lines.append("  ─────────────────────────────────────────────────────────────────────────────────────────")

        all_ordered = sources + passives_and_ics + [gc for gc in grounded_comps if gc not in sources and gc not in passives_and_ics]
        if not all_ordered:
            all_ordered = list(components.items())

        for cname, comp in all_ordered:
            val = getattr(comp, "value", getattr(comp, "dc", ""))
            val_str = format_eng_unit(val) if isinstance(val, (int, float)) and val != 0 else str(val)
            pins = pin_map.get(cname, getattr(comp, "nodes", []))
            pins_clean = [p if p != "0" else "GND" for p in pins]

            comp_type = type(comp).__name__
            probe_str = None
            if probed_values and cname in probed_values:
                pv = probed_values[cname]
                p_parts = []
                if "v" in pv:
                    p_parts.append(f"V={pv['v']:.2f}V")
                if "i" in pv:
                    p_parts.append(f"I={format_eng_unit(pv['i'], 'A')}")
                if "p" in pv:
                    p_parts.append(f"P={format_eng_unit(pv['p'], 'W')}")
                probe_str = " | ".join(p_parts)

            block_str = cls.get_component_short_block(cname, comp_type, val_str, probe_str)
            p1_str = f"Net [{pins_clean[0]}]" if len(pins_clean) > 0 else "Net [?]"
            
            if len(pins_clean) > 1:
                p2_name = pins_clean[1]
                if p2_name == "GND":
                    p2_str = "⏚ GND"
                elif p2_name in ("VCC", "VDD", "+5V", "+3V3", "+12V"):
                    p2_str = f"⏉ {p2_name}"
                else:
                    p2_str = f"Net [{p2_name}]"
            else:
                p2_str = "⏚ GND" if len(pins_clean) == 1 and pins_clean[0] != "GND" else ""

            line = f"  {p1_str:<14} ───► {block_str} ───► {p2_str}"
            lines.append(line)

        # 2. Net junction list (Wires & Interconnects)
        lines.append("\n  [bold green]Circuit Net Connectivity (Wires & Junctions):[/bold green]")
        for net, endpoints in net_connections.items():
            end_str = " ──┼── ".join(endpoints)
            net_icon = "⏚" if net == "GND" else ("⏉" if net in ("VCC", "VDD", "+5V", "+3V3") else "•")
            lines.append(f"    {net_icon} Net [{net:<8}]:  {end_str}")

        return "\n".join(lines)

    @classmethod
    def render_pcb_board(
        cls,
        width_mm: float,
        height_mm: float,
        components: Dict[str, Any],
        tracks: List[Any],
        vias: List[Any],
        grid_cols: int = 50,
        grid_rows: int = 16
    ) -> str:
        """Renders high-clarity 2D terminal PCB layout canvas with board outline, components, traces, and vias."""
        canvas = AsciiCanvas(width=grid_cols, height=grid_rows)
        canvas.draw_box(0, 0, grid_cols, grid_rows, title=f"PCB: {width_mm:.0f}x{height_mm:.0f}mm", as_obstacle=False)

        def to_grid(x_mm: float, y_mm: float) -> Tuple[int, int]:
            gx = int(1 + (x_mm / max(1.0, width_mm)) * (grid_cols - 3))
            gy = int(1 + (y_mm / max(1.0, height_mm)) * (grid_rows - 3))
            return max(1, min(grid_cols - 2, gx)), max(1, min(grid_rows - 2, gy))

        # 1. Draw routed copper tracks
        for t in tracks:
            sx, sy = getattr(t, "start_x_mm", 0), getattr(t, "start_y_mm", 0)
            ex, ey = getattr(t, "end_x_mm", 0), getattr(t, "end_y_mm", 0)
            p1 = to_grid(sx, sy)
            p2 = to_grid(ex, ey)
            canvas.draw_line(p1[0], p1[1], p2[0], p2[1], char="─" if p1[1] == p2[1] else "│" if p1[0] == p2[0] else "╱")

        # 2. Draw vias
        for v in vias:
            vx, vy = getattr(v, "x_mm", 0), getattr(v, "y_mm", 0)
            gx, gy = to_grid(vx, vy)
            canvas.set_char(gx, gy, "◎")

        # 3. Draw component footprints and short labels
        for ref, comp in components.items():
            cx = getattr(comp, "x_mm", 0)
            cy = getattr(comp, "y_mm", 0)
            gx, gy = to_grid(cx, cy)
            tag = ref[:4]
            canvas.draw_text(max(1, gx - 1), gy, f"[{tag}]")

        board_art = canvas.render()
        lines = [
            board_art,
            f"[dim]  * Placed Components: {len(components)} | Routed Tracks: {len(tracks)} | Vias: {len(vias)}[/dim]",
            f"[dim]  * Layers: F.Cu (Top Red), B.Cu (Bottom Blue), Edge.Cuts (Outline), SilkS (Labels)[/dim]"
        ]
        return "\n".join(lines)

    @classmethod
    def render_dynamic_block_diagram(cls, blocks: Dict[str, Any], connections: List[Tuple[Tuple[str, int], Tuple[str, int]]]) -> str:
        """Draws visual block diagram of connected dynamic system blocks."""
        if not blocks:
            return "[Empty Dynamic System Diagram]"

        lines: List[str] = [
            "[bold cyan]┌────────────────────────────────────────────────────────┐[/bold cyan]",
            "[bold cyan]│            DYNAMIC SYSTEM BLOCK FLOW DIAGRAM           │[/bold cyan]",
            "[bold cyan]└────────────────────────────────────────────────────────┘[/bold cyan]"
        ]

        # Draw blocks
        for bname, block in blocks.items():
            btype = type(block).__name__.replace("Block", "")
            lines.append(f"  ┌──────────────────────┐")
            lines.append(f"  │ {bname:<8} ({btype:<9}) │")
            lines.append(f"  └──────────────────────┘")

        # Draw connection flows
        lines.append("\n  [bold green]Signal Flow Wires:[/bold green]")
        for (src_b, src_p), (dst_b, dst_p) in connections:
            lines.append(f"    [{src_b}.out{src_p}] ────────────────► [{dst_b}.in{dst_p}]")

        return "\n".join(lines)
