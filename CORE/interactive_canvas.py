"""2D Interactive ASCII Canvas Engine for TerminusECE.

Provides positionable, movable component blocks ([ R1 10k ], [ C1 100n ], [ U1 OP-AMP ]),
orthogonal pin-to-pin wiring, interactive cursor selection, and probed value overlays.
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any


def format_compact_val(val: Any) -> str:
    """Formats numeric or string values into clean engineering representations (e.g. 1k, 100n, 10u)."""
    if val is None or val == "":
        return ""
    if isinstance(val, (int, float)):
        v = float(val)
        if v == 0:
            return "0"
        abs_v = abs(v)
        for mult, prefix in [
            (1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'k'), (1.0, ''),
            (1e-3, 'm'), (1e-6, 'u'), (1e-9, 'n'), (1e-12, 'p'), (1e-15, 'f')
        ]:
            if abs_v >= mult * 0.9999:
                scaled = v / mult
                if abs(scaled - round(scaled)) < 1e-4:
                    return f"{int(round(scaled))}{prefix}"
                else:
                    return f"{scaled:.3g}{prefix}"
        return f"{v:.3g}"

    s_val = str(val).strip()
    try:
        # Check if standard float or exponent
        f_val = float(s_val)
        return format_compact_val(f_val)
    except Exception:
        pass
    return s_val


@dataclass
class CanvasBlock:
    """Positionable component block on the 2D schematic grid."""
    ref: str                     # e.g. R1, C1, U1, INT1, GAIN1
    label: str                   # e.g. [ R1 10k ], [ U1 OP-AMP ], [ GAIN K=5 ]
    x: int                       # Grid column coordinate
    y: int                       # Grid row coordinate
    width: int = 12              # Block character width
    height: int = 3              # Block character height
    pins: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # pin_name -> (rel_x, rel_y)
    selected: bool = False
    probed_v: Optional[float] = None
    probed_i: Optional[float] = None
    probed_p: Optional[float] = None
    category: str = "PASSIVE"


@dataclass
class CanvasWire:
    """Orthogonal wire connection between component pins."""
    src_block: str
    src_pin: str
    dst_block: str
    dst_pin: str
    net_name: str
    points: List[Tuple[int, int]] = field(default_factory=list)


class InteractiveSchematicCanvas:
    """Manages 2D movable blocks, wire routing, and screen drawing for 60% left panel."""

    def __init__(self, width_chars: int = 68, height_chars: int = 22):
        self.width = width_chars
        self.height = height_chars
        self.blocks: Dict[str, CanvasBlock] = {}
        self.wires: List[CanvasWire] = []
        self.selected_ref: Optional[str] = None
        self.cursor_x = 2
        self.cursor_y = 2

    def clear(self):
        self.blocks.clear()
        self.wires.clear()
        self.selected_ref = None

    def add_or_update_block(
        self,
        ref: str,
        label: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
        category: str = "PASSIVE",
        pins: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> CanvasBlock:
        """Adds or updates a movable block on the 2D grid."""
        clean_ref = ref.strip().upper()
        if clean_ref in self.blocks:
            block = self.blocks[clean_ref]
            block.label = label
            if x is not None: block.x = max(1, min(self.width - block.width - 1, x))
            if y is not None: block.y = max(1, min(self.height - block.height - 1, y))
            return block

        # Auto position if none provided based on topology
        if x is None or y is None:
            idx = len(self.blocks)
            # Layout horizontally with vertical branch offset for ground / loads
            col = (idx % 3) * 18 + 2
            row = (idx // 3) * 5 + 2
            x = max(1, min(self.width - 16, col))
            y = max(1, min(self.height - 4, row))

        # Default pins (Pin 1 on Left, Pin 2 on Right)
        if not pins:
            pins = {
                "1": (0, 1),
                "2": (len(label) - 1, 1) if len(label) > 6 else (10, 1)
            }

        block_w = max(len(label) + 2, 10)
        block = CanvasBlock(
            ref=clean_ref,
            label=label,
            x=x,
            y=y,
            width=block_w,
            height=3,
            pins=pins,
            category=category
        )
        self.blocks[clean_ref] = block
        return block

    def move_block(self, ref: str, new_x: int, new_y: int) -> bool:
        """Reposition a component block to absolute coordinates."""
        clean_ref = ref.strip().upper()
        if clean_ref not in self.blocks:
            return False
        block = self.blocks[clean_ref]
        block.x = max(1, min(self.width - block.width - 1, new_x))
        block.y = max(1, min(self.height - block.height - 1, new_y))
        self.route_all_wires()
        return True

    def move_block_relative(self, ref: str, dx: int, dy: int) -> bool:
        """Reposition a component block relative to its current location."""
        clean_ref = ref.strip().upper()
        if clean_ref not in self.blocks:
            return False
        block = self.blocks[clean_ref]
        return self.move_block(clean_ref, block.x + dx, block.y + dy)

    def select_block(self, ref_or_coord: Any) -> Optional[CanvasBlock]:
        """Select a block by reference name or (x, y) coordinate."""
        for b in self.blocks.values():
            b.selected = False

        if isinstance(ref_or_coord, str):
            clean_ref = ref_or_coord.strip().upper()
            if clean_ref in self.blocks:
                self.blocks[clean_ref].selected = True
                self.selected_ref = clean_ref
                return self.blocks[clean_ref]
        elif isinstance(ref_or_coord, (tuple, list)) and len(ref_or_coord) == 2:
            cx, cy = ref_or_coord
            for ref, b in self.blocks.items():
                if b.x <= cx <= b.x + b.width and b.y <= cy <= b.y + b.height:
                    b.selected = True
                    self.selected_ref = ref
                    return b

        self.selected_ref = None
        return None

    def add_wire(self, src_b: str, src_p: str, dst_b: str, dst_p: str, net_name: str = "WIRE"):
        """Connects two block pins with an orthogonal wire."""
        wire = CanvasWire(src_block=src_b, src_pin=src_p, dst_block=dst_b, dst_pin=dst_p, net_name=net_name)
        self.wires.append(wire)
        self.route_all_wires()

    def route_all_wires(self):
        """Calculates orthogonal Manhattan paths for all active wires."""
        for w in self.wires:
            b1 = self.blocks.get(w.src_block.upper())
            b2 = self.blocks.get(w.dst_block.upper())
            if not b1 or not b2:
                continue

            p1_rel = b1.pins.get(w.src_pin, (b1.width, 1))
            p2_rel = b2.pins.get(w.dst_pin, (0, 1))

            start_x = b1.x + p1_rel[0]
            start_y = b1.y + p1_rel[1]
            end_x = b2.x + p2_rel[0]
            end_y = b2.y + p2_rel[1]

            # Orthogonal path: (x1, y1) -> (mid_x, y1) -> (mid_x, y2) -> (x2, y2)
            mid_x = (start_x + end_x) // 2
            path = []

            # Segment 1: start_x to mid_x at start_y
            step_x = 1 if mid_x >= start_x else -1
            for px in range(start_x, mid_x + step_x, step_x):
                path.append((px, start_y))

            # Segment 2: start_y to end_y at mid_x
            step_y = 1 if end_y >= start_y else -1
            for py in range(start_y, end_y + step_y, step_y):
                path.append((mid_x, py))

            # Segment 3: mid_x to end_x at end_y
            step_x2 = 1 if end_x >= mid_x else -1
            for px in range(mid_x, end_x + step_x2, step_x2):
                path.append((px, end_y))

            w.points = path

    def render_grid_lines(self, width: Optional[int] = None, height: Optional[int] = None) -> List[str]:
        """Renders only the 2D grid character rows without outer box frames."""
        w_cols = width or self.width
        h_rows = height or self.height
        grid = [[" " for _ in range(w_cols)] for _ in range(h_rows)]

        # 1. Draw Wires
        for w in self.wires:
            for i, (px, py) in enumerate(w.points):
                if 0 <= px < w_cols and 0 <= py < h_rows:
                    prev_p = w.points[i - 1] if i > 0 else None
                    next_p = w.points[i + 1] if i < len(w.points) - 1 else None
                    if prev_p and next_p:
                        if prev_p[1] == py == next_p[1]:
                            grid[py][px] = "─"
                        elif prev_p[0] == px == next_p[0]:
                            grid[py][px] = "│"
                        else:
                            # Corner junction
                            if (prev_p[0] < px and next_p[1] > py) or (next_p[0] < px and prev_p[1] > py):
                                grid[py][px] = "┐"
                            elif (prev_p[0] < px and next_p[1] < py) or (next_p[0] < px and prev_p[1] < py):
                                grid[py][px] = "┘"
                            elif (prev_p[0] > px and next_p[1] > py) or (next_p[0] > px and prev_p[1] > py):
                                grid[py][px] = "┌"
                            elif (prev_p[0] > px and next_p[1] < py) or (next_p[0] > px and prev_p[1] < py):
                                grid[py][px] = "└"
                            else:
                                grid[py][px] = "┼"
                    elif prev_p:
                        grid[py][px] = "─" if prev_p[1] == py else "│"
                    else:
                        grid[py][px] = "●"

        # 2. Draw Component Blocks
        for ref, b in self.blocks.items():
            bx, by = b.x, b.y
            bw, bh = b.width, b.height

            # Draw block border
            border_h = "═" if b.selected else "─"
            border_v = "║" if b.selected else "│"
            c_tl = "╔" if b.selected else "┌"
            c_tr = "╗" if b.selected else "┐"
            c_bl = "╚" if b.selected else "└"
            c_br = "╝" if b.selected else "┘"

            if 0 <= by < h_rows and 0 <= bx < w_cols:
                # Top edge
                for x in range(bx, min(w_cols, bx + bw)):
                    grid[by][x] = border_h
                grid[by][bx] = c_tl
                if bx + bw - 1 < w_cols:
                    grid[by][bx + bw - 1] = c_tr

                # Bottom edge
                if by + bh - 1 < h_rows:
                    for x in range(bx, min(w_cols, bx + bw)):
                        grid[by + bh - 1][x] = border_h
                    grid[by + bh - 1][bx] = c_bl
                    if bx + bw - 1 < w_cols:
                        grid[by + bh - 1][bx + bw - 1] = c_br

                # Sides & Inner content
                for y in range(by + 1, min(h_rows, by + bh - 1)):
                    grid[y][bx] = border_v
                    if bx + bw - 1 < w_cols:
                        grid[y][bx + bw - 1] = border_v

                # Draw Label centered inside block
                mid_y = by + 1
                if 0 <= mid_y < h_rows:
                    disp_label = f" {b.label} "
                    start_x = max(bx + 1, bx + (bw - len(disp_label)) // 2)
                    for i, ch in enumerate(disp_label):
                        if 0 <= start_x + i < bx + bw - 1 and start_x + i < w_cols:
                            grid[mid_y][start_x + i] = ch

                # Probed overlay if available (e.g. 12.0V, 1.2mA)
                if b.probed_v is not None or b.probed_i is not None:
                    p_info = ""
                    if b.probed_v is not None: p_info += f"{b.probed_v:.2g}V "
                    if b.probed_i is not None: p_info += f"{b.probed_i*1e3:.2g}mA"
                    p_info = p_info.strip()
                    py_tag = min(h_rows - 1, by + bh)
                    for i, ch in enumerate(f"({p_info})"):
                        if 0 <= bx + i < w_cols:
                            grid[py_tag][bx + i] = ch

        return ["".join(row) for row in grid]

    def render(self, mode_title: str = "Circuit Schematic") -> str:
        """Renders the 2D interactive canvas as framed ASCII text."""
        grid_lines = self.render_grid_lines()
        rendered_lines = []
        header = f"┌─── [ 60% Interactive Canvas: {mode_title} ] "
        header += "─" * max(0, self.width - len(header) - 1) + "┐"
        rendered_lines.append(f"[bold cyan]{header}[/bold cyan]")

        for row_str in grid_lines:
            rendered_lines.append("│" + row_str + "│")

        # Footer info line
        sel_text = f"Selected: {self.selected_ref} ('move {self.selected_ref} <x> <y>')" if self.selected_ref else "Click/Move components"
        footer = f"└─── [ {sel_text} ] "
        footer += "─" * max(0, self.width - len(footer) - 1) + "┘"
        rendered_lines.append(f"[bold cyan]{footer}[/bold cyan]")

        return "\n".join(rendered_lines)

