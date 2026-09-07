"""Lookup Tables, Multi-dimensional interpolation, and Prelookup blocks for Dynamic Systems."""

from __future__ import annotations
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.interpolate as interp
from .base import Block


class LookupTable1DBlock(Block):
    """1D Interpolated Lookup Table: y = f(u) via linear interpolation."""

    def __init__(self, name: str, x_data: Sequence[float], y_data: Sequence[float]):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.x_pts = np.asarray(x_data, dtype=float)
        self.y_pts = np.asarray(y_data, dtype=float)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        self.outputs[0] = float(np.interp(u, self.x_pts, self.y_pts))
        return self.outputs


class LookupTable2DBlock(Block):
    """2D Interpolated Lookup Table: z = f(x, y)."""

    def __init__(
        self,
        name: str,
        x_data: Sequence[float],
        y_data: Sequence[float],
        table_data: Sequence[Sequence[float]]
    ):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.x_pts = np.asarray(x_data, dtype=float)
        self.y_pts = np.asarray(y_data, dtype=float)
        self.table = np.asarray(table_data, dtype=float)
        self.rgi = interp.RegularGridInterpolator(
            (self.x_pts, self.y_pts),
            self.table,
            bounds_error=False,
            fill_value=None
        )
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        x = float(self.inputs[0])
        y = float(self.inputs[1]) if len(self.inputs) > 1 else 0.0
        val = self.rgi(np.array([[x, y]]))
        self.outputs[0] = float(val[0])
        return self.outputs


class LookupTableNDBlock(Block):
    """N-Dimensional Interpolated Lookup Table."""

    def __init__(self, name: str, grid_points: Sequence[Sequence[float]], table_data: np.ndarray):
        num_in = len(grid_points)
        super().__init__(name, num_inputs=num_in, num_outputs=1)
        self.grid = tuple(np.asarray(g, dtype=float) for g in grid_points)
        self.table = np.asarray(table_data, dtype=float)
        self.rgi = interp.RegularGridInterpolator(
            self.grid,
            self.table,
            bounds_error=False,
            fill_value=None
        )
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        coords = [float(x) for x in self.inputs]
        val = self.rgi(np.array([coords]))
        self.outputs[0] = float(val[0])
        return self.outputs


class DirectLookupTableNDBlock(Block):
    """Direct Indexing into N-D Table without interpolation (zero-order hold lookup)."""

    def __init__(self, name: str, table_data: np.ndarray):
        self.table = np.asarray(table_data)
        num_in = self.table.ndim
        super().__init__(name, num_inputs=num_in, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        indices = tuple(
            max(0, min(self.table.shape[i] - 1, int(round(float(self.inputs[i])))))
            for i in range(len(self.inputs))
        )
        res = self.table[indices]
        self.outputs[0] = float(res) if np.isscalar(res) else res
        return self.outputs


class LookupTableDynamicBlock(Block):
    """Dynamic 1D Lookup Table where x_data and y_data are provided as inputs:
    Input 0: u (query point)
    Input 1: x_data (breakpoints array)
    Input 2: y_data (table values array)
    """

    def __init__(self, name: str):
        super().__init__(name, num_inputs=3, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        x_pts = np.asarray(self.inputs[1], dtype=float) if len(self.inputs) > 1 else np.array([0.0, 1.0])
        y_pts = np.asarray(self.inputs[2], dtype=float) if len(self.inputs) > 2 else np.array([0.0, 1.0])
        self.outputs[0] = float(np.interp(u, x_pts, y_pts))
        return self.outputs


class PrelookupBlock(Block):
    """Compute index and fraction for Interpolation Using Prelookup block.
    Outputs: [index (int), fraction (float in [0, 1])].
    """

    def __init__(self, name: str, breakpoints: Sequence[float]):
        super().__init__(name, num_inputs=1, num_outputs=2)
        self.bp = np.asarray(breakpoints, dtype=float)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        u = float(self.inputs[0])
        idx = np.searchsorted(self.bp, u) - 1
        idx = max(0, min(len(self.bp) - 2, idx))
        x0 = self.bp[idx]
        x1 = self.bp[idx + 1]
        frac = (u - x0) / (x1 - x0) if abs(x1 - x0) > 1e-12 else 0.0
        frac = max(0.0, min(1.0, frac))
        self.outputs[0] = float(idx)
        self.outputs[1] = float(frac)
        return self.outputs


class InterpolationUsingPrelookupBlock(Block):
    """Use precalculated index and fraction to calculate interpolated table value:
    Input 0: index
    Input 1: fraction
    """

    def __init__(self, name: str, table_data: Sequence[float]):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.table = np.asarray(table_data, dtype=float)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        idx = max(0, min(len(self.table) - 2, int(round(float(self.inputs[0])))))
        frac = max(0.0, min(1.0, float(self.inputs[1]))) if len(self.inputs) > 1 else 0.0
        y0 = self.table[idx]
        y1 = self.table[idx + 1]
        self.outputs[0] = float(y0 + frac * (y1 - y0))
        return self.outputs


class SineLookupBlock(Block):
    """Implement quarter-wave symmetry Sine / Cosine lookup table approximation."""

    def __init__(self, name: str, num_points: int = 100, is_cosine: bool = False):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.num_pts = num_points
        self.is_cos = is_cosine
        # Quarter wave table [0, pi/2]
        self.quarter_x = np.linspace(0.0, math.pi / 2.0, num_points)
        self.quarter_y = np.sin(self.quarter_x)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        rad = float(self.inputs[0])
        if self.is_cos:
            rad += math.pi / 2.0
        # Wrap to [0, 2*pi)
        rad = rad % (2.0 * math.pi)

        # Exploit quarter-wave symmetry
        if rad <= math.pi / 2.0:
            val = np.interp(rad, self.quarter_x, self.quarter_y)
        elif rad <= math.pi:
            val = np.interp(math.pi - rad, self.quarter_x, self.quarter_y)
        elif rad <= 1.5 * math.pi:
            val = -np.interp(rad - math.pi, self.quarter_x, self.quarter_y)
        else:
            val = -np.interp(2.0 * math.pi - rad, self.quarter_x, self.quarter_y)

        self.outputs[0] = float(val)
        return self.outputs
