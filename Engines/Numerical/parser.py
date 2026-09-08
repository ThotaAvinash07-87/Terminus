"""AST Parser and interactive Workspace for MATLAB-style numerical expressions."""

from __future__ import annotations
import ast
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from CORE.common_math import Waveform, parse_eng_unit
from CORE.figure_renderer import active_figure, TechnicalReportMetrics
from .matrix_ops import (
    matrix_det,
    matrix_inv,
    matrix_rank,
    matrix_eig,
    matrix_svd,
    matrix_solve,
    matrix_lu,
    matrix_qr,
    poly_roots,
    poly_conv,
)
from .transforms import TransferFunction, DiscreteTransferFunction, fourier_transform
from .dsp_engine import DSPSignalEngine


class NumericalWorkspace:
    """Variable memory store and computation context for the Numerical engine."""

    def __init__(self):
        self.variables: Dict[str, Any] = {
            "pi": math.pi,
            "e": math.e,
            "j": 1j,
            "i": 1j,
        }
        self._init_builtins()

    def _init_builtins(self) -> None:
        self.builtins: Dict[str, Any] = {
            # Constants
            "pi": math.pi,
            "e": math.e,
            "j": 1j,
            "i": 1j,
            # Math
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "asin": np.arcsin,
            "acos": np.arccos,
            "atan": np.arctan,
            "exp": np.exp,
            "log": np.log,
            "log10": np.log10,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "angle": lambda x: np.angle(x, deg=True),
            "real": np.real,
            "imag": np.imag,
            "conj": np.conj,
            # Array creation
            "linspace": np.linspace,
            "arange": np.arange,
            "zeros": lambda *args: np.zeros(args if len(args) > 1 else (args[0], args[0])),
            "ones": lambda *args: np.ones(args if len(args) > 1 else (args[0], args[0])),
            "eye": np.eye,
            # Matrix operations
            "det": matrix_det,
            "inv": matrix_inv,
            "rank": matrix_rank,
            "eig": matrix_eig,
            "svd": matrix_svd,
            "solve": matrix_solve,
            "lu": matrix_lu,
            "qr": matrix_qr,
            "roots": poly_roots,
            "conv": poly_conv,
            # Statistics & Metrics
            "sum": np.sum,
            "mean": np.mean,
            "std": np.std,
            "min": np.min,
            "max": np.max,
            # Transforms & Control
            "tf": lambda num, den, name="H(s)": TransferFunction(num, den, name=name),
            "tf_d": lambda b, a, dt=1.0, name="H(z)": DiscreteTransferFunction(b, a, dt=dt, name=name),
            "fft": lambda wf: wf.compute_fft() if isinstance(wf, Waveform) else np.fft.rfft(wf),
            "ifft": np.fft.irfft,
            # Advanced DSP built-ins from DSPSignalEngine
            "butter": DSPSignalEngine.butterworth_filter,
            "firwin": DSPSignalEngine.fir_window_filter,
            "filter": DSPSignalEngine.apply_filter,
            "lfilter": DSPSignalEngine.apply_filter,
            "psd": DSPSignalEngine.power_spectral_density,
            "pwelch": DSPSignalEngine.power_spectral_density,
            "spectrogram": DSPSignalEngine.spectrogram,
            "chirp": DSPSignalEngine.chirp_signal,
            "square": DSPSignalEngine.square_wave,
            "sawtooth": DSPSignalEngine.sawtooth_wave,
            "bode": DSPSignalEngine.bode_response,
            "metrics": TechnicalReportMetrics.compute_from_waveform,
        }

    def get_eval_context(self) -> Dict[str, Any]:
        ctx = dict(self.builtins)
        ctx.update(self.variables)
        return ctx

    def clear(self) -> None:
        self.variables = {
            "pi": math.pi,
            "e": math.e,
            "j": 1j,
            "i": 1j,
        }

    def list_vars(self) -> Dict[str, str]:
        res = {}
        for k, v in self.variables.items():
            if k in ("pi", "e", "j", "i"):
                continue
            if isinstance(v, np.ndarray):
                res[k] = f"Array {v.shape} ({v.dtype})"
            elif isinstance(v, (TransferFunction, DiscreteTransferFunction)):
                res[k] = repr(v)
            elif isinstance(v, Waveform):
                res[k] = f"Waveform ({len(v.x)} pts)"
            elif isinstance(v, (int, float, complex)):
                res[k] = f"Scalar: {v}"
            else:
                res[k] = f"{type(v).__name__}: {v}"
        return res


class NumericalASTParser:
    """Evaluates mathematical strings, matrix syntax, assignments, expressions, and full scripts."""

    def __init__(self, workspace: Optional[NumericalWorkspace] = None):
        self.workspace = workspace or NumericalWorkspace()

    def _preprocess_syntax(self, code: str) -> str:
        """Translates MATLAB-style matrix syntax like `[1 2; 3 4]` or `0:0.01:1` to valid Python."""
        text = code.strip()

        # Handle colon range: start:step:stop or start:stop (with decimals, scientific notation, or expressions)
        def colon_repl(match: re.Match) -> str:
            raw = match.group(0)
            parts = raw.split(":")
            if len(parts) == 2:
                return f"np.arange({parts[0]}, float({parts[1]}) + 1e-12, 1.0)"
            elif len(parts) == 3:
                return f"np.arange({parts[0]}, float({parts[2]}) + (float({parts[1]}) * 0.5), float({parts[1]}))"
            return raw

        text = re.sub(r'(?<![a-zA-Z0-9_\.])\b(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\d+):(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\d+)(?::(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\d+))?\b', colon_repl, text)

        # Handle matrix brackets: e.g. [1 2; 3 4] -> np.array([[1, 2], [3, 4]])
        def matrix_bracket_repl(match: re.Match) -> str:
            inner = match.group(1).strip()
            if not inner:
                return "np.array([])"
            rows = inner.split(";")
            row_lists = []
            for r in rows:
                r_clean = r.strip()
                elements = [e.strip() for e in re.split(r'[\s,]+', r_clean) if e.strip()]
                row_lists.append(f"[{', '.join(elements)}]")
            if len(row_lists) == 1 and not inner.endswith(";"):
                return f"np.array({row_lists[0]})"
            return f"np.array([{', '.join(row_lists)}])"

        text = re.sub(r'\[([^\]]+)\]', matrix_bracket_repl, text)

        # Element-wise power .^ -> **
        text = text.replace(".^", "**")
        # Element-wise multiply .* -> *
        text = text.replace(".*", "*")
        # Element-wise divide ./ -> /
        text = text.replace("./", "/")
        # MATLAB power ^ -> **
        text = re.sub(r'(?<=[a-zA-Z0-9_\)\]])\^(?=[a-zA-Z0-9_\(\[])', '**', text)

        return text

    def execute(self, code: str) -> Any:
        """Executes a single MATLAB-like line or statement."""
        line = code.strip()
        if not line or line.startswith("%"):
            return None

        # Strip trailing comments
        if "%" in line:
            line = line[:line.find("%")].strip()

        # Strip trailing semicolon (MATLAB suppress output)
        line = line.rstrip(";").strip()
        if not line:
            return None

        # Check special commands
        if line.lower() in ("clear", "clear all"):
            self.workspace.clear()
            return "Workspace cleared."
        if line.lower() in ("who", "whos"):
            return self.workspace.list_vars()

        # Multi-variable assignment detection: [v1, v2, ...] = expr or [v1 v2] = expr
        multi_assign = re.match(r'^\[\s*([a-zA-Z0-9_,\s]+)\s*\]\s*=\s*(.+)$', line)
        if multi_assign:
            vars_str, expr_str = multi_assign.groups()
            var_names = [v.strip() for v in re.split(r'[\s,]+', vars_str) if v.strip()]
            py_code = self._preprocess_syntax(expr_str)
            ctx = self.workspace.get_eval_context()
            ctx["np"] = np
            ctx["float"] = float
            ctx["int"] = int
            ctx["len"] = len
            val = eval(py_code, {"__builtins__": {
                "float": float, "int": int, "len": len, "range": range, "abs": abs,
                "max": max, "min": min, "sum": sum, "round": round, "print": print,
                "complex": complex, "bool": bool, "str": str
            }}, ctx)
            if isinstance(val, (tuple, list)):
                for i, vname in enumerate(var_names):
                    if i < len(val):
                        self.workspace.variables[vname] = val[i]
            elif isinstance(val, np.ndarray) and len(var_names) == 1:
                self.workspace.variables[var_names[0]] = val
            else:
                self.workspace.variables[var_names[0]] = val
            return val

        # Single variable assignment detection: var_name = expr
        assign_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$', line)
        if assign_match:
            var_name, expr_str = assign_match.groups()
            py_code = self._preprocess_syntax(expr_str)
            ctx = self.workspace.get_eval_context()
            ctx["np"] = np
            ctx["float"] = float
            ctx["int"] = int
            ctx["len"] = len
            val = eval(py_code, {"__builtins__": {
                "float": float, "int": int, "len": len, "range": range, "abs": abs,
                "max": max, "min": min, "sum": sum, "round": round, "print": print,
                "complex": complex, "bool": bool, "str": str
            }}, ctx)
            self.workspace.variables[var_name] = val
            return val

        # Pure expression evaluation
        py_code = self._preprocess_syntax(line)
        ctx = self.workspace.get_eval_context()
        ctx["np"] = np
        ctx["float"] = float
        ctx["int"] = int
        ctx["len"] = len
        return eval(py_code, {"__builtins__": {
            "float": float, "int": int, "len": len, "range": range, "abs": abs,
            "max": max, "min": min, "sum": sum, "round": round, "print": print,
            "complex": complex, "bool": bool, "str": str
        }}, ctx)

    def execute_script(self, script_text: str) -> List[Tuple[int, str, Any]]:
        """Executes a multi-line MATLAB script line-by-line, returning line execution logs."""
        results = []
        lines = script_text.splitlines()
        for idx, l in enumerate(lines, 1):
            clean_l = l.strip()
            if not clean_l or clean_l.startswith("%"):
                continue
            try:
                res = self.execute(clean_l)
                results.append((idx, clean_l, res))
            except Exception as e:
                results.append((idx, clean_l, f"Error (Line {idx}): {str(e)}"))
        return results
