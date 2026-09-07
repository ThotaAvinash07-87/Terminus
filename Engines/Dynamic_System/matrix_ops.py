"""Matrix and Multidimensional Array operations for Dynamic Systems."""

from __future__ import annotations
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from .base import Block


class MatrixConcatenateBlock(Block):
    """Concatenate input matrices along specified axis (0 for vertical/rows, 1 for horizontal/cols)."""

    def __init__(self, name: str, axis: int = 0, num_inputs: int = 2):
        super().__init__(name, num_inputs=num_inputs, num_outputs=1)
        self.axis = int(axis)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        arrs = [np.asarray(x) for x in self.inputs]
        res = np.concatenate(arrs, axis=self.axis)
        self.outputs[0] = res
        return self.outputs


class VectorConcatenateBlock(Block):
    """Concatenate input vectors into a single 1D vector."""

    def __init__(self, name: str, num_inputs: int = 2):
        super().__init__(name, num_inputs=num_inputs, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        arrs = [np.atleast_1d(x) for x in self.inputs]
        res = np.concatenate(arrs)
        self.outputs[0] = res
        return self.outputs


class CreateDiagonalMatrixBlock(Block):
    """Create square diagonal matrix from diagonal vector."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        v = np.asarray(self.inputs[0])
        self.outputs[0] = np.diag(v)
        return self.outputs


class ExtractDiagonalBlock(Block):
    """Extract main diagonal of input matrix."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        m = np.asarray(self.inputs[0])
        self.outputs[0] = np.diag(m)
        return self.outputs


class TransposeBlock(Block):
    """Compute transpose of matrix."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        m = np.asarray(self.inputs[0])
        self.outputs[0] = m.T
        return self.outputs


class HermitianTransposeBlock(Block):
    """Compute conjugate / hermitian transpose of matrix."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        m = np.asarray(self.inputs[0], dtype=complex)
        self.outputs[0] = m.conj().T
        return self.outputs


class MatrixSquareBlock(Block):
    """Compute square of matrix (M @ M)."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        m = np.asarray(self.inputs[0])
        self.outputs[0] = m @ m
        return self.outputs


class IdentityMatrixBlock(Block):
    """Generate NxN identity matrix."""

    def __init__(self, name: str, size: int = 3):
        super().__init__(name, num_inputs=0, num_outputs=1)
        self.size = int(size)
        self.direct_feedthrough = False

    def compute_output(self, t: float) -> List[Any]:
        self.outputs[0] = np.eye(self.size)
        return self.outputs


class IsHermitianBlock(Block):
    """Check if matrix is Hermitian (M == M.conj().T) or skew-Hermitian."""

    def __init__(self, name: str, check_skew: bool = False):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.skew = check_skew
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        m = np.asarray(self.inputs[0], dtype=complex)
        target = -m.conj().T if self.skew else m.conj().T
        is_h = np.allclose(m, target)
        self.outputs[0] = 1.0 if is_h else 0.0
        return self.outputs


class IsSymmetricBlock(Block):
    """Check if matrix is symmetric (M == M.T) or skew-symmetric (M == -M.T)."""

    def __init__(self, name: str, check_skew: bool = False):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.skew = check_skew
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        m = np.asarray(self.inputs[0])
        target = -m.T if self.skew else m.T
        is_sym = np.allclose(m, target)
        self.outputs[0] = 1.0 if is_sym else 0.0
        return self.outputs


class IsTriangularBlock(Block):
    """Check if matrix is upper or lower triangular."""

    def __init__(self, name: str, triangular_type: str = "Upper"):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.tri_type = triangular_type.lower()
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[float]:
        m = np.asarray(self.inputs[0])
        if "low" in self.tri_type:
            is_tri = np.allclose(m, np.tril(m))
        else:
            is_tri = np.allclose(m, np.triu(m))
        self.outputs[0] = 1.0 if is_tri else 0.0
        return self.outputs


class CrossProductBlock(Block):
    """Cross product of two 3D vectors."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        v1 = np.asarray(self.inputs[0])
        v2 = np.asarray(self.inputs[1]) if len(self.inputs) > 1 else np.zeros(3)
        self.outputs[0] = np.cross(v1, v2)
        return self.outputs


class ExpandScalarBlock(Block):
    """Create matrix of shape (rows, cols) filled with input scalar."""

    def __init__(self, name: str, rows: int = 3, cols: int = 3):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.rows = int(rows)
        self.cols = int(cols)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        val = float(self.inputs[0])
        self.outputs[0] = np.full((self.rows, self.cols), val)
        return self.outputs


class SubmatrixBlock(Block):
    """Select subset of elements (submatrix) from matrix input: [row_start:row_end, col_start:col_end]."""

    def __init__(self, name: str, row_span: Tuple[int, int] = (0, 1), col_span: Tuple[int, int] = (0, 1)):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.r_span = row_span
        self.c_span = col_span
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        m = np.asarray(self.inputs[0])
        self.outputs[0] = m[self.r_span[0]:self.r_span[1]+1, self.c_span[0]:self.c_span[1]+1]
        return self.outputs


class ReshapeBlock(Block):
    """Change dimensionality and shape of signal."""

    def __init__(self, name: str, output_shape: Tuple[int, ...]):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.target_shape = output_shape
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        arr = np.asarray(self.inputs[0])
        self.outputs[0] = arr.reshape(self.target_shape)
        return self.outputs


class SqueezeBlock(Block):
    """Remove singleton dimensions from multidimensional signal."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        arr = np.asarray(self.inputs[0])
        self.outputs[0] = np.squeeze(arr)
        return self.outputs


class PermuteMatrixBlock(Block):
    """Permute/Reorder dimensions of multidimensional array."""

    def __init__(self, name: str, order: Tuple[int, ...]):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.order = order
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        arr = np.asarray(self.inputs[0])
        self.outputs[0] = np.transpose(arr, self.order)
        return self.outputs


PermuteDimensionsBlock = PermuteMatrixBlock


class ProductMatrixMultiplyBlock(Block):
    """Matrix Multiplication M1 @ M2."""

    def __init__(self, name: str):
        super().__init__(name, num_inputs=2, num_outputs=1)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        m1 = np.asarray(self.inputs[0])
        m2 = np.asarray(self.inputs[1]) if len(self.inputs) > 1 else np.eye(m1.shape[-1])
        self.outputs[0] = m1 @ m2
        return self.outputs


class ArrayProcessingSubsystemBlock(Block):
    """Apply elementwise function algorithm to each element of matrix."""

    def __init__(self, name: str, element_func: Callable[[Any], Any] = lambda x: x * 2):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.func = np.vectorize(element_func)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        m = np.asarray(self.inputs[0])
        self.outputs[0] = self.func(m)
        return self.outputs


class NeighborhoodProcessingSubsystemBlock(Block):
    """Apply 2D spatial filter kernel (convolution) across neighborhood."""

    def __init__(self, name: str, kernel: Sequence[Sequence[float]]):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.kernel = np.asarray(kernel, dtype=float)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        import scipy.signal as sig
        img = np.asarray(self.inputs[0], dtype=float)
        if img.ndim == 2:
            self.outputs[0] = sig.convolve2d(img, self.kernel, mode='same', boundary='symm')
        else:
            self.outputs[0] = img
        return self.outputs


class PixelProcessingSubsystemBlock(Block):
    """Convert multichannel image to single channel (e.g., RGB to Grayscale)."""

    def __init__(self, name: str, weights: Tuple[float, float, float] = (0.2989, 0.5870, 0.1140)):
        super().__init__(name, num_inputs=1, num_outputs=1)
        self.w = np.asarray(weights, dtype=float)
        self.direct_feedthrough = True

    def compute_output(self, t: float) -> List[Any]:
        img = np.asarray(self.inputs[0], dtype=float)
        if img.ndim == 3 and img.shape[-1] == 3:
            self.outputs[0] = np.dot(img[..., :3], self.w)
        else:
            self.outputs[0] = img
        return self.outputs
