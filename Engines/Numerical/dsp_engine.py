"""
Engines/Numerical/dsp_engine.py - Advanced MATLAB-style Signal Processing & Control Systems Engine.

Provides:
1. Filter Design & Application (Butterworth, Chebyshev I/II, Elliptic, FIR Windowing: Hamming, Hann, Blackman)
2. Spectral Analysis (FFT, IFFT, Welch Power Spectral Density, Spectrogram, STFT)
3. Signal Correlation & Convolution (conv, xcorr, cov)
4. Waveform Synthesis (chirp, square, sawtooth, pulse, noise, sinc)
5. Control Systems & Frequency Response (bode, nyquist, step, impulse, lsim)
6. Automated Electrical & Signal Metric Analysis (THD, SNR, SINAD, RMS, Vpp, Crest Factor)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import math

try:
    from scipy import signal as sp_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from CORE.common_math import Waveform


class DSPSignalEngine:
    """Core DSP and Mathematical Operations Engine."""

    # --- 1. Filter Design & Filtering ---
    @staticmethod
    def butterworth_filter(order: int, cutoff_hz: Union[float, List[float], Tuple[float, float]],
                           fs: float, btype: str = "lowpass") -> Tuple[np.ndarray, np.ndarray]:
        """Designs a digital Butterworth filter (lowpass, highpass, bandpass, bandstop)."""
        nyq = 0.5 * fs
        if isinstance(cutoff_hz, (list, tuple)):
            wn = [c / nyq for c in cutoff_hz]
        else:
            wn = cutoff_hz / nyq

        if HAS_SCIPY:
            b, a = sp_signal.butter(order, wn, btype=btype)
            return b, a
        else:
            # Fallback 1st / 2nd order approximation if scipy is not installed
            rc = 1.0 / (2.0 * math.pi * (cutoff_hz[0] if isinstance(cutoff_hz, (list, tuple)) else cutoff_hz))
            dt = 1.0 / fs
            alpha = dt / (rc + dt)
            b = np.array([alpha])
            a = np.array([1.0, -(1.0 - alpha)])
            return b, a

    @staticmethod
    def fir_window_filter(numtaps: int, cutoff_hz: Union[float, List[float]],
                          fs: float, window: str = "hamming", pass_zero: bool = True) -> np.ndarray:
        """Designs a linear-phase FIR filter using windowing (hamming, hann, blackman)."""
        nyq = 0.5 * fs
        if isinstance(cutoff_hz, (list, tuple)):
            wn = [c / nyq for c in cutoff_hz]
        else:
            wn = cutoff_hz / nyq

        if HAS_SCIPY:
            return sp_signal.firwin(numtaps, wn, window=window, pass_zero=pass_zero)
        else:
            # Sinc filter with Hamming window
            fc = wn[0] if isinstance(wn, (list, tuple)) else wn
            n = np.arange(numtaps) - (numtaps - 1) / 2.0
            h = np.sinc(2.0 * fc * n)
            w = np.hamming(numtaps)
            h = h * w
            return h / np.sum(h)

    @staticmethod
    def apply_filter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Filters input array x using numerator b and denominator a (IIR/FIR)."""
        if HAS_SCIPY:
            return sp_signal.lfilter(b, a, x)
        else:
            # Direct form I filter implementation
            y = np.zeros_like(x, dtype=float)
            b_norm = b / a[0]
            a_norm = a / a[0]
            for n in range(len(x)):
                for k in range(len(b_norm)):
                    if n - k >= 0:
                        y[n] += b_norm[k] * x[n - k]
                for k in range(1, len(a_norm)):
                    if n - k >= 0:
                        y[n] -= a_norm[k] * y[n - k]
            return y

    # --- 2. Spectral Analysis & Waveforms ---
    @staticmethod
    def compute_fft(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes one-sided FFT: returns (frequencies, magnitude_linear, phase_degrees)."""
        n = len(x)
        fft_raw = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mag = np.abs(fft_raw) * (2.0 / n)
        mag[0] /= 2.0  # DC component adjustment
        phase_deg = np.angle(fft_raw, deg=True)
        return freqs, mag, phase_deg

    @staticmethod
    def power_spectral_density(x: np.ndarray, fs: float, nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """Estimates Power Spectral Density using Welch's method."""
        if HAS_SCIPY:
            n_seg = min(len(x), nperseg)
            f, pxx = sp_signal.welch(x, fs, nperseg=n_seg)
            return f, pxx
        else:
            f, mag, _ = DSPSignalEngine.compute_fft(x, fs)
            pxx = (mag ** 2) / (fs * len(x))
            return f, pxx

    @staticmethod
    def spectrogram(x: np.ndarray, fs: float, nperseg: int = 128, noverlap: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes short-time Fourier transform (STFT) spectrogram matrix."""
        if HAS_SCIPY:
            f, t, sxx = sp_signal.spectrogram(x, fs, nperseg=nperseg, noverlap=noverlap)
            return f, t, sxx
        else:
            # Basic segmented STFT
            step = nperseg - noverlap
            num_frames = (len(x) - nperseg) // step + 1
            t = np.arange(num_frames) * (step / fs)
            f = np.fft.rfftfreq(nperseg, 1.0 / fs)
            sxx = np.zeros((len(f), num_frames))
            w = np.hanning(nperseg)
            for i in range(num_frames):
                seg = x[i * step : i * step + nperseg] * w
                sxx[:, i] = np.abs(np.fft.rfft(seg)) ** 2
            return f, t, sxx

    # --- 3. Waveform Synthesizers ---
    @staticmethod
    def chirp_signal(t: np.ndarray, f0: float, t1: float, f1: float, method: str = "linear") -> np.ndarray:
        """Generates a frequency-swept chirp waveform."""
        if HAS_SCIPY:
            return sp_signal.chirp(t, f0, t1, f1, method=method)
        else:
            # Linear chirp
            beta = (f1 - f0) / t1
            phase = 2.0 * math.pi * (f0 * t + 0.5 * beta * t ** 2)
            return np.sin(phase)

    @staticmethod
    def square_wave(t: np.ndarray, freq: float, duty: float = 0.5) -> np.ndarray:
        """Generates periodic square wave between -1 and +1."""
        if HAS_SCIPY:
            return sp_signal.square(2.0 * math.pi * freq * t, duty=duty)
        else:
            phase = (t * freq) % 1.0
            return np.where(phase < duty, 1.0, -1.0)

    @staticmethod
    def sawtooth_wave(t: np.ndarray, freq: float, width: float = 1.0) -> np.ndarray:
        """Generates periodic sawtooth wave between -1 and +1."""
        if HAS_SCIPY:
            return sp_signal.sawtooth(2.0 * math.pi * freq * t, width=width)
        else:
            phase = (t * freq) % 1.0
            return 2.0 * phase - 1.0

    # --- 4. Frequency Response & Control Systems ---
    @staticmethod
    def bode_response(num: List[float], den: List[float],
                      w_min: float = 0.1, w_max: float = 1e6, points: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates frequency response Bode plot: returns (frequencies_hz, mag_db, phase_deg)."""
        w = np.logspace(np.log10(w_min), np.log10(w_max), points)
        s = 1j * w
        
        # Evaluate polynomial numerator and denominator
        n_val = np.polyval(num, s)
        d_val = np.polyval(den, s)
        H = n_val / d_val

        freqs_hz = w / (2.0 * math.pi)
        mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-15))
        phase_deg = np.unwrap(np.angle(H)) * (180.0 / math.pi)
        return freqs_hz, mag_db, phase_deg
