"""
Unit & Integration Tests for MATLAB/DSP Signal Processing, Multi-Graph PNG Figure Renderer,
and In-Terminal Truecolor Previewer.

Tests:
1. TerminusFigure & SubplotLayout (1x1, 1x2, 2x1, 2x2 grids, trace styles)
2. TechnicalReportMetrics (Fs, Vrms, Vpp, THD, SNR, dominant frequency extraction)
3. PNG Export & TerminalImagePreviewer (24-bit Truecolor ANSI Half-Block preview)
4. DSPSignalEngine (Butterworth, FIR, FFT, PSD, Spectrogram, Chirp, Square, Bode)
5. MatlabProjectManager (Documents/Terminus Files/Numerical/<proj>/<proj>.m management)
6. TerminusEngineBridge CLI Integration across subsystems
"""

import os
import sys
import unittest
import tempfile
import shutil
import time
from pathlib import Path
import numpy as np

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CORE.figure_renderer import TerminusFigure, TerminalImagePreviewer, TechnicalReportMetrics, active_figure
from CORE.storage_manager import StorageManager
from Engines.Numerical.dsp_engine import DSPSignalEngine
from Engines.Numerical.matlab_manager import MatlabProjectManager, MatlabProjectFile
from Engines.Numerical.parser import NumericalWorkspace, NumericalASTParser
from UI.app import TerminusEngineBridge


class TestFigureAndPreviewRenderer(unittest.TestCase):
    def setUp(self):
        self.fig = TerminusFigure(title="Test Signal Processing Figure")
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_multi_subplot_configuration(self):
        self.fig.set_subplot_grid(2, 2, 1)
        t = np.linspace(0, 1, 100)
        self.fig.add_trace(t, np.sin(2 * np.pi * 5 * t), label="Sine 5Hz")

        self.fig.set_subplot_grid(2, 2, 4)
        self.fig.add_trace(t, np.cos(2 * np.pi * 10 * t), label="Cosine 10Hz", color="#ffab00")

        self.assertEqual(self.fig.rows, 2)
        self.assertEqual(self.fig.cols, 2)
        self.assertIn(0, self.fig.subplots)
        self.assertIn(3, self.fig.subplots)
        self.assertEqual(len(self.fig.subplots[0].traces), 1)
        self.assertEqual(len(self.fig.subplots[3].traces), 1)

    def test_technical_report_metrics_computation(self):
        t = np.linspace(0, 0.1, 1000)
        sig = 2.0 * np.sin(2 * np.pi * 50 * t)  # 50 Hz pure sine wave
        metrics = TechnicalReportMetrics.compute_from_waveform(t, sig, mode="MATLAB", project="TestProj")

        self.assertAlmostEqual(metrics.v_peak, 2.0, places=1)
        self.assertAlmostEqual(metrics.v_rms, 2.0 / np.sqrt(2), places=1)
        self.assertAlmostEqual(metrics.v_pp, 4.0, places=1)
        self.assertIsNotNone(metrics.sampling_freq)
        self.assertAlmostEqual(metrics.sampling_freq, 10000.0, delta=100.0)
        self.assertIsNotNone(metrics.dominant_freq)
        self.assertAlmostEqual(metrics.dominant_freq, 50.0, delta=15.0)

    def test_png_export_and_in_terminal_preview(self):
        t = np.linspace(0, 0.05, 500)
        sig = np.sin(2 * np.pi * 60 * t) + 0.3 * np.sin(2 * np.pi * 180 * t)
        self.fig.add_trace(t, sig, label="Harmonic Signal")
        self.fig.metrics = TechnicalReportMetrics.compute_from_waveform(t, sig)

        out_png = Path(self.temp_dir) / "test_report.png"
        ok, msg = self.fig.export_to_png(out_png, dpi=100)
        self.assertTrue(ok)
        self.assertTrue(out_png.exists())
        self.assertGreater(out_png.stat().st_size, 500)

        # Test Truecolor terminal half-block preview
        preview_str = TerminalImagePreviewer.preview_image(out_png, max_cols=30, max_rows=10)
        self.assertIn("Truecolor Graphic Preview", preview_str)
        self.assertIn("▀", preview_str)


class TestDSPSignalEngine(unittest.TestCase):
    def test_butterworth_filter_design_and_apply(self):
        fs = 1000.0
        cutoff = 100.0
        b, a = DSPSignalEngine.butterworth_filter(4, cutoff, fs, btype="lowpass")
        self.assertEqual(len(b), len(a))

        # Filter a test signal
        t = np.linspace(0, 0.2, 200)
        sig = np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 300 * t)
        filtered = DSPSignalEngine.apply_filter(b, a, sig)
        self.assertEqual(len(filtered), len(sig))

    def test_fir_window_filter(self):
        fs = 1000.0
        h = DSPSignalEngine.fir_window_filter(31, 150.0, fs, window="hamming")
        self.assertEqual(len(h), 31)

    def test_fft_and_psd(self):
        fs = 1000.0
        t = np.linspace(0, 1.0, 1000, endpoint=False)
        sig = np.sin(2 * np.pi * 75 * t)
        freqs, mag, phase = DSPSignalEngine.compute_fft(sig, fs)
        peak_freq = freqs[np.argmax(mag)]
        self.assertAlmostEqual(peak_freq, 75.0, delta=2.0)

        f_psd, pxx = DSPSignalEngine.power_spectral_density(sig, fs, nperseg=256)
        self.assertGreater(len(f_psd), 0)
        self.assertGreater(len(pxx), 0)

    def test_waveform_synthesizers(self):
        t = np.linspace(0, 1.0, 500)
        chirp = DSPSignalEngine.chirp_signal(t, 10, 1.0, 100)
        self.assertEqual(len(chirp), 500)

        sq = DSPSignalEngine.square_wave(t, 5.0)
        self.assertEqual(len(sq), 500)
        self.assertTrue(np.all(np.abs(sq) <= 1.0))

    def test_bode_response(self):
        # 2nd order lowpass filter H(s) = 1 / (s^2 + 1.414s + 1)
        num = [1.0]
        den = [1.0, 1.414, 1.0]
        f_hz, mag_db, phase_deg = DSPSignalEngine.bode_response(num, den, w_min=0.01, w_max=100.0, points=100)
        self.assertEqual(len(f_hz), 100)
        self.assertEqual(len(mag_db), 100)
        self.assertEqual(len(phase_deg), 100)
        self.assertAlmostEqual(mag_db[0], 0.0, delta=0.5)


class TestMatlabProjectAndParser(unittest.TestCase):
    def setUp(self):
        self.mgr = MatlabProjectManager("DSPFilterLab")
        self.parser = NumericalASTParser()

    def test_project_folder_management(self):
        self.assertEqual(self.mgr.project_name, "DSPFilterLab")
        self.assertTrue(self.mgr.active_filename.endswith("DSPFilterLab.m"))

        ok, msg = self.mgr.save_project()
        self.assertTrue(ok)
        main_p = self.mgr.get_main_filepath()
        self.assertTrue(main_p.exists())
        self.assertEqual(main_p.name, "DSPFilterLab.m")

    def test_colon_and_matrix_syntax(self):
        # Colon range
        res_range = self.parser.execute("t = 0:0.01:0.1")
        self.assertEqual(len(res_range), 11)

        # Matrix bracket syntax
        res_mat = self.parser.execute("M = [1 2; 3 4]")
        self.assertEqual(res_mat.shape, (2, 2))
        self.assertEqual(res_mat[0, 1], 2)

    def test_script_execution(self):
        script = """
        % Filter script
        Fs = 1000;
        t = 0:0.001:0.05;
        x = sin(2*pi*50*t);
        [b, a] = butter(2, 50, Fs, 'lowpass');
        y = filter(b, a, x);
        mean_val = mean(x);
        """
        logs = self.parser.execute_script(script)
        self.assertGreaterEqual(len(logs), 6)
        self.assertIn("x", self.parser.workspace.variables)
        self.assertIn("b", self.parser.workspace.variables)
        self.assertIn("a", self.parser.workspace.variables)
        self.assertIn("y", self.parser.workspace.variables)
        self.assertIn("mean_val", self.parser.workspace.variables)


class TestTerminusBridgeMatlabAndFigureCommands(unittest.TestCase):
    def setUp(self):
        self.bridge = TerminusEngineBridge()
        self.bridge.execute_command("mode numerical")

    def test_matlab_project_and_code_commands(self):
        res_new = self.bridge.execute_command("project SignalLab")
        self.assertIn("Created MATLAB project 'SignalLab'", res_new)

        res_code = self.bridge.execute_command("code")
        self.assertIn("SignalLab.m", res_code)

        res_append = self.bridge.execute_command("code append % New signal line")
        self.assertIn("Appended line", res_append)

    def test_plot_and_stem_commands(self):
        self.bridge.execute_command("t = 0:0.005:0.1")
        self.bridge.execute_command("y = sin(2*pi*20*t)")

        out_plot = self.bridge.execute_command("plot(t, y)")
        self.assertIn("MATLAB Plot", out_plot)
        self.assertIn("Vpk", out_plot)

        out_stem = self.bridge.execute_command("stem(t, y)")
        self.assertIn("MATLAB Plot", out_stem)

    def test_subplot_and_bode_commands(self):
        out_sub = self.bridge.execute_command("subplot 2 1 1")
        self.assertIn("Set active subplot", out_sub)

        self.bridge.execute_command("H = tf([1], [1, 2, 1])")
        out_bode = self.bridge.execute_command("bode(H)")
        self.assertIn("Bode Plot", out_bode)

    def test_universal_png_export_and_preview(self):
        # Numerical mode PNG
        self.bridge.execute_command("t = 0:0.01:0.1; s = cos(2*pi*10*t)")
        self.bridge.execute_command("plot(t, s)")
        out_png = self.bridge.execute_command("png matlab_test.png")
        self.assertIn("High-Resolution PNG Graphic Generated", out_png)
        self.assertIn("matlab_test.png", out_png)

        # Preview command
        out_prev = self.bridge.execute_command("preview matlab_test.png")
        self.assertIn("Truecolor Graphic Preview", out_prev)

        # Circuit mode PNG
        self.bridge.execute_command("mode circuit")
        self.bridge.execute_command("add V1 10V")
        self.bridge.execute_command("add R1 1k")
        self.bridge.execute_command("connect V1.p | R1.a")
        self.bridge.execute_command("connect R1.b | 0")
        self.bridge.execute_command("connect V1.n | 0")
        self.bridge.execute_command("run .op")
        out_ckt_png = self.bridge.execute_command("png circuit_test.png")
        self.assertIn("High-Resolution PNG Graphic Generated", out_ckt_png)


if __name__ == '__main__':
    unittest.main()
