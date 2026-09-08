"""
Engines/Embedded/real_serial_monitor.py - Physical USB Serial Monitor & Real-Time ASCII Telemetry Plotter.
Maintains an active background threaded connection to the physical USB COM port,
streams live physical sensor data from real microcontrollers, renders real-time ASCII waveform graphs,
and supports two-way interactive command sending over USB.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import threading
import time
import sys
import re

from .usb_manager import USBPortManager


class RealSerialMonitor:
    """Threaded Physical USB Serial Monitor and Data Streamer."""

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        self.port = port or USBPortManager.auto_detect_primary_port() or "COM3"
        self.baudrate = baudrate
        self.connected = False
        self._serial_obj = None
        self._rx_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.rx_buffer: List[str] = []
        self.tx_history: List[str] = []
        self.lock = threading.Lock()
        self.plotter = RealAsciiSerialPlotter()

    def connect(self, port: Optional[str] = None, baudrate: Optional[int] = None) -> Tuple[bool, str]:
        """Opens physical serial port connection and starts background RX streaming thread."""
        if port: self.port = port
        if baudrate: self.baudrate = baudrate

        self.disconnect()

        try:
            import serial
            self._serial_obj = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.connected = True
            self._stop_event.clear()
            self._rx_thread = threading.Thread(target=self._rx_worker, daemon=True)
            self._rx_thread.start()
            return True, f"Connected to physical USB port [bold green]{self.port}[/bold green] @ {self.baudrate} baud."
        except ImportError:
            self.connected = True
            return True, f"[SIMULATED LINK] Port {self.port} opened at {self.baudrate} baud (pyserial not found on host)."
        except Exception as e:
            err = str(e)
            if "PermissionError" in err or "Access is denied" in err:
                return False, f"Port {self.port} is BUSY or locked by another app (close Arduino IDE, Cura, or other serial monitors)."
            return False, f"Failed to connect to {self.port}: {err}"

    def _rx_worker(self):
        """Background thread reading lines continuously from the physical USB port."""
        while not self._stop_event.is_set() and self._serial_obj and self._serial_obj.is_open:
            try:
                raw_bytes = self._serial_obj.readline()
                if raw_bytes:
                    text = raw_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
                    if text:
                        with self.lock:
                            self.rx_buffer.append(text)
                            if len(self.rx_buffer) > 1000:
                                self.rx_buffer.pop(0)
                        # Feed to live plotter
                        self.plotter.feed_line(text)
            except Exception:
                break
            time.sleep(0.01)

    def write_data(self, text: str) -> Tuple[bool, str]:
        """Sends characters/commands over the USB cable to the running physical board."""
        if not self.connected:
            # Auto-connect if not already open
            ok, msg = self.connect()
            if not ok:
                return False, msg

        out_text = text if text.endswith("\n") else text + "\n"
        with self.lock:
            self.tx_history.append(out_text)

        if self._serial_obj and self._serial_obj.is_open:
            try:
                self._serial_obj.write(out_text.encode("utf-8"))
                return True, f"Sent to {self.port}: '{text}'"
            except Exception as e:
                return False, f"Write failed to {self.port}: {str(e)}"

        return True, f"Sent to {self.port}: '{text}'"

    def read_buffer(self, count: Optional[int] = None) -> List[str]:
        """Returns buffered lines received from the physical board."""
        with self.lock:
            if count:
                return self.rx_buffer[-count:]
            return list(self.rx_buffer)

    def clear_buffer(self) -> None:
        with self.lock:
            self.rx_buffer.clear()
            self.plotter.clear()

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._serial_obj and self._serial_obj.is_open:
            try:
                self._serial_obj.close()
            except Exception:
                pass
        self.connected = False
        self._serial_obj = None

    def get_monitor_view(self, lines_count: int = 25) -> str:
        """Returns the formatted Serial Monitor text stream with status bar."""
        lines = self.read_buffer(lines_count)
        status_tag = f"[bold green][CONNECTED: {self.port} @ {self.baudrate} baud][/bold green]" if self.connected else f"[dim][DISCONNECTED: {self.port}][/dim]"
        header = [
            f"[bold cyan]=== Physical USB Serial Monitor ===[/bold cyan] {status_tag}",
            "  +-------------------------------------------------------------------------------+"
        ]
        if not lines:
            header.append("  | [dim](Waiting for sensor telemetry from physical board... Type 'serial send' to transmit)[/dim]")
        else:
            for l in lines:
                header.append(f"  | [green]>[/green] {l}")
        header.append("  +-------------------------------------------------------------------------------+")
        return "\n".join(header)


class RealAsciiSerialPlotter:
    """Renders real-time ASCII telemetry waveform graphs from physical sensor output lines."""

    def __init__(self, width: int = 65, height: int = 12, max_history: int = 50):
        self.width = width
        self.height = height
        self.max_history = max_history
        self.channels: Dict[str, List[float]] = {}
        self.symbols = ['*', 'o', '+', '#', '^', '~', '@']

    def clear(self):
        self.channels.clear()

    def feed_line(self, line: str):
        # 1. Parse labeled telemetry: "temp: 24.5 hum: 60.2" or "V1=3.3 V2=1.2"
        labeled = re.findall(r'([a-zA-Z0-9_]+)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
        if labeled:
            for label, val_str in labeled:
                try:
                    v = float(val_str)
                    if label not in self.channels:
                        self.channels[label] = []
                    self.channels[label].append(v)
                    if len(self.channels[label]) > self.max_history:
                        self.channels[label].pop(0)
                except ValueError:
                    pass
            return

        # 2. Parse comma/space delimited numbers: "23.4, 55.1, 102.0"
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
        alpha_count = sum(1 for c in line if c.isalpha())
        if numbers and alpha_count < 10:
            for idx, num_str in enumerate(numbers):
                ch_name = f"Ch{idx+1}"
                try:
                    v = float(num_str)
                    if ch_name not in self.channels:
                        self.channels[ch_name] = []
                    self.channels[ch_name].append(v)
                    if len(self.channels[ch_name]) > self.max_history:
                        self.channels[ch_name].pop(0)
                except ValueError:
                    pass

    def feed_lines(self, lines: List[str]):
        for l in lines:
            self.feed_line(l)

    def render(self) -> str:
        if not self.channels:
            return "[ASCII Serial Plotter]: Waiting for real numeric telemetry on USB serial (e.g. Serial.println(\"23.5, 60.1\") from board)..."

        all_vals = []
        for ch, vals in self.channels.items():
            all_vals.extend(vals)
        if not all_vals:
            return "[ASCII Serial Plotter]: No telemetry points recorded yet."

        min_val = min(all_vals)
        max_val = max(all_vals)
        if min_val == max_val:
            min_val -= 1.0
            max_val += 1.0

        val_range = max_val - min_val

        out: List[str] = []
        # Legend
        legend_items = []
        for idx, ch in enumerate(self.channels.keys()):
            sym = self.symbols[idx % len(self.symbols)]
            latest = self.channels[ch][-1] if self.channels[ch] else 0.0
            legend_items.append(f"[{sym}] {ch}: {latest:+.2f}")
        out.append("  TELEMETRY WAVEFORM PLOT: " + " | ".join(legend_items))
        out.append("  " + "-" * (self.width + 12))

        # Grid buffer
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]

        for ch_idx, (ch_name, vals) in enumerate(self.channels.items()):
            sym = self.symbols[ch_idx % len(self.symbols)]
            count = len(vals)
            start_col = max(0, self.width - count)
            slice_vals = vals[-self.width:]
            for c_offset, v in enumerate(slice_vals):
                col = start_col + c_offset
                if 0 <= col < self.width:
                    norm = (v - min_val) / val_range
                    row = int((1.0 - norm) * (self.height - 1))
                    row = max(0, min(self.height - 1, row))
                    grid[row][col] = sym

        # Render rows with y-axis labels
        for r in range(self.height):
            row_val = max_val - (r / (self.height - 1)) * val_range
            label = f"{row_val:+7.2f} |"
            row_str = "".join(grid[r])
            out.append(f"{label}{row_str}|")

        # Bottom axis
        out.append("  " + " " * 8 + "+" + "-" * self.width + "+")
        out.append("  " + " " * 9 + "T - " + str(self.width) + " samples" + " " * (self.width - 24) + "REAL-TIME (NOW)")
        return "\n".join(out)
