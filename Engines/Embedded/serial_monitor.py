"""
Engines/Embedded/serial_monitor.py - Serial Monitor, COM Port Bridge & ASCII Waveform Plotter.
Handles serial stream capture, bidirectional serial I/O, COM port hardware detection, and ASCII plotting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import threading
import time
import sys
import re


@dataclass
class SerialPortInfo:
    port: str
    description: str
    hwid: str = ""
    is_usb: bool = False


def scan_com_ports() -> List[SerialPortInfo]:
    """Scans system for available COM / serial ports using serial.tools or PowerShell/system fallback."""
    ports: List[SerialPortInfo] = []
    
    # 1. Try pyserial if installed
    try:
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            desc = p.description or "Generic Serial Port"
            is_usb = "USB" in desc.upper() or "CH340" in desc.upper() or "FTDI" in desc.upper() or "CP210" in desc.upper() or "ARDUINO" in desc.upper()
            ports.append(SerialPortInfo(port=p.device, description=desc, hwid=p.hwid or "", is_usb=is_usb))
        if ports:
            return ports
    except Exception:
        pass

    # 2. Windows PowerShell fallback
    if sys.platform.startswith("win"):
        try:
            cmd = "powershell -Command \"Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match '\\(COM\\d+\\)' } | Select-Object -ExpandProperty Name\""
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3)
            lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
            for line in lines:
                m = re.search(r'\((COM\d+)\)', line)
                if m:
                    com = m.group(1)
                    ports.append(SerialPortInfo(
                        port=com,
                        description=line,
                        is_usb="USB" in line.upper() or "CH340" in line.upper() or "ARDUINO" in line.upper()
                    ))
        except Exception:
            pass

    # 3. Default fallback if none detected (simulated devices)
    if not ports:
        ports.append(SerialPortInfo(port="COM3 (Simulated)", description="Virtual USB-Serial CH340 Driver", is_usb=True))
    
    return ports


class HardwareSerialBridge:
    """Provides bidirectional serial communication to physical COM ports."""

    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.connected = False
        self._serial_obj = None
        self._rx_thread = None
        self._stop_event = threading.Event()
        self.rx_buffer: List[str] = []
        self.lock = threading.Lock()

    def connect(self) -> Tuple[bool, str]:
        try:
            import serial
            self._serial_obj = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.connected = True
            self._stop_event.clear()
            self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
            self._rx_thread.start()
            return True, f"Connected to {self.port} at {self.baudrate} baud."
        except ImportError:
            # Pyserial not installed - fallback mock
            self.connected = True
            return True, f"[SIMULATED] Connected to {self.port} at {self.baudrate} baud (pyserial not installed)."
        except Exception as e:
            return False, f"Failed to open {self.port}: {str(e)}"

    def _rx_loop(self):
        while not self._stop_event.is_set() and self._serial_obj and self._serial_obj.is_open:
            try:
                line = self._serial_obj.readline().decode('utf-8', errors='replace')
                if line:
                    with self.lock:
                        self.rx_buffer.append(line.rstrip('\r\n'))
                        if len(self.rx_buffer) > 500:
                            self.rx_buffer.pop(0)
            except Exception:
                break

    def write(self, data: str) -> bool:
        if not self.connected:
            return False
        if self._serial_obj and self._serial_obj.is_open:
            try:
                self._serial_obj.write(data.encode('utf-8'))
                return True
            except Exception:
                return False
        return True

    def read_all(self) -> List[str]:
        with self.lock:
            lines = list(self.rx_buffer)
            self.rx_buffer.clear()
            return lines

    def disconnect(self):
        self._stop_event.set()
        if self._serial_obj and self._serial_obj.is_open:
            try:
                self._serial_obj.close()
            except Exception:
                pass
        self.connected = False


class AsciiSerialPlotter:
    """
    Renders multi-channel ASCII telemetry waveform graphs from serial output lines.
    Parses lines containing comma-, space-, or colon-separated numerical values.
    Example input: "23.4, 55.1, 102.0" or "temp: 24.5 hum: 60.2"
    """

    def __init__(self, width: int = 60, height: int = 12, max_history: int = 50):
        self.width = width
        self.height = height
        self.max_history = max_history
        self.channels: Dict[str, List[float]] = {}
        self.symbols = ['*', 'o', '+', '#', '^', '~', '@']

    def feed_line(self, line: str):
        """Extracts numeric readings from a serial line."""
        # Try labeled pairs like "temp:25.4,hum:60" or "V1=3.3 V2=1.2"
        labeled = re.findall(r'([a-zA-Z0-9_]+)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
        if labeled:
            for label, val_str in labeled:
                try:
                    val = float(val_str)
                    if label not in self.channels:
                        self.channels[label] = []
                    self.channels[label].append(val)
                    if len(self.channels[label]) > self.max_history:
                        self.channels[label].pop(0)
                except ValueError:
                    pass
            return

        # Try pure numbers separated by comma, space, tab
        # Exclude lines that have lots of letters
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
        # Check if line isn't just pure non-numeric text
        alpha_count = sum(1 for c in line if c.isalpha())
        if numbers and alpha_count < 10:
            for idx, num_str in enumerate(numbers):
                ch_name = f"Ch{idx+1}"
                try:
                    val = float(num_str)
                    if ch_name not in self.channels:
                        self.channels[ch_name] = []
                    self.channels[ch_name].append(val)
                    if len(self.channels[ch_name]) > self.max_history:
                        self.channels[ch_name].pop(0)
                except ValueError:
                    pass

    def feed_lines(self, lines: List[str]):
        for l in lines:
            self.feed_line(l)

    def render(self) -> str:
        """Draws the ASCII telemetry plot."""
        if not self.channels:
            return "[ASCII Serial Plotter]: Waiting for numeric telemetry data on Serial (e.g. Serial.println(\"23.5, 60.1\"))..."

        # Determine global or per-channel min and max
        all_vals = []
        for ch, vals in self.channels.items():
            all_vals.extend(vals)
        if not all_vals:
            return "[ASCII Serial Plotter]: No data points recorded yet."

        min_val = min(all_vals)
        max_val = max(all_vals)
        if min_val == max_val:
            min_val -= 1.0
            max_val += 1.0

        val_range = max_val - min_val

        out: List[str] = []
        # Header / Legend
        legend_items = []
        for idx, ch in enumerate(self.channels.keys()):
            sym = self.symbols[idx % len(self.symbols)]
            latest = self.channels[ch][-1] if self.channels[ch] else 0.0
            legend_items.append(f"[{sym}] {ch}: {latest:+.2f}")
        out.append("  PLOT LEGEND: " + " | ".join(legend_items))
        out.append("  " + "-" * (self.width + 12))

        # Grid buffer initialization (height rows x width cols)
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]

        for ch_idx, (ch_name, vals) in enumerate(self.channels.items()):
            sym = self.symbols[ch_idx % len(self.symbols)]
            # Map recent vals to grid columns
            count = len(vals)
            start_col = max(0, self.width - count)
            slice_vals = vals[-self.width:]
            for c_offset, v in enumerate(slice_vals):
                col = start_col + c_offset
                if 0 <= col < self.width:
                    # Normalize y into [0, height-1] where height-1 is top (row 0)
                    norm = (v - min_val) / val_range
                    row = int((1.0 - norm) * (self.height - 1))
                    row = max(0, min(self.height - 1, row))
                    grid[row][col] = sym

        # Render rows with y-axis labels
        for r in range(self.height):
            # Calculate value at this row
            row_val = max_val - (r / (self.height - 1)) * val_range
            label = f"{row_val:+7.2f} |"
            row_str = "".join(grid[r])
            out.append(f"{label}{row_str}|")

        # Bottom axis
        out.append("  " + " " * 8 + "+" + "-" * self.width + "+")
        out.append("  " + " " * 9 + "T - " + str(self.width) + " steps" + " " * (self.width - 20) + "NOW (T-0)")
        return "\n".join(out)
