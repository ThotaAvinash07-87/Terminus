"""
Engines/Embedded/flasher.py - Firmware Flashing & Bootloader Deployment.
Provides animated bootloader flashing simulation and real hardware toolchain dispatch
(arduino-cli, esptool.py, avrdude, picotool, st-flash).
"""

from typing import Dict, List, Optional, Tuple, Callable
import subprocess
import shutil
import time
import sys

from .boards import BoardSpecification


class FirmwareFlasher:
    """Manages the compilation verification and flashing / deployment lifecycle to target hardware."""

    def __init__(self, board: BoardSpecification):
        self.board = board
        self.last_flash_log: List[str] = []

    def flash_simulation(
        self,
        binary_size: int,
        eeprom_size: int = 0,
        progress_cb: Optional[Callable[[int, str], None]] = None
    ) -> Tuple[bool, str]:
        """
        Simulates the full bootloader handshake, flash erasing, page-by-page writing,
        and verification with realistic delays and memory boundary checks.
        """
        self.last_flash_log = []

        def log(msg: str):
            self.last_flash_log.append(msg)

        # 1. Check flash limits
        flash_limit = self.board.flash_kb * 1024
        if binary_size > flash_limit:
            err = f"[FLASH ERROR]: Binary size ({binary_size} bytes) exceeds {self.board.name} flash capacity ({flash_limit} bytes)!"
            log(err)
            return False, err

        # 2. Simulate toolchain initiation
        tool = self.board.upload_tool
        log(f"[{tool.upper()}]: Initializing connection to target {self.board.name} ({self.board.mcu})...")
        log(f"[{tool.upper()}]: Protocol: {self.board.upload_protocol} @ {self.board.default_baud} baud")

        steps = [
            (10, f"Attempting reset pulse via DTR/RTS..."),
            (25, f"Bootloader response acknowledged. Chip ID: 0x{self.board.signature:04X} ({self.board.mcu})"),
            (40, f"Erasing {self.board.flash_kb} KB flash memory sectors..."),
            (65, f"Writing {binary_size} bytes to flash address {self.board.flash_base_addr}..."),
            (85, f"Reading and verifying {binary_size} bytes (CRC32 checksum verified)..."),
            (95, f"Writing fuse bits / configuration registers..."),
            (100, f"Device reset into user application. Target running at {self.board.clock_mhz} MHz!"),
        ]

        for pct, desc in steps:
            if progress_cb:
                progress_cb(pct, desc)
            log(f"  [{pct:3d}%] {desc}")

        pct_flash = (binary_size / flash_limit) * 100.0
        summary = (
            f"\n[FLASH SUCCESSFUL]:\n"
            f"  Target Board : {self.board.name} ({self.board.mcu})\n"
            f"  Flash Memory : {binary_size:,} / {flash_limit:,} bytes ({pct_flash:.1f}% used)\n"
            f"  Protocol     : {self.board.upload_protocol} via {tool}\n"
            f"  Status       : Application active & running"
        )
        log(summary)
        return True, summary

    def flash_hardware(
        self,
        sketch_path: str,
        port: str,
        extra_args: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        Attempts to invoke real external toolchains (arduino-cli, esptool.py, avrdude) if installed on the host.
        If the tool is missing, provides clear installation instructions and fallback simulation.
        """
        tool = self.board.upload_tool
        if not shutil.which(tool) and not shutil.which(f"{tool}.exe"):
            # Check python module fallback for esptool
            if tool == "esptool":
                try:
                    res = subprocess.run([sys.executable, "-m", "esptool", "version"], capture_output=True, text=True)
                    if res.returncode == 0:
                        cmd = [sys.executable, "-m", "esptool", "--port", port, "write_flash", "0x10000", sketch_path]
                        run_res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        return (run_res.returncode == 0), run_res.stdout + run_res.stderr
                except Exception:
                    pass

            return False, (
                f"Toolchain '{tool}' not found in system PATH.\n"
                f"To flash physical hardware:\n"
                f"  - For Arduino: Install 'arduino-cli' or 'avrdude'\n"
                f"  - For ESP32/ESP8266: Run 'pip install esptool'\n"
                f"  - For STM32: Install 'st-flash' (ST-Link tools)\n"
                f"  - For RP2040: Install 'picotool'\n\n"
                f"Terminus fallback: Simulated flash mode is fully operational in this session."
            )

        # If tool found, execute real command
        try:
            if tool == "arduino-cli":
                cmd = ["arduino-cli", "upload", "-p", port, "--fqbn", self.board.fqbn or "arduino:avr:uno", sketch_path]
            elif tool == "avrdude":
                cmd = ["avrdude", "-p", self.board.mcu.lower(), "-c", self.board.upload_protocol, "-P", port, "-b", str(self.board.default_baud), "-U", f"flash:w:{sketch_path}:i"]
            else:
                cmd = [tool, "--port", port, sketch_path]

            if extra_args:
                cmd.extend(extra_args)

            res = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            if res.returncode == 0:
                return True, f"Upload successful via {tool}:\n" + res.stdout
            else:
                return False, f"Upload failed via {tool} (exit code {res.returncode}):\n" + res.stderr
        except Exception as e:
            return False, f"Execution failed: {str(e)}"
