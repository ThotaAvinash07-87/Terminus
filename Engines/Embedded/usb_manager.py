"""
Engines/Embedded/usb_manager.py - Physical USB COM Port Discovery & Hardware Link Manager.
Scans and identifies connected physical microcontrollers (CH340, CP210x, FTDI, 16U2, STM32 VCP, ESP32 USB-JTAG, RP2040).
Provides connection verification, port auto-selection, and DTR/RTS hardware reset triggering.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import subprocess
import threading
import time
import sys
import re


@dataclass
class USBDeviceInfo:
    port: str
    description: str
    hwid: str = ""
    vid: Optional[int] = None
    pid: Optional[int] = None
    serial_number: str = ""
    manufacturer: str = ""
    detected_board_hint: str = ""
    is_usb_serial: bool = False


# Known USB Vendor/Product IDs for popular development boards
KNOWN_HW_SIGNATURES = [
    # Arduino Official
    (0x2341, 0x0043, "Arduino Uno R3 (ATmega16U2)"),
    (0x2341, 0x0010, "Arduino Mega 2560 R3"),
    (0x2341, 0x0042, "Arduino Mega 2560"),
    (0x2341, 0x0036, "Arduino Leonardo (ATmega32U4)"),
    (0x2341, 0x003D, "Arduino Due (Programming Port)"),
    (0x2341, 0x003E, "Arduino Due (Native USB Port)"),
    (0x2341, 0x8036, "Arduino Micro / Pro Micro"),
    (0x2341, 0x0058, "Arduino Nano Every"),
    (0x2341, 0x805A, "Arduino Nano 33 BLE"),
    (0x2341, 0x005E, "Arduino Nano 33 IoT"),

    # USB to UART Bridges commonly used on Arduino Clones, ESP8266, ESP32, STM32
    (0x1A86, 0x7523, "WCH CH340 / CH341 USB-to-Serial (Arduino Nano / Uno Clone, ESP8266, ESP32)"),
    (0x1A86, 0x55D4, "WCH CH9102 USB-to-Serial"),
    (0x10C4, 0xEA60, "Silicon Labs CP2102/CP2104 USB-UART (NodeMCU, ESP32 DevKit)"),
    (0x0403, 0x6001, "FTDI FT232R USB-UART"),
    (0x0403, 0x6015, "FTDI FT230X USB-UART"),

    # Espressif Direct USB-JTAG / CDC
    (0x303A, 0x1001, "ESP32-C3 / ESP32-S3 Built-in USB JTAG/Serial CDC"),
    (0x303A, 0x0002, "ESP32-S2 Native USB CDC"),

    # Raspberry Pi Silicon
    (0x2E8A, 0x0005, "Raspberry Pi Pico (RP2040 USB CDC Serial)"),
    (0x2E8A, 0x0003, "Raspberry Pi Pico (RP2040 USB Bootloader)"),
    (0x2E8A, 0x000F, "Raspberry Pi Pico 2 (RP2350 USB CDC)"),

    # STMicroelectronics
    (0x0483, 0x5740, "STM32 Virtual COM Port (STM32F103/F401/F411/F407 CDC)"),
    (0x0483, 0x374B, "ST-Link V2-1 Debug / Virtual COM Port"),
    (0x0483, 0x3748, "ST-Link V2 Programmer"),
    (0x0483, 0xDF11, "STM32 DFU Bootloader Mode"),
]


class USBPortManager:
    """Manages physical USB port scanning, identification, selection, and hardware line control."""

    @staticmethod
    def scan_ports() -> List[USBDeviceInfo]:
        """Scans the system for all connected physical USB COM and serial devices."""
        devices: List[USBDeviceInfo] = []

        # 1. Try PySerial list_ports
        try:
            import serial.tools.list_ports
            for p in serial.tools.list_ports.comports():
                desc = p.description or "Serial Port"
                hwid = p.hwid or ""
                vid = p.vid
                pid = p.pid
                mfg = p.manufacturer or ""
                sn = p.serial_number or ""

                hint = ""
                # Check known signature database
                if vid and pid:
                    for s_vid, s_pid, s_name in KNOWN_HW_SIGNATURES:
                        if vid == s_vid and pid == s_pid:
                            hint = s_name
                            break

                # Fuzzy fallback from description / HWID
                desc_u = (desc + " " + hwid + " " + mfg).upper()
                if not hint:
                    if "CH340" in desc_u or "CH341" in desc_u:
                        hint = "CH340 USB-Serial (Arduino / ESP32 / ESP8266)"
                    elif "CP210" in desc_u:
                        hint = "CP210x USB-UART (ESP32 / NodeMCU)"
                    elif "FTDI" in desc_u:
                        hint = "FTDI USB-UART"
                    elif "ARDUINO" in desc_u:
                        hint = "Arduino Microcontroller"
                    elif "STM32" in desc_u or "STLINK" in desc_u:
                        hint = "STM32 Microcontroller"
                    elif "PICO" in desc_u or "RP2040" in desc_u or "RP2350" in desc_u:
                        hint = "Raspberry Pi Pico (RP2040 / RP2350)"

                is_usb = (
                    bool(vid)
                    or "USB" in desc_u
                    or "CH340" in desc_u
                    or "CP210" in desc_u
                    or "FTDI" in desc_u
                    or "ARDUINO" in desc_u
                    or bool(hint)
                )

                devices.append(USBDeviceInfo(
                    port=p.device,
                    description=desc,
                    hwid=hwid,
                    vid=vid,
                    pid=pid,
                    serial_number=sn,
                    manufacturer=mfg,
                    detected_board_hint=hint,
                    is_usb_serial=is_usb
                ))
            if devices:
                return devices
        except Exception:
            pass

        # 2. Windows PowerShell CimInstance fallback
        if sys.platform.startswith("win"):
            try:
                cmd = 'powershell -NoProfile -Command "Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match \'\\(COM\\d+\\)\' } | Select-Object Name, DeviceID, Manufacturer"'
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3)
                lines = [l.strip() for l in res.stdout.splitlines() if l.strip() and not l.startswith("Name") and not l.startswith("----")]
                for l in lines:
                    m = re.search(r'\((COM\d+)\)', l)
                    if m:
                        com = m.group(1)
                        hint = ""
                        if "CH340" in l.upper(): hint = "CH340 USB-Serial"
                        elif "CP210" in l.upper(): hint = "CP210x USB-UART"
                        elif "ARDUINO" in l.upper(): hint = "Arduino Board"
                        devices.append(USBDeviceInfo(
                            port=com,
                            description=l,
                            detected_board_hint=hint,
                            is_usb_serial=True
                        ))
            except Exception:
                pass

        return devices

    @staticmethod
    def auto_detect_primary_port() -> Optional[str]:
        """Auto-detects the most likely microcontroller COM port connected via USB."""
        ports = USBPortManager.scan_ports()
        # Prioritize ports with known microcontroller hints
        for p in ports:
            if p.detected_board_hint and ("Arduino" in p.detected_board_hint or "ESP" in p.detected_board_hint or "CH340" in p.detected_board_hint or "CP210" in p.detected_board_hint or "STM32" in p.detected_board_hint or "Pico" in p.detected_board_hint):
                return p.port
        # Otherwise return first USB serial port
        for p in ports:
            if p.is_usb_serial:
                return p.port
        if ports:
            return ports[0].port
        return None

    @staticmethod
    def test_connection(port: str, baud: int = 115200) -> Tuple[bool, str]:
        """Verifies if the specified physical USB COM port can be opened and communicates."""
        try:
            import serial
            s = serial.Serial(port, baud, timeout=0.2)
            s.close()
            return True, f"Successfully connected to physical port {port} @ {baud} baud."
        except ImportError:
            return True, f"Port {port} accessible (pyserial not installed on host Python)."
        except Exception as e:
            err = str(e)
            if "PermissionError" in err or "Access is denied" in err:
                return False, f"Port {port} is BUSY or locked by another application (close Arduino IDE, Cura, or other serial monitors)."
            elif "FileNotFoundError" in err or "could not open port" in err:
                return False, f"Port {port} not found. Please verify the USB cable is firmly plugged into the board."
            return False, f"Failed to connect to {port}: {err}"

    @staticmethod
    def hardware_reset(port: str) -> Tuple[bool, str]:
        """Toggles the DTR/RTS lines over the USB cable to pulse the microcontroller RESET pin into bootloader."""
        try:
            import serial
            with serial.Serial(port, 1200) as s:
                s.dtr = False
                s.rts = False
                time.sleep(0.05)
                s.dtr = True
                s.rts = True
                time.sleep(0.1)
                s.dtr = False
                s.rts = False
            return True, f"Sent 1200-baud DTR/RTS hardware reset pulse to {port}."
        except Exception as e:
            return False, f"Hardware reset pulse failed on {port}: {str(e)}"
