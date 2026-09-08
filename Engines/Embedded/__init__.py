"""
Engines/Embedded/__init__.py - Embedded Hardware & Arduino IDE Workbench Subsystem.
"""

from .boards import BoardSpecification, BoardCatalog, board_catalog
from .usb_manager import USBPortManager, USBDeviceInfo
from .sketch_manager import ProjectSketchManager, ProjectFile
from .real_flasher import HardwareFirmwareFlasher, HardwareFlashResult
from .real_serial_monitor import RealSerialMonitor, RealAsciiSerialPlotter
from .virtual_devices import (
    VirtualDevice, LEDDevice, RGBLEDDevice, ButtonDevice,
    PotentiometerDevice, DHTSensorDevice, UltrasonicSensorDevice,
    ServoDevice, LCD16x2Device, OLEDSSD1306Device, DeviceManager
)
from .arduino_runtime import (
    ArduinoPinState, ArduinoSerialStream, ArduinoRuntimeEngine,
    CompilationResult, WireSimulator, SPISimulator, ServoSimulator, EEPROMSimulator
)
from .serial_monitor import (
    SerialPortInfo, scan_com_ports, HardwareSerialBridge, AsciiSerialPlotter
)
from .flasher import FirmwareFlasher

__all__ = [
    "BoardSpecification",
    "BoardCatalog",
    "board_catalog",
    "USBPortManager",
    "USBDeviceInfo",
    "ProjectSketchManager",
    "ProjectFile",
    "HardwareFirmwareFlasher",
    "HardwareFlashResult",
    "RealSerialMonitor",
    "RealAsciiSerialPlotter",
    "VirtualDevice",
    "LEDDevice",
    "RGBLEDDevice",
    "ButtonDevice",
    "PotentiometerDevice",
    "DHTSensorDevice",
    "UltrasonicSensorDevice",
    "ServoDevice",
    "LCD16x2Device",
    "OLEDSSD1306Device",
    "DeviceManager",
    "ArduinoPinState",
    "ArduinoSerialStream",
    "ArduinoRuntimeEngine",
    "CompilationResult",
    "WireSimulator",
    "SPISimulator",
    "ServoSimulator",
    "EEPROMSimulator",
    "SerialPortInfo",
    "scan_com_ports",
    "HardwareSerialBridge",
    "AsciiSerialPlotter",
    "FirmwareFlasher",
]
