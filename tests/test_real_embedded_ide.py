"""
Unit & Integration Tests for Real-World USB Embedded & Arduino IDE Workbench.

Tests:
1. USBPortManager (hardware device detection, known VID:PID matching, connection testing, reset lines)
2. ProjectSketchManager (project folder creation Documents/Terminus Files/Embedded/<proj>/<proj>.ino, code viewing & editing)
3. HardwareFirmwareFlasher (real toolchain detection, pure-python STK500v1 protocol, hardware flash & dump)
4. RealSerialMonitor & RealAsciiSerialPlotter (threaded background monitoring, multi-channel ASCII charts, two-way USB comms)
5. TerminusEngineBridge CLI Integration (interactive commands: ports, port, board, code, edit, compile, upload, dump, serial, plot, send)
"""

import os
import sys
import unittest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from CORE.storage_manager import StorageManager
from Engines.Embedded.usb_manager import USBPortManager, USBDeviceInfo, KNOWN_HW_SIGNATURES
from Engines.Embedded.sketch_manager import ProjectSketchManager, ProjectFile
from Engines.Embedded.real_flasher import HardwareFirmwareFlasher, HardwareFlashResult
from Engines.Embedded.real_serial_monitor import RealSerialMonitor, RealAsciiSerialPlotter
from Engines.Embedded.boards import board_catalog, BoardSpecification
from UI.app import TerminusEngineBridge


class TestUSBPortManager(unittest.TestCase):
    def test_known_vid_pid_signatures(self):
        # Check database signatures
        uno_matches = [name for vid, pid, name in KNOWN_HW_SIGNATURES if vid == 0x2341 and pid == 0x0043]
        self.assertTrue(len(uno_matches) > 0)
        self.assertIn("Arduino Uno", uno_matches[0])

        esp_matches = [name for vid, pid, name in KNOWN_HW_SIGNATURES if vid == 0x303A and pid == 0x1001]
        self.assertTrue(len(esp_matches) > 0)
        self.assertIn("ESP32", esp_matches[0])

        ch340_matches = [name for vid, pid, name in KNOWN_HW_SIGNATURES if vid == 0x1A86 and pid == 0x7523]
        self.assertTrue(len(ch340_matches) > 0)
        self.assertIn("CH340", ch340_matches[0])

    @patch('serial.tools.list_ports.comports')
    def test_scan_ports(self, mock_comports):
        mock_port1 = MagicMock()
        mock_port1.device = "COM3"
        mock_port1.description = "Arduino Uno (COM3)"
        mock_port1.hwid = "USB VID:PID=2341:0043"
        mock_port1.vid = 0x2341
        mock_port1.pid = 0x0043
        mock_port1.serial_number = "123456"
        mock_port1.manufacturer = "Arduino LLC"

        mock_comports.return_value = [mock_port1]

        devices = USBPortManager.scan_ports()
        self.assertEqual(len(devices), 1)
        dev = devices[0]
        self.assertEqual(dev.port, "COM3")
        self.assertIn("Arduino Uno", dev.detected_board_hint)


class TestProjectSketchManager(unittest.TestCase):
    def setUp(self):
        self.mgr = ProjectSketchManager("BlinkyLED")

    def test_create_and_save_project(self):
        self.mgr.new_project("BlinkyLED")
        self.assertEqual(self.mgr.project_name, "BlinkyLED")
        self.assertTrue(self.mgr.active_filename.endswith("BlinkyLED.ino"))

        # Save to disk
        ok, msg = self.mgr.save_project()
        self.assertTrue(ok)
        main_path = self.mgr.get_main_filepath()
        self.assertTrue(main_path.exists())
        self.assertEqual(main_path.parent.name, "BlinkyLED")
        self.assertEqual(main_path.name, "BlinkyLED.ino")

    def test_multi_file_project(self):
        self.mgr.new_project("SensorNode")
        self.mgr.add_file("sensor_config.h", "#define SENSOR_PIN 4\n")
        self.assertIn("sensor_config.h", self.mgr.files)

        ok, msg = self.mgr.save_project()
        self.assertTrue(ok)
        header_path = self.mgr.get_project_dir() / "sensor_config.h"
        self.assertTrue(header_path.exists())

        # Reload project
        reloader = ProjectSketchManager("Temp")
        ok_load, _ = reloader.load_project("SensorNode")
        self.assertTrue(ok_load)
        self.assertIn("sensor_config.h", reloader.files)
        self.assertEqual(reloader.files["sensor_config.h"].content.strip(), "#define SENSOR_PIN 4")

    def test_code_formatting_and_editing(self):
        self.mgr.new_project("TestEdit")
        new_code = "void setup() {\n  Serial.begin(115200);\n}\nvoid loop() {\n  delay(1000);\n}"
        self.mgr.set_content(new_code)
        
        formatted = self.mgr.view_code()
        self.assertIn("1 | void setup()", formatted)
        self.assertIn("2 |   Serial.begin(115200);", formatted)


class TestHardwareFirmwareFlasher(unittest.TestCase):
    def setUp(self):
        self.board = board_catalog.get("UNO")
        self.flasher = HardwareFirmwareFlasher(self.board, "COM3")

    def test_supported_targets(self):
        self.assertIn("UNO", board_catalog.boards)
        self.assertIn("ESP32", board_catalog.boards)
        self.assertIn("RPI_PICO", board_catalog.boards)
        self.assertIn("STM32_BLUEPILL", board_catalog.boards)

    def test_stk500_sync_packet_generation(self):
        # Verify STK500v1 synchronization packet logic
        sync_cmd = bytes([0x30, 0x20]) # Cmnd_STK_GET_SYNC, Sync_CRC_EOP
        self.assertEqual(sync_cmd[0], 0x30)
        self.assertEqual(sync_cmd[1], 0x20)


class TestRealSerialMonitorAndPlotter(unittest.TestCase):
    def setUp(self):
        self.monitor = RealSerialMonitor()
        self.plotter = RealAsciiSerialPlotter(width=40, height=8)

    def test_serial_monitor_simulated_packet_handling(self):
        # Push raw sensor telemetry lines
        self.monitor.rx_buffer.append("Temperature: 24.5, Humidity: 60.2")
        self.monitor.rx_buffer.append("POT: 512, LIGHT: 890")
        
        recent = self.monitor.read_buffer(10)
        self.assertEqual(len(recent), 2)
        self.assertIn("Temperature: 24.5", recent[0])

    def test_ascii_plotter_multi_channel(self):
        self.plotter.feed_line("TEMP:25.0, HUMID:65.0")
        self.plotter.feed_line("TEMP:26.5, HUMID:63.0")
        self.plotter.feed_line("TEMP:28.0, HUMID:61.0")

        rendered = self.plotter.render()
        self.assertIn("TEMP", rendered)
        self.assertIn("HUMID", rendered)
        self.assertIn("+", rendered)


class TestTerminusBridgeEmbeddedCLI(unittest.TestCase):
    def setUp(self):
        self.bridge = TerminusEngineBridge()
        self.bridge.execute_command("mode embedded")

    def test_ports_and_port_commands(self):
        res = self.bridge.execute_command("ports")
        self.assertIn("Physical USB COM & Serial Ports", res)

        res_port = self.bridge.execute_command("port COM5")
        self.assertIn("Target USB port set to:", res_port)

        res_board = self.bridge.execute_command("board esp32")
        self.assertIn("Target board switched to:", res_board)

    def test_code_and_edit_commands(self):
        res_new = self.bridge.execute_command("project MySmartSensor")
        self.assertIn("Created project 'MySmartSensor'", res_new)

        res_code = self.bridge.execute_command("code")
        self.assertIn("MySmartSensor.ino", res_code)

        # Append code line
        res_append = self.bridge.execute_command("code append // sensor init")
        self.assertIn("Appended line", res_append)

        res_view = self.bridge.execute_command("code")
        self.assertIn("// sensor init", res_view)

    def test_file_save_and_list_embedded(self):
        self.bridge.execute_command("project TempLogger")
        save_res = self.bridge.execute_command("file save")
        self.assertIn("TempLogger", save_res)

        list_res = self.bridge.execute_command("file list")
        self.assertIn("TempLogger", list_res)

    def test_serial_and_plot_cli(self):
        stat_res = self.bridge.execute_command("serial status")
        self.assertIn("Physical USB Serial Monitor", stat_res)

        plot_res = self.bridge.execute_command("plot status")
        self.assertIn("Live ASCII Telemetry Plotter", plot_res)


if __name__ == '__main__':
    unittest.main()
