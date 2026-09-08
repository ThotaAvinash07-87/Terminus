"""
Unit tests for the Embedded Hardware & Arduino IDE Workbench Subsystem in TerminusECE.
"""

import unittest
import os
import shutil
from pathlib import Path

from Engines.Embedded.boards import BoardCatalog, board_catalog, BoardSpecification
from Engines.Embedded.virtual_devices import (
    LEDDevice, RGBLEDDevice, ButtonDevice, PotentiometerDevice,
    DHTSensorDevice, UltrasonicSensorDevice, ServoDevice,
    LCD16x2Device, OLEDSSD1306Device, DeviceManager
)
from Engines.Embedded.arduino_runtime import (
    ArduinoRuntimeEngine, ArduinoPinState, ArduinoSerialStream,
    CompilationResult, WireSimulator, SPISimulator, ServoSimulator, EEPROMSimulator
)
from Engines.Embedded.serial_monitor import (
    scan_com_ports, AsciiSerialPlotter, HardwareSerialBridge
)
from Engines.Embedded.flasher import FirmwareFlasher
from UI.app import TerminusEngineBridge


class TestEmbeddedBoards(unittest.TestCase):
    """Test microcontroller board catalog and hardware specifications."""

    def test_catalog_population(self):
        self.assertGreaterEqual(len(board_catalog.boards), 25)
        self.assertIn("UNO", board_catalog.boards)
        self.assertIn("ESP32", board_catalog.boards)
        self.assertIn("STM32_BLUEPILL", board_catalog.boards)
        self.assertIn("RPI_PICO", board_catalog.boards)
        self.assertIn("TI_C2000_F28379D", board_catalog.boards)
        self.assertIn("RISCV_GD32VF103", board_catalog.boards)

    def test_board_specs(self):
        uno = board_catalog.get("UNO")
        self.assertEqual(uno.mcu, "ATmega328P")
        self.assertEqual(uno.clock_mhz, 16.0)
        self.assertEqual(uno.flash_kb, 32)
        self.assertEqual(uno.ram_kb, 2)
        self.assertEqual(uno.voltage_v, 5.0)

        esp32 = board_catalog.get("ESP32")
        self.assertEqual(esp32.mcu, "ESP32-D0WDQ6")
        self.assertEqual(esp32.clock_mhz, 240.0)
        self.assertEqual(esp32.flash_kb, 4096)
        self.assertTrue(esp32.has_wifi)
        self.assertTrue(esp32.has_ble)

    def test_board_search(self):
        results = board_catalog.search("STM32")
        self.assertGreaterEqual(len(results), 4)
        summary = board_catalog.generate_summary_table("ARDUINO")
        self.assertIn("Arduino Uno R3", summary)


class TestVirtualDevices(unittest.TestCase):
    """Test virtual breadboard peripherals and input/output synchronization."""

    def test_led_device(self):
        led = LEDDevice("LED1", pin=13, color="Green")
        self.assertFalse(led.is_on)
        self.assertIn("OFF", led.get_ascii_visual())
        led.update_state(1, 255, 3.3)
        self.assertTrue(led.is_on)
        self.assertIn("ON", led.get_ascii_visual())

    def test_rgb_led_device(self):
        rgb = RGBLEDDevice("RGB1", r_pin=9, g_pin=10, b_pin=11)
        rgb.update_state(1, 255, 3.3, pin=9)
        rgb.update_state(1, 128, 1.65, pin=10)
        self.assertEqual(rgb.red_pwm, 255)
        self.assertEqual(rgb.green_pwm, 128)
        self.assertIn("RGB", rgb.get_ascii_visual())

    def test_button_device(self):
        btn = ButtonDevice("BTN1", pin=2, pullup=True)
        self.assertEqual(btn.get_digital_level(), 1)  # Released = HIGH (pullup)
        btn.press()
        self.assertEqual(btn.get_digital_level(), 0)  # Pressed = LOW
        btn.release()
        self.assertEqual(btn.get_digital_level(), 1)

    def test_potentiometer_device(self):
        pot = PotentiometerDevice("POT1", pin="A0", voltage=2.5)
        self.assertEqual(pot.get_raw_adc(), 512)
        pot.set_voltage(5.0)
        self.assertEqual(pot.get_raw_adc(), 1023)
        pot.set_raw_adc(256)
        self.assertAlmostEqual(pot.voltage, 1.25, places=1)

    def test_sensor_devices(self):
        dht = DHTSensorDevice("DHT1", pin=7, temperature=24.5, humidity=55.0)
        self.assertIn("24.5", dht.get_ascii_visual())
        self.assertIn("55.0", dht.get_ascii_visual())

        sonar = UltrasonicSensorDevice("US1", trig_pin=8, echo_pin=9, distance_cm=18.4)
        self.assertIn("18.4", sonar.get_ascii_visual())

    def test_servo_device(self):
        srv = ServoDevice("SRV1", pin=9)
        srv.set_angle(90)
        self.assertEqual(srv.angle, 90)
        self.assertIn("90.0", srv.get_ascii_visual())
        self.assertIn("Servo", srv.get_ascii_visual())

    def test_display_devices(self):
        lcd = LCD16x2Device("LCD1")
        lcd.write_line(0, "Terminus ECE")
        lcd.write_line(1, "Ready")
        vis = lcd.get_ascii_visual()
        self.assertIn("Terminus ECE", vis)
        self.assertIn("Ready", vis)

        oled = OLEDSSD1306Device("OLED1")
        oled.set_text_line(0, "System OK")
        self.assertIn("System OK", oled.get_ascii_visual())


class TestArduinoRuntimeEngine(unittest.TestCase):
    """Test Arduino C++ parsing, compilation, execution, and memory calculation."""

    def test_compile_memory_computation(self):
        engine = ArduinoRuntimeEngine("UNO")
        sketch = """
        const int led = 13;
        int sensorVal = 0;
        void setup() {
            pinMode(led, OUTPUT);
            Serial.begin(9600);
        }
        void loop() {
            digitalWrite(led, HIGH);
            delay(100);
            digitalWrite(led, LOW);
            delay(100);
        }
        """
        engine.load_sketch(sketch)
        res = engine.compile()
        self.assertTrue(res.success)
        self.assertGreater(res.flash_used_bytes, 0)
        self.assertLessEqual(res.flash_used_bytes, 32 * 1024)
        self.assertGreater(res.ram_used_bytes, 0)
        self.assertLessEqual(res.ram_used_bytes, 2 * 1024)

    def test_runtime_execution_blink(self):
        engine = ArduinoRuntimeEngine("UNO")
        led = LEDDevice("LED1", pin=13)
        engine.devices.add_device(led)

        sketch = """
        void setup() {
            pinMode(13, OUTPUT);
            Serial.begin(115200);
            Serial.println("Init done");
        }
        void loop() {
            digitalWrite(13, HIGH);
            Serial.println("Blink ON");
            delay(500);
            digitalWrite(13, LOW);
            Serial.println("Blink OFF");
            delay(500);
        }
        """
        engine.load_sketch(sketch)
        ok, logs = engine.run_continuous(cycles=2)
        self.assertTrue(ok)
        self.assertIn("Init done", logs)
        self.assertIn("Blink ON", logs)
        self.assertIn("Blink OFF", logs)
        self.assertEqual(engine.current_millis, 2000)

    def test_analog_read_pot_interaction(self):
        engine = ArduinoRuntimeEngine("UNO")
        pot = PotentiometerDevice("POT1", pin="A0", voltage=3.3)
        engine.devices.add_device(pot)

        sketch = """
        void setup() {
            Serial.begin(9600);
        }
        void loop() {
            int val = analogRead(A0);
            Serial.print("ADC: ");
            Serial.println(val);
        }
        """
        engine.load_sketch(sketch)
        ok, logs = engine.run_continuous(cycles=1)
        self.assertTrue(ok)
        self.assertTrue(any("ADC: 675" in l for l in logs))

    def test_servo_wire_eeprom_simulators(self):
        engine = ArduinoRuntimeEngine("UNO")
        sketch = """
        #include <Servo.h>
        Servo myservo;
        void setup() {
            myservo.attach(9);
            myservo.write(120);
            EEPROM.write(0x10, 42);
            byte b = EEPROM.read(0x10);
            Serial.print("EEPROM Read: ");
            Serial.println(b);
        }
        void loop() {
        }
        """
        engine.load_sketch(sketch)
        ok, logs = engine.run_continuous(cycles=1)
        self.assertTrue(ok)
        self.assertEqual(engine.servos[9].angle, 120)
        self.assertEqual(engine.eeprom.read(0x10), 42)
        self.assertTrue(any("EEPROM Read: 42" in l for l in logs))


class TestSerialPlotterAndFlasher(unittest.TestCase):
    """Test ASCII serial telemetry waveform plotter and firmware flasher."""

    def test_ascii_plotter(self):
        plotter = AsciiSerialPlotter(width=40, height=8)
        plotter.feed_lines([
            "10.0, 20.0",
            "15.0, 25.0",
            "20.0, 30.0",
            "25.0, 35.0"
        ])
        plot_str = plotter.render()
        self.assertIn("PLOT LEGEND", plot_str)
        self.assertIn("Ch1", plot_str)
        self.assertIn("Ch2", plot_str)

    def test_flasher_simulation(self):
        board = board_catalog.get("ESP32")
        flasher = FirmwareFlasher(board)
        ok, msg = flasher.flash_simulation(binary_size=150000)
        self.assertTrue(ok)
        self.assertIn("ESP32", msg)
        self.assertIn("FLASH SUCCESSFUL", msg)

        # Test flash limit exceed
        ok_fail, err = flasher.flash_simulation(binary_size=10 * 1024 * 1024)
        self.assertFalse(ok_fail)
        self.assertIn("exceeds", err)


class TestTerminusEngineBridgeEmbeddedCommands(unittest.TestCase):
    """Test real hardware CLI command dispatch and integration in TerminusEngineBridge."""

    def setUp(self):
        self.bridge = TerminusEngineBridge()
        self.bridge.switch_mode("EMBEDDED")

    def test_board_commands(self):
        out = self.bridge.execute_command("board list")
        self.assertIn("Arduino Uno", out)
        self.assertIn("ESP32", out)

        out_set = self.bridge.execute_command("board ESP32")
        self.assertIn("Target board switched to", out_set)
        self.assertEqual(self.bridge.target_board.id, "ESP32")

        out_info = self.bridge.execute_command("board info")
        self.assertIn("Microcontroller Specification", out_info)
        self.assertIn("ESP32", out_info)

    def test_sketch_example_compile_flash_run(self):
        # Load built-in blink example
        out_ex = self.bridge.execute_command("template blink")
        self.assertIn("Loaded hardware template 'blink'", out_ex)

        # View code
        out_code = self.bridge.execute_command("code")
        self.assertIn("void setup()", out_code)

        # Compile
        out_comp = self.bridge.execute_command("compile")
        self.assertIn("Sketch verified", out_comp)

    def test_serial_monitor_and_plot(self):
        out_mon = self.bridge.execute_command("serial status")
        self.assertIn("Physical USB Serial Monitor", out_mon)

        out_plot = self.bridge.execute_command("plot status")
        self.assertIn("Live ASCII Telemetry Plotter", out_plot)

    def test_com_ports_scan(self):
        out_ports = self.bridge.execute_command("ports")
        self.assertIn("Physical USB COM & Serial Ports", out_ports)

    def test_file_save_and_open_sketch(self):
        self.bridge.execute_command("project test_blink_sketch arduino")
        out_save = self.bridge.execute_command("file save")
        self.assertIn("test_blink_sketch", out_save)

        out_open = self.bridge.execute_command("file open test_blink_sketch")
        self.assertIn("test_blink_sketch", out_open)


if __name__ == "__main__":
    unittest.main()
