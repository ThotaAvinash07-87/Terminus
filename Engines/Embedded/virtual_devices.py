"""
Engines/Embedded/virtual_devices.py - Interactive Virtual Peripherals and Breadboard Components.
Simulates LEDs, RGB LEDs, Pushbuttons, Potentiometers, Temperature/Humidity (DHT11),
Ultrasonic Distance (HC-SR04), I2C LCD 16x2, SSD1306 OLED, Servos, and DC Motors.
"""

from __future__ import annotations
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union


class VirtualDevice:
    """Base class for all virtual breadboard peripherals wired to board GPIO pins."""

    def __init__(self, name: str, device_type: str):
        self.name = name.strip()
        self.device_type = device_type
        self.pin_connections: Dict[str, str] = {}
        self.state: Dict[str, Any] = {}

    def connect_pin(self, dev_pin: str, board_pin: str) -> None:
        self.pin_connections[dev_pin.lower()] = str(board_pin).upper()

    def update_from_board(self, pin_states: Dict[str, Any]) -> None:
        pass

    def get_input_to_board(self) -> Dict[str, Any]:
        return {}

    def render_visual(self) -> str:
        return f"[{self.device_type}: {self.name}]"

    def get_ascii_visual(self) -> str:
        return self.render_visual()


class LEDDevice(VirtualDevice):
    """Standard Light Emitting Diode (Red, Green, Blue, Yellow, White)."""

    def __init__(self, name: str, pin: Union[int, str] = 13, color: str = "RED"):
        super().__init__(name, "LED")
        self.pin = pin
        self.color = color.upper()
        self.is_on: bool = False
        self.brightness: float = 0.0  # 0.0 to 1.0 (PWM duty cycle)
        self.voltage: float = 0.0
        self.connect_pin("anode", str(pin))
        self.state = {"is_on": False, "brightness": 0.0, "color": self.color}

    def update_state(self, digital: int, pwm: float = 0.0, voltage: float = 0.0, pin: Optional[Union[int, str]] = None) -> None:
        self.is_on = (digital == 1 or pwm > 0.05 or voltage >= 1.8)
        self.brightness = pwm if pwm > 0 else (1.0 if self.is_on else 0.0)
        self.voltage = voltage
        self.state["is_on"] = self.is_on
        self.state["brightness"] = self.brightness

    def update_from_board(self, pin_states: Dict[str, Any]) -> None:
        anode_pin = self.pin_connections.get("anode") or str(self.pin).upper()
        if anode_pin in pin_states:
            ps = pin_states[anode_pin]
            if isinstance(ps, dict):
                is_high = ps.get("digital", 0) == 1 or ps.get("voltage", 0.0) >= 2.0
                pwm = ps.get("pwm", 1.0 if is_high else 0.0)
                volt = ps.get("voltage", 3.3 if is_high else 0.0)
            else:
                is_high = bool(ps)
                pwm = 1.0 if is_high else 0.0
                volt = 3.3 if is_high else 0.0
            self.update_state(1 if is_high else 0, pwm, volt)

    def render_visual(self) -> str:
        symbol = "(*)" if self.is_on else "( )"
        col = self.color.lower()
        if self.is_on:
            return f"[{col} bold]{symbol} LED '{self.name}' (Pin {self.pin}) [ON - {int(self.brightness * 100)}%][/]"
        return f"[dim]{symbol} LED '{self.name}' (Pin {self.pin}) [OFF][/dim]"

    def get_ascii_visual(self) -> str:
        return self.render_visual()


class RGBLEDDevice(VirtualDevice):
    """Common-Cathode RGB LED with Red, Green, Blue channels."""

    def __init__(self, name: str, r_pin: Union[int, str] = 9, g_pin: Union[int, str] = 10, b_pin: Union[int, str] = 11):
        super().__init__(name, "RGB_LED")
        self.r_pin = r_pin
        self.g_pin = g_pin
        self.b_pin = b_pin
        self.red_pwm: int = 0
        self.green_pwm: int = 0
        self.blue_pwm: int = 0
        self.r_val: float = 0.0
        self.g_val: float = 0.0
        self.b_val: float = 0.0
        self.connect_pin("r", str(r_pin))
        self.connect_pin("g", str(g_pin))
        self.connect_pin("b", str(b_pin))

    def update_state(self, digital: int, pwm: Union[float, int] = 0, voltage: float = 0.0, pin: Optional[Union[int, str]] = None) -> None:
        p_str = str(pin).upper() if pin is not None else ""
        norm_pwm = int(pwm) if pwm > 1.0 else int(pwm * 255)
        if p_str == str(self.r_pin).upper() or pin == self.r_pin:
            self.red_pwm = norm_pwm
            self.r_val = norm_pwm / 255.0
        elif p_str == str(self.g_pin).upper() or pin == self.g_pin:
            self.green_pwm = norm_pwm
            self.g_val = norm_pwm / 255.0
        elif p_str == str(self.b_pin).upper() or pin == self.b_pin:
            self.blue_pwm = norm_pwm
            self.b_val = norm_pwm / 255.0

    def update_from_board(self, pin_states: Dict[str, Any]) -> None:
        for ch, p_val in (("r", self.r_pin), ("g", self.g_pin), ("b", self.b_pin)):
            p_str = str(p_val).upper()
            if p_str in pin_states:
                ps = pin_states[p_str]
                val = ps.get("pwm", 1.0 if ps.get("digital", 0) == 1 else 0.0) if isinstance(ps, dict) else float(bool(ps))
                self.update_state(1 if val > 0 else 0, val * 255, 3.3 * val, pin=p_val)

    def render_visual(self) -> str:
        return f"[bold]RGB LED '{self.name}'[/bold]: (R: {self.red_pwm:3d}, G: {self.green_pwm:3d}, B: {self.blue_pwm:3d}) | #[b]{self.red_pwm:02X}{self.green_pwm:02X}{self.blue_pwm:02X}[/b]"

    def get_ascii_visual(self) -> str:
        return self.render_visual()


class ButtonDevice(VirtualDevice):
    """Momentary Pushbutton with internal pull-up / pull-down support."""

    def __init__(self, name: str, pin: Union[int, str] = 2, pullup: bool = True):
        super().__init__(name, "PUSHBUTTON")
        self.pin = pin
        self.pullup = pullup
        self.is_pressed: bool = False
        self.connect_pin("out", str(pin))

    def press(self) -> None:
        self.is_pressed = True

    def release(self) -> None:
        self.is_pressed = False

    def toggle(self) -> bool:
        self.is_pressed = not self.is_pressed
        return self.is_pressed

    def get_digital_level(self) -> int:
        if self.pullup:
            return 0 if self.is_pressed else 1
        return 1 if self.is_pressed else 0

    def get_input_to_board(self) -> Dict[str, Any]:
        p_str = str(self.pin).upper()
        val = self.get_digital_level()
        volt = 0.0 if val == 0 else 5.0
        return {p_str: {"digital": val, "voltage": volt}}

    def render_visual(self) -> str:
        state_str = "[bold green][PRESSED][/bold green]" if self.is_pressed else "[dim][RELEASED][/dim]"
        return f"Button '{self.name}' (Pin {self.pin}): {state_str} (Pull-up: {self.pullup})"

    def get_ascii_visual(self) -> str:
        return self.render_visual()


class PotentiometerDevice(VirtualDevice):
    """Analog Potentiometer / Variable Voltage Source."""

    def __init__(self, name: str, pin: Union[int, str] = "A0", max_voltage: float = 5.0, voltage: float = 2.5, current_voltage: float = 2.5):
        super().__init__(name, "POTENTIOMETER")
        self.pin = pin
        self.max_voltage = float(max_voltage)
        self.voltage = float(voltage if voltage != 2.5 else current_voltage)
        self.connect_pin("wiper", str(pin))

    def set_voltage(self, v: float) -> None:
        self.voltage = min(self.max_voltage, max(0.0, float(v)))

    def set_raw_adc(self, raw_counts: int, resolution_bits: int = 10) -> None:
        max_counts = (1 << resolution_bits) - 1
        ratio = max(0, min(max_counts, int(raw_counts))) / max_counts
        self.voltage = ratio * self.max_voltage

    def get_raw_adc(self, resolution_bits: int = 10) -> int:
        max_counts = (1 << resolution_bits) - 1
        ratio = self.voltage / self.max_voltage if self.max_voltage > 0 else 0
        return int(round(ratio * max_counts))

    def get_input_to_board(self) -> Dict[str, Any]:
        p_str = str(self.pin).upper()
        return {p_str: {"voltage": self.voltage, "analog": self.get_raw_adc(), "max_voltage": self.max_voltage}}

    def render_visual(self) -> str:
        pct = (self.voltage / self.max_voltage) * 100.0 if self.max_voltage > 0 else 0
        bar_len = 16
        filled = int((pct / 100.0) * bar_len)
        bar = "=" * filled + "-" * (bar_len - filled)
        return f"Potentiometer '{self.name}' (Pin {self.pin}): [{bar}] {self.voltage:.2f}V ({self.get_raw_adc()} ADC, {pct:.0f}%)"

    def get_ascii_visual(self) -> str:
        return self.render_visual()


class DHTSensorDevice(VirtualDevice):
    """DHT11 / DHT22 Temperature & Humidity Sensor."""

    def __init__(self, name: str, pin: Union[int, str] = 7, temperature: float = 24.5, humidity: float = 55.0, temperature_c: float = 24.5, humidity_pct: float = 55.0):
        super().__init__(name, "DHT_SENSOR")
        self.pin = pin
        self.temperature = float(temperature if temperature != 24.5 else temperature_c)
        self.humidity = float(humidity if humidity != 55.0 else humidity_pct)
        self.temperature_c = self.temperature
        self.humidity_pct = self.humidity
        self.connect_pin("data", str(pin))

    def set_readings(self, temp_c: float, humidity: float) -> None:
        self.temperature = float(temp_c)
        self.humidity = float(humidity)
        self.temperature_c = self.temperature
        self.humidity_pct = self.humidity

    def get_input_to_board(self) -> Dict[str, Any]:
        p_str = str(self.pin).upper()
        return {
            p_str: {
                "temperature": self.temperature,
                "humidity": self.humidity,
                "temperature_c": self.temperature_c,
                "humidity_pct": self.humidity_pct,
                "sensor_type": "DHT11"
            }
        }

    def render_visual(self) -> str:
        return f"DHT11 '{self.name}' (Pin {self.pin}): Temp: [bold yellow]{self.temperature:.1f} °C[/bold yellow] | Humidity: [bold cyan]{self.humidity:.1f} %[/bold cyan]"

    def get_ascii_visual(self) -> str:
        return self.render_visual()


class UltrasonicSensorDevice(VirtualDevice):
    """HC-SR04 Ultrasonic Distance Sensor."""

    def __init__(self, name: str, trig_pin: Union[int, str] = 8, echo_pin: Union[int, str] = 9, distance_cm: float = 35.0):
        super().__init__(name, "ULTRASONIC_HCSR04")
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.distance_cm = float(distance_cm)
        self.connect_pin("trig", str(trig_pin))
        self.connect_pin("echo", str(echo_pin))

    def set_distance(self, dist_cm: float) -> None:
        self.distance_cm = max(2.0, min(400.0, float(dist_cm)))

    def get_input_to_board(self) -> Dict[str, Any]:
        p_str = str(self.echo_pin).upper()
        pulse_us = self.distance_cm * 58.2
        return {p_str: {"distance_cm": self.distance_cm, "pulse_duration_us": pulse_us}}

    def render_visual(self) -> str:
        return f"HC-SR04 '{self.name}' (Trig:{self.trig_pin}, Echo:{self.echo_pin}): [bold cyan]{self.distance_cm:.1f} cm[/bold cyan]"

    def get_ascii_visual(self) -> str:
        return self.render_visual()


class ServoDevice(VirtualDevice):
    """Standard Micro Servo Motor (0 to 180 degrees)."""

    def __init__(self, name: str, pin: Union[int, str] = 9, angle: float = 90.0, angle_deg: float = 90.0):
        super().__init__(name, "SERVO_MOTOR")
        self.pin = pin
        self.angle: float = float(angle if angle != 90.0 else angle_deg)
        self.angle_deg: float = self.angle
        self.connect_pin("signal", str(pin))

    def set_angle(self, a: float) -> None:
        self.angle = max(0.0, min(180.0, float(a)))
        self.angle_deg = self.angle

    def update_from_board(self, pin_states: Dict[str, Any]) -> None:
        p_str = str(self.pin).upper()
        if p_str in pin_states:
            ps = pin_states[p_str]
            if isinstance(ps, dict):
                if "servo_angle" in ps:
                    self.set_angle(ps["servo_angle"])
                elif "pwm" in ps:
                    self.set_angle(ps["pwm"] * 180.0)

    def render_visual(self) -> str:
        pointer = "-->" if self.angle > 135 else ("^" if 45 <= self.angle <= 135 else "<--")
        return f"Servo '{self.name}' (Pin {self.pin}): Angle = [bold green]{self.angle:.1f}°[/bold green] [dim]({pointer} 0°..180°)[/dim]"

    def get_ascii_visual(self) -> str:
        return self.render_visual()


class LCD16x2Device(VirtualDevice):
    """HD44780 / I2C 16x2 Character LCD Display."""

    def __init__(self, name: str, i2c_addr: int = 0x27):
        super().__init__(name, "LCD_16X2")
        self.i2c_addr = i2c_addr
        self.lines: List[str] = [" " * 16, " " * 16]
        self.cursor_pos: Tuple[int, int] = (0, 0)

    def write_line(self, line_idx: int, text: str) -> None:
        if 0 <= line_idx < 2:
            padded = text.ljust(16)[:16]
            self.lines[line_idx] = padded

    def write_char(self, ch: str) -> None:
        r, c = self.cursor_pos
        if r < 2 and c < 16:
            line = list(self.lines[r])
            line[c] = ch[0]
            self.lines[r] = "".join(line)
            self.cursor_pos = (r, c + 1)

    def write_text(self, text: str) -> None:
        for ch in text:
            if ch == "\n":
                self.cursor_pos = (min(1, self.cursor_pos[0] + 1), 0)
            else:
                self.write_char(ch)

    def set_cursor(self, col: int, row: int) -> None:
        self.cursor_pos = (min(1, max(0, row)), min(15, max(0, col)))

    def clear(self) -> None:
        self.lines = [" " * 16, " " * 16]
        self.cursor_pos = (0, 0)

    def render_visual(self) -> str:
        sep = "+" + "-" * 18 + "+"
        l1 = f"| {self.lines[0]} |"
        l2 = f"| {self.lines[1]} |"
        return f"[bold cyan]LCD 16x2 ({self.name} @ 0x{self.i2c_addr:02X})[/bold cyan]:\n  {sep}\n  {l1}\n  {l2}\n  {sep}"

    def get_ascii_visual(self) -> str:
        return self.render_visual()


class OLEDSSD1306Device(VirtualDevice):
    """128x64 Monochrome I2C/SPI OLED Graphic Display."""

    def __init__(self, name: str, i2c_addr: int = 0x3C):
        super().__init__(name, "OLED_SSD1306")
        self.i2c_addr = i2c_addr
        self.text_buffer: List[str] = []

    def set_text_line(self, line_idx: int, text: str) -> None:
        while len(self.text_buffer) <= line_idx:
            self.text_buffer.append("")
        self.text_buffer[line_idx] = text[:21]

    def print_text(self, text: str) -> None:
        for line in str(text).splitlines():
            self.text_buffer.append(line[:21])
        if len(self.text_buffer) > 8:
            self.text_buffer = self.text_buffer[-8:]

    def clear(self) -> None:
        self.text_buffer.clear()

    def render_visual(self) -> str:
        sep = "+" + "-" * 23 + "+"
        display_lines = []
        for i in range(max(4, min(8, len(self.text_buffer)))):
            txt = self.text_buffer[i] if i < len(self.text_buffer) else ""
            display_lines.append(f"  | {txt:<21s} |")
        header = f"[bold magenta]OLED SSD1306 ({self.name} @ 0x{self.i2c_addr:02X})[/bold magenta]:"
        return "\n".join([header, "  " + sep] + display_lines + ["  " + sep])

    def get_ascii_visual(self) -> str:
        return self.render_visual()


class DeviceManager:
    """Manages collection of attached virtual devices and interfaces with the Arduino Runtime."""

    def __init__(self):
        self.devices: Dict[str, VirtualDevice] = {}

    def add_device(self, dev: VirtualDevice) -> None:
        self.devices[dev.name] = dev

    def remove_device(self, name: str) -> bool:
        if name in self.devices:
            del self.devices[name]
            return True
        return False

    def get(self, name: str) -> Optional[VirtualDevice]:
        return self.devices.get(name)

    def sync_outputs(self, runtime: Any) -> None:
        pin_states: Dict[str, Any] = {}
        for p_name, p_obj in runtime.pins.items():
            pin_states[p_name] = p_obj.to_dict()
            # Also add without D prefix, e.g. "13"
            if p_name.startswith("D") and p_name[1:].isdigit():
                pin_states[p_name[1:]] = p_obj.to_dict()
        for dev in self.devices.values():
            dev.update_from_board(pin_states)

    def inject_inputs(self, runtime: Any) -> None:
        for dev in self.devices.values():
            in_dict = dev.get_input_to_board()
            for p_str, vals in in_dict.items():
                p_norm = p_str.upper()
                if p_norm.isdigit():
                    p_norm = f"D{p_norm}"
                if p_norm in runtime.pins:
                    pin_obj = runtime.pins[p_norm]
                    if "digital" in vals:
                        pin_obj.digital_value = vals["digital"]
                    if "voltage" in vals:
                        pin_obj.voltage = vals["voltage"]
                    if "analog" in vals:
                        pin_obj.analog_value = vals["analog"]
