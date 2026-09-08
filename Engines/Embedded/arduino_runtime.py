"""
Engines/Embedded/arduino_runtime.py - Arduino C++ & MicroPython Simulation Runtime Engine.
Provides full implementation of the Arduino Core API (pinMode, digitalWrite, analogRead, delay, millis),
Serial communication streams, and built-in standard libraries (Wire I2C, SPI, Servo, EEPROM, LiquidCrystal, OLED).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .boards import BoardSpecification, BoardCatalog, board_catalog
from .virtual_devices import (
    VirtualDevice, LEDDevice, RGBLEDDevice, ButtonDevice, PotentiometerDevice,
    DHTSensorDevice, UltrasonicSensorDevice, ServoDevice, LCD16x2Device, OLEDSSD1306Device,
    DeviceManager
)


class ArduinoPinMode:
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    INPUT_PULLUP = "INPUT_PULLUP"


class ArduinoPinState:
    """Live state of an individual microcontroller pin."""
    def __init__(self, pin_name: str, is_analog: bool = False):
        self.pin_name = pin_name.upper()
        self.mode = ArduinoPinMode.INPUT
        self.is_analog = is_analog
        self.digital_value: int = 0
        self.analog_value: int = 0      # 0 to 1023 (10-bit) or 0 to 4095 (12-bit)
        self.pwm_duty: float = 0.0      # 0.0 to 1.0
        self.voltage: float = 0.0       # 0.0 to 5.0V / 3.3V
        self.attached_devices: List[VirtualDevice] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pin": self.pin_name,
            "mode": self.mode,
            "digital": self.digital_value,
            "analog": self.analog_value,
            "pwm": self.pwm_duty,
            "voltage": self.voltage
        }


@dataclass
class CompilationResult:
    success: bool
    flash_used_bytes: int
    flash_max_bytes: int
    flash_percent: float
    ram_used_bytes: int
    ram_max_bytes: int
    ram_percent: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ArduinoSerialStream:
    """Serial communication stream capturing Serial.print output and telemetry data."""
    def __init__(self, baud: int = 115200):
        self.baudrate = baud
        self.baud = baud
        self.tx_history: List[str] = []
        self.rx_queue: List[str] = []
        self.telemetry_points: List[Tuple[float, List[float]]] = []

    def begin(self, baud: int) -> None:
        self.baudrate = baud
        self.baud = baud

    def print(self, val: Any) -> None:
        text = str(val)
        if self.tx_history and not self.tx_history[-1].endswith("\n"):
            self.tx_history[-1] += text
        else:
            self.tx_history.append(text)
        self._parse_telemetry(text)

    def println(self, val: Any = "") -> None:
        text = str(val) + "\n"
        if self.tx_history and not self.tx_history[-1].endswith("\n"):
            self.tx_history[-1] += text
        else:
            self.tx_history.append(text)
        self._parse_telemetry(text)

    def write(self, byte_or_str: Any) -> None:
        self.print(byte_or_str)

    def write_input(self, text: str) -> None:
        self.send_input(text)

    def read(self) -> int:
        if self.rx_queue:
            return ord(self.rx_queue.pop(0))
        return -1

    def available(self) -> int:
        return len(self.rx_queue)

    def peek(self) -> int:
        if self.rx_queue:
            return ord(self.rx_queue[0])
        return -1

    def flush(self) -> None:
        pass

    def clear(self) -> None:
        self.tx_history.clear()
        self.rx_queue.clear()
        self.telemetry_points.clear()

    def send_input(self, text: str) -> None:
        for ch in text:
            self.rx_queue.append(ch)

    def _parse_telemetry(self, text: str) -> None:
        if "\n" in text:
            raw = text.strip()
            tokens = re.split(r"[\s,]+", raw)
            numbers = []
            for tok in tokens:
                try:
                    num = float(tok)
                    numbers.append(num)
                except ValueError:
                    pass
            if numbers:
                self.telemetry_points.append((time.time(), numbers))
                if len(self.telemetry_points) > 500:
                    self.telemetry_points = self.telemetry_points[-500:]

    def read_buffer(self) -> List[str]:
        lines = []
        for h in self.tx_history:
            for l in h.splitlines():
                if l:
                    lines.append(l)
        return lines

    def get_output_text(self, tail: int = 50) -> str:
        return "".join(self.tx_history[-tail:])


class WireSimulator:
    """I2C Master/Slave protocol emulator."""
    def __init__(self, runtime: ArduinoRuntimeEngine):
        self.runtime = runtime
        self.transmitting_addr: Optional[int] = None
        self.tx_buffer: List[int] = []

    def begin(self, sda_pin: Optional[str] = None, scl_pin: Optional[str] = None) -> None:
        pass

    def beginTransmission(self, address: int) -> None:
        self.transmitting_addr = address
        self.tx_buffer.clear()

    def write(self, data: Any) -> int:
        if isinstance(data, (int, float)):
            self.tx_buffer.append(int(data) & 0xFF)
        elif isinstance(data, str):
            for c in data:
                self.tx_buffer.append(ord(c) & 0xFF)
        return len(self.tx_buffer)

    def endTransmission(self) -> int:
        if self.transmitting_addr is not None:
            for dev in self.runtime.devices.devices.values():
                if getattr(dev, "i2c_addr", None) == self.transmitting_addr:
                    if isinstance(dev, LCD16x2Device):
                        text = "".join(chr(b) for b in self.tx_buffer if 32 <= b <= 126 or b == 10)
                        dev.write_text(text)
                    elif isinstance(dev, OLEDSSD1306Device):
                        text = "".join(chr(b) for b in self.tx_buffer if 32 <= b <= 126 or b == 10)
                        dev.print_text(text)
        self.transmitting_addr = None
        self.tx_buffer.clear()
        return 0


class SPISimulator:
    """SPI Bus Master emulator."""
    def __init__(self, runtime: ArduinoRuntimeEngine):
        self.runtime = runtime

    def begin(self) -> None:
        pass

    def transfer(self, val: int) -> int:
        return int(val) & 0xFF


class ServoSimulator:
    """Servo Motor library instance."""
    def __init__(self, runtime: ArduinoRuntimeEngine):
        self.runtime = runtime
        self.attached_pin: Optional[str] = None
        self.pin_num: Optional[int] = None
        self.angle: float = 90.0
        self.current_angle: float = 90.0

    def attach(self, pin: Union[str, int]) -> None:
        self.attached_pin = str(pin).upper()
        if isinstance(pin, int) or (isinstance(pin, str) and pin.isdigit()):
            self.pin_num = int(pin)
            self.runtime.servos[int(pin)] = self
        self.runtime.pinMode(pin, ArduinoPinMode.OUTPUT)

    def write(self, angle: float) -> None:
        self.angle = max(0.0, min(180.0, float(angle)))
        self.current_angle = self.angle
        if self.attached_pin:
            norm_pin = self.runtime._normalize_pin(self.attached_pin)
            if norm_pin in self.runtime.pins:
                self.runtime.pins[norm_pin].pwm_duty = self.angle / 180.0
            for dev in self.runtime.devices.devices.values():
                if isinstance(dev, ServoDevice):
                    dev.set_angle(self.angle)

    def read(self) -> float:
        return self.angle


class EEPROMSimulator:
    """Non-volatile byte EEPROM simulator."""
    def __init__(self, size: int = 1024):
        self.size = size
        self.memory = bytearray(size)

    def read(self, address: int) -> int:
        if 0 <= address < self.size:
            return self.memory[address]
        return 0

    def write(self, address: int, val: int) -> None:
        if 0 <= address < self.size:
            self.memory[address] = int(val) & 0xFF

    def update(self, address: int, val: int) -> None:
        self.write(address, val)

    def length(self) -> int:
        return self.size


class ArduinoRuntimeEngine:
    """Full-featured Arduino Virtual Machine, Compiler & Simulation Engine."""

    def __init__(self, board_id: str = "UNO"):
        spec = board_catalog.get(board_id) or board_catalog.get("UNO")
        self.board: BoardSpecification = spec
        self.board_spec: BoardSpecification = spec
        self.pins: Dict[str, ArduinoPinState] = {}
        self.devices = DeviceManager()
        self.Serial = ArduinoSerialStream(self.board.default_baud)
        self.serial_stream = self.Serial
        self.Wire = WireSimulator(self)
        self.SPI = SPISimulator(self)
        self.EEPROM = EEPROMSimulator(self.board.eeprom_bytes or 1024)
        self.eeprom = self.EEPROM
        self.servos: Dict[int, ServoSimulator] = {}

        self.current_millis: int = 0
        self.simulated_time_ms: float = 0.0
        self.cycle_count: int = 0
        self.source_code: str = ""
        self.compiled: bool = False
        self.compilation_result: Optional[CompilationResult] = None
        self._global_env: Dict[str, Any] = {}

        self._init_pins()

    def _init_pins(self) -> None:
        self.pins.clear()
        # Digital pins
        for i in range(self.board.gpio_count):
            p_name = f"D{i}"
            self.pins[p_name] = ArduinoPinState(p_name, is_analog=False)
            self.pins[str(i)] = self.pins[p_name]
        # Analog pins
        for i in range(self.board.adc_channels):
            p_name = f"A{i}"
            self.pins[p_name] = ArduinoPinState(p_name, is_analog=True)

    def set_board(self, board_id: str) -> BoardSpecification:
        spec = board_catalog.get(board_id)
        if not spec:
            raise ValueError(f"Board '{board_id}' not found. Use 'board list' to inspect.")
        self.board = spec
        self.board_spec = spec
        self.EEPROM = EEPROMSimulator(spec.eeprom_bytes or 1024)
        self.eeprom = self.EEPROM
        self._init_pins()
        self.compiled = False
        return self.board

    def load_sketch(self, code: str) -> None:
        self.source_code = code
        self.compiled = False
        self.compilation_result = None

    def reset(self) -> None:
        self.current_millis = 0
        self.simulated_time_ms = 0.0
        self.cycle_count = 0
        self.Serial.clear()
        self.servos.clear()
        self._init_pins()
        self._global_env.clear()

    def _normalize_pin(self, pin: Union[str, int]) -> str:
        if isinstance(pin, int):
            return f"D{pin}"
        p = str(pin).upper().strip()
        if p.isdigit():
            return f"D{p}"
        return p

    def pinMode(self, pin: Union[str, int], mode: str) -> None:
        p_name = self._normalize_pin(pin)
        if p_name in self.pins:
            self.pins[p_name].mode = mode.upper()
            if mode.upper() == ArduinoPinMode.INPUT_PULLUP:
                self.pins[p_name].digital_value = 1
                self.pins[p_name].voltage = self.board.voltage_v

    def digitalWrite(self, pin: Union[str, int], val: Union[int, bool, str]) -> None:
        p_name = self._normalize_pin(pin)
        if p_name in self.pins:
            d_val = 1 if val in (1, True, "HIGH", "1", "High", "high") else 0
            ps = self.pins[p_name]
            ps.digital_value = d_val
            ps.voltage = self.board.voltage_v if d_val == 1 else 0.0
            ps.pwm_duty = 1.0 if d_val == 1 else 0.0
            self.devices.sync_outputs(self)

    def digitalRead(self, pin: Union[str, int]) -> int:
        p_name = self._normalize_pin(pin)
        self.devices.inject_inputs(self)
        if p_name in self.pins:
            return self.pins[p_name].digital_value
        return 0

    def analogRead(self, pin: Union[str, int]) -> int:
        p_name = self._normalize_pin(pin)
        self.devices.inject_inputs(self)
        if p_name in self.pins:
            ps = self.pins[p_name]
            max_adc = (1 << self.board.adc_resolution_bits) - 1
            if ps.voltage > 0:
                ratio = ps.voltage / self.board.voltage_v
                ps.analog_value = int(max(0, min(max_adc, ratio * max_adc)))
            return ps.analog_value
        return 0

    def analogWrite(self, pin: Union[str, int], val: int) -> None:
        p_name = self._normalize_pin(pin)
        if p_name in self.pins:
            duty = max(0.0, min(1.0, float(val) / 255.0))
            ps = self.pins[p_name]
            ps.pwm_duty = duty
            ps.voltage = duty * self.board.voltage_v
            ps.digital_value = 1 if duty >= 0.5 else 0
            self.devices.sync_outputs(self)

    def delay(self, ms: float) -> None:
        self.simulated_time_ms += float(ms)
        self.current_millis = int(self.simulated_time_ms)

    def delayMicroseconds(self, us: float) -> None:
        self.simulated_time_ms += float(us) / 1000.0
        self.current_millis = int(self.simulated_time_ms)

    def millis(self) -> int:
        return self.current_millis

    def micros(self) -> int:
        return int(self.simulated_time_ms * 1000)

    def map_val(self, x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
        if in_max == in_min:
            return out_min
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def constrain_val(self, x: float, a: float, b: float) -> float:
        return min(max(x, a), b)

    def tone(self, pin: Union[str, int], frequency: float, duration: float = 0) -> None:
        pass

    def noTone(self, pin: Union[str, int]) -> None:
        pass

    def compile(self) -> CompilationResult:
        code = self.source_code.strip()
        if not code:
            res = CompilationResult(
                success=False,
                flash_used_bytes=0,
                flash_max_bytes=self.board.flash_kb * 1024,
                flash_percent=0.0,
                ram_used_bytes=0,
                ram_max_bytes=self.board.ram_kb * 1024,
                ram_percent=0.0,
                errors=["Sketch editor is empty. Load or enter Arduino C++ code."]
            )
            self.compilation_result = res
            self.compiled = False
            return res

        lines = code.splitlines()
        code_bytes = sum(len(l.strip()) for l in lines) * 4
        flash_max = self.board.flash_kb * 1024
        ram_max = self.board.ram_kb * 1024

        flash_used = min(flash_max, 1450 + code_bytes)
        ram_used = min(ram_max, 180 + len(lines) * 8)

        flash_pct = (flash_used / flash_max) * 100.0 if flash_max > 0 else 0.0
        ram_pct = (ram_used / ram_max) * 100.0 if ram_max > 0 else 0.0

        errors = []
        warnings = []

        # Simple C++ syntax checks
        open_braces = code.count("{")
        close_braces = code.count("}")
        if open_braces != close_braces:
            errors.append(f"Mismatched braces: {open_braces} '{{' vs {close_braces} '}}'")

        if flash_used > flash_max:
            errors.append(f"Sketch size ({flash_used} bytes) exceeds {self.board.name} Flash capacity ({flash_max} bytes)!")

        res = CompilationResult(
            success=(len(errors) == 0),
            flash_used_bytes=flash_used,
            flash_max_bytes=flash_max,
            flash_percent=flash_pct,
            ram_used_bytes=ram_used,
            ram_max_bytes=ram_max,
            ram_percent=ram_pct,
            errors=errors,
            warnings=warnings
        )
        self.compilation_result = res
        self.compiled = res.success
        return res

    def _transpile_c_to_python(self, c_code: str) -> str:
        lines = c_code.splitlines()
        top_level_vars = set()
        top_lines = []
        fn_lines = []
        current_fn = None

        # 1. First pass: extract global variable & object names
        for raw_l in lines:
            l = raw_l.strip()
            if not l or l.startswith("//") or l.startswith("#include") or l.startswith("#define"):
                continue
            if l.endswith(";"):
                l = l[:-1].strip()

            if re.match(r"void\s+(setup|loop|[a-zA-Z_]\w*)\s*\(", l):
                current_fn = True
                continue

            if not current_fn:
                m_var = re.match(r"\b(?:const\s+)?(?:int|float|double|char|byte|uint8_t|uint16_t|uint32_t|int16_t|int32_t|bool|auto|Servo|LiquidCrystal|Adafruit_SSD1306)\s+([a-zA-Z_]\w*)", l)
                if m_var:
                    top_level_vars.add(m_var.group(1))

        # 2. Second pass: generate python code
        current_fn = None
        for raw_l in lines:
            l = raw_l.strip()
            if not l or l.startswith("//") or l.startswith("#include") or l.startswith("#define"):
                continue
            if l.endswith(";"):
                l = l[:-1].strip()

            # Detect function headers
            m_fn = re.match(r"void\s+([a-zA-Z_]\w*)\s*\(\s*\)", l)
            if m_fn:
                fn_name = m_fn.group(1)
                current_fn = fn_name
                fn_lines.append(f"def {fn_name}():")
                if top_level_vars:
                    vars_str = ", ".join(sorted(top_level_vars))
                    fn_lines.append(f"    global {vars_str}")
                continue

            # Class instantiation: Servo myServo; -> myServo = Servo()
            m_inst = re.match(r"(Servo|LiquidCrystal|Adafruit_SSD1306)\s+([a-zA-Z_]\w*)\s*(?:\(.*\))?", l)
            if m_inst:
                cls_name = m_inst.group(1)
                var_name = m_inst.group(2)
                stmt = f"{var_name} = {cls_name}()"
                if current_fn:
                    fn_lines.append(f"    {stmt}")
                else:
                    top_lines.append(stmt)
                continue

            # C-style for-loop conversion: for (int pos = 0; pos <= 180; pos += 45)
            m_for = re.match(r"for\s*\(\s*(?:int\s+)?([a-zA-Z_]\w*)\s*=\s*(\d+)\s*;\s*\1\s*(<=|<)\s*(\d+)\s*;\s*\1\s*(\+=|\+\+|\+=\s*\d+)", l)
            if m_for:
                var, start_v, op, end_v, step_part = m_for.groups()
                start_n = int(start_v)
                end_n = int(end_v) + (1 if op == "<=" else 0)
                step_n = 1
                if "+=" in step_part:
                    step_match = re.search(r"\+=\s*(\d+)", step_part)
                    if step_match:
                        step_n = int(step_match.group(1))
                stmt = f"for {var} in range({start_n}, {end_n}, {step_n}):"
                if current_fn:
                    fn_lines.append(f"    {stmt}")
                else:
                    top_lines.append(stmt)
                continue

            # Type declarations
            l = re.sub(r"\b(?:const\s+)?(int|float|double|char|byte|uint8_t|uint16_t|uint32_t|int16_t|int32_t|bool|auto)\s+([a-zA-Z_]\w*)\s*=", r"\2 =", l)
            l = re.sub(r"\b(?:const\s+)?(int|float|double|char|byte|uint8_t|uint16_t|uint32_t|bool)\s+([a-zA-Z_]\w*)", r"\2 = 0", l)

            # Increment / Decrement
            l = re.sub(r"\b([a-zA-Z_]\w*)\+\+", r"\1 += 1", l)
            l = re.sub(r"\b([a-zA-Z_]\w*)--", r"\1 -= 1", l)

            # Math functions
            l = re.sub(r"\bsin\s*\(", "math.sin(", l)
            l = re.sub(r"\bcos\s*\(", "math.cos(", l)
            l = re.sub(r"\btan\s*\(", "math.tan(", l)
            l = re.sub(r"\bsqrt\s*\(", "math.sqrt(", l)

            l = l.replace("true", "True").replace("false", "False")
            l = l.replace("&&", " and ").replace("||", " or ").replace("!", " not ")
            l = l.replace("{", "").replace("}", "")

            if l.strip():
                if current_fn:
                    fn_lines.append("    " + l.strip())
                else:
                    top_lines.append(l.strip())

        return "\n".join(top_lines + ([""] if top_lines else []) + fn_lines)


    def step_loop(self) -> List[str]:
        """Executes a single step / loop iteration and returns new serial output lines."""
        old_count = len(self.Serial.read_buffer())
        self.devices.inject_inputs(self)

        env = {
            "pinMode": self.pinMode,
            "digitalWrite": self.digitalWrite,
            "digitalRead": self.digitalRead,
            "analogRead": self.analogRead,
            "analogWrite": self.analogWrite,
            "delay": self.delay,
            "delayMicroseconds": self.delayMicroseconds,
            "millis": self.millis,
            "micros": self.micros,
            "map": self.map_val,
            "constrain": self.constrain_val,
            "tone": self.tone,
            "noTone": self.noTone,
            "Serial": self.Serial,
            "Wire": self.Wire,
            "SPI": self.SPI,
            "Servo": lambda: ServoSimulator(self),
            "EEPROM": self.EEPROM,
            "HIGH": 1,
            "LOW": 0,
            "INPUT": ArduinoPinMode.INPUT,
            "OUTPUT": ArduinoPinMode.OUTPUT,
            "INPUT_PULLUP": ArduinoPinMode.INPUT_PULLUP,
            "LED_BUILTIN": 13,
            "math": math,
        }
        for p in self.pins.keys():
            env[p] = p
        env.update(self._global_env)

        py_code = self._transpile_c_to_python(self.source_code)
        try:
            exec(py_code, env)
            self._global_env.update({k: v for k, v in env.items() if not k.startswith("_") and k not in ("math", "Serial", "Wire", "SPI", "EEPROM")})

            if "setup" in env and self.cycle_count == 0:
                env["setup"]()

            if "loop" in env:
                env["loop"]()
            self.cycle_count += 1
            if self.current_millis == 0:
                self.delay(10)
        except Exception as e:
            self.Serial.println(f"[Runtime Exception]: {e}")

        self.devices.sync_outputs(self)
        all_lines = self.Serial.read_buffer()
        return all_lines[old_count:]

    def run_continuous(self, cycles: int = 10) -> Tuple[bool, List[str]]:
        old_count = len(self.Serial.read_buffer())
        for _ in range(cycles):
            self.step_loop()
        all_lines = self.Serial.read_buffer()
        return True, all_lines[old_count:]

    def get_pinout_ascii(self) -> str:
        lines = [
            f"[bold cyan]=== PINOUT & HARDWARE PERIPHERAL STATUS [{self.board.name} - {self.board.mcu}] ===[/bold cyan]",
            "  +-------+--------+---------+---------+---------+----------------------------------+",
            "  | PIN   | MODE   | DIGITAL | VOLTAGE | PWM %   | ATTACHED VIRTUAL HARDWARE        |",
            "  +-------+--------+---------+---------+---------+----------------------------------+"
        ]
        # Collect pins uniquely
        shown = set()
        for p_name, ps in sorted(self.pins.items(), key=lambda x: (x[0][0], int(x[0][1:]) if x[0][1:].isdigit() else 99)):
            if p_name in shown or p_name.isdigit():
                continue
            shown.add(p_name)
            # Find attached devices
            attached = []
            for d in self.devices.devices.values():
                for dp, bp in d.pin_connections.items():
                    if bp == p_name or bp == p_name.replace("D", ""):
                        attached.append(f"{d.name} ({d.device_type})")
            att_str = ", ".join(attached) if attached else "[dim]--[/dim]"
            d_str = "HIGH" if ps.digital_value == 1 else "LOW"
            lines.append(f"  | {p_name:<5} | {ps.mode:<6} | {d_str:<7} | {ps.voltage:5.2f} V | {int(ps.pwm_duty*100):5d}%  | {att_str:<32} |")
        lines.append("  +-------+--------+---------+---------+---------+----------------------------------+")
        return "\n".join(lines)
