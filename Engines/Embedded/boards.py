"""Classified Board Database and Hardware Architecture Catalog for Embedded Engine.
Covers Arduino (AVR/ARM), Espressif (ESP8266/ESP32), STMicroelectronics (STM32),
Raspberry Pi (RP2040/RP2350/SBC), Texas Instruments (C2000), Microchip PIC, 8051, and RISC-V.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class BoardSpecification:
    """Comprehensive hardware profile and pinout mapping for a microcontroller development board."""
    board_id: str
    name: str
    family: str                          # "Arduino", "ESP", "STM32", "Raspberry Pi", "TI", "PIC", "8051", "RISC-V"
    mcu_chip: str
    architecture: str                    # "AVR_8", "ARM_Cortex_M0+", "ARM_Cortex_M3", "ARM_Cortex_M4F", "Xtensa_LX6", "Xtensa_LX7", "RISC_V_32", "C2000_DSP"
    clock_freq_hz: float
    supply_voltage: float                # 5.0 or 3.3 V
    flash_bytes: int
    sram_bytes: int
    eeprom_bytes: int = 0
    digital_pins: List[str] = field(default_factory=list)
    analog_pins: List[str] = field(default_factory=list)
    pwm_pins: List[str] = field(default_factory=list)
    uart_pins: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # {"UART0": ("D0", "D1")}
    i2c_pins: Dict[str, Tuple[str, str]] = field(default_factory=dict)   # {"I2C0": ("SDA", "SCL")}
    spi_pins: Dict[str, Tuple[str, str, str]] = field(default_factory=dict) # {"SPI0": ("MOSI", "MISO", "SCK")}
    adc_resolution_bits: int = 10
    dac_pins: List[str] = field(default_factory=list)
    default_baud: int = 115200
    flasher_tool: str = "avrdude"         # "avrdude", "esptool", "st-flash", "picotool", "openocd"
    fqbn: str = ""                       # Fully Qualified Board Name for arduino-cli
    description: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def clock_freq_mhz(self) -> float:
        return self.clock_freq_hz / 1e6

    @property
    def clock_mhz(self) -> float:
        return self.clock_freq_mhz

    @property
    def id(self) -> str:
        return self.board_id

    @property
    def mcu(self) -> str:
        return self.mcu_chip

    @property
    def flash_kb(self) -> int:
        return int(self.flash_bytes / 1024)

    @property
    def sram_kb(self) -> int:
        return int(self.sram_bytes / 1024)

    @property
    def ram_kb(self) -> int:
        return self.sram_kb

    @property
    def voltage_v(self) -> float:
        return self.supply_voltage

    @property
    def gpio_count(self) -> int:
        return len(self.digital_pins)

    @property
    def adc_channels(self) -> int:
        return len(self.analog_pins)

    @property
    def pwm_channels(self) -> int:
        return len(self.pwm_pins)

    @property
    def upload_tool(self) -> str:
        return self.flasher_tool

    @property
    def upload_protocol(self) -> str:
        if self.family == "ESP": return "esptool"
        if self.family == "STM32": return "stlink"
        if self.family == "Raspberry Pi": return "picoboot"
        return "serial"

    @property
    def flash_base_addr(self) -> str:
        if self.family == "STM32": return "0x08000000"
        if self.family == "ESP": return "0x10000"
        if self.family == "Raspberry Pi": return "0x10000000"
        return "0x0000"

    @property
    def ram_base_addr(self) -> str:
        if self.family == "STM32": return "0x20000000"
        if self.family == "ESP": return "0x3FFE8000"
        return "0x0100"

    @property
    def signature(self) -> int:
        return 0x1E95

    @property
    def has_uart(self) -> bool:
        return bool(self.uart_pins)

    @property
    def has_i2c(self) -> bool:
        return bool(self.i2c_pins)

    @property
    def has_spi(self) -> bool:
        return bool(self.spi_pins)

    @property
    def has_can(self) -> bool:
        return "can" in self.tags or "automotive" in self.tags or self.family in ("STM32", "TI")

    @property
    def has_wifi(self) -> bool:
        return "wifi" in self.tags or "wireless" in self.tags or self.family == "ESP"

    @property
    def has_ble(self) -> bool:
        return "ble" in self.tags or "bluetooth" in self.tags or "wireless" in self.tags or self.family in ("ESP", "Nordic")

    @property
    def pinout_map(self) -> Dict[str, str]:
        p_map = {}
        for p in self.digital_pins:
            p_map[p] = "GPIO"
        for p in self.analog_pins:
            p_map[p] = "ADC"
        for p in self.pwm_pins:
            p_map[p] = "PWM"
        for k, v in self.i2c_pins.items():
            p_map[v[0]] = f"{k}_SDA"
            p_map[v[1]] = f"{k}_SCL"
        return p_map

    def summary(self) -> str:
        lines = [
            f"[bold cyan]=== Development Board: {self.name} ({self.board_id}) ===[/bold cyan]",
            f"Family & MCU      : [bold yellow]{self.family}[/bold yellow] | Chip: [bold green]{self.mcu_chip}[/bold green] ({self.architecture})",
            f"Clock Frequency   : {self.clock_freq_mhz:.1f} MHz",
            f"Supply / Logic    : {self.supply_voltage:.1f} V logic level",
            f"Memory Capacity   : Flash: {self.flash_kb} KB | SRAM: {self.sram_kb} KB | EEPROM: {self.eeprom_bytes} B",
            f"Digital I/O Pins  : {len(self.digital_pins)} pins ({', '.join(self.digital_pins[:8])}...)",
            f"Analog Inputs     : {len(self.analog_pins)} channels ({self.adc_resolution_bits}-bit ADC) - {', '.join(self.analog_pins)}",
            f"PWM Output Pins   : {len(self.pwm_pins)} pins ({', '.join(self.pwm_pins)})",
            f"Hardware Buses    : UART: {list(self.uart_pins.keys())} | I2C: {list(self.i2c_pins.keys())} | SPI: {list(self.spi_pins.keys())}",
            f"Flasher / Tool    : {self.flasher_tool} (FQBN: {self.fqbn or 'N/A'})",
            f"Description       : {self.description}"
        ]
        return "\n".join(lines)


class BoardCatalog:
    """Comprehensive repository of 25+ industry standard development boards and architectures."""

    def __init__(self):
        self._boards: Dict[str, BoardSpecification] = {}
        self._init_catalog()

    def _register(self, spec: BoardSpecification) -> None:
        self._boards[spec.board_id.upper()] = spec

    def _init_catalog(self) -> None:
        # ==========================================
        # 1. ARDUINO FAMILY (AVR & ARM)
        # ==========================================
        self._register(BoardSpecification(
            board_id="UNO",
            name="Arduino Uno R3",
            family="Arduino",
            mcu_chip="ATmega328P",
            architecture="AVR_8",
            clock_freq_hz=16e6,
            supply_voltage=5.0,
            flash_bytes=32768,
            sram_bytes=2048,
            eeprom_bytes=1024,
            digital_pins=[f"D{i}" for i in range(14)],
            analog_pins=[f"A{i}" for i in range(6)],
            pwm_pins=["D3", "D5", "D6", "D9", "D10", "D11"],
            uart_pins={"UART0": ("D0", "D1")},
            i2c_pins={"I2C0": ("A4", "A5")},
            spi_pins={"SPI0": ("D11", "D12", "D13")},
            adc_resolution_bits=10,
            default_baud=9600,
            flasher_tool="avrdude",
            fqbn="arduino:avr:uno",
            description="The classic industry-standard 8-bit AVR development board.",
            tags=["arduino", "uno", "atmega328p", "avr", "5v", "beginner"]
        ))

        self._register(BoardSpecification(
            board_id="MEGA",
            name="Arduino Mega 2560",
            family="Arduino",
            mcu_chip="ATmega2560",
            architecture="AVR_8",
            clock_freq_hz=16e6,
            supply_voltage=5.0,
            flash_bytes=262144,
            sram_bytes=8192,
            eeprom_bytes=4096,
            digital_pins=[f"D{i}" for i in range(54)],
            analog_pins=[f"A{i}" for i in range(16)],
            pwm_pins=[f"D{i}" for i in range(2, 14)] + ["D44", "D45", "D46"],
            uart_pins={
                "UART0": ("D0", "D1"),
                "UART1": ("D19", "D18"),
                "UART2": ("D17", "D16"),
                "UART3": ("D15", "D14")
            },
            i2c_pins={"I2C0": ("D20", "D21")},
            spi_pins={"SPI0": ("D51", "D50", "D52")},
            adc_resolution_bits=10,
            default_baud=115200,
            flasher_tool="avrdude",
            fqbn="arduino:avr:mega",
            description="High pin-count 8-bit AVR board with 54 digital I/O lines and 4 hardware UARTs.",
            tags=["arduino", "mega", "atmega2560", "robotics", "3d-printing"]
        ))

        self._register(BoardSpecification(
            board_id="NANO",
            name="Arduino Nano R3",
            family="Arduino",
            mcu_chip="ATmega328P",
            architecture="AVR_8",
            clock_freq_hz=16e6,
            supply_voltage=5.0,
            flash_bytes=32768,
            sram_bytes=2048,
            eeprom_bytes=1024,
            digital_pins=[f"D{i}" for i in range(14)],
            analog_pins=[f"A{i}" for i in range(8)],
            pwm_pins=["D3", "D5", "D6", "D9", "D10", "D11"],
            uart_pins={"UART0": ("D0", "D1")},
            i2c_pins={"I2C0": ("A4", "A5")},
            spi_pins={"SPI0": ("D11", "D12", "D13")},
            adc_resolution_bits=10,
            default_baud=115200,
            flasher_tool="avrdude",
            fqbn="arduino:avr:nano",
            description="Compact breadboard-friendly 8-bit development board with 8 analog inputs.",
            tags=["arduino", "nano", "breadboard", "compact"]
        ))

        self._register(BoardSpecification(
            board_id="DUE",
            name="Arduino Due",
            family="Arduino",
            mcu_chip="ATSAM3X8E",
            architecture="ARM_Cortex_M3",
            clock_freq_hz=84e6,
            supply_voltage=3.3,
            flash_bytes=524288,
            sram_bytes=98304,
            digital_pins=[f"D{i}" for i in range(54)],
            analog_pins=[f"A{i}" for i in range(12)],
            pwm_pins=[f"D{i}" for i in range(2, 14)],
            uart_pins={"UART0": ("D0", "D1"), "UART1": ("D19", "D18")},
            i2c_pins={"I2C0": ("D20", "D21"), "I2C1": ("SDA1", "SCL1")},
            spi_pins={"SPI0": ("MOSI", "MISO", "SCK")},
            dac_pins=["DAC0", "DAC1"],
            adc_resolution_bits=12,
            default_baud=115200,
            flasher_tool="bossac",
            fqbn="arduino:sam:arduino_due_x",
            description="32-bit ARM Cortex-M3 Arduino with 84 MHz CPU and dual true analog DAC outputs.",
            tags=["arduino", "due", "sam3x8e", "arm", "32bit", "cortex-m3"]
        ))

        # ==========================================
        # 2. ESPRESSIF ESP FAMILY (ESP8266 & ESP32)
        # ==========================================
        self._register(BoardSpecification(
            board_id="ESP8266",
            name="ESP8266 NodeMCU / D1 Mini",
            family="ESP",
            mcu_chip="ESP8266EX",
            architecture="Xtensa_L106",
            clock_freq_hz=80e6,
            supply_voltage=3.3,
            flash_bytes=4194304,  # 4 MB
            sram_bytes=81920,     # 80 KB
            digital_pins=["D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"],
            analog_pins=["A0"],
            pwm_pins=["D1", "D2", "D5", "D6", "D7", "D8"],
            uart_pins={"UART0": ("TX", "RX")},
            i2c_pins={"I2C0": ("D2", "D1")},
            spi_pins={"SPI0": ("D7", "D6", "D5")},
            adc_resolution_bits=10,
            default_baud=115200,
            flasher_tool="esptool",
            fqbn="esp8266:esp8266:nodemcuv2",
            description="Ultra low-cost 80 MHz Wi-Fi IoT microcontroller.",
            tags=["esp", "esp8266", "wifi", "iot", "nodemcu"]
        ))

        self._register(BoardSpecification(
            board_id="ESP32",
            name="ESP32 DevKit V1",
            family="ESP",
            mcu_chip="ESP32-D0WDQ6",
            architecture="Xtensa_LX6",
            clock_freq_hz=240e6,
            supply_voltage=3.3,
            flash_bytes=4194304,  # 4 MB
            sram_bytes=532480,    # 520 KB
            digital_pins=[f"GPIO{i}" for i in (0, 2, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33, 34, 35)],
            analog_pins=[f"ADC{i}" for i in range(1, 9)],
            pwm_pins=[f"GPIO{i}" for i in (2, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27)],
            uart_pins={"UART0": ("GPIO1", "GPIO3"), "UART2": ("GPIO17", "GPIO16")},
            i2c_pins={"I2C0": ("GPIO21", "GPIO22")},
            spi_pins={"VSPI": ("GPIO23", "GPIO19", "GPIO18")},
            dac_pins=["GPIO25", "GPIO26"],
            adc_resolution_bits=12,
            default_baud=115200,
            flasher_tool="esptool",
            fqbn="esp32:esp32:esp32",
            description="Dual-core 240 MHz SoC with integrated Wi-Fi and Bluetooth Low Energy (BLE).",
            tags=["esp", "esp32", "wifi", "bluetooth", "dual-core", "iot"]
        ))

        self._register(BoardSpecification(
            board_id="ESP32S3",
            name="ESP32-S3 Dev Module",
            family="ESP",
            mcu_chip="ESP32-S3",
            architecture="Xtensa_LX7",
            clock_freq_hz=240e6,
            supply_voltage=3.3,
            flash_bytes=8388608,  # 8 MB
            sram_bytes=524288,    # 512 KB
            digital_pins=[f"GPIO{i}" for i in range(1, 22)],
            analog_pins=[f"ADC{i}" for i in range(1, 11)],
            pwm_pins=[f"GPIO{i}" for i in range(1, 22)],
            uart_pins={"UART0": ("GPIO43", "GPIO44")},
            i2c_pins={"I2C0": ("GPIO8", "GPIO9")},
            spi_pins={"SPI0": ("GPIO11", "GPIO13", "GPIO12")},
            adc_resolution_bits=12,
            default_baud=115200,
            flasher_tool="esptool",
            fqbn="esp32:esp32:esp32s3",
            description="Dual-core 240 MHz Xtensa LX7 MCU with vector instructions for TinyML AI acceleration.",
            tags=["esp", "esp32-s3", "tinyml", "ai", "dual-core"]
        ))

        self._register(BoardSpecification(
            board_id="ESP32C3",
            name="ESP32-C3 RISC-V DevKit",
            family="ESP",
            mcu_chip="ESP32-C3",
            architecture="RISC_V_32",
            clock_freq_hz=160e6,
            supply_voltage=3.3,
            flash_bytes=4194304,
            sram_bytes=409600,
            digital_pins=[f"GPIO{i}" for i in range(22)],
            analog_pins=[f"ADC{i}" for i in range(1, 6)],
            pwm_pins=[f"GPIO{i}" for i in range(22)],
            uart_pins={"UART0": ("GPIO21", "GPIO20")},
            i2c_pins={"I2C0": ("GPIO8", "GPIO9")},
            spi_pins={"SPI0": ("GPIO6", "GPIO5", "GPIO4")},
            adc_resolution_bits=12,
            default_baud=115200,
            flasher_tool="esptool",
            fqbn="esp32:esp32:esp32c3",
            description="Single-core 32-bit RISC-V 160 MHz microcontroller with Wi-Fi & BLE 5.0.",
            tags=["esp", "esp32-c3", "risc-v", "wifi", "ble"]
        ))

        # ==========================================
        # 3. STMICROELECTRONICS STM32 FAMILY
        # ==========================================
        self._register(BoardSpecification(
            board_id="STM32F103",
            name="STM32F103C8T6 (Blue Pill)",
            family="STM32",
            mcu_chip="STM32F103C8T6",
            architecture="ARM_Cortex_M3",
            clock_freq_hz=72e6,
            supply_voltage=3.3,
            flash_bytes=65536,    # 64 KB
            sram_bytes=20480,     # 20 KB
            digital_pins=[f"PA{i}" for i in range(16)] + [f"PB{i}" for i in range(16)] + ["PC13", "PC14", "PC15"],
            analog_pins=[f"PA{i}" for i in range(8)] + [f"PB{i}" for i in range(2)],
            pwm_pins=["PA0", "PA1", "PA2", "PA3", "PA6", "PA7", "PA8", "PA9", "PA10", "PB0", "PB1", "PB6", "PB7"],
            uart_pins={"USART1": ("PA9", "PA10"), "USART2": ("PA2", "PA3"), "USART3": ("PB10", "PB11")},
            i2c_pins={"I2C1": ("PB6", "PB7"), "I2C2": ("PB10", "PB11")},
            spi_pins={"SPI1": ("PA7", "PA6", "PA5"), "SPI2": ("PB15", "PB14", "PB13")},
            adc_resolution_bits=12,
            default_baud=115200,
            flasher_tool="st-flash",
            fqbn="STMicroelectronics:stm32:GenF1:pnum=BLUEPILL_F103C8",
            description="72 MHz 32-bit ARM Cortex-M3 microcontroller widely used in robotics and industrial controls.",
            tags=["stm32", "bluepill", "arm", "cortex-m3", "stm32f103"]
        ))

        self._register(BoardSpecification(
            board_id="STM32F401",
            name="STM32F401CCU6 (Black Pill)",
            family="STM32",
            mcu_chip="STM32F401CCU6",
            architecture="ARM_Cortex_M4F",
            clock_freq_hz=84e6,
            supply_voltage=3.3,
            flash_bytes=262144,   # 256 KB
            sram_bytes=65536,     # 64 KB
            digital_pins=[f"PA{i}" for i in range(16)] + [f"PB{i}" for i in range(16)] + ["PC13"],
            analog_pins=[f"PA{i}" for i in range(8)] + ["PB0", "PB1"],
            pwm_pins=["PA0", "PA1", "PA2", "PA3", "PA6", "PA7", "PA8", "PA9", "PB0", "PB1", "PB6", "PB7", "PB8", "PB9"],
            uart_pins={"USART1": ("PA9", "PA10"), "USART2": ("PA2", "PA3")},
            i2c_pins={"I2C1": ("PB6", "PB7"), "I2C2": ("PB10", "PB3")},
            spi_pins={"SPI1": ("PA7", "PA6", "PA5"), "SPI2": ("PB15", "PB14", "PB13")},
            adc_resolution_bits=12,
            default_baud=115200,
            flasher_tool="st-flash",
            fqbn="STMicroelectronics:stm32:GenF4:pnum=BLACKPILL_F401CC",
            description="84 MHz ARM Cortex-M4F with Hardware Floating-Point Unit (FPU) and DSP instructions.",
            tags=["stm32", "blackpill", "arm", "cortex-m4f", "fpu", "dsp"]
        ))

        self._register(BoardSpecification(
            board_id="STM32F407",
            name="STM32F407G-DISC1 Discovery",
            family="STM32",
            mcu_chip="STM32F407VGT6",
            architecture="ARM_Cortex_M4F",
            clock_freq_hz=168e6,
            supply_voltage=3.3,
            flash_bytes=1048576,  # 1 MB
            sram_bytes=196608,    # 192 KB
            digital_pins=[f"PD{i}" for i in range(16)] + [f"PE{i}" for i in range(16)],
            analog_pins=[f"PA{i}" for i in range(8)] + [f"PC{i}" for i in range(6)],
            pwm_pins=[f"PD{i}" for i in (12, 13, 14, 15)],
            uart_pins={"USART1": ("PB6", "PB7"), "USART2": ("PA2", "PA3"), "USART3": ("PB10", "PB11")},
            i2c_pins={"I2C1": ("PB6", "PB9")},
            spi_pins={"SPI1": ("PA7", "PA6", "PA5")},
            dac_pins=["PA4", "PA5"],
            adc_resolution_bits=12,
            default_baud=115200,
            flasher_tool="st-flash",
            fqbn="STMicroelectronics:stm32:DISCO_F407VG",
            description="168 MHz ARM Cortex-M4F with high SRAM, onboard DACs, MEMS sensor, and audio DAC.",
            tags=["stm32", "discovery", "cortex-m4f", "audio", "dsp", "168mhz"]
        ))

        # ==========================================
        # 4. RASPBERRY PI FAMILY (RP2040 & SBC)
        # ==========================================
        self._register(BoardSpecification(
            board_id="PICO",
            name="Raspberry Pi Pico",
            family="Raspberry Pi",
            mcu_chip="RP2040",
            architecture="ARM_Cortex_M0+",
            clock_freq_hz=133e6,
            supply_voltage=3.3,
            flash_bytes=2097152,  # 2 MB Flash
            sram_bytes=268288,    # 264 KB SRAM
            digital_pins=[f"GP{i}" for i in range(29)],
            analog_pins=["GP26_A0", "GP27_A1", "GP28_A2", "GP29_A3"],
            pwm_pins=[f"GP{i}" for i in range(29)],
            uart_pins={"UART0": ("GP0", "GP1"), "UART1": ("GP4", "GP5")},
            i2c_pins={"I2C0": ("GP4", "GP5"), "I2C1": ("GP2", "GP3")},
            spi_pins={"SPI0": ("GP3", "GP4", "GP2"), "SPI1": ("GP11", "GP12", "GP10")},
            adc_resolution_bits=12,
            default_baud=115200,
            flasher_tool="picotool",
            fqbn="rp2040:rp2040:rpipico",
            description="Dual-core 133 MHz ARM Cortex-M0+ with unique Programmable I/O (PIO) state machines.",
            tags=["raspberrypi", "pico", "rp2040", "dual-core", "pio", "micropython", "c++"]
        ))

        self._register(BoardSpecification(
            board_id="PICO2",
            name="Raspberry Pi Pico 2",
            family="Raspberry Pi",
            mcu_chip="RP2350",
            architecture="ARM_Cortex_M33",
            clock_freq_hz=150e6,
            supply_voltage=3.3,
            flash_bytes=4194304,  # 4 MB
            sram_bytes=524288,    # 512 KB
            digital_pins=[f"GP{i}" for i in range(29)],
            analog_pins=["GP26_A0", "GP27_A1", "GP28_A2", "GP29_A3"],
            pwm_pins=[f"GP{i}" for i in range(29)],
            uart_pins={"UART0": ("GP0", "GP1")},
            i2c_pins={"I2C0": ("GP4", "GP5")},
            spi_pins={"SPI0": ("GP3", "GP4", "GP2")},
            adc_resolution_bits=12,
            default_baud=115200,
            flasher_tool="picotool",
            fqbn="rp2040:rp2040:rpipico2",
            description="Dual-core ARM Cortex-M33 / RISC-V Hazard3 selectable cores at 150 MHz with security architecture.",
            tags=["raspberrypi", "pico2", "rp2350", "cortex-m33", "risc-v"]
        ))

        # ==========================================
        # 5. TI C2000, 8051, PIC, NORDIC
        # ==========================================
        self._register(BoardSpecification(
            board_id="C2000",
            name="TI C2000 Delfino LaunchPad (F28379D)",
            family="TI",
            mcu_chip="TMS320F28379D",
            architecture="C2000_DSP",
            clock_freq_hz=200e6,
            supply_voltage=3.3,
            flash_bytes=1048576,  # 1 MB
            sram_bytes=208896,    # 204 KB
            digital_pins=[f"GPIO{i}" for i in range(64)],
            analog_pins=[f"ADCINA{i}" for i in range(6)] + [f"ADCINB{i}" for i in range(6)],
            pwm_pins=[f"EPWM{i}A" for i in range(1, 13)] + [f"EPWM{i}B" for i in range(1, 13)],
            uart_pins={"SCIA": ("GPIO28", "GPIO29"), "SCIB": ("GPIO18", "GPIO19")},
            i2c_pins={"I2CA": ("GPIO32", "GPIO33")},
            spi_pins={"SPIA": ("GPIO16", "GPIO17", "GPIO18")},
            adc_resolution_bits=16,
            default_baud=115200,
            flasher_tool="openocd",
            fqbn="ti:c2000:f28379d",
            description="Dual-core 200 MHz 32-bit floating-point C28x DSP for high-frequency digital power & motor control.",
            tags=["ti", "c2000", "f28379d", "dsp", "power-electronics", "motor-control"]
        ))

        self._register(BoardSpecification(
            board_id="8051",
            name="Classic Intel 8051 / AT89C51",
            family="8051",
            mcu_chip="AT89C51",
            architecture="8051_8",
            clock_freq_hz=12e6,
            supply_voltage=5.0,
            flash_bytes=4096,     # 4 KB
            sram_bytes=128,       # 128 Bytes
            digital_pins=[f"P0.{i}" for i in range(8)] + [f"P1.{i}" for i in range(8)] + [f"P2.{i}" for i in range(8)] + [f"P3.{i}" for i in range(8)],
            analog_pins=[],
            pwm_pins=[],
            uart_pins={"UART": ("P3.0", "P3.1")},
            default_baud=9600,
            flasher_tool="avrdude",
            description="The foundational 8-bit Harvard architecture microcontroller for educational & embedded fundamentals.",
            tags=["8051", "intel", "at89c51", "classic", "8bit"]
        ))

        self._register(BoardSpecification(
            board_id="PIC16",
            name="Microchip PIC16F877A",
            family="PIC",
            mcu_chip="PIC16F877A",
            architecture="PIC_MidRange",
            clock_freq_hz=20e6,
            supply_voltage=5.0,
            flash_bytes=14336,    # 8K words (14 KB)
            sram_bytes=368,
            eeprom_bytes=256,
            digital_pins=[f"RB{i}" for i in range(8)] + [f"RD{i}" for i in range(8)],
            analog_pins=[f"AN{i}" for i in range(8)],
            pwm_pins=["CCP1", "CCP2"],
            uart_pins={"USART": ("RC6", "RC7")},
            i2c_pins={"MSSP": ("RC3", "RC4")},
            spi_pins={"SPI": ("RC5", "RC4", "RC3")},
            adc_resolution_bits=10,
            default_baud=9600,
            flasher_tool="pk2cmd",
            description="Classic 8-bit Microchip PIC microcontroller widely used in automotive and legacy industrial systems.",
            tags=["pic", "microchip", "pic16f877a", "embedded"]
        ))

        self._register(BoardSpecification(
            board_id="STM32_BLUEPILL",
            name="STM32F103C8T6 (Blue Pill)",
            family="STM32",
            mcu_chip="STM32F103C8T6",
            architecture="ARM_Cortex_M3",
            clock_freq_hz=72e6,
            supply_voltage=3.3,
            flash_bytes=65536,
            sram_bytes=20480,
            digital_pins=[f"PA{i}" for i in range(16)] + [f"PB{i}" for i in range(16)] + ["PC13", "PC14", "PC15"],
            analog_pins=[f"PA{i}" for i in range(8)] + [f"PB{i}" for i in range(2)],
            pwm_pins=["PA0", "PA1", "PA2", "PA3", "PA6", "PA7", "PA8", "PA9", "PA10", "PB0", "PB1", "PB6", "PB7"],
            uart_pins={"USART1": ("PA9", "PA10"), "USART2": ("PA2", "PA3"), "USART3": ("PB10", "PB11")},
            i2c_pins={"I2C1": ("PB6", "PB7"), "I2C2": ("PB10", "PB11")},
            spi_pins={"SPI1": ("PA7", "PA6", "PA5"), "SPI2": ("PB15", "PB14", "PB13")},
            adc_resolution_bits=12,
            default_baud=115200,
            flasher_tool="st-flash",
            fqbn="STMicroelectronics:stm32:GenF1:pnum=BLUEPILL_F103C8",
            description="72 MHz 32-bit ARM Cortex-M3 microcontroller widely used in robotics and industrial controls.",
            tags=["stm32", "bluepill", "arm", "cortex-m3", "stm32f103"]
        ))

        self._register(BoardSpecification(
            board_id="STM32_BLACKPILL",
            name="STM32F401CCU6 (Black Pill)",
            family="STM32",
            mcu_chip="STM32F401CCU6",
            architecture="ARM_Cortex_M4F",
            clock_freq_hz=84e6,
            supply_voltage=3.3,
            flash_bytes=262144,
            sram_bytes=65536,
            digital_pins=[f"PA{i}" for i in range(16)] + [f"PB{i}" for i in range(16)] + ["PC13"],
            analog_pins=[f"PA{i}" for i in range(8)] + ["PB0", "PB1"],
            pwm_pins=["PA0", "PA1", "PA2", "PA3", "PA6", "PA7", "PA8", "PA9", "PB0", "PB1", "PB6", "PB7", "PB8", "PB9"],
            uart_pins={"USART1": ("PA9", "PA10"), "USART2": ("PA2", "PA3")},
            i2c_pins={"I2C1": ("PB6", "PB7"), "I2C2": ("PB10", "PB3")},
            spi_pins={"SPI1": ("PA7", "PA6", "PA5"), "SPI2": ("PB15", "PB14", "PB13")},
            adc_resolution_bits=12,
            default_baud=115200,
            flasher_tool="st-flash",
            fqbn="STMicroelectronics:stm32:GenF4:pnum=BLACKPILL_F401CC",
            description="84 MHz ARM Cortex-M4F with Hardware Floating-Point Unit (FPU) and DSP instructions.",
            tags=["stm32", "blackpill", "arm", "cortex-m4f", "fpu", "dsp"]
        ))

        self._register(BoardSpecification(
            board_id="STM32F411",
            name="STM32F411CEU6 Black Pill",
            family="STM32",
            mcu_chip="STM32F411CEU6",
            architecture="ARM_Cortex_M4F",
            clock_freq_hz=100e6,
            supply_voltage=3.3,
            flash_bytes=524288,
            sram_bytes=131072,
            digital_pins=[f"PA{i}" for i in range(16)] + [f"PB{i}" for i in range(16)],
            analog_pins=[f"PA{i}" for i in range(8)],
            pwm_pins=["PA0", "PA1", "PA2", "PA3", "PA6", "PA7", "PA8", "PA9"],
            default_baud=115200,
            flasher_tool="st-flash",
            tags=["stm32", "stm32f411", "blackpill", "arm"]
        ))

        self._register(BoardSpecification(
            board_id="STM32H743",
            name="STM32 Nucleo-H743ZI2",
            family="STM32",
            mcu_chip="STM32H743ZIT6",
            architecture="ARM_Cortex_M7",
            clock_freq_hz=480e6,
            supply_voltage=3.3,
            flash_bytes=2097152,
            sram_bytes=1048576,
            digital_pins=[f"PG{i}" for i in range(16)] + [f"PD{i}" for i in range(16)],
            analog_pins=[f"PA{i}" for i in range(16)],
            pwm_pins=[f"PD{i}" for i in range(8)],
            default_baud=115200,
            flasher_tool="st-flash",
            tags=["stm32", "h7", "cortex-m7", "high-performance"]
        ))

        # ==========================================
        # 4. RASPBERRY PI SILICON (RP2040 & RP2350)
        # ==========================================
        self._register(BoardSpecification(
            board_id="RPI_PICO",
            name="Raspberry Pi Pico",
            family="Raspberry Pi",
            mcu_chip="RP2040",
            architecture="ARM_Cortex_M0+",
            clock_freq_hz=133e6,
            supply_voltage=3.3,
            flash_bytes=2097152,
            sram_bytes=268288,
            digital_pins=[f"GP{i}" for i in range(30)],
            analog_pins=["GP26", "GP27", "GP28"],
            pwm_pins=[f"GP{i}" for i in range(30)],
            uart_pins={"UART0": ("GP0", "GP1"), "UART1": ("GP4", "GP5")},
            i2c_pins={"I2C0": ("GP4", "GP5"), "I2C1": ("GP6", "GP7")},
            spi_pins={"SPI0": ("GP19", "GP16", "GP18")},
            adc_resolution_bits=12,
            default_baud=115200,
            flasher_tool="picotool",
            fqbn="rp2040:rp2040:rpipico",
            description="Dual-core ARM Cortex-M0+ @ 133 MHz with Programmable I/O (PIO) state machines.",
            tags=["raspberrypi", "pico", "rp2040", "dual-core", "pio"]
        ))

        self._register(BoardSpecification(
            board_id="RPI_PICO_W",
            name="Raspberry Pi Pico W",
            family="Raspberry Pi",
            mcu_chip="RP2040",
            architecture="ARM_Cortex_M0+",
            clock_freq_hz=133e6,
            supply_voltage=3.3,
            flash_bytes=2097152,
            sram_bytes=268288,
            digital_pins=[f"GP{i}" for i in range(30)],
            analog_pins=["GP26", "GP27", "GP28"],
            pwm_pins=[f"GP{i}" for i in range(30)],
            default_baud=115200,
            flasher_tool="picotool",
            tags=["raspberrypi", "pico-w", "wifi", "rp2040"]
        ))

        self._register(BoardSpecification(
            board_id="RPI_PICO2",
            name="Raspberry Pi Pico 2",
            family="Raspberry Pi",
            mcu_chip="RP2350",
            architecture="ARM_Cortex_M33",
            clock_freq_hz=150e6,
            supply_voltage=3.3,
            flash_bytes=4194304,
            sram_bytes=524288,
            digital_pins=[f"GP{i}" for i in range(30)],
            analog_pins=["GP26", "GP27", "GP28", "GP29"],
            pwm_pins=[f"GP{i}" for i in range(30)],
            default_baud=115200,
            flasher_tool="picotool",
            tags=["raspberrypi", "pico2", "rp2350", "arm-m33", "riscv-hazard3"]
        ))

        # ==========================================
        # 5. TEXAS INSTRUMENTS & DSP
        # ==========================================
        self._register(BoardSpecification(
            board_id="TI_C2000_F28379D",
            name="TI C2000 LaunchPad (TMS320F28379D)",
            family="TI",
            mcu_chip="TMS320F28379D",
            architecture="C2000_DSP",
            clock_freq_hz=200e6,
            supply_voltage=3.3,
            flash_bytes=1048576,
            sram_bytes=204800,
            digital_pins=[f"GPIO{i}" for i in range(40)],
            analog_pins=[f"ADCINA{i}" for i in range(6)] + [f"ADCINB{i}" for i in range(6)],
            pwm_pins=[f"EPWM{i}A" for i in range(1, 13)],
            adc_resolution_bits=16,
            default_baud=115200,
            flasher_tool="openocd",
            tags=["ti", "c2000", "dsp", "f28379d", "motor-control", "power-electronics"]
        ))

        self._register(BoardSpecification(
            board_id="TI_C2000_F280049C",
            name="TI C2000 LaunchPad (TMS320F280049C)",
            family="TI",
            mcu_chip="TMS320F280049C",
            architecture="C2000_DSP",
            clock_freq_hz=100e6,
            supply_voltage=3.3,
            flash_bytes=262144,
            sram_bytes=102400,
            digital_pins=[f"GPIO{i}" for i in range(32)],
            analog_pins=[f"ADC{i}" for i in range(8)],
            pwm_pins=[f"EPWM{i}A" for i in range(1, 9)],
            default_baud=115200,
            flasher_tool="openocd",
            tags=["ti", "c2000", "f280049c", "dsp"]
        ))

        # ==========================================
        # 6. RISC-V ARCHITECTURES
        # ==========================================
        self._register(BoardSpecification(
            board_id="RISCV_GD32VF103",
            name="GigaDevice GD32VF103 (Longan Nano)",
            family="RISC-V",
            mcu_chip="GD32VF103CBT6",
            architecture="RISC_V_32",
            clock_freq_hz=108e6,
            supply_voltage=3.3,
            flash_bytes=131072,
            sram_bytes=32768,
            digital_pins=[f"PA{i}" for i in range(16)] + [f"PB{i}" for i in range(16)],
            analog_pins=[f"PA{i}" for i in range(8)],
            pwm_pins=["PA0", "PA1", "PA2", "PA3"],
            default_baud=115200,
            flasher_tool="openocd",
            tags=["riscv", "gd32v", "longan-nano", "risc-v-32"]
        ))

        self._register(BoardSpecification(
            board_id="CH32V003",
            name="WCH CH32V003 DevBoard (10-cent RISC-V)",
            family="RISC-V",
            mcu_chip="CH32V003F4P6",
            architecture="RISC_V_32",
            clock_freq_hz=48e6,
            supply_voltage=3.3,
            flash_bytes=16384,
            sram_bytes=2048,
            digital_pins=[f"PD{i}" for i in range(8)] + [f"PC{i}" for i in range(8)],
            analog_pins=[f"A{i}" for i in range(8)],
            pwm_pins=["PD2", "PD3", "PD4", "PC3"],
            default_baud=115200,
            flasher_tool="wlink",
            tags=["riscv", "ch32v003", "wch", "budget"]
        ))

        self._register(BoardSpecification(
            board_id="NRF52840",
            name="Nordic nRF52840 Dongle / DevKit",
            family="Nordic",
            mcu_chip="nRF52840",
            architecture="ARM_Cortex_M4F",
            clock_freq_hz=64e6,
            supply_voltage=3.3,
            flash_bytes=1048576,
            sram_bytes=262144,
            digital_pins=[f"P0.{i}" for i in range(32)] + [f"P1.{i}" for i in range(16)],
            analog_pins=[f"AIN{i}" for i in range(8)],
            pwm_pins=[f"P0.{i}" for i in (6, 8, 12, 13, 14, 15, 16)],
            uart_pins={"UART0": ("P0.6", "P0.8")},
            i2c_pins={"TWI0": ("P0.26", "P0.27")},
            spi_pins={"SPI0": ("P0.29", "P0.30", "P0.31")},
            adc_resolution_bits=12,
            default_baud=115200,
            flasher_tool="nrfutil",
            fqbn="nordic:nrf52:pca10056",
            description="Bluetooth 5.3, Thread, Zigbee, and USB-enabled ARM Cortex-M4F SoC for ultra-low power wireless IoT.",
            tags=["nordic", "nrf52840", "bluetooth", "ble", "zigbee", "thread", "iot"]
        ))


    @property
    def boards(self) -> Dict[str, BoardSpecification]:
        return self._boards

    def get(self, identifier: str) -> Optional[BoardSpecification]:
        return self.get_board(identifier)

    def get_board(self, identifier: str) -> Optional[BoardSpecification]:
        """Looks up a board by ID, name, MCU chip, or tag."""
        target = identifier.upper().strip()
        if target in self._boards:
            return self._boards[target]
        for spec in self._boards.values():
            if (
                target in spec.name.upper()
                or target in spec.mcu_chip.upper()
                or target in spec.family.upper()
                or any(target.lower() == t.lower() for t in spec.tags)
            ):
                return spec
        return None

    def search(self, query: str) -> List[BoardSpecification]:
        q = query.lower().strip()
        results = []
        for b in self._boards.values():
            if (
                q in b.board_id.lower()
                or q in b.name.lower()
                or q in b.mcu_chip.lower()
                or q in b.family.lower()
                or any(q in t.lower() for t in b.tags)
            ):
                results.append(b)
        return results

    def list_all_boards(self, family: Optional[str] = None) -> List[BoardSpecification]:
        """Returns all registered boards, optionally filtered by family."""
        if not family or family.lower() in ("all", "*"):
            return list(self._boards.values())
        fam_u = family.upper().strip()
        return [b for b in self._boards.values() if b.family.upper() == fam_u]

    def get_all_families(self) -> List[str]:
        """Returns sorted list of all unique board manufacturer families."""
        fams = set(b.family for b in self._boards.values())
        return sorted(list(fams))

    def generate_summary_table(self, family: Optional[str] = None) -> str:
        boards = self.list_all_boards(family)
        lines = [
            f"[bold cyan]=== MICROCONTROLLER & BOARD CATALOG ({len(boards)} models) ===[/bold cyan]",
            "  +-------------------+----------------------+-------------------+----------+-----------+----------+--------+",
            "  | BOARD ID          | NAME                 | MCU ARCH          | CLOCK    | FLASH     | SRAM     | FAMILY |",
            "  +-------------------+----------------------+-------------------+----------+-----------+----------+--------+"
        ]
        for b in boards:
            lines.append(f"  | {b.board_id:<17} | {b.name:<20} | {b.mcu_chip:<17} | {b.clock_freq_mhz:5.0f}MHz | {b.flash_kb:6d} KB | {b.sram_kb:5d} KB | {b.family:<6} |")
        lines.append("  +-------------------+----------------------+-------------------+----------+-----------+----------+--------+")
        return "\n".join(lines)


# Singleton Global Catalog
board_catalog = BoardCatalog()

