"""Harvard-Architecture Microcontroller (MCU) Core and execution engine."""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from .toolchain import Assembler, Instruction
from .peripherals import PeripheralBus


class CPUFlags:
    """Status flags register bits."""
    def __init__(self):
        self.Z: bool = False  # Zero
        self.C: bool = False  # Carry
        self.N: bool = False  # Negative
        self.V: bool = False  # Overflow

    def to_int(self) -> int:
        return (int(self.Z) << 0) | (int(self.C) << 1) | (int(self.N) << 2) | (int(self.V) << 3)

    def from_int(self, val: int) -> None:
        self.Z = bool(val & 1)
        self.C = bool((val >> 1) & 1)
        self.N = bool((val >> 2) & 1)
        self.V = bool((val >> 3) & 1)

    def __repr__(self) -> str:
        z = "Z" if self.Z else "-"
        c = "C" if self.C else "-"
        n = "N" if self.N else "-"
        v = "V" if self.V else "-"
        return f"[{z}{c}{n}{v}]"


class RegisterFile:
    """MCU Register set: ACC, R0-R7, PC, SP, STATUS."""

    def __init__(self):
        self.ACC: int = 0
        self.R: List[int] = [0] * 8
        self.PC: int = 0
        self.SP: int = 0x0FFF  # Top of stack in SRAM
        self.flags = CPUFlags()
        self.cycles: int = 0
        self.halted: bool = False

    def reset(self) -> None:
        self.ACC = 0
        self.R = [0] * 8
        self.PC = 0
        self.SP = 0x0FFF
        self.flags = CPUFlags()
        self.cycles = 0
        self.halted = False

    def get_register(self, reg_name: str) -> int:
        r = reg_name.upper().strip()
        if r == "ACC":
            return self.ACC
        if r == "PC":
            return self.PC
        if r == "SP":
            return self.SP
        if r.startswith("R") and len(r) == 2 and r[1].isdigit():
            idx = int(r[1])
            if 0 <= idx < 8:
                return self.R[idx]
        raise ValueError(f"Unknown register name '{reg_name}'")

    def set_register(self, reg_name: str, value: int) -> None:
        val_32 = value & 0xFFFFFFFF
        r = reg_name.upper().strip()
        if r == "ACC":
            self.ACC = val_32
        elif r == "PC":
            self.PC = val_32
        elif r == "SP":
            self.SP = val_32 & 0xFFFF
        elif r.startswith("R") and len(r) == 2 and r[1].isdigit():
            idx = int(r[1])
            if 0 <= idx < 8:
                self.R[idx] = val_32
        else:
            raise ValueError(f"Unknown register name '{reg_name}'")


class MCUCore:
    """16/32-bit Harvard DSP/RISC Microcontroller Emulator."""

    def __init__(self, sram_size: int = 65536):
        self.regs = RegisterFile()
        self.data_mem = [0] * sram_size
        self.prog_mem: List[Instruction] = []
        self.labels: Dict[str, int] = {}
        self.peripherals = PeripheralBus()
        self.call_stack: List[int] = []
        self.device_spec: Any = {
            "device_name": "GENERIC_RISC_DSP",
            "name": "GENERIC_RISC_DSP",
            "clock_freq_hz": 60e6,
            "supply_voltage": 3.3,
            "flash_bytes": 65536,
            "sram_bytes": sram_size,
            "adc_bits": 12,
            "adc_channels": 16,
            "epwm_channels": 8,
            "gpio_count": 32
        }

    @property
    def clock_freq_mhz(self) -> float:
        if hasattr(self.device_spec, "clock_freq_mhz"):
            return self.device_spec.clock_freq_mhz
        if isinstance(self.device_spec, dict):
            return self.device_spec.get("clock_freq_hz", 60e6) / 1e6
        return 60.0

    @property
    def flash_kb(self) -> int:
        if hasattr(self.device_spec, "flash_kb"):
            return self.device_spec.flash_kb
        if isinstance(self.device_spec, dict):
            return int(self.device_spec.get("flash_bytes", 65536) / 1024)
        return 64

    def apply_specification(self, spec: Any) -> None:
        """Applies hardware characteristics parsed from .lib / .txt definition file."""
        self.device_spec = spec

        # Resize data memory if requested
        new_sram = getattr(spec, "sram_bytes", None) or (spec.get("sram_bytes") if isinstance(spec, dict) else len(self.data_mem))
        if new_sram != len(self.data_mem) and new_sram > 0:
            self.data_mem = [0] * int(new_sram)

    def reset(self) -> None:
        self.regs.reset()
        self.data_mem = [0] * len(self.data_mem)
        self.call_stack.clear()

    def load_program(self, asm_text: str) -> int:
        """Assembles and loads program into execution memory."""
        self.reset()
        self.prog_mem, self.labels = Assembler.assemble(asm_text)
        return len(self.prog_mem)

    def _resolve_operand(self, op_str: str) -> int:
        """Resolves operand string (immediate #10/0x10, label, or register name) to numerical value."""
        s = op_str.strip()
        # Immediate prefix '#' or standard numbers
        if s.startswith("#"):
            s = s[1:]
        
        # Check hex/oct/bin/decimal
        if s.startswith("0x") or s.startswith("0X"):
            return int(s, 16)
        if s.startswith("0b") or s.startswith("0B"):
            return int(s, 2)
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
        
        # Label resolution
        if s.upper() in self.labels:
            return self.labels[s.upper()]

        # Register resolution
        try:
            return self.regs.get_register(s)
        except ValueError:
            pass

        raise ValueError(f"Cannot resolve operand: '{op_str}'")

    def _read_memory(self, addr: int) -> int:
        if self.peripherals.is_peripheral_addr(addr):
            return self.peripherals.read(addr)
        if 0 <= addr < len(self.data_mem):
            return self.data_mem[addr]
        return 0

    def _write_memory(self, addr: int, val: int) -> None:
        if self.peripherals.is_peripheral_addr(addr):
            self.peripherals.write(addr, val)
        elif 0 <= addr < len(self.data_mem):
            self.data_mem[addr] = val & 0xFFFFFFFF

    def _update_flags(self, result: int) -> None:
        r = result & 0xFFFFFFFF
        self.regs.flags.Z = (r == 0)
        self.regs.flags.N = bool(r & 0x80000000)

    def step(self) -> bool:
        """Executes one instruction cycle. Returns False if halted or end of program."""
        if self.regs.halted:
            return False

        if self.regs.PC >= len(self.prog_mem):
            self.regs.halted = True
            return False

        inst = self.prog_mem[self.regs.PC]
        self.regs.PC += 1
        self.regs.cycles += 1
        self.peripherals.timer.tick()

        op = inst.opcode
        op1 = inst.op1
        op2 = inst.op2

        if op == "NOP":
            pass

        elif op == "HALT":
            self.regs.halted = True
            return False

        elif op == "MOV":
            if op1 and op2:
                val = self._resolve_operand(op2)
                self.regs.set_register(op1, val)

        elif op == "LDR":
            # LDR reg, [addr] or LDR reg, addr
            if op1 and op2:
                addr_str = op2.strip("[]")
                addr = self._resolve_operand(addr_str)
                val = self._read_memory(addr)
                self.regs.set_register(op1, val)

        elif op == "STR":
            # STR reg, [addr] or STR reg, addr
            if op1 and op2:
                addr_str = op2.strip("[]")
                addr = self._resolve_operand(addr_str)
                val = self._resolve_operand(op1)
                self._write_memory(addr, val)

        elif op == "ADD":
            if op1 and op2:
                v1 = self.regs.get_register(op1)
                v2 = self._resolve_operand(op2)
                res = v1 + v2
                self._update_flags(res)
                self.regs.set_register(op1, res)

        elif op == "SUB":
            if op1 and op2:
                v1 = self.regs.get_register(op1)
                v2 = self._resolve_operand(op2)
                res = v1 - v2
                self._update_flags(res)
                self.regs.set_register(op1, res)

        elif op == "MUL":
            if op1 and op2:
                v1 = self.regs.get_register(op1)
                v2 = self._resolve_operand(op2)
                res = v1 * v2
                self._update_flags(res)
                self.regs.set_register(op1, res)

        elif op == "CMP":
            if op1 and op2:
                v1 = self._resolve_operand(op1)
                v2 = self._resolve_operand(op2)
                diff = v1 - v2
                self._update_flags(diff)
                self.regs.flags.C = (v1 >= v2)

        elif op == "JMP":
            if op1:
                target = self._resolve_operand(op1)
                self.regs.PC = target

        elif op == "JZ":
            if op1 and self.regs.flags.Z:
                self.regs.PC = self._resolve_operand(op1)

        elif op == "JNZ":
            if op1 and not self.regs.flags.Z:
                self.regs.PC = self._resolve_operand(op1)

        elif op == "JC":
            if op1 and self.regs.flags.C:
                self.regs.PC = self._resolve_operand(op1)

        elif op == "JNC":
            if op1 and not self.regs.flags.C:
                self.regs.PC = self._resolve_operand(op1)

        elif op == "CALL":
            if op1:
                self.call_stack.append(self.regs.PC)
                self.regs.PC = self._resolve_operand(op1)

        elif op == "RET":
            if self.call_stack:
                self.regs.PC = self.call_stack.pop()
            else:
                self.regs.halted = True

        return True

    def run(self, max_cycles: int = 10000) -> int:
        """Runs continuous execution until HALT or max_cycles."""
        count = 0
        while count < max_cycles and self.step():
            count += 1
        return count

    def dump_state(self) -> str:
        """Returns formatted string of CPU state."""
        flags = repr(self.regs.flags)
        lines = [
            f"=== MCU Status (Cycles: {self.regs.cycles}, PC: 0x{self.regs.PC:04X}, Flags: {flags}) ===",
            f"  ACC: 0x{self.regs.ACC:08X} ({self.regs.ACC})",
            f"  R0:  0x{self.regs.R[0]:08X}   R1: 0x{self.regs.R[1]:08X}   R2: 0x{self.regs.R[2]:08X}   R3: 0x{self.regs.R[3]:08X}",
            f"  R4:  0x{self.regs.R[4]:08X}   R5: 0x{self.regs.R[5]:08X}   R6: 0x{self.regs.R[6]:08X}   R7: 0x{self.regs.R[7]:08X}",
            f"  ePWM1A: Period={self.peripherals.epwm.tbprd}, Duty={self.peripherals.epwm.cmpa} ({self.peripherals.epwm.duty_cycle*100:.1f}%)",
            f"  GPIO Port A Data: 0x{self.peripherals.gpio.data:08X}"
        ]
        return "\n".join(lines)
