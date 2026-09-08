"""Assembler, Disassembler, and Instruction parser for the MCU emulator."""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple, Union


class Instruction:
    """Represents a single parsed assembly instruction."""

    def __init__(
        self,
        opcode: str,
        op1: Optional[str] = None,
        op2: Optional[str] = None,
        label: Optional[str] = None,
        line_num: int = 0
    ):
        self.opcode = opcode.upper().strip()
        self.op1 = op1.strip() if op1 else None
        self.op2 = op2.strip() if op2 else None
        self.label = label
        self.line_num = line_num

    def __repr__(self) -> str:
        parts = [self.opcode]
        if self.op1:
            parts.append(self.op1)
        if self.op2:
            parts.append(self.op2)
        return " ".join(parts)


class Assembler:
    """Translates assembly text into an instruction memory stream with label resolutions."""

    VALID_OPCODES = {
        "MOV", "LDR", "STR", "PUSH", "POP",
        "ADD", "SUB", "MUL", "DIV", "AND", "OR", "XOR", "NOT", "SHL", "SHR",
        "CMP", "JMP", "JZ", "JNZ", "JC", "JNC", "JG", "JL",
        "CALL", "RET", "OUT", "IN", "NOP", "HALT"
    }

    @classmethod
    def assemble(cls, asm_text: str) -> Tuple[List[Instruction], Dict[str, int]]:
        """Two-pass assembler: pass 1 records labels, pass 2 builds instructions."""
        lines = asm_text.strip().splitlines()
        instructions: List[Instruction] = []
        labels: Dict[str, int] = {}

        # Pass 1: Parse labels and raw instruction lines
        raw_entries: List[Tuple[Optional[str], str, int]] = []
        pc_counter = 0

        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            # Strip comments (; or //)
            if ";" in line:
                line = line.split(";", 1)[0].strip()
            if "//" in line:
                line = line.split("//", 1)[0].strip()
            if not line:
                continue

            current_label = None
            if ":" in line:
                label_part, rest = line.split(":", 1)
                current_label = label_part.strip().upper()
                labels[current_label] = pc_counter
                line = rest.strip()

            if not line:
                continue

            raw_entries.append((current_label, line, line_num))
            pc_counter += 1

        # Pass 2: Parse opcodes and operands
        for label, line, line_num in raw_entries:
            tokens = re.split(r'[\s,]+', line)
            tokens = [t for t in tokens if t]
            if not tokens:
                continue

            opcode = tokens[0].upper()
            if opcode not in cls.VALID_OPCODES:
                raise ValueError(f"Line {line_num}: Unknown opcode '{opcode}'")

            op1 = tokens[1] if len(tokens) > 1 else None
            op2 = tokens[2] if len(tokens) > 2 else None

            inst = Instruction(opcode, op1, op2, label=label, line_num=line_num)
            instructions.append(inst)

        return instructions, labels


class Disassembler:
    """Formats instruction objects back into formatted assembly strings."""

    @classmethod
    def disassemble(cls, instructions: List[Any], pc_highlight: int = -1) -> str:
        lines: List[str] = []
        for i, inst in enumerate(instructions):
            prefix = "► " if i == pc_highlight else "  "
            if hasattr(inst, "opcode"):
                opcode = inst.opcode
                label_str = f"{inst.label + ':':<10}" if getattr(inst, "label", None) else " " * 10
                operands = []
                if getattr(inst, "op1", None):
                    operands.append(str(inst.op1))
                if getattr(inst, "op2", None):
                    operands.append(str(inst.op2))
                op_str = ", ".join(operands)
                lines.append(f"{prefix}0x{i:04X}: {label_str} {opcode:<6} {op_str}")
            elif isinstance(inst, (tuple, list)):
                op_code = f"OP_{inst[0]}" if isinstance(inst[0], int) else str(inst[0])
                ops = [str(x) for x in inst[1:] if x is not None]
                lines.append(f"{prefix}0x{i:04X}: {' ' * 10} {op_code:<6} {', '.join(ops)}")
            else:
                lines.append(f"{prefix}0x{i:04X}: {' ' * 10} {str(inst)}")
        return "\n".join(lines)
