"""Test suite for Team Room codes, Cross-Mode signal validation, Custom .lib/MCU specifications, and Universal Standard Exporters."""

import unittest
import tempfile
from pathlib import Path

from CORE.ipc_router import (
    IPCRouter, IPCClient, CrossModeSignalValidator, CrossModeValidationResult
)
from CORE.lib_importer import CustomModelImporter, MCUSpecification
from CORE.universal_exporter import UniversalStandardExporter
from Engines.Circuit.catalog import CircuitComponentCatalog
from Engines.Circuit.netlist_parser import Netlist, CircuitParser
from Engines.Dynamic_System.scheduler import SystemDiagram
from Engines.Dynamic_System.blocks import GainBlock, IntegratorBlock, SumBlock, StepSourceBlock
from Engines.Digital_Logic.hdl_parser import LogicCircuit, HDLParser
from Engines.Numerical.parser import NumericalWorkspace, NumericalASTParser
from Engines.Embedded.mcu_core import MCUCore
from UI.app import TerminusEngineBridge


class TestCrossModeSignalValidator(unittest.TestCase):
    """Tests semantic validation rules between like and unlike modes."""

    def test_same_mode_compatible(self):
        v = CrossModeSignalValidator.validate_connection(
            src_mode="CIRCUIT", src_signal="V(out)", src_meta={"voltage_max": 5.0},
            dst_mode="CIRCUIT", dst_signal="Vin", dst_meta={}
        )
        self.assertEqual(v.status, CrossModeValidationResult.COMPATIBLE)
        self.assertIn("Direct Passthrough", v.message)

    def test_circuit_to_mcu_adc_validation(self):
        # Normal ADC range
        v_ok = CrossModeSignalValidator.validate_connection(
            src_mode="CIRCUIT", src_signal="V(node1)", src_meta={"voltage_max": 3.0},
            dst_mode="EMBEDDED", dst_signal="ADC_IN0", dst_meta={"adc_vmax": 3.3, "adc_bits": 12}
        )
        self.assertEqual(v_ok.status, CrossModeValidationResult.VALID_WITH_ADAPTER)

        # Over-voltage range warning
        v_warn = CrossModeSignalValidator.validate_connection(
            src_mode="CIRCUIT", src_signal="V(node1)", src_meta={"voltage_max": 12.0},
            dst_mode="EMBEDDED", dst_signal="ADC_IN0", dst_meta={"adc_vmax": 3.3, "adc_bits": 12}
        )
        self.assertEqual(v_warn.status, CrossModeValidationResult.VALID_WITH_ADAPTER)
        self.assertTrue(any("exceeds MCU ADC max" in w for w in v_warn.warnings))

    def test_mcu_pwm_to_circuit_validation(self):
        v = CrossModeSignalValidator.validate_connection(
            src_mode="EMBEDDED", src_signal="EPWM1A", src_meta={"duty": 0.5},
            dst_mode="CIRCUIT", dst_signal="Gate_Drive", dst_meta={}
        )
        self.assertEqual(v.status, CrossModeValidationResult.VALID_WITH_ADAPTER)
        self.assertIn("PWM Duty Cycle", v.adapter_applied)

    def test_dynamic_to_circuit_validation(self):
        v = CrossModeSignalValidator.validate_connection(
            src_mode="DYNAMIC", src_signal="Controller_Out", src_meta={},
            dst_mode="CIRCUIT", dst_signal="V_ctrl", dst_meta={}
        )
        self.assertEqual(v.status, CrossModeValidationResult.VALID_WITH_ADAPTER)
        self.assertIn("Behavioral Source", v.adapter_applied)


class TestCustomModelImporter(unittest.TestCase):
    """Tests importing custom SPICE .lib and MCU .lib/.txt characteristic files."""

    def setUp(self):
        self.catalog = CircuitComponentCatalog()

    def test_spice_lib_importer(self):
        spice_content = """
* Custom Power MOSFET and Diode Library
.MODEL SCHOTTKY_MBR10100 D(IS=1e-7 RS=0.03 N=1.05 BV=100 IBV=1e-4 CJO=400p)
.MODEL GAN_TRANSISTOR NMOS(LEVEL=1 VTO=1.8 KP=45.0 LAMBDA=0.01 RD=0.02)

.SUBCKT BUCK_STAGE IN OUT GND SW
L1 IN SW 10u
D1 GND SW SCHOTTKY_MBR10100
C1 SW OUT 100u
.ENDS BUCK_STAGE
"""
        models, subckts, _ = CustomModelImporter.import_spice_lib(spice_content, self.catalog)
        self.assertEqual(models, 2)
        self.assertEqual(subckts, 1)

        mbr = self.catalog.get_component("SCHOTTKY_MBR10100")
        self.assertIsNotNone(mbr)
        self.assertIn("Custom", mbr.category)

        gan = self.catalog.get_component("GAN_TRANSISTOR")
        self.assertIsNotNone(gan)

    def test_mcu_spec_importer(self):
        mcu_txt = """
# Hardware Specification for Texas Instruments TMS320F28379D
DEVICE = TMS320F28379D
ARCHITECTURE = C2000_Delfino_DualCore
CLOCK_FREQ_MHZ = 200
FLASH_KB = 1024
SRAM_KB = 204
ADC_RESOLUTION_BITS = 16
ADC_INPUT_RANGE = 0.0, 3.0
ADC_CHANNELS = 24
EPWM_CHANNELS = 24
GPIO_COUNT = 169
"""
        spec = CustomModelImporter.import_mcu_spec(mcu_txt)
        self.assertIsNotNone(spec)
        self.assertEqual(spec.name, "TMS320F28379D")
        self.assertEqual(spec.clock_freq_mhz, 200.0)
        self.assertEqual(spec.flash_kb, 1024)
        self.assertEqual(spec.sram_kb, 204)
        self.assertEqual(spec.adc_resolution_bits, 16)
        self.assertEqual(spec.adc_channels, 24)

        # Apply to MCU core
        mcu = MCUCore()
        mcu.apply_specification(spec)
        self.assertEqual(mcu.device_spec.name, "TMS320F28379D")
        self.assertEqual(mcu.clock_freq_mhz, 200.0)
        self.assertEqual(mcu.flash_kb, 1024)


class TestUniversalStandardExporter(unittest.TestCase):
    """Tests export fidelity across LTspice, Simulink, Verilog, VHDL, C, and HEX."""

    def test_circuit_ltspice_cir_and_asc_export(self):
        netlist = Netlist("RLC_Filter")
        CircuitParser.parse_command(netlist, "add V1 12V ac=1")
        CircuitParser.parse_command(netlist, "add R1 100")
        CircuitParser.parse_command(netlist, "add L1 10m")
        CircuitParser.parse_command(netlist, "add C1 10u")
        CircuitParser.parse_command(netlist, "connect V1.p | R1.a")
        CircuitParser.parse_command(netlist, "connect R1.b | L1.a | C1.a | out")
        CircuitParser.parse_command(netlist, "connect V1.n | L1.b | C1.b | 0")

        # SPICE Netlist (.cir)
        cir_txt = UniversalStandardExporter.export_circuit_ltspice_cir(netlist)
        self.assertIn("Standard Netlist", cir_txt)
        self.assertIn("V1", cir_txt)
        self.assertIn("R1", cir_txt)
        self.assertIn(".end", cir_txt)

        # LTspice Schematic (.asc)
        asc_txt = UniversalStandardExporter.export_circuit_ltspice_asc(netlist)
        self.assertIn("Version 4", asc_txt)
        self.assertIn("SHEET 1 880 680", asc_txt)
        self.assertIn("SYMBOL res", asc_txt)
        self.assertIn("SYMBOL cap", asc_txt)
        self.assertIn("SYMBOL ind", asc_txt)

    def test_dynamic_system_simulink_script_export(self):
        diagram = SystemDiagram("ClosedLoopDC")
        step = StepSourceBlock("Step1", step_time=1.0, initial_value=0.0, final_value=10.0)
        gain = GainBlock("Gain1", k=2.5)
        integ = IntegratorBlock("Integ1", x0=0.0)
        diagram.add_block(step)
        diagram.add_block(gain)
        diagram.add_block(integ)
        diagram.connect("Step1", 0, "Gain1", 0)
        diagram.connect("Gain1", 0, "Integ1", 0)

        m_script = UniversalStandardExporter.export_dynamic_system_simulink_script(diagram)
        self.assertIn("function build_ClosedLoopDC_simulink()", m_script)
        self.assertIn("new_system(model_name);", m_script)
        self.assertIn("add_block('simulink/Sources/Step'", m_script)
        self.assertIn("add_block('simulink/Math Operations/Gain'", m_script)
        self.assertIn("add_block('simulink/Continuous/Integrator'", m_script)
        self.assertIn("add_line(model_name,", m_script)

    def test_digital_logic_verilog_and_vhdl_export(self):
        circuit = LogicCircuit()
        HDLParser.parse_line(circuit, "gate G1 AND a b -> out_and")
        HDLParser.parse_line(circuit, "gate G2 OR out_and c -> final_out")

        # Verilog HDL (.v)
        v_code = UniversalStandardExporter.export_digital_verilog(circuit, "MyLogicModule")
        self.assertIn("module MyLogicModule (", v_code)
        self.assertIn("input wire a,", v_code)
        self.assertIn("output wire final_out", v_code)
        self.assertIn("and G1 (out_and, a, b);", v_code)
        self.assertIn("or G2 (final_out, out_and, c);", v_code)
        self.assertIn("endmodule", v_code)

        # VHDL (.vhd)
        vhd_code = UniversalStandardExporter.export_digital_vhdl(circuit, "MyLogicModule")
        self.assertIn("library IEEE;", vhd_code)
        self.assertIn("entity MyLogicModule is", vhd_code)
        self.assertIn("architecture Structural of MyLogicModule is", vhd_code)

    def test_embedded_c_and_intel_hex_export(self):
        mcu = MCUCore()
        mcu.load_program("MOV R0, #5\nMOV R1, R0\nADD R0, R1\nNOP")

        c_code = UniversalStandardExporter.export_embedded_c_firmware(mcu, "TestFirmware")
        self.assertIn("#include <stdint.h>", c_code)
        self.assertIn("void c2000_main(void)", c_code)
        self.assertIn("MOV", c_code)

        hex_code = UniversalStandardExporter.export_embedded_intel_hex(mcu)
        self.assertTrue(hex_code.startswith(":"))
        self.assertIn(":00000001FF", hex_code) # Standard EOF record


class TestBridgeImportExportAndRooms(unittest.TestCase):
    """Tests CLI commands for importing, exporting, and session rooms in TerminusEngineBridge."""

    def setUp(self):
        self.bridge = TerminusEngineBridge()

    def test_import_spice_and_mcu_via_bridge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create SPICE lib
            spice_file = Path(tmpdir) / "test_diodes.lib"
            spice_file.write_text(".MODEL ULTRA_FAST_MUR460 D(IS=1e-8 RS=0.02 N=1.02 BV=600)")
            out_spice = self.bridge.execute_command(f"import '{spice_file}'")
            self.assertIn("Successfully imported model library", out_spice)
            self.assertIn("SPICE Models Registered", out_spice)

            # Create MCU spec
            mcu_file = Path(tmpdir) / "stm32f4.txt"
            mcu_file.write_text("DEVICE = STM32F407VG\nCLOCK_FREQ_MHZ = 168\nFLASH_KB = 1024\nSRAM_KB = 192\nADC_RESOLUTION_BITS = 12\n")
            out_mcu = self.bridge.execute_command(f"import '{mcu_file}'")
            self.assertIn("STM32F407VG", out_mcu)
            self.assertIn("168.0 MHz", out_mcu)
            self.assertEqual(self.bridge.mcu.clock_freq_mhz, 168.0)

    def test_export_commands_via_bridge(self):
        # 1. Circuit export
        self.bridge.switch_mode("circuit")
        self.bridge.execute_command("add R1 1k")
        self.bridge.execute_command("add C1 1u")
        out_cir = self.bridge.execute_command("export ltspice my_circuit")
        self.assertIn("Exported Industry-Standard SPICE/LTspice Netlist", out_cir)

        out_asc = self.bridge.execute_command("export asc my_schematic")
        self.assertIn("Exported LTspice Schematic File", out_asc)

        # 2. Dynamic system export
        self.bridge.switch_mode("dynamic")
        self.bridge.execute_command("add step S1")
        self.bridge.execute_command("add gain G1 k=5")
        self.bridge.execute_command("connect S1.0 G1.0")
        out_slx = self.bridge.execute_command("export simulink my_sim_model")
        self.assertIn("Exported MATLAB Simulink Model Generator Script", out_slx)

        # 3. Digital logic export
        self.bridge.switch_mode("digital")
        self.bridge.execute_command("gate G1 AND a b -> y")
        out_v = self.bridge.execute_command("export verilog my_alu")
        self.assertIn("Exported IEEE 1364-2005 Verilog HDL Module", out_v)

        # 4. Embedded MCU export
        self.bridge.switch_mode("embedded")
        out_c = self.bridge.execute_command("export c firmware_v1")
        self.assertIn("Exported ANSI C/C++ Embedded Firmware", out_c)


if __name__ == "__main__":
    unittest.main()
