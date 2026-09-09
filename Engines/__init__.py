"""Computation engines for TerminusECE.

Industry-Standard Engine Architecture:
- AnalogDevices_LTspice (Circuit Simulation Engine)
- MathWorks_Simulink   (Dynamic Systems & Block Diagram Engine)
- MathWorks_MATLAB     (Numerical AST, DSP & Matrix Engine)
- AMD_Xilinx           (Digital Logic & HDL Discrete Event Engine)
- KiCad                (Schematic, Footprints, PCB Layout & Gerber Engine)
- Embedded_IDE         (Microcontroller Firmware, Telemetry & Real-Hardware IDE)
"""

from . import Circuit as AnalogDevices_LTspice
from . import Circuit
from . import Dynamic_System as MathWorks_Simulink
from . import Dynamic_System
from . import Numerical as MathWorks_MATLAB
from . import Numerical
from . import Digital_Logic as AMD_Xilinx
from . import Digital_Logic
from . import Embedded as Embedded_IDE
from . import Embedded
from . import KiCad

__all__ = [
    "AnalogDevices_LTspice",
    "Circuit",
    "MathWorks_Simulink",
    "Dynamic_System",
    "MathWorks_MATLAB",
    "Numerical",
    "AMD_Xilinx",
    "Digital_Logic",
    "Embedded_IDE",
    "Embedded",
    "KiCad",
]

