"""KiCad Schematics & PCB Design Engine for TerminusECE."""

from Engines.KiCad.footprints import FootprintCatalog, FootprintSpec, PadSpec
from Engines.KiCad.schematic import KiCadSchematic, SchematicComponent, SchematicPin
from Engines.KiCad.pcb_engine import KiCadPCB, PlacedComponent, CopperTrack, PCBVia, PCBPadLocation
from Engines.KiCad.gerber_exporter import GerberExporter
from Engines.KiCad.calculator import KiCadCalculator
from Engines.KiCad.project_manager import KiCadProject

__all__ = [
    "FootprintCatalog",
    "FootprintSpec",
    "PadSpec",
    "KiCadSchematic",
    "SchematicComponent",
    "SchematicPin",
    "KiCadPCB",
    "PlacedComponent",
    "CopperTrack",
    "PCBVia",
    "PCBPadLocation",
    "GerberExporter",
    "KiCadCalculator",
    "KiCadProject",
]
