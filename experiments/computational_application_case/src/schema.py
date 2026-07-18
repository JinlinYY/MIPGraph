"""Single source of truth for the six-property MIPGraph output schema."""

from __future__ import annotations

from types import MappingProxyType


PROPERTY_SCHEMA = (
    ("Density", "kg m^-3"),
    ("ElectricalConductivity", "S m^-1"),
    ("HeatCapacity", "J mol^-1 K^-1"),
    ("SurfaceTension", "N m^-1"),
    ("ThermalConductivity", "W m^-1 K^-1"),
    ("Viscosity", "Pa s"),
)
PROPERTY_NAMES = tuple(name for name, _ in PROPERTY_SCHEMA)
PROPERTY_UNITS = MappingProxyType(dict(PROPERTY_SCHEMA))
