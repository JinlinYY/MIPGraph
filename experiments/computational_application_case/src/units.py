"""Audited unit conversion and thermophysical proxy primitives.

All public calculations in this module consume and return SI units.  The
functions reject non-finite or non-positive values where the underlying
physical definition requires positivity; callers can then record the failed
candidate instead of silently clipping it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from .schema import PROPERTY_UNITS


Numeric = Union[float, np.ndarray, pd.Series]
SI_UNITS = dict(PROPERTY_UNITS)
_UNIT_MARKER = "mipgraph_property_units"


class UnitConversionError(ValueError):
    """Raised when a conversion is ambiguous, unsupported, or repeated."""


def _positive_finite(value: Numeric, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only finite positive values")
    return array


def _return_like(original: Numeric, value: np.ndarray) -> Numeric:
    if np.ndim(original) == 0:
        return float(np.asarray(value).reshape(-1)[0])
    if isinstance(original, pd.Series):
        return pd.Series(value, index=original.index, name=original.name)
    return value


def molar_mass_kg_per_mol(ion_pair_smiles: str) -> float:
    """Return complete ion-pair molecular mass in kg mol^-1 using RDKit."""

    molecule = Chem.MolFromSmiles(str(ion_pair_smiles))
    if molecule is None:
        raise ValueError(f"RDKit could not parse complete ion pair: {ion_pair_smiles}")
    mass = float(Descriptors.MolWt(molecule)) / 1000.0
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"Invalid ion-pair molar mass for {ion_pair_smiles}: {mass}")
    return mass


def molar_heat_capacity_to_mass_specific(
    heat_capacity_j_per_mol_k: Numeric,
    molar_mass_kg_per_mol_value: Numeric,
) -> Numeric:
    """Convert J mol^-1 K^-1 to J kg^-1 K^-1 exactly once."""

    heat_capacity = _positive_finite(heat_capacity_j_per_mol_k, "molar heat capacity")
    molar_mass = _positive_finite(molar_mass_kg_per_mol_value, "molar mass")
    return _return_like(heat_capacity_j_per_mol_k, heat_capacity / molar_mass)


def millilitres_to_cubic_metres(volume_ml: Numeric) -> Numeric:
    """Convert mL to m^3 using 1 mL = 1e-6 m^3."""

    volume = _positive_finite(volume_ml, "volume")
    return _return_like(volume_ml, volume * 1.0e-6)


def volumetric_heat_capacity(
    density_kg_per_m3: Numeric,
    mass_heat_capacity_j_per_kg_k: Numeric,
) -> Numeric:
    """Compute C_vol = rho * cp in J m^-3 K^-1."""

    density = _positive_finite(density_kg_per_m3, "density")
    heat_capacity = _positive_finite(
        mass_heat_capacity_j_per_kg_k, "mass-specific heat capacity"
    )
    return _return_like(density_kg_per_m3, density * heat_capacity)


def thermal_diffusivity(
    thermal_conductivity_w_per_m_k: Numeric,
    volumetric_heat_capacity_j_per_m3_k: Numeric,
) -> Numeric:
    """Compute alpha = lambda / C_vol in m^2 s^-1."""

    conductivity = _positive_finite(
        thermal_conductivity_w_per_m_k, "thermal conductivity"
    )
    heat_capacity = _positive_finite(
        volumetric_heat_capacity_j_per_m3_k, "volumetric heat capacity"
    )
    return _return_like(
        thermal_conductivity_w_per_m_k, conductivity / heat_capacity
    )


def simplified_thermal_diffusion_timescale(
    thermal_length_m: Numeric,
    thermal_diffusivity_m2_per_s: Numeric,
) -> Numeric:
    """Compute the simplified thermal-diffusion timescale L^2 / alpha in s."""

    length = _positive_finite(thermal_length_m, "thermal length")
    diffusivity = _positive_finite(
        thermal_diffusivity_m2_per_s, "thermal diffusivity"
    )
    return _return_like(thermal_diffusivity_m2_per_s, length**2 / diffusivity)


def convert_property_frame_to_si(
    frame: pd.DataFrame,
    source_units: Mapping[str, str],
) -> pd.DataFrame:
    """Convert a six-property frame to audited SI units with repeat protection."""

    if frame.attrs.get(_UNIT_MARKER):
        raise UnitConversionError("Property frame has already been converted to SI")
    missing = [name for name in SI_UNITS if name not in frame.columns]
    if missing:
        raise UnitConversionError(f"Missing property columns: {missing}")
    factors: dict[str, dict[str, float]] = {
        "Density": {"kg m^-3": 1.0, "g cm^-3": 1000.0},
        "ElectricalConductivity": {
            "S m^-1": 1.0,
            "mS cm^-1": 0.1,
        },
        "HeatCapacity": {
            "J mol^-1 K^-1": 1.0,
            "J kg^-1 K^-1": 1.0,
        },
        "SurfaceTension": {"N m^-1": 1.0, "mN m^-1": 1.0e-3},
        "ThermalConductivity": {"W m^-1 K^-1": 1.0},
        "Viscosity": {"Pa s": 1.0, "mPa s": 1.0e-3, "cP": 1.0e-3},
    }
    output = frame.copy(deep=True)
    for name, target_unit in SI_UNITS.items():
        source = str(source_units.get(name, ""))
        if source not in factors[name]:
            raise UnitConversionError(
                f"Unsupported or missing unit for {name}: {source!r}"
            )
        output[name] = pd.to_numeric(output[name], errors="coerce") * factors[name][source]
        if name == "HeatCapacity" and source == "J kg^-1 K^-1":
            target_unit = source
        output.attrs.setdefault("source_units", {})[name] = source
        output.attrs.setdefault("output_units", {})[name] = target_unit
    output.attrs[_UNIT_MARKER] = True
    return output
