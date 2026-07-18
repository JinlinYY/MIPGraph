from __future__ import annotations

import math

import pandas as pd
import pytest

from experiments.computational_application_case.src.units import (
    UnitConversionError,
    convert_property_frame_to_si,
    molar_heat_capacity_to_mass_specific,
    molar_mass_kg_per_mol,
    millilitres_to_cubic_metres,
    simplified_thermal_diffusion_timescale,
    thermal_diffusivity,
    volumetric_heat_capacity,
)


def test_molar_heat_capacity_is_converted_with_kg_per_mol() -> None:
    assert molar_heat_capacity_to_mass_specific(500.0, 0.250) == pytest.approx(2000.0)


def test_molar_mass_uses_kg_per_mol_for_the_complete_ion_pair() -> None:
    assert molar_mass_kg_per_mol("[Na+].[Cl-]") == pytest.approx(0.05844, rel=2e-3)


def test_one_millilitre_is_one_cubic_centimetre_in_si() -> None:
    assert millilitres_to_cubic_metres(1.0) == pytest.approx(1.0e-6)


def test_thermal_proxy_units_follow_worked_example() -> None:
    c_vol = volumetric_heat_capacity(1000.0, 2000.0)
    alpha = thermal_diffusivity(0.2, c_vol)
    tau = simplified_thermal_diffusion_timescale(1.0e-3, alpha)
    assert c_vol == pytest.approx(2.0e6)
    assert alpha == pytest.approx(1.0e-7)
    assert tau == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (molar_heat_capacity_to_mass_specific, (500.0, 0.0)),
        (volumetric_heat_capacity, (-1.0, 1000.0)),
        (thermal_diffusivity, (0.2, 0.0)),
        (simplified_thermal_diffusion_timescale, (1.0e-3, -1.0e-7)),
    ],
)
def test_nonpositive_physical_inputs_are_rejected(function, args) -> None:
    with pytest.raises(ValueError):
        function(*args)


def test_property_frame_cannot_be_converted_twice() -> None:
    frame = pd.DataFrame(
        {
            "Density": [1.2],
            "ElectricalConductivity": [2.0],
            "HeatCapacity": [500.0],
            "SurfaceTension": [35.0],
            "ThermalConductivity": [0.15],
            "Viscosity": [50.0],
        }
    )
    units = {
        "Density": "g cm^-3",
        "ElectricalConductivity": "S m^-1",
        "HeatCapacity": "J mol^-1 K^-1",
        "SurfaceTension": "mN m^-1",
        "ThermalConductivity": "W m^-1 K^-1",
        "Viscosity": "mPa s",
    }
    converted = convert_property_frame_to_si(frame, units)
    assert converted.loc[0, "Density"] == pytest.approx(1200.0)
    assert converted.loc[0, "SurfaceTension"] == pytest.approx(0.035)
    assert converted.loc[0, "Viscosity"] == pytest.approx(0.05)
    assert all(math.isfinite(float(value)) for value in converted.iloc[0])
    with pytest.raises(UnitConversionError):
        convert_property_frame_to_si(converted, units)

