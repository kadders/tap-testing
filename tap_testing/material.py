"""
Workpiece material and cutting parameters for visuals and guidance.

6061 aluminum is the default (most common use case). Other materials can be
selected via config or --material. All units are metric (mm, mm/min, N/mm², g, Hz, rpm).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# Default material (most common use case)
DEFAULT_MATERIAL_NAME = "6061 aluminum"

# Units string for chart footnotes
METRIC_UNITS_LABEL = "Units: mm, mm/min, Hz, rpm, g (metric)"


@dataclass
class Material:
    """Cutting parameters for a workpiece material (metric)."""

    name: str
    specific_force_ks_n_per_mm2: float  # N/mm²
    force_angle_deg: float
    chip_load_mm_per_tooth: float  # typical mm/tooth for reference
    radial_depth_mm_ref: float = 5.0  # mm, for reference (axial b in force formulas)


# Registry of known materials. Add entries as needed; 6061 is the primary default.
MATERIALS: dict[str, Material] = {
    "6061 aluminum": Material(
        name="6061 aluminum",
        specific_force_ks_n_per_mm2=791.0,  # ≈ 750 kt, 250 kn N/mm²
        force_angle_deg=71.6,
        chip_load_mm_per_tooth=0.1,
        radial_depth_mm_ref=5.0,
    ),
    "7075 aluminum": Material(
        name="7075 aluminum",
        specific_force_ks_n_per_mm2=900.0,  # reference; higher than 6061
        force_angle_deg=70.0,
        chip_load_mm_per_tooth=0.08,
        radial_depth_mm_ref=5.0,
    ),
    "A36 steel": Material(
        name="A36 steel",
        specific_force_ks_n_per_mm2=1500.0,  # reference
        force_angle_deg=75.0,
        chip_load_mm_per_tooth=0.05,
        radial_depth_mm_ref=5.0,
    ),
}


def get_material(name: str) -> Optional[Material]:
    """Return the Material for the given name, or None if unknown."""
    return MATERIALS.get(name.strip() if name else "")


def get_default_material() -> Material:
    """Return the default material (6061 aluminum)."""
    m = MATERIALS.get(DEFAULT_MATERIAL_NAME)
    if m is None:
        raise KeyError(f"Default material '{DEFAULT_MATERIAL_NAME}' not in MATERIALS")
    return m


def get_material_or_default(name: Optional[str]) -> Material:
    """Return the Material for name if known, otherwise the default material."""
    if name:
        m = get_material(name)
        if m is not None:
            return m
    return get_default_material()


def list_material_names() -> list[str]:
    """Return sorted list of registered material names (for CLI/UI)."""
    return sorted(MATERIALS.keys())


def cutting_coefficients_kt_kn_N_per_mm2(material: Material) -> tuple[float, float]:
    """
    Tangential and normal specific forces (N/mm²) for milling force calculations.

    Derived from Ks and force angle β (deg): kt = Ks·sin(β), kn = Ks·cos(β).
    Matches textbook convention (e.g. 6061: kt ≈ 750, kn ≈ 250 N/mm²). Use with
    Eq. 4.8 (Ft = kt·b·h, Fn = kn·b·h) and Eqs. 4.11–4.12 for Fx, Fy.

    Args:
        material: Material with specific_force_ks_n_per_mm2 and force_angle_deg.

    Returns:
        (kt, kn) in N/mm².
    """
    beta_rad = math.radians(material.force_angle_deg)
    Ks = material.specific_force_ks_n_per_mm2
    kt = Ks * math.sin(beta_rad)
    kn = Ks * math.cos(beta_rad)
    return (kt, kn)


def default_material_label(material_name: Optional[str] = None) -> str:
    """Return a short label for the material (e.g. for chart titles)."""
    mat = get_material_or_default(material_name)
    return mat.name


def metric_note(material_name: Optional[str] = None) -> str:
    """Return a short note for charts: material + metric units."""
    mat = get_material_or_default(material_name)
    return f"Material: {mat.name}  ·  {METRIC_UNITS_LABEL}"


# ----- Tool material (cutter material: carbide, HSS, etc.) -----
# Used for chart labels and future speed-limit logic. Workpiece material is separate.

DEFAULT_TOOL_MATERIAL = "carbide"

# Registry of known tool materials (cutter material). Keys lowercase for CLI/env; values are display labels.
TOOL_MATERIALS: dict[str, str] = {
    "carbide": "Carbide",
    "hss": "HSS",
    "high speed steel": "HSS",
    "ceramic": "Ceramic",
    "cermet": "Cermet",
    "cbn": "CBN",
    "pcd": "PCD",
}


def get_tool_material_label(name: Optional[str] = None) -> str:
    """Return display label for tool material (e.g. 'Carbide'). Unknown names returned as-is."""
    if not name:
        return TOOL_MATERIALS.get(DEFAULT_TOOL_MATERIAL, DEFAULT_TOOL_MATERIAL)
    key = name.strip().lower()
    return TOOL_MATERIALS.get(key, name.strip())


def list_tool_material_names() -> list[str]:
    """Return sorted list of registered tool material names (for CLI/UI)."""
    return sorted(set(TOOL_MATERIALS.keys()))


def normalize_tool_material(name: Optional[str] = None) -> str:
    """Return normalized key for tool material (e.g. 'carbide'). Default if name is empty or unknown."""
    if not name or not name.strip():
        return DEFAULT_TOOL_MATERIAL
    key = name.strip().lower()
    if key in TOOL_MATERIALS:
        return key
    if key == "high speed steel":
        return "hss"
    return key
