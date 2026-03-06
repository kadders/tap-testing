"""
Tests for tap_testing.material (default 6061 aluminum, other materials, metric).
"""
import pytest

from tap_testing.material import (
    DEFAULT_MATERIAL_NAME,
    MATERIALS,
    cutting_coefficients_kt_kn_N_per_mm2,
    get_default_material,
    get_material,
    get_material_or_default,
    list_material_names,
    default_material_label,
    metric_note,
)


class TestMaterialRegistry:
    def test_6061_is_default(self):
        assert DEFAULT_MATERIAL_NAME == "6061 aluminum"

    def test_6061_in_registry(self):
        assert "6061 aluminum" in MATERIALS
        m = MATERIALS["6061 aluminum"]
        assert m.name == "6061 aluminum"
        assert m.specific_force_ks_n_per_mm2 > 0
        assert m.chip_load_mm_per_tooth > 0

    def test_list_material_names_includes_6061(self):
        names = list_material_names()
        assert "6061 aluminum" in names
        assert len(names) >= 1

    def test_get_material_known(self):
        m = get_material("6061 aluminum")
        assert m is not None
        assert m.name == "6061 aluminum"

    def test_get_material_unknown_returns_none(self):
        assert get_material("unknown alloy") is None

    def test_get_default_material(self):
        m = get_default_material()
        assert m.name == DEFAULT_MATERIAL_NAME

    def test_get_material_or_default_with_none(self):
        m = get_material_or_default(None)
        assert m.name == DEFAULT_MATERIAL_NAME

    def test_get_material_or_default_with_unknown_falls_back(self):
        m = get_material_or_default("unknown")
        assert m.name == DEFAULT_MATERIAL_NAME

    def test_get_material_or_default_with_other_material(self):
        if "7075 aluminum" in MATERIALS:
            m = get_material_or_default("7075 aluminum")
            assert m.name == "7075 aluminum"


class TestMaterialLabels:
    def test_default_material_label_no_arg(self):
        assert "6061" in default_material_label() or "aluminum" in default_material_label()

    def test_default_material_label_with_name(self):
        if "A36 steel" in MATERIALS:
            assert "A36" in default_material_label("A36 steel") or "steel" in default_material_label("A36 steel")

    def test_metric_note_contains_material_and_metric(self):
        s = metric_note()
        assert "Material:" in s
        assert "metric" in s.lower()


class TestCuttingCoefficients:
    """kt, kn from Ks and force angle (for Eq. 4.8 and Eqs. 4.11–4.12)."""

    def test_6061_kt_kn_approx_750_250(self):
        m = get_default_material()
        kt, kn = cutting_coefficients_kt_kn_N_per_mm2(m)
        assert kt == pytest.approx(750.0, rel=0.02)
        assert kn == pytest.approx(250.0, rel=0.02)

    def test_kt_kn_positive_for_registered_materials(self):
        for name in list_material_names():
            m = get_material(name)
            assert m is not None
            kt, kn = cutting_coefficients_kt_kn_N_per_mm2(m)
            assert kt > 0 and kn > 0
