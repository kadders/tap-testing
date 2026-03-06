"""
Tests for tap_testing.feeds_speeds (chip thinning, TEA, MRR, SFM, power/torque).
"""
import math
import pytest

from tap_testing.feeds_speeds import (
    DEFAULT_SPINDLE_POWER_W,
    MIN_CHIPLOAD_MM_PER_TOOTH,
    chipload_adjusted_for_stepover_mm,
    chipload_from_feed_mm_per_tooth,
    cutting_force_n,
    cutting_power_w,
    cutting_torque_nm,
    cutting_zone_status,
    feed_rate_for_chipload_mm_min,
    mrr_mm3_per_min,
    rpm_from_sfm,
    sfm_from_rpm,
    tool_engagement_angle_deg,
    UNIT_POWER_W_MIN_PER_MM3,
)


def test_chipload_adjusted_at_50_percent_stepover_unchanged():
    # At 50% stepover, adjusted chipload should equal nominal (no thinning).
    diam = 10.0
    stepover = 5.0  # 50%
    nominal = 0.05
    adj = chipload_adjusted_for_stepover_mm(diam, stepover, nominal)
    assert adj == nominal


def test_chipload_adjusted_below_50_increases():
    # Below 50% stepover, adjusted chipload > nominal.
    diam = 10.0
    stepover = 2.0  # 20%
    nominal = 0.05
    adj = chipload_adjusted_for_stepover_mm(diam, stepover, nominal)
    assert adj > nominal
    # Doc example: 12.5% stepover, 0.25" tool, 0.002" chipload → ~1.5× → 0.003"
    # 0.25" = 6.35 mm, 12.5% = 0.79375 mm, 0.002" = 0.0508 mm
    adj_ex = chipload_adjusted_for_stepover_mm(6.35, 6.35 * 0.125, 0.0508)
    assert 1.3 * 0.0508 <= adj_ex <= 1.7 * 0.0508


def test_tool_engagement_angle_50_percent_is_90():
    # 50% stepover → TEA = 90°.
    tea = tool_engagement_angle_deg(10.0, 5.0)
    assert abs(tea - 90.0) < 0.1


def test_tool_engagement_angle_slotting_is_180():
    # Full slotting (stepover = diameter) → TEA = 180°.
    tea = tool_engagement_angle_deg(10.0, 10.0)
    assert abs(tea - 180.0) < 0.1


def test_tool_engagement_angle_25_percent():
    # 25% stepover → doc says 60°.
    tea = tool_engagement_angle_deg(10.0, 2.5)
    assert abs(tea - 60.0) < 1.0


def test_mrr_formula():
    # MRR = WOC × DOC × Feed.
    assert mrr_mm3_per_min(10.0, 2.0, 500.0) == 10_000.0


def test_feed_and_chipload_roundtrip():
    fz = 0.05
    n = 3
    rpm = 12_000.0
    feed = feed_rate_for_chipload_mm_min(fz, n, rpm)
    assert feed == fz * n * rpm
    back = chipload_from_feed_mm_per_tooth(feed, n, rpm)
    assert abs(back - fz) < 1e-9


def test_sfm_rpm_roundtrip():
    rpm = 18_000.0
    diam_mm = 6.35  # 1/4"
    sfm = sfm_from_rpm(rpm, diam_mm)
    rpm_back = rpm_from_sfm(sfm, diam_mm)
    assert abs(rpm_back - rpm) < 1.0


def test_cutting_power_torque_force_chain():
    # Power = MRR * unit_power; torque = power / omega; force = torque / radius.
    mrr = 1000.0  # mm³/min
    up = UNIT_POWER_W_MIN_PER_MM3["6061 aluminum"]
    power = cutting_power_w(mrr, up)
    assert power > 0
    torque = cutting_torque_nm(power, 18_000.0)
    assert torque > 0
    force = cutting_force_n(torque, 3.175)  # 6.35 mm dia → 3.175 mm radius
    assert force > 0
    # F = T / r  (r in m): force ≈ torque / 0.003175
    assert abs(force - torque / (3.175 / 1000)) < 0.1


def test_cutting_zone_status_ok():
    z = cutting_zone_status(0.05, 800.0, spindle_power_w=DEFAULT_SPINDLE_POWER_W, min_chipload_mm=MIN_CHIPLOAD_MM_PER_TOOTH)
    assert z["chip_evac_ok"] is True
    assert z["spindle_ok"] is True
    assert z["label"] == "OK"


def test_cutting_zone_status_low_fz():
    z = cutting_zone_status(0.01, 200.0, spindle_power_w=DEFAULT_SPINDLE_POWER_W, min_chipload_mm=MIN_CHIPLOAD_MM_PER_TOOTH)
    assert z["chip_evac_ok"] is False
    assert z["spindle_ok"] is True
    assert z["label"] == "Low fz"


def test_cutting_zone_status_overload():
    z = cutting_zone_status(0.05, 2000.0, spindle_power_w=DEFAULT_SPINDLE_POWER_W, min_chipload_mm=MIN_CHIPLOAD_MM_PER_TOOTH)
    assert z["chip_evac_ok"] is True
    assert z["spindle_ok"] is False
    assert z["label"] == "Overload"


def test_cutting_zone_status_power_unknown():
    z = cutting_zone_status(0.02, None, spindle_power_w=DEFAULT_SPINDLE_POWER_W, min_chipload_mm=MIN_CHIPLOAD_MM_PER_TOOTH)
    assert z["chip_evac_ok"] is False
    assert "power n/a" in z["label"]
