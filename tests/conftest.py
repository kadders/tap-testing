"""
Pytest fixtures for tap-testing: synthetic data so tests run without hardware.
"""
import sys
from pathlib import Path

# Add project root so "tap_testing" is importable when running pytest from tap-testing/
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pytest

# Default tooling for test suite: 6 mm, 3 flute endmill
DEFAULT_FLUTE_COUNT = 3
DEFAULT_TOOL_DIAMETER_MM = 6.0


@pytest.fixture
def default_flute_count():
    return DEFAULT_FLUTE_COUNT


@pytest.fixture
def default_tool_diameter_mm():
    return DEFAULT_TOOL_DIAMETER_MM


@pytest.fixture
def tmp_path_here(tmp_path):
    """Use pytest's tmp_path as the working dir for CSV fixtures."""
    return tmp_path


@pytest.fixture
def sample_rate_hz():
    return 800.0


@pytest.fixture
def synthetic_tap_csv(tmp_path, sample_rate_hz):
    """
    Write a tap-test CSV with a single known frequency (100 Hz) in the signal.
    Returns path to the CSV. Used to verify load_tap_csv and FFT detection.
    """
    duration_s = 0.5
    n = int(sample_rate_hz * duration_s)
    t = np.arange(n) / sample_rate_hz
    freq_hz = 100.0
    # One axis as sine at freq_hz, others small/noise so magnitude peak is at freq_hz
    ax = np.sin(2 * np.pi * freq_hz * t) + 0.1 * np.random.randn(n)
    ay = 0.1 * np.random.randn(n)
    az = 0.1 * np.random.randn(n)
    path = tmp_path / "synthetic_tap.csv"
    with path.open("w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(("t_s", "ax_g", "ay_g", "az_g"))
        w.writerow(("# sample_rate_hz", sample_rate_hz, "", ""))
        for i in range(n):
            w.writerow((t[i], ax[i], ay[i], az[i]))
    return path


@pytest.fixture
def synthetic_tap_csv_no_comment(tmp_path, sample_rate_hz):
    """Tap CSV without the # sample_rate_hz comment row (tests infer from dt)."""
    duration_s = 0.2
    n = int(sample_rate_hz * duration_s)
    t = np.arange(n) / sample_rate_hz
    ax = np.zeros(n)
    ay = np.zeros(n)
    az = np.ones(n) * 0.5
    path = tmp_path / "synthetic_no_comment.csv"
    with path.open("w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(("t_s", "ax_g", "ay_g", "az_g"))
        for i in range(n):
            w.writerow((t[i], ax[i], ay[i], az[i]))
    return path


@pytest.fixture
def two_tap_csvs(tmp_path, sample_rate_hz):
    """Two short tap CSVs for testing _combine_tap_csvs."""
    def write_one(path, t0, n):
        t = t0 + np.arange(n) / sample_rate_hz
        ax = np.sin(2 * np.pi * 50 * t)
        ay = np.zeros(n)
        az = np.zeros(n)
        with path.open("w", newline="") as f:
            import csv
            w = csv.writer(f)
            w.writerow(("t_s", "ax_g", "ay_g", "az_g"))
            w.writerow(("# sample_rate_hz", sample_rate_hz, "", ""))
            for i in range(n):
                w.writerow((t[i], ax[i], ay[i], az[i]))

    n = 100
    p1 = tmp_path / "tap_1.csv"
    p2 = tmp_path / "tap_2.csv"
    write_one(p1, 0.0, n)
    write_one(p2, 0.0, n)  # same t start; combine will offset
    return [p1, p2]
