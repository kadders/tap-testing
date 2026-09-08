"""Tests for motion sample builder and publish filter."""

from tap_testing.motion_sample import MotionPublishFilter, build_motion_sample


def test_build_motion_sample_xyz():
    s = build_motion_sample(
        session_id="s1",
        device_id="pi",
        t_s=1.0,
        axis_positions_mm={"X": 10.0, "Y": -3.0, "Z": 1.5, "A": 45.0},
        feed_mm_min=1200,
        file_position=100,
        rrf_status="processing",
        tool_number=3,
    )
    assert s is not None
    assert s["x"] == 10.0
    assert s["a"] == 45.0
    assert s["source"] == "tap_rrf_poll"


def test_build_motion_sample_missing_xyz():
    assert (
        build_motion_sample(
            session_id="s1",
            device_id="pi",
            t_s=1.0,
            axis_positions_mm={"X": 1.0},
        )
        is None
    )


def test_motion_publish_filter_delta_and_heartbeat():
    filt = MotionPublishFilter(pos_eps_mm=0.05, rot_eps_deg=0.5, heartbeat_s=10.0)
    base = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "file_position": 1,
        "rrf_status": "processing",
    }
    assert filt.should_publish({**base, "t_s": 0.0}, now_mono=0.0)
    assert not filt.should_publish({**base, "t_s": 0.1}, now_mono=0.1)
    assert filt.should_publish({**base, "x": 0.1, "t_s": 0.2}, now_mono=0.2)
    assert filt.should_publish({**base, "x": 0.1, "t_s": 0.2}, now_mono=10.5)
