from __future__ import annotations

from pathlib import Path

import pytest

import mblt_tracker.device_tracker_jetson as jetson_module
from mblt_tracker import JetsonDeviceTracker

# Mirrors an AGX Orin devkit: two INA3221 chips, a "sum of shunt voltages"
# channel to skip, an unlabeled channel, and a disabled channel.
_RAILS = {
    "1-0040/hwmon/hwmon1": {
        1: ("VDD_GPU_SOC", 20000, 1000, None),
        2: ("VDD_CPU_CV", 20000, 500, None),
        3: ("VIN_SYS_5V0", 5000, 1000, None),
        7: ("sum of shunt voltages", 840, None, None),
    },
    "1-0041/hwmon/hwmon2": {
        1: (None, 5000, 10, None),
        2: ("VDDQ_VDD2_1V8AO", 5000, 200, None),
        3: ("DISABLED_RAIL", 5000, 100, 0),
    },
}


# Mirrors an Orin NX/Nano: VDD_IN is total module power and already includes
# the other two channels.
_ORIN_NX_RAILS = {
    "1-0040/hwmon/hwmon1": {
        1: ("VDD_IN", 5000, 3000, None),
        2: ("VDD_CPU_GPU_CV", 5000, 1000, None),
        3: ("VDD_SOC", 5000, 800, None),
        7: ("sum of shunt voltages", 840, None, None),
    },
}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_rails(root: Path, rails: dict) -> None:
    for hwmon, channels in rails.items():
        base = root / "sys/bus/i2c/drivers/ina3221" / hwmon
        for channel, (label, voltage_mv, current_ma, enable) in channels.items():
            if label is not None:
                _write(base / f"in{channel}_label", f"{label}\n")
            _write(base / f"in{channel}_input", f"{voltage_mv}\n")
            if current_ma is not None:
                _write(base / f"curr{channel}_input", f"{current_ma}\n")
            if enable is not None:
                _write(base / f"in{channel}_enable", f"{enable}\n")


@pytest.fixture()
def orin_nx_root(tmp_path: Path) -> Path:
    _write_rails(tmp_path, _ORIN_NX_RAILS)
    return tmp_path


@pytest.fixture()
def jetson_root(tmp_path: Path) -> Path:
    _write_rails(tmp_path, _RAILS)
    _write(tmp_path / "sys/devices/platform/bus@0/17000000.gpu/load", "625\n")
    _write(tmp_path / "sys/class/thermal/thermal_zone0/type", "cpu-thermal\n")
    _write(tmp_path / "sys/class/thermal/thermal_zone0/temp", "53000\n")
    _write(tmp_path / "sys/class/thermal/thermal_zone1/type", "gpu-thermal\n")
    _write(tmp_path / "sys/class/thermal/thermal_zone1/temp", "48500\n")
    _write(
        tmp_path / "proc/meminfo",
        "MemTotal:       32768000 kB\nMemFree:  1000 kB\nMemAvailable:   24576000 kB\n",
    )
    _write(
        tmp_path / "proc/device-tree/model",
        "NVIDIA Jetson AGX Orin Developer Kit\x00",
    )
    _write(
        tmp_path / "proc/device-tree/compatible",
        "nvidia,p3737-0000+p3701-0005\x00nvidia,tegra234\x00",
    )
    _write(
        tmp_path / "etc/nv_tegra_release",
        "# R36 (release), REVISION: 5.2, GCID: 46426093, BOARD: generic\n",
    )
    return tmp_path


def test_discovers_labeled_enabled_rails_only(jetson_root: Path) -> None:
    tracker = JetsonDeviceTracker(interval=0.1, sysfs_root=str(jetson_root))

    assert tracker.rails == [
        "VDD_GPU_SOC",
        "VDD_CPU_CV",
        "VIN_SYS_5V0",
        "VDDQ_VDD2_1V8AO",
    ]
    # VDDQ_VDD2_1V8AO is already included in VIN_SYS_5V0.
    assert tracker.total_rails == ["VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"]


def test_sample_sums_rail_power_and_reads_load_memory_temperature(
    jetson_root: Path, monkeypatch
) -> None:
    tracker = JetsonDeviceTracker(interval=0.1, sysfs_root=str(jetson_root))
    monkeypatch.setattr(jetson_module.time, "time", lambda: 10.0)

    tracker._func_for_sched()
    metrics = tracker.get_metric()

    # 20 W + 10 W + 5 W; the 1 W VDDQ sub-rail is inside VIN_SYS_5V0.
    assert tracker.get_trace() == [(10.0, 35.0)]
    assert metrics["avg_power_w"] == pytest.approx(35.0)
    assert metrics["rails"]["VDD_GPU_SOC"]["avg_power_w"] == pytest.approx(20.0)
    assert metrics["rails"]["VDDQ_VDD2_1V8AO"]["avg_power_w"] == pytest.approx(1.0)
    assert metrics["avg_gpu_util_pct"] == pytest.approx(62.5)
    assert metrics["avg_utilization_pct"] == pytest.approx(62.5)
    assert metrics["total_memory_mb"] == pytest.approx(32000.0)
    assert metrics["avg_memory_used_mb"] == pytest.approx(8000.0)
    assert metrics["avg_memory_used_pct"] == pytest.approx(25.0)
    assert metrics["avg_temperature_c"] == pytest.approx(48.5)
    assert metrics["samples"] == 1
    assert metrics["util_samples"] == 1


def test_module_input_rail_is_the_default_total(orin_nx_root: Path) -> None:
    tracker = JetsonDeviceTracker(interval=0.1, sysfs_root=str(orin_nx_root))

    tracker._func_for_sched()
    metrics = tracker.get_metric()

    assert tracker.rails == ["VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"]
    assert tracker.total_rails == ["VDD_IN"]
    # VDD_IN = 15 W; the 5 W and 4 W sub-rails are not added on top.
    assert metrics["avg_power_w"] == pytest.approx(15.0)
    assert metrics["rails"]["VDD_CPU_GPU_CV"]["avg_power_w"] == pytest.approx(5.0)
    assert metrics["rails"]["VDD_SOC"]["avg_power_w"] == pytest.approx(4.0)


def test_selected_rails_limit_total_power(jetson_root: Path) -> None:
    tracker = JetsonDeviceTracker(
        interval=0.1, rails=["VDD_GPU_SOC"], sysfs_root=str(jetson_root)
    )

    tracker._func_for_sched()

    assert tracker.total_rails == ["VDD_GPU_SOC"]
    assert tracker.get_metric()["avg_power_w"] == pytest.approx(20.0)
    # Per-rail statistics still cover every discovered rail.
    assert list(tracker.get_metric()["rails"]) == tracker.rails


def test_explicit_rails_are_summed_as_requested(jetson_root: Path) -> None:
    tracker = JetsonDeviceTracker(
        interval=0.1,
        rails=["VIN_SYS_5V0", "VDDQ_VDD2_1V8AO"],
        sysfs_root=str(jetson_root),
    )

    tracker._func_for_sched()

    # The caller asked for an overlapping pair; honor it: 5 W + 1 W.
    assert tracker.total_rails == ["VIN_SYS_5V0", "VDDQ_VDD2_1V8AO"]
    assert tracker.get_metric()["avg_power_w"] == pytest.approx(6.0)


def test_unknown_rail_is_rejected(jetson_root: Path) -> None:
    with pytest.raises(ValueError, match="Unknown Jetson power rail"):
        JetsonDeviceTracker(rails=["VDD_IN"], sysfs_root=str(jetson_root))


def test_missing_rails_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No Jetson INA3221 power rails found"):
        JetsonDeviceTracker(sysfs_root=str(tmp_path))


def test_unreadable_sensors_keep_other_measurements(
    jetson_root: Path, monkeypatch
) -> None:
    tracker = JetsonDeviceTracker(interval=0.1, sysfs_root=str(jetson_root))
    (jetson_root / "sys/class/thermal/thermal_zone1/temp").unlink()
    (
        jetson_root / "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/curr1_input"
    ).unlink()

    tracker._func_for_sched()
    metrics = tracker.get_metric()

    assert metrics["avg_power_w"] == pytest.approx(15.0)
    assert metrics["rails"]["VDD_GPU_SOC"]["avg_power_w"] is None
    assert metrics["avg_temperature_c"] is None
    assert metrics["avg_gpu_util_pct"] == pytest.approx(62.5)


def test_reset_clears_samples(jetson_root: Path) -> None:
    tracker = JetsonDeviceTracker(interval=0.1, sysfs_root=str(jetson_root))
    tracker._func_for_sched()

    tracker.reset()
    metrics = tracker.get_metric()

    assert metrics["samples"] == 0
    assert metrics["avg_power_w"] is None
    assert tracker.get_trace() == []


def test_is_available(jetson_root: Path, tmp_path_factory) -> None:
    assert JetsonDeviceTracker.is_available(str(jetson_root))
    assert not JetsonDeviceTracker.is_available(str(tmp_path_factory.mktemp("x86")))


def test_static_info(jetson_root: Path) -> None:
    tracker = JetsonDeviceTracker(interval=0.1, sysfs_root=str(jetson_root))

    info = tracker.get_static_info()

    assert info["hardware"]["gpu"]["devices"] == [
        {"device_index": 0, "name": "NVIDIA Jetson AGX Orin Developer Kit"}
    ]
    assert info["hardware"]["gpu"]["power_rails"] == tracker.rails
    assert info["hardware"]["gpu"]["power_total_rails"] == [
        "VDD_GPU_SOC",
        "VDD_CPU_CV",
        "VIN_SYS_5V0",
    ]
    assert info["inference"]["gpu"]["l4t"] == {"version": "36.5.2"}


def test_repeated_stop_does_not_extend_the_window(
    jetson_root: Path, monkeypatch
) -> None:
    tracker = JetsonDeviceTracker(interval=60.0, sysfs_root=str(jetson_root))
    now = [10.0]
    monkeypatch.setattr(jetson_module.time, "time", lambda: now[0])

    tracker.start()
    now[0] = 10.2
    tracker.stop()
    now[0] = 18.0
    tracker.stop()

    assert [ts for ts, _ in tracker.get_trace()] == [10.0, 10.2]
