from __future__ import annotations

import json
import os
import subprocess

import pytest

from mblt_tracker.device_tracker_npu import (
    NPUDeviceTracker,
    _parse_mobilint_status_query_metric_samples,
    _parse_mobilint_status_query_metrics,
    _parse_mobilint_status_static_info,
)
from mblt_tracker.static_info import parse_mobilint_status_query_output

STATUS_OUTPUT = """2026-04-15 16:06:59
+------------------------------------------------------------------------------------------+
| Mobilint-NPU-Monitor                           Drivers - Aries: 1.12.0  Regulus: N/A     |
+------------------------------------------------------------------------------------------+
| NPU  Name                     |   Pwr:NPU/Total |     Clock:NPU/Bus |       Memory-Usage |
| Sig  Temp    Firmware Version |   Cur:NPU/Total |                   |           NPU-Util |
|===============================+=================+===================+====================|
|   0  Aries(aries0)            |   2.11W   7.87W |   50MHz /  150MHz |      0MB / 16384MB |
|   0  49 C               1.2.4 |   0.17A   0.65A |                   |              0.00% |
+-------------------------------+-----------------+-------------------+--------------------+
"""

STATUS_QUERY_OUTPUT = """Timestamp                     : 2026-05-07 04:10:46
Driver Version (Aries)        : 1.12.0 (Rev: 1)
Driver Version (Regulus)      : N/A
Connected NPUs                : 1
/dev/aries0
    Product                   : Aries
    Firmware
        Version               : 1.1 (Rev: 0)
        CRC                   : 0xFB9A5980
    Temperature               : 39 C
    Signal Type               : Interrupt
    Clock
        NPU                   : 1250 MHz
        Bus                   : 1000 MHz
    Power
        Total                 : 12.85 W
        NPU                   : 3.90 W
        DDR                   : 3.92 W
        PMIC                  : 3.92 W
    Current
        Total                 : 1.05 A
        NPU                   : 0.32 A
        DDR                   : 0.32 A
        PMIC                  : 0.32 A
    Voltage
        Total                 : 12.20 V
        NPU                   : 12.19 V
        DDR                   : 12.19 V
        PMIC                  : 12.19 V
    Memory
        Usage                 : 0 MB
        Total                 : 16384 MB
    Utilization
        Total                 : 0.00 %
        Cluster0
            GlobalCore        : 0.00 %
            Core0             : 0.00 %
            Core1             : 0.00 %
            Core2             : 0.00 %
            Core3             : 0.00 %
        Cluster1
            GlobalCore        : 0.00 %
            Core0             : 0.00 %
            Core1             : 0.00 %
            Core2             : 0.00 %
            Core3             : 0.00 %
    Fan Duty                  : 34 %
    PCI Express
        Vendor ID             : 0x209F
        Device ID             : 0x0
        Sub Vendor ID         : 0x401
        Sub Device ID         : 0x1093
        PCIe Generation       : 4
        PCIe Lanes            : 8
        PCIe Revision         : 0x2
        PCIe Class Code       : 0x7800002
    Processes
"""

MLA400_STATUS_QUERY_OUTPUT = """Timestamp                     : 2026-05-07 15:29:48
Driver Version (Aries)        : 1.12.0 (Rev: 1)
Driver Version (Regulus)      : N/A
Connected NPUs                : 4
/dev/aries0
    Product                   : Aries
    Firmware
        Version               : 1.2.5 (Rev: 0)
        CRC                   : 0xCCCC0005
    Temperature               : 45 C
    Power
        Total                 : 51.55 W
        NPU                   : 4.32 W
        DDR                   : 0.00 W
        PMIC                  : 0.00 W
        GOLDFINGER            : 27.79 W
    Memory
        Usage                 : 14363 MB
        Total                 : 16384 MB
    Utilization
        Total                 : 0.00 %
    PCI Express
        Vendor ID             : 0x209F
        Device ID             : 0x0
        Sub Vendor ID         : 0x402
        Sub Device ID         : 0x108B
/dev/aries1
    Product                   : Aries
    Firmware
        Version               : 1.2.5 (Rev: 0)
    Temperature               : 46 C
    Power
        Total                 : 0.00 W
        NPU                   : 4.58 W
        DDR                   : 0.00 W
        PMIC                  : 0.00 W
        GOLDFINGER            : 0.00 W
    Memory
        Usage                 : 14299 MB
        Total                 : 16384 MB
    Utilization
        Total                 : 0.00 %
    PCI Express
        Vendor ID             : 0x209F
        Device ID             : 0x0
        Sub Vendor ID         : 0x402
        Sub Device ID         : 0x108B
/dev/aries2
    Product                   : Aries
    Firmware
        Version               : 1.2.5 (Rev: 0)
    Temperature               : 48 C
    Power
        Total                 : 0.00 W
        NPU                   : 4.41 W
        DDR                   : 0.00 W
        PMIC                  : 0.00 W
        GOLDFINGER            : 0.00 W
    Memory
        Usage                 : 12375 MB
        Total                 : 16384 MB
    Utilization
        Total                 : 0.00 %
    PCI Express
        Vendor ID             : 0x209F
        Device ID             : 0x0
        Sub Vendor ID         : 0x402
        Sub Device ID         : 0x108B
/dev/aries3
    Product                   : Aries
    Firmware
        Version               : 1.2.5 (Rev: 0)
    Temperature               : 46 C
    Power
        Total                 : 0.00 W
        NPU                   : 4.50 W
        DDR                   : 0.00 W
        PMIC                  : 0.00 W
        GOLDFINGER            : 0.00 W
    Memory
        Usage                 : 10756 MB
        Total                 : 16384 MB
    Utilization
        Total                 : 0.00 %
    PCI Express
        Vendor ID             : 0x209F
        Device ID             : 0x0
        Sub Vendor ID         : 0x402
        Sub Device ID         : 0x108B
"""

MLA100_STATUS_QUERY_DEVICE_4 = """/dev/aries4
    Product                   : Aries
    Firmware
        Version               : 1.1 (Rev: 0)
    Temperature               : 39 C
    Power
        Total                 : 12.85 W
        NPU                   : 3.90 W
        DDR                   : 3.92 W
        PMIC                  : 3.92 W
    Memory
        Usage                 : 0 MB
        Total                 : 16384 MB
    Utilization
        Total                 : 0.00 %
    PCI Express
        Vendor ID             : 0x209F
        Device ID             : 0x0
        Sub Vendor ID         : 0x401
        Sub Device ID         : 0x1093
"""


def _renumber_mla400_status_output(device_offset: int, power_offset: float) -> str:
    output = MLA400_STATUS_QUERY_OUTPUT
    for dev_no in range(3, -1, -1):
        output = output.replace(f"/dev/aries{dev_no}", f"/dev/aries{dev_no + device_offset}")
    return output.replace(
        "Total                 : 51.55 W",
        f"Total                 : {51.55 + power_offset:.2f} W",
        1,
    ).replace(
        "GOLDFINGER            : 27.79 W",
        f"GOLDFINGER            : {27.79 + power_offset:.2f} W",
        1,
    )


def _make_tracker() -> NPUDeviceTracker:
    tracker = object.__new__(NPUDeviceTracker)
    tracker._npu_power_glance = []
    tracker._ddr_power_glance = []
    tracker._pmic_power_glance = []
    tracker._goldfinger_power_glance = []
    tracker._total_power_glance = []
    tracker._npu_util_glance = []
    tracker._npu_mem_used_mb_glance = []
    tracker._npu_mem_used_pct_glance = []
    tracker._npu_temp_glance = []
    tracker._npu_mem_total_mb = None
    tracker._power_trace = []
    tracker._npu_power_trace = []
    tracker._ddr_power_trace = []
    tracker._pmic_power_trace = []
    tracker._goldfinger_power_trace = []
    tracker._util_trace = []
    tracker._mem_used_trace = []
    tracker._mem_used_pct_trace = []
    tracker._temp_trace = []
    tracker._npu_metric_glance = {}
    tracker._npu_memory_total_mb = {}
    tracker._npu_card_model = {}
    return tracker


def test_npu_shell_parser_reads_temperature_from_status_output(
    monkeypatch, tmp_path
) -> None:
    if os.name == "nt":
        pytest.skip("bash PATH injection for the shell parser test is Linux-only")

    cli_path = tmp_path / "mobilint-cli"
    cli_path.write_text(
        """#!/usr/bin/env bash
cat <<'EOF'
2026-04-15 16:06:59
+------------------------------------------------------------------------------------------+
| Mobilint-NPU-Monitor                           Drivers - Aries: 1.12.0  Regulus: N/A     |
+------------------------------------------------------------------------------------------+
| NPU  Name                     |   Pwr:NPU/Total |     Clock:NPU/Bus |       Memory-Usage |
| Sig  Temp    Firmware Version |   Cur:NPU/Total |                   |           NPU-Util |
|===============================+=================+===================+====================|
|   0  Aries(aries0)            |   2.11W   7.87W |   50MHz /  150MHz |      0MB / 16384MB |
|   0  49 C               1.2.4 |   0.17A   0.65A |                   |              0.00% |
+-------------------------------+-----------------+-------------------+--------------------+
EOF
""",
        encoding="utf-8",
    )
    cli_path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")

    result = subprocess.run(
        ["bash", "mblt_tracker/device_tracker_npu.sh", "--sample-once", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["npu_power_w"] == 2.11
    assert payload["total_power_w"] == 7.87
    assert payload["npu_util_pct"] == 0.0
    assert payload["npu_mem_used_mb"] == 0
    assert payload["npu_mem_total_mb"] == 16384
    assert payload["npu_temp_c"] == 49


def test_parse_mobilint_status_static_info() -> None:
    info = _parse_mobilint_status_static_info(STATUS_OUTPUT)

    assert info["inference"]["npu_driver_version"] == "1.12.0"
    assert info["hardware"]["npus"] == [
        {"dev_no": 0, "board_name": "aries0", "firmware": {"version": "1.2.4"}}
    ]
    assert info["inference"]["driver"] == {
        "aries_version": "1.12.0",
        "regulus_version": "N/A",
    }


def test_parse_mobilint_status_query_output_to_nested_dict() -> None:
    parsed = parse_mobilint_status_query_output(STATUS_QUERY_OUTPUT)

    assert parsed["Driver Version (Aries)"] == "1.12.0 (Rev: 1)"
    assert parsed["Connected NPUs"] == "1"
    devices = parsed["devices"]
    assert isinstance(devices, list)
    device = devices[0]
    assert device["path"] == "/dev/aries0"
    assert device["Product"] == "Aries"
    assert device["Firmware"]["Version"] == "1.1 (Rev: 0)"
    assert device["Power"]["Total"] == "12.85 W"
    assert device["Utilization"]["Cluster0"]["Core3"] == "0.00 %"


def test_parse_mobilint_status_query_metrics() -> None:
    metrics = _parse_mobilint_status_query_metrics(STATUS_QUERY_OUTPUT)

    assert metrics == (3.90, 12.85, 3.92, 3.92, 0.0, 0.0, 16384.0, 0.0, 39.0)


def test_parse_mobilint_status_query_metric_samples_classifies_mla100() -> None:
    samples = _parse_mobilint_status_query_metric_samples(STATUS_QUERY_OUTPUT)

    assert samples is not None
    assert len(samples) == 1
    assert samples[0]["card_model"] == "MLA100"
    assert samples[0]["card_id"] == 0
    assert samples[0]["goldfinger_power_w"] is None


def test_parse_mobilint_status_query_metric_samples_groups_mla400() -> None:
    samples = _parse_mobilint_status_query_metric_samples(MLA400_STATUS_QUERY_OUTPUT)

    assert samples is not None
    assert len(samples) == 1
    sample = samples[0]
    assert sample["card_model"] == "MLA400"
    assert sample["chip_count"] == 4
    assert sample["total_power_w"] == pytest.approx(51.55)
    assert sample["npu_power_w"] == pytest.approx(17.81)
    assert sample["goldfinger_power_w"] == pytest.approx(27.79)
    assert sample["npu_mem_used_mb"] == pytest.approx(51793.0)
    assert sample["npu_mem_total_mb"] == pytest.approx(65536.0)


def test_parse_mobilint_status_query_metric_samples_keeps_mixed_mla100_separate() -> None:
    status_output = MLA400_STATUS_QUERY_OUTPUT + MLA100_STATUS_QUERY_DEVICE_4
    samples = _parse_mobilint_status_query_metric_samples(status_output)

    assert samples is not None
    assert len(samples) == 2
    mla400_sample, mla100_sample = samples
    assert mla400_sample["card_id"] == 0
    assert mla400_sample["card_model"] == "MLA400"
    assert mla400_sample["chip_count"] == 4
    assert mla400_sample["total_power_w"] == pytest.approx(51.55)
    assert mla400_sample["npu_power_w"] == pytest.approx(17.81)
    assert mla100_sample["card_id"] == 1
    assert mla100_sample["card_model"] == "MLA100"
    assert mla100_sample["dev_no"] == 4
    assert mla100_sample["total_power_w"] == pytest.approx(12.85)
    assert mla100_sample["npu_power_w"] == pytest.approx(3.90)


def test_parse_mobilint_status_query_metric_samples_groups_two_mla400_cards() -> None:
    status_output = MLA400_STATUS_QUERY_OUTPUT + _renumber_mla400_status_output(
        device_offset=4,
        power_offset=10.0,
    )
    samples = _parse_mobilint_status_query_metric_samples(status_output)

    assert samples is not None
    assert len(samples) == 2
    first_card, second_card = samples
    assert first_card["card_id"] == 0
    assert first_card["card_model"] == "MLA400"
    assert first_card["chip_count"] == 4
    assert first_card["total_power_w"] == pytest.approx(51.55)
    assert first_card["goldfinger_power_w"] == pytest.approx(27.79)
    assert second_card["card_id"] == 1
    assert second_card["card_model"] == "MLA400"
    assert second_card["chip_count"] == 4
    assert second_card["total_power_w"] == pytest.approx(61.55)
    assert second_card["goldfinger_power_w"] == pytest.approx(37.79)


def test_npu_sampling_records_mla400_goldfinger_and_per_card_stats(monkeypatch) -> None:
    tracker = _make_tracker()
    tracker._npu_id = None
    monkeypatch.setattr(
        tracker,
        "_fetch_metric_samples",
        lambda: _parse_mobilint_status_query_metric_samples(MLA400_STATUS_QUERY_OUTPUT),
    )
    monkeypatch.setattr("mblt_tracker.device_tracker_npu.time.time", lambda: 789.0)

    tracker._func_for_sched()

    assert tracker.get_trace() == [(789.0, 51.55)]
    assert tracker.get_npu_power_trace() == [(789.0, pytest.approx(17.81))]
    assert tracker.get_goldfinger_power_trace() == [(789.0, 27.79)]
    metrics = tracker.get_metric()
    assert metrics["avg_goldfinger_power_w"] == 27.79
    assert metrics["total_memory_mb"] == 65536.0
    assert metrics["npu"][0]["card_model"] == "MLA400"
    assert metrics["npu"][0]["avg_power_w"] == 51.55


def test_npu_sampling_records_ddr_and_pmic_power_traces(monkeypatch) -> None:
    tracker = _make_tracker()
    monkeypatch.setattr(
        tracker,
        "_fetch_metrics",
        lambda: (3.90, 12.85, 3.92, 3.93, 5.0, 128.0, 16384.0, None, 39.0),
    )
    monkeypatch.setattr("mblt_tracker.device_tracker_npu.time.time", lambda: 123.0)

    tracker._func_for_sched()

    assert tracker.get_trace() == [(123.0, 12.85)]
    assert tracker.get_npu_power_trace() == [(123.0, 3.90)]
    assert tracker.get_ddr_power_trace() == [(123.0, 3.92)]
    assert tracker.get_pmic_power_trace() == [(123.0, 3.93)]
    metrics = tracker.get_metric()
    assert metrics["avg_npu_power_w"] == 3.90
    assert metrics["avg_ddr_power_w"] == 3.92
    assert metrics["avg_pmic_power_w"] == 3.93
    assert metrics["ddr_power_samples"] == 1
    assert metrics["pmic_power_samples"] == 1


def test_npu_sampling_keeps_existing_metrics_when_ddr_and_pmic_missing(
    monkeypatch,
) -> None:
    tracker = _make_tracker()
    monkeypatch.setattr(
        tracker,
        "_fetch_metrics",
        lambda: (3.90, 12.85, None, None, 5.0, 128.0, 16384.0, 0.78125, 39.0),
    )
    monkeypatch.setattr("mblt_tracker.device_tracker_npu.time.time", lambda: 456.0)

    tracker._func_for_sched()

    assert tracker.get_trace() == [(456.0, 12.85)]
    assert tracker.get_npu_power_trace() == [(456.0, 3.90)]
    assert tracker.get_ddr_power_trace() == []
    assert tracker.get_pmic_power_trace() == []
    metrics = tracker.get_metric()
    assert metrics["avg_ddr_power_w"] is None
    assert metrics["avg_pmic_power_w"] is None
    assert metrics["ddr_power_samples"] == 0
    assert metrics["pmic_power_samples"] == 0


def test_npu_fetch_metrics_reads_legacy_json_ddr_and_pmic_power(monkeypatch) -> None:
    tracker = _make_tracker()
    tracker._status_cmd = "fake-status"

    class Result:
        returncode = 0
        stdout = json.dumps(
            {
                "ok": True,
                "npu_power_w": 3.9,
                "total_power_w": 12.85,
                "ddr_power_w": 3.92,
                "pmic_power_w": 3.93,
            }
        )

    monkeypatch.setattr(
        "mblt_tracker.device_tracker_npu.subprocess.run",
        lambda *args, **kwargs: Result(),
    )

    assert tracker._fetch_metrics() == (
        3.9,
        12.85,
        3.92,
        3.93,
        None,
        None,
        None,
        None,
        None,
    )


def test_npu_fetch_metrics_falls_back_when_status_query_fails(monkeypatch) -> None:
    tracker = _make_tracker()
    tracker._status_cmd = "mobilint-cli status -q"
    calls = []

    class Result:
        def __init__(self, returncode: int, stdout: str):
            self.returncode = returncode
            self.stdout = stdout

    def fake_run(command, *args, **kwargs):
        calls.append(command)
        if command == ["mobilint-cli", "status", "-q"]:
            return Result(returncode=2, stdout="")
        assert command[-2:] == ["--sample-once", "--json"]
        return Result(
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "npu_power_w": 2.11,
                    "total_power_w": 7.87,
                    "npu_util_pct": 0.0,
                    "npu_mem_used_mb": 0,
                    "npu_mem_total_mb": 16384,
                    "npu_temp_c": 49,
                }
            ),
        )

    monkeypatch.setattr("mblt_tracker.device_tracker_npu.subprocess.run", fake_run)

    assert tracker._fetch_metrics() == (
        2.11,
        7.87,
        None,
        None,
        0.0,
        0.0,
        16384.0,
        0.0,
        49.0,
    )
    assert calls[0] == ["mobilint-cli", "status", "-q"]
    assert calls[1][-2:] == ["--sample-once", "--json"]


def test_npu_json_fallback_honors_selected_npu_id(monkeypatch) -> None:
    tracker = _make_tracker()
    tracker._status_cmd = "mobilint-cli status -q"
    tracker._npu_id = [1]

    class Result:
        def __init__(self, returncode: int, stdout: str):
            self.returncode = returncode
            self.stdout = stdout

    def fake_run(command, *args, **kwargs):
        if command == ["mobilint-cli", "status", "-q"]:
            return Result(returncode=2, stdout="")
        assert command[-2:] == ["--sample-once", "--json"]
        return Result(
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "npu_power_w": 2.11,
                    "total_power_w": 7.87,
                    "npu_util_pct": 0.0,
                    "npu_mem_used_mb": 0,
                    "npu_mem_total_mb": 16384,
                    "npu_temp_c": 49,
                }
            ),
        )

    monkeypatch.setattr("mblt_tracker.device_tracker_npu.subprocess.run", fake_run)

    assert tracker._fetch_metric_samples() == []

    tracker._func_for_sched()

    metrics = tracker.get_metric()
    assert metrics["samples"] == 0
    assert tracker.get_trace() == []
    assert metrics["npu"] == {}


def test_npu_fetch_metrics_falls_back_when_status_query_unparsable(
    monkeypatch,
) -> None:
    tracker = _make_tracker()
    tracker._status_cmd = "mobilint-cli status -q"

    class Result:
        def __init__(self, returncode: int, stdout: str):
            self.returncode = returncode
            self.stdout = stdout

    def fake_run(command, *args, **kwargs):
        if command == ["mobilint-cli", "status", "-q"]:
            return Result(returncode=0, stdout="not a supported status output")
        return Result(
            returncode=0,
            stdout=json.dumps(
                {"ok": True, "npu_power_w": 3.9, "total_power_w": 12.85}
            ),
        )

    monkeypatch.setattr("mblt_tracker.device_tracker_npu.subprocess.run", fake_run)

    assert tracker._fetch_metrics() == (
        3.9,
        12.85,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def test_npu_reset_clears_ddr_and_pmic_power_traces() -> None:
    tracker = _make_tracker()
    tracker._ddr_power_glance = [1.0]
    tracker._pmic_power_glance = [2.0]
    tracker._npu_power_trace = [(1.0, 3.0)]
    tracker._ddr_power_trace = [(1.0, 1.0)]
    tracker._pmic_power_trace = [(1.0, 2.0)]

    tracker.reset()

    assert tracker.get_npu_power_trace() == []
    assert tracker.get_ddr_power_trace() == []
    assert tracker.get_pmic_power_trace() == []
    assert tracker.get_metric()["ddr_power_samples"] == 0


def test_parse_mobilint_status_static_info_from_query_output() -> None:
    info = _parse_mobilint_status_static_info(STATUS_QUERY_OUTPUT)

    assert info["inference"] == {
        "npu_driver_version": "1.12.0",
        "driver": {"aries_version": "1.12.0", "regulus_version": "N/A"},
    }
    assert info["hardware"]["npus"] == [
        {
            "dev_no": 0,
            "board_name": "aries0",
            "product": "Aries",
            "firmware": {
                "version": "1.1",
                "revision": "0",
            },
            "vendor_id": "0x209F",
            "device_id": "0x0",
            "subsystem_vendor_id": "0x401",
            "subsystem_device_id": "0x1093",
            "card_model": "MLA100",
            "card_id": 0,
            "link_generation": "4",
            "lane_width": "8",
            "revision": "0x2",
            "class": "0x7800002",
            "memory_total_bytes": 17179869184,
        }
    ]


def test_npu_get_static_info_uses_mobilint_pci_vendor_by_default(monkeypatch) -> None:
    tracker = object.__new__(NPUDeviceTracker)
    captured = {}
    pcie_devices = []

    def fake_get_pcie_static_info(
        vendor_id=None, device_id=None, class_filter=None, devices=None
    ):
        captured["vendor_id"] = vendor_id
        captured["device_id"] = device_id
        captured["class_filter"] = class_filter
        captured["devices"] = devices
        return {"hardware": {"pcie": {"devices": []}}}

    monkeypatch.setattr(
        "mblt_tracker.device_tracker_npu.get_all_pcie_devices",
        lambda: pcie_devices,
    )
    monkeypatch.setattr(
        "mblt_tracker.device_tracker_npu.get_pcie_static_info",
        fake_get_pcie_static_info,
    )
    monkeypatch.setattr("mblt_tracker.device_tracker_npu.platform.system", lambda: "Linux")
    monkeypatch.setattr("mblt_tracker.device_tracker_npu.run_command", lambda command: None)
    monkeypatch.delenv("MBLT_TRACKER_NPU_PCI_VENDOR_ID", raising=False)
    monkeypatch.delenv("MBLT_TRACKER_NPU_PCI_DEVICE_ID", raising=False)
    monkeypatch.delenv("MBLT_TRACKER_NPU_PCI_CLASS_FILTER", raising=False)

    info = tracker.get_static_info()

    assert info == {"hardware": {"pcie": {"devices": []}}}
    assert captured == {
        "vendor_id": None,
        "device_id": None,
        "class_filter": None,
        "devices": pcie_devices,
    }


def test_npu_get_static_info_uses_windows_pnp_metadata_without_mobilint_cli(
    monkeypatch,
) -> None:
    tracker = object.__new__(NPUDeviceTracker)
    commands = []
    pcie_devices = [{"vendor_id": "0x209f", "device_id": "0x0000"}]
    captured: dict[str, object] = {}

    monkeypatch.setattr("mblt_tracker.device_tracker_npu.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "mblt_tracker.device_tracker_npu.get_all_pcie_devices",
        lambda: pcie_devices,
    )
    monkeypatch.setattr(
        "mblt_tracker.device_tracker_npu.get_pcie_static_info",
        lambda vendor_id=None, device_id=None, class_filter=None, devices=None: {
            "hardware": {"npus": [{"vendor_id": "0x209f"}]}
        },
    )

    def fake_windows_metadata(**kwargs):
        captured.update(kwargs)
        return {"inference": {"npu_driver_version": "1.8.1.1348"}}

    monkeypatch.setattr(
        "mblt_tracker.device_tracker_npu.get_windows_npu_driver_firmware_info",
        fake_windows_metadata,
    )
    monkeypatch.setattr(
        "mblt_tracker.device_tracker_npu.run_command",
        lambda command: commands.append(command) or None,
    )

    info = tracker.get_static_info()

    assert info == {
        "hardware": {"npus": [{"vendor_id": "0x209f"}]},
        "inference": {"npu_driver_version": "1.8.1.1348"},
    }
    assert commands == []
    assert captured == {
        "vendor_id": None,
        "device_id": None,
        "class_filter": None,
        "devices": pcie_devices,
    }


def test_npu_get_static_info_passes_filters_and_devices_to_windows_metadata(
    monkeypatch,
) -> None:
    tracker = object.__new__(NPUDeviceTracker)
    pcie_devices = [
        {"vendor_id": "0x1ed5", "device_id": "0x0100", "class": "0x120000"},
        {"vendor_id": "0x209f", "device_id": "0x0000", "class": "0x120000"},
    ]
    captured_pcie: dict[str, object] = {}
    captured_windows: dict[str, object] = {}

    monkeypatch.setenv("MBLT_TRACKER_NPU_PCI_VENDOR_ID", "1ed5")
    monkeypatch.setenv("MBLT_TRACKER_NPU_PCI_DEVICE_ID", "0100")
    monkeypatch.setenv("MBLT_TRACKER_NPU_PCI_CLASS_FILTER", "0x12")
    monkeypatch.setattr("mblt_tracker.device_tracker_npu.platform.system", lambda: "Windows")
    monkeypatch.setattr(
        "mblt_tracker.device_tracker_npu.get_all_pcie_devices",
        lambda: pcie_devices,
    )

    def fake_get_pcie_static_info(**kwargs):
        captured_pcie.update(kwargs)
        return {"hardware": {"npus": [{"dev_no": 0, "vendor_id": "0x1ed5"}]}}

    def fake_windows_metadata(**kwargs):
        captured_windows.update(kwargs)
        return {"inference": {"npu_driver_version": "1.8.1.1348"}}

    monkeypatch.setattr(
        "mblt_tracker.device_tracker_npu.get_pcie_static_info",
        fake_get_pcie_static_info,
    )
    monkeypatch.setattr(
        "mblt_tracker.device_tracker_npu.get_windows_npu_driver_firmware_info",
        fake_windows_metadata,
    )
    monkeypatch.setattr(
        "mblt_tracker.device_tracker_npu.run_command",
        lambda _command: None,
    )

    info = tracker.get_static_info()

    assert info == {
        "hardware": {"npus": [{"dev_no": 0, "vendor_id": "0x1ed5"}]},
        "inference": {"npu_driver_version": "1.8.1.1348"},
    }
    expected = {
        "vendor_id": "1ed5",
        "device_id": "0100",
        "class_filter": "0x12",
        "devices": pcie_devices,
    }
    assert captured_pcie == expected
    assert captured_windows == expected


def test_npu_get_static_info_preserves_filtered_pcie_npus_when_merging_status(
    monkeypatch,
) -> None:
    tracker = object.__new__(NPUDeviceTracker)
    status_output = """
Drivers - Aries: 1.8.1 Regulus: 1.12.0 |
| 0 Aries(Board-A) | 42 C 2.0.1 |
| 0 42 C 2.0.1 |
| 1 Aries(Board-B) | 43 C 2.0.2 |
| 1 43 C 2.0.2 |
"""

    monkeypatch.setenv("MBLT_TRACKER_NPU_PCI_DEVICE_ID", "0000")
    monkeypatch.setattr("mblt_tracker.device_tracker_npu.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "mblt_tracker.device_tracker_npu.get_all_pcie_devices",
        lambda: [],
    )
    monkeypatch.setattr(
        "mblt_tracker.device_tracker_npu.get_pcie_static_info",
        lambda vendor_id=None, device_id=None, class_filter=None, devices=None: {
            "hardware": {
                "npus": [
                    {
                        "dev_no": 1,
                        "bus_address": "0000:02:00.0",
                        "vendor_id": "0x209f",
                    }
                ]
            }
        },
    )
    monkeypatch.setattr(
        "mblt_tracker.device_tracker_npu.run_command",
        lambda _command: status_output,
    )

    info = tracker.get_static_info()

    assert info["hardware"]["npus"] == [
        {
            "dev_no": 1,
            "bus_address": "0000:02:00.0",
            "vendor_id": "0x209f",
            "board_name": "Board-B",
            "firmware": {"version": "2.0.2"},
        }
    ]
    assert info["inference"] == {
        "npu_driver_version": "1.8.1",
        "driver": {"aries_version": "1.8.1", "regulus_version": "1.12.0"},
    }
