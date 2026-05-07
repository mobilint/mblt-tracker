from __future__ import annotations

import json
import os
import subprocess

import pytest

from mblt_tracker.device_tracker_npu import NPUDeviceTracker
from mblt_tracker.device_tracker_npu import _parse_mobilint_status_static_info


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
