from __future__ import annotations

from mblt_tracker.static_info import get_pcie_static_info


def _write(path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def test_get_pcie_static_info_reads_sysfs_and_selects_matching_device(
    monkeypatch, tmp_path
) -> None:
    sysfs = tmp_path / "pci"
    # Windows cannot create directories containing ':', so use sanitized names
    # while still exercising the sysfs reader logic.
    npu = sysfs / "0000_01_00.0"
    other = sysfs / "0000_02_00.0"
    npu.mkdir(parents=True)
    other.mkdir(parents=True)

    _write(npu / "vendor", "0x1ed5\n")
    _write(npu / "device", "0x0100\n")
    _write(npu / "class", "0x120000\n")
    _write(npu / "current_link_speed", "16.0 GT/s PCIe\n")
    _write(npu / "current_link_width", "8\n")
    _write(other / "vendor", "0x10de\n")
    _write(other / "device", "0x2684\n")

    monkeypatch.setenv("MBLT_TRACKER_PCI_SYSFS", str(sysfs))

    info = get_pcie_static_info(vendor_id="1ed5", device_id="0100")

    assert len(info["hardware.pcie.devices"]) == 2
    assert info["hardware.pcie.npu.bus_address"] == "0000_01_00.0"
    assert info["hardware.pcie.npu.vendor_id"] == "0x1ed5"
    assert info["hardware.pcie.npu.device_id"] == "0x0100"
    assert info["hardware.pcie.npu.link_speed"] == "16.0 GT/s PCIe"
    assert info["hardware.pcie.npu.link_generation"] == "Gen4"
    assert info["hardware.pcie.npu.lane_width"] == "x8"