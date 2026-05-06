from __future__ import annotations

import mblt_tracker.static_info as static_info
from mblt_tracker.static_info import (
    _calculate_theoretical_bandwidth_gbps,
    _normalize_windows_power_plan_name,
    _parse_linux_dmidecode_memory,
    _parse_windows_active_power_scheme,
    _parse_windows_power_setting_ac_value,
    _read_dram_dimms_linux,
    _read_dram_dimms_windows,
    get_pcie_static_info,
    get_windows_power_policy,
)


def _write(path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def test_read_dram_dimms_windows_parses_cim_json(monkeypatch) -> None:
    output = """
    [
      {
        "Manufacturer": "Samsung",
        "PartNumber": "M378A1K43EB2-CWE   ",
        "SerialNumber": "12345678",
        "Capacity": "8589934592",
        "Speed": 3200,
        "ConfiguredClockSpeed": 3200,
        "DataWidth": 64,
        "TotalWidth": 72,
        "SMBIOSMemoryType": 26
      },
      {
        "Manufacturer": "SK Hynix",
        "PartNumber": "HMA81GU6CJR8N-XN",
        "SerialNumber": "87654321",
        "Capacity": "8589934592",
        "Speed": 3200,
        "ConfiguredClockSpeed": 3200,
        "DataWidth": 64,
        "TotalWidth": 64,
        "SMBIOSMemoryType": 26
      }
    ]
    """

    monkeypatch.setattr(static_info, "run_command", lambda _command: output)

    dimms = _read_dram_dimms_windows()

    assert dimms == [
        {
            "manufacturer": "Samsung",
            "part_number": "M378A1K43EB2-CWE",
            "serial_number": "12345678",
            "capacity_bytes": 8589934592,
            "speed_mhz": 3200,
            "configured_speed_mhz": 3200,
            "data_width_bits": 64,
            "total_width_bits": 72,
            "type": "DDR4",
        },
        {
            "manufacturer": "SK Hynix",
            "part_number": "HMA81GU6CJR8N-XN",
            "serial_number": "87654321",
            "capacity_bytes": 8589934592,
            "speed_mhz": 3200,
            "configured_speed_mhz": 3200,
            "data_width_bits": 64,
            "total_width_bits": 64,
            "type": "DDR4",
        },
    ]


def test_read_dram_dimms_windows_returns_empty_list_on_command_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(static_info, "run_command", lambda _command: None)

    assert _read_dram_dimms_windows() == []


def test_parse_linux_dmidecode_memory() -> None:
    output = """
Handle 0x0038, DMI type 17, 40 bytes
Memory Device
        Total Width: 64 bits
        Data Width: 64 bits
        Size: 16 GB
        Type: DDR5
        Speed: 5600 MT/s
        Manufacturer: Samsung
        Serial Number: 12345678
        Part Number: M425R2GA3BB0-CWM
        Configured Memory Speed: 5600 MT/s

Handle 0x0039, DMI type 17, 40 bytes
Memory Device
        Total Width: Unknown
        Data Width: Unknown
        Size: No Module Installed
        Type: Unknown
        Speed: Unknown
        Manufacturer: Not Specified
        Part Number: Not Specified
    """

    dimms = _parse_linux_dmidecode_memory(output)

    assert dimms == [
        {
            "manufacturer": "Samsung",
            "part_number": "M425R2GA3BB0-CWM",
            "serial_number": "12345678",
            "capacity_bytes": 16 * 1024**3,
            "speed_mhz": 5600,
            "configured_speed_mhz": 5600,
            "data_width_bits": 64,
            "total_width_bits": 64,
            "type": "DDR5",
        }
    ]


def test_read_dram_dimms_linux_returns_empty_list_on_command_failure(monkeypatch) -> None:
    monkeypatch.setattr(static_info, "run_command", lambda _command: None)

    assert _read_dram_dimms_linux() == []


def test_calculate_theoretical_bandwidth_gbps() -> None:
    dimms = [
        {"configured_speed_mhz": 3200, "data_width_bits": 64},
        {"configured_speed_mhz": 3200, "data_width_bits": 64},
    ]

    assert _calculate_theoretical_bandwidth_gbps(dimms) == 51.2


def test_calculate_theoretical_bandwidth_gbps_falls_back_to_nominal_speed() -> None:
    dimms = [{"speed_mhz": 5600, "data_width_bits": 64}]

    assert _calculate_theoretical_bandwidth_gbps(dimms) == 44.8


def test_calculate_theoretical_bandwidth_gbps_returns_none_without_required_fields() -> None:
    assert _calculate_theoretical_bandwidth_gbps([{"speed_mhz": 3200}]) is None


def test_get_pcie_static_info_reads_sysfs_and_selects_matching_device(
    monkeypatch, tmp_path
) -> None:
    sysfs = tmp_path / "pci"
    # Windows cannot create directories containing ':', so use sanitized names
    # while still exercising the sysfs reader logic.
    npu = sysfs / "0000_01_00.0"
    npu2 = sysfs / "0000_03_00.0"
    other = sysfs / "0000_02_00.0"
    npu.mkdir(parents=True)
    npu2.mkdir(parents=True)
    other.mkdir(parents=True)

    _write(npu / "vendor", "0x1ed5\n")
    _write(npu / "device", "0x0100\n")
    _write(npu / "class", "0x120000\n")
    _write(npu / "current_link_speed", "16.0 GT/s PCIe\n")
    _write(npu / "current_link_width", "8\n")
    _write(npu2 / "vendor", "0x1ed5\n")
    _write(npu2 / "device", "0x0100\n")
    _write(npu2 / "class", "0x120000\n")
    _write(npu2 / "current_link_speed", "8.0 GT/s PCIe\n")
    _write(npu2 / "current_link_width", "4\n")
    _write(other / "vendor", "0x10de\n")
    _write(other / "device", "0x2684\n")

    monkeypatch.setenv("MBLT_TRACKER_PCI_SYSFS", str(sysfs))

    info = get_pcie_static_info(vendor_id="1ed5", device_id="0100")

    assert "devices" not in info["hardware"]["pcie"]
    npus = info["hardware"]["pcie"]["npus"]
    assert len(npus) == 2
    assert npus[0]["dev_no"] == 0
    assert npus[0]["bus_address"] == "0000_01_00.0"
    assert npus[0]["vendor_id"] == "0x1ed5"
    assert npus[0]["device_id"] == "0x0100"
    assert npus[0]["current_link_speed"] == "16.0 GT/s PCIe"
    assert npus[0]["link_generation"] == "Gen4"
    assert npus[0]["lane_width"] == "x8"
    assert npus[1]["dev_no"] == 1
    assert npus[1]["bus_address"] == "0000_03_00.0"
    assert npus[1]["current_link_speed"] == "8.0 GT/s PCIe"
    assert npus[1]["link_generation"] == "Gen3"
    assert npus[1]["lane_width"] == "x4"


def test_get_pcie_static_info_can_include_all_devices(monkeypatch, tmp_path) -> None:
    sysfs = tmp_path / "pci"
    npu = sysfs / "0000_01_00.0"
    other = sysfs / "0000_02_00.0"
    npu.mkdir(parents=True)
    other.mkdir(parents=True)

    _write(npu / "vendor", "0x1ed5\n")
    _write(npu / "device", "0x0100\n")
    _write(npu / "class", "0x120000\n")
    _write(other / "vendor", "0x8086\n")
    _write(other / "device", "0x1234\n")
    _write(other / "class", "0x060400\n")

    monkeypatch.setenv("MBLT_TRACKER_PCI_SYSFS", str(sysfs))

    info = get_pcie_static_info(include_all_devices=True)

    assert len(info["hardware"]["pcie"]["devices"]) == 2


def test_get_pcie_static_info_keeps_gpu_and_accelerator_devices(
    monkeypatch, tmp_path
) -> None:
    sysfs = tmp_path / "pci"
    gpu = sysfs / "0000_01_00.0"
    accelerator = sysfs / "0000_02_00.0"
    bridge = sysfs / "0000_03_00.0"
    gpu.mkdir(parents=True)
    accelerator.mkdir(parents=True)
    bridge.mkdir(parents=True)

    _write(gpu / "vendor", "0x10de\n")
    _write(gpu / "device", "0x2684\n")
    _write(gpu / "class", "0x030000\n")
    _write(accelerator / "vendor", "0x1234\n")
    _write(accelerator / "device", "0xabcd\n")
    _write(accelerator / "class", "0x120000\n")
    _write(bridge / "vendor", "0x8086\n")
    _write(bridge / "device", "0x5678\n")
    _write(bridge / "class", "0x060400\n")

    monkeypatch.setenv("MBLT_TRACKER_PCI_SYSFS", str(sysfs))

    info = get_pcie_static_info(include_all_devices=True)

    pcie = info["hardware"]["pcie"]
    assert [device["bus_address"] for device in pcie["devices"]] == [
        "0000_01_00.0",
        "0000_02_00.0",
        "0000_03_00.0",
    ]
    assert [device["bus_address"] for device in pcie["gpus"]] == ["0000_01_00.0"]
    assert "npus" not in pcie


def test_get_pcie_static_info_omits_raw_devices_by_default(
    monkeypatch, tmp_path
) -> None:
    sysfs = tmp_path / "pci"
    gpu = sysfs / "0000_01_00.0"
    bridge = sysfs / "0000_02_00.0"
    gpu.mkdir(parents=True)
    bridge.mkdir(parents=True)

    _write(gpu / "vendor", "0x10de\n")
    _write(gpu / "device", "0x2684\n")
    _write(gpu / "class", "0x030000\n")
    _write(bridge / "vendor", "0x8086\n")
    _write(bridge / "device", "0x5678\n")
    _write(bridge / "class", "0x060400\n")

    monkeypatch.setenv("MBLT_TRACKER_PCI_SYSFS", str(sysfs))

    info = get_pcie_static_info()

    pcie = info["hardware"]["pcie"]
    assert "devices" not in pcie
    assert [device["bus_address"] for device in pcie["gpus"]] == ["0000_01_00.0"]


def test_parse_windows_active_power_scheme_english_output() -> None:
    output = (
        "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e "
        "(Balanced)\n"
    )

    scheme_guid, power_plan = _parse_windows_active_power_scheme(output)

    assert scheme_guid == "381b4222-f694-41f0-9685-ff5bb260df2e"
    assert power_plan == "Balanced"


def test_parse_windows_active_power_scheme_korean_output() -> None:
    output = (
        "전원 구성표 GUID: 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c "
        "(고성능)\n"
    )

    scheme_guid, power_plan = _parse_windows_active_power_scheme(output)

    assert scheme_guid == "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
    assert power_plan == "고성능"


def test_parse_windows_power_setting_ac_value() -> None:
    output = "    Current AC Power Setting Index: 0x00000064\n"

    assert _parse_windows_power_setting_ac_value(output) == 100


def test_parse_windows_power_setting_ac_value_korean_output() -> None:
    output = "    현재 AC 전원 설정 인덱스: 0x00000005\n"

    assert _parse_windows_power_setting_ac_value(output) == 5


def test_normalize_windows_power_plan_name_uses_builtin_english_name() -> None:
    power_plan = _normalize_windows_power_plan_name(
        "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",
        "고성능",
    )

    assert power_plan == "High performance"


def test_normalize_windows_power_plan_name_keeps_custom_name() -> None:
    power_plan = _normalize_windows_power_plan_name(
        "00000000-0000-0000-0000-000000000000",
        "Custom Plan",
    )

    assert power_plan == "Custom Plan"


def test_get_windows_power_policy_uses_windows_native_names(monkeypatch) -> None:
    scheme_guid = "381b4222-f694-41f0-9685-ff5bb260df2e"
    commands = []

    def fake_run_command(command):
        commands.append(command)
        if command == ["powercfg", "/getactivescheme"]:
            return f"Power Scheme GUID: {scheme_guid} (Balanced)\n"
        if command[-1] == "893dee8e-2bef-41e0-89c6-b55d0929964c":
            return "    Current AC Power Setting Index: 0x00000005\n"
        if command[-1] == "bc5038f7-23e0-4960-96da-33abaf5935ec":
            return "    Current AC Power Setting Index: 0x00000064\n"
        return None

    monkeypatch.setattr(static_info, "run_command", fake_run_command)

    policy = get_windows_power_policy()

    assert policy == {
        "power_plan": "Balanced",
        "min_processor_state_pct": 5,
        "max_processor_state_pct": 100,
    }
    assert commands[0] == ["powercfg", "/getactivescheme"]


def test_get_windows_power_policy_normalizes_builtin_plan_to_english(
    monkeypatch,
) -> None:
    scheme_guid = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"

    def fake_run_command(command):
        if command == ["powercfg", "/getactivescheme"]:
            return f"전원 구성표 GUID: {scheme_guid} (고성능)\n"
        if command[-1] == "893dee8e-2bef-41e0-89c6-b55d0929964c":
            return "    현재 AC 전원 설정 인덱스: 0x00000064\n"
        if command[-1] == "bc5038f7-23e0-4960-96da-33abaf5935ec":
            return "    현재 AC 전원 설정 인덱스: 0x00000064\n"
        return None

    monkeypatch.setattr(static_info, "run_command", fake_run_command)

    policy = get_windows_power_policy()

    assert policy == {
        "power_plan": "High performance",
        "min_processor_state_pct": 100,
        "max_processor_state_pct": 100,
    }
