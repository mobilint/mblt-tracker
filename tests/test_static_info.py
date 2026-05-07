from __future__ import annotations

import json
import sys
import types
from typing import cast

import mblt_tracker.static_info as static_info
from mblt_tracker.static_info import (
    _calculate_theoretical_bandwidth_gbps,
    _deep_merge,
    _format_nvml_cuda_driver_version,
    _get_cuda_version,
    _get_python_package_version,
    get_cpu_power_policy,
    get_nvml_gpu_static_info,
    get_linux_npu_driver_firmware_info,
    _parse_nvcc_cuda_version,
    _read_windows_pci_link_properties,
    _normalize_windows_power_plan_name,
    _parse_linux_dmidecode_memory,
    _parse_windows_active_power_scheme,
    _parse_windows_pci_id,
    _parse_windows_power_setting_ac_value,
    _read_dram_dimms_linux,
    _read_lspci_device_metadata,
    _read_dram_dimms_windows,
    get_pcie_static_info,
    get_windows_npu_driver_firmware_info,
    get_windows_power_policy,
    parse_mobilint_status_static_info,
)


def _write(path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def test_parse_nvcc_cuda_version() -> None:
    output = "Cuda compilation tools, release 12.4, V12.4.131\n"

    assert _parse_nvcc_cuda_version(output) == "12.4"


def test_get_cuda_version_prefers_torch_cuda(monkeypatch) -> None:
    torch_module = types.SimpleNamespace(version=types.SimpleNamespace(cuda="12.1"))
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setattr(static_info, "run_command", lambda _command: "release 11.8")

    assert _get_cuda_version() == "12.1"


def test_get_cuda_version_falls_back_to_nvcc(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(
        static_info,
        "run_command",
        lambda command: "Cuda compilation tools, release 12.4, V12.4.131\n"
        if command == ["nvcc", "--version"]
        else None,
    )

    assert _get_cuda_version() == "12.4"


def test_format_nvml_cuda_driver_version() -> None:
    assert _format_nvml_cuda_driver_version(12080) == "12.8"


def test_get_nvml_gpu_static_info_returns_metadata(monkeypatch) -> None:
    class FakePciInfo:
        busId = b"00000000:17:00.0"

    class FakeMemoryInfo:
        total = 24 * 1024**3

    class FakeNvml:
        def __init__(self) -> None:
            self.shutdown_called = False

        def nvmlInit(self) -> None:
            return None

        def nvmlShutdown(self) -> None:
            self.shutdown_called = True

        def nvmlDeviceGetCount(self) -> int:
            return 1

        def nvmlSystemGetDriverVersion(self) -> bytes:
            return b"580.95.05"

        def nvmlSystemGetCudaDriverVersion(self) -> int:
            return 12080

        def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
            return f"handle-{index}"

        def nvmlDeviceGetName(self, handle: str) -> bytes:
            assert handle == "handle-0"
            return b"NVIDIA RTX Test"

        def nvmlDeviceGetPciInfo(self, handle: str) -> FakePciInfo:
            assert handle == "handle-0"
            return FakePciInfo()

        def nvmlDeviceGetMemoryInfo(self, handle: str) -> FakeMemoryInfo:
            assert handle == "handle-0"
            return FakeMemoryInfo()

        def nvmlDeviceGetArchitecture(self, handle: str) -> int:
            assert handle == "handle-0"
            return 8

        def nvmlDeviceGetCurrPcieLinkGeneration(self, handle: str) -> int:
            assert handle == "handle-0"
            return 1

        def nvmlDeviceGetCurrPcieLinkWidth(self, handle: str) -> int:
            assert handle == "handle-0"
            return 16

    fake_nvml = FakeNvml()
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
    monkeypatch.setattr(static_info.platform, "system", lambda: "Linux")

    pcie_devices = [
        {
            "bus_address": "0000:17:00.0",
            "vendor_id": "0x10de",
            "device_id": "0x2bb1",
            "class": "0x030000",
            "current_link_speed": "2.5 GT/s PCIe",
            "current_link_width": "16",
        }
    ]

    info = get_nvml_gpu_static_info(pcie_devices=pcie_devices)

    assert fake_nvml.shutdown_called is True
    assert info == {
        "hardware": {
            "gpus": [
                {
                    "dev_no": 0,
                    "bus_address": "0000:17:00.0",
                    "class": "0x030000",
                    "device_id": "0x2bb1",
                    "driver_version": "580.95.05",
                    "architecture": "Ada Lovelace",
                    "lane_width": "x16",
                    "link_generation": "Gen1",
                    "memory_total_bytes": 24 * 1024**3,
                    "name": "NVIDIA RTX Test",
                    "vendor_id": "0x10de",
                }
            ],
        },
        "inference": {
            "gpu": {
                "driver": {"version": "580.95.05"},
                "cuda_driver": {"version": "12.8"},
            }
        },
    }


def test_get_nvml_gpu_static_info_returns_empty_on_unavailable_nvml(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)

    assert get_nvml_gpu_static_info() == {}
    assert "Warning: NVML not available" in capsys.readouterr().err


def test_get_nvml_gpu_static_info_matches_windows_pcie_by_nvidia_order(
    monkeypatch,
) -> None:
    class FakePciInfo:
        busId = b"00000000:03:00.0"

    class FakeMemoryInfo:
        total = 24 * 1024**3

    class FakeNvml:
        def nvmlInit(self) -> None:
            return None

        def nvmlShutdown(self) -> None:
            return None

        def nvmlDeviceGetCount(self) -> int:
            return 1

        def nvmlSystemGetDriverVersion(self) -> str:
            return "595.97"

        def nvmlSystemGetCudaDriverVersion(self) -> int:
            return 13020

        def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
            return f"handle-{index}"

        def nvmlDeviceGetName(self, _handle: str) -> str:
            return "NVIDIA GeForce RTX 3090"

        def nvmlDeviceGetPciInfo(self, _handle: str) -> FakePciInfo:
            return FakePciInfo()

        def nvmlDeviceGetMemoryInfo(self, _handle: str) -> FakeMemoryInfo:
            return FakeMemoryInfo()

        def nvmlDeviceGetArchitecture(self, _handle: str) -> int:
            return 7

        def nvmlDeviceGetCurrPcieLinkGeneration(self, _handle: str) -> int:
            return 3

        def nvmlDeviceGetCurrPcieLinkWidth(self, _handle: str) -> int:
            return 4

    monkeypatch.setitem(sys.modules, "pynvml", FakeNvml())
    monkeypatch.setattr(static_info.platform, "system", lambda: "Windows")

    pcie_devices = [
        {
            "dev_no": 0,
            "bus_address": "PCI\\VEN_10DE&DEV_1AEF",
            "vendor_id": "0x10de",
            "device_id": "0x1aef",
            "driver_description": "High Definition Audio Controller",
            "manufacturer": "Microsoft",
            "current_link_speed": "8.0 GT/s PCIe",
            "current_link_width": "4",
        },
        {
            "dev_no": 1,
            "bus_address": "PCI\\VEN_10DE&DEV_2204",
            "vendor_id": "0x10de",
            "device_id": "0x2204",
            "driver_description": "NVIDIA GeForce RTX 3090",
            "manufacturer": "NVIDIA",
            "current_link_speed": "8.0 GT/s PCIe",
            "current_link_width": "4",
        }
    ]

    info = get_nvml_gpu_static_info(pcie_devices=pcie_devices)

    assert info["hardware"]["gpus"] == [
        {
            "dev_no": 0,
            "bus_address": "0000:03:00.0",
            "device_id": "0x2204",
            "driver_description": "NVIDIA GeForce RTX 3090",
            "driver_version": "595.97",
            "architecture": "Ampere",
            "lane_width": "x4",
            "link_generation": "Gen3",
            "manufacturer": "NVIDIA",
            "memory_total_bytes": 24 * 1024**3,
            "name": "NVIDIA GeForce RTX 3090",
            "vendor_id": "0x10de",
        }
    ]


def test_deep_merge_matches_gpu_by_bus_address_before_index() -> None:
    base: dict[str, object] = {
        "hardware": {
            "gpus": [
                {"dev_no": 0, "bus_address": "0000:02:00.0", "name": "ASPEED"},
                {"dev_no": 1, "bus_address": "0000:17:00.0", "name": "Device"},
            ]
        }
    }
    overlay = {
        "hardware": {
            "gpus": [
                {
                    "bus_address": "0000:17:00.0",
                    "driver_version": "580.95.05",
                    "name": "NVIDIA RTX Test",
                }
            ]
        }
    }

    _deep_merge(base, overlay)

    gpus = base["hardware"]["gpus"]
    assert gpus == [
        {"dev_no": 0, "bus_address": "0000:02:00.0", "name": "ASPEED"},
        {
            "dev_no": 1,
            "bus_address": "0000:17:00.0",
            "name": "NVIDIA RTX Test",
            "driver_version": "580.95.05",
        },
    ]


def test_get_python_package_version_reads_module_version(monkeypatch) -> None:
    module = types.SimpleNamespace(__version__="1.2.3")
    monkeypatch.setitem(sys.modules, "qbruntime", module)

    assert _get_python_package_version("qbruntime") == "1.2.3"


def test_get_python_package_version_returns_none_when_not_installed(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "qbcompiler", None)

    assert _get_python_package_version("qbcompiler") is None


def test_get_python_package_version_falls_back_to_metadata_on_import_runtime_failure(
    monkeypatch,
) -> None:
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "qbruntime":
            raise OSError("missing native dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(static_info, "__import__", fake_import, raising=False)
    monkeypatch.setattr(
        static_info.metadata,
        "version",
        lambda package_name: "2.3.4"
        if package_name == "mobilint-qb-runtime"
        else None,
    )

    assert _get_python_package_version("qbruntime") == "2.3.4"


def test_get_python_package_version_returns_none_when_import_fails_and_metadata_missing(
    monkeypatch,
) -> None:
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "qbruntime":
            raise OSError("missing native dependency")
        return real_import(name, *args, **kwargs)

    def fake_metadata_version(_package_name):
        raise static_info.metadata.PackageNotFoundError

    monkeypatch.setattr(static_info, "__import__", fake_import, raising=False)
    monkeypatch.setattr(static_info.metadata, "version", fake_metadata_version)

    assert _get_python_package_version("qbruntime") is None


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


def test_read_dram_dimms_linux_tries_non_interactive_sudo(monkeypatch) -> None:
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
    """
    commands = []

    def fake_run_command(command):
        commands.append(command)
        if command == ["sudo", "-n", "dmidecode", "-t", "memory"]:
            return output
        return None

    monkeypatch.setattr(static_info, "run_command", fake_run_command)

    dimms = _read_dram_dimms_linux()

    assert commands == [
        ["dmidecode", "-t", "memory"],
        ["sudo", "-n", "dmidecode", "-t", "memory"],
    ]
    assert dimms[0]["manufacturer"] == "Samsung"


def test_read_dram_dimms_linux_skips_password_provider_when_direct_succeeds(
    monkeypatch,
) -> None:
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
    """
    commands = []

    def fake_run_command(command):
        commands.append(command)
        if command == ["dmidecode", "-t", "memory"]:
            return output
        return None

    def fail_password_provider():
        raise AssertionError("password provider should not be called")

    monkeypatch.setattr(static_info, "run_command", fake_run_command)

    dimms = _read_dram_dimms_linux(sudo_password_provider=fail_password_provider)

    assert commands == [["dmidecode", "-t", "memory"]]
    assert dimms[0]["manufacturer"] == "Samsung"


def test_read_dram_dimms_linux_uses_password_for_sudo(monkeypatch) -> None:
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
    """
    commands = []

    def fake_run_command(command):
        commands.append((command, None, None))
        return None

    def fake_run_command_with_input(command, input_text, timeout):
        commands.append((command, input_text, timeout))
        return output

    monkeypatch.setattr(static_info, "run_command", fake_run_command)
    monkeypatch.setattr(static_info, "run_command_with_input", fake_run_command_with_input)

    dimms = _read_dram_dimms_linux(sudo_password="secret")

    assert commands == [
        (["dmidecode", "-t", "memory"], None, None),
        (["sudo", "-n", "dmidecode", "-t", "memory"], None, None),
        (["sudo", "-S", "-p", "", "dmidecode", "-t", "memory"], "secret\n", 30),
    ]
    assert dimms[0]["manufacturer"] == "Samsung"


def test_read_dram_dimms_linux_defers_password_provider_until_direct_fails(
    monkeypatch,
) -> None:
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
    """
    commands = []

    def fake_run_command(command):
        commands.append((command, None, None))
        return None

    def fake_run_command_with_input(command, input_text, timeout):
        commands.append((command, input_text, timeout))
        return output

    monkeypatch.setattr(static_info, "run_command", fake_run_command)
    monkeypatch.setattr(static_info, "run_command_with_input", fake_run_command_with_input)

    dimms = _read_dram_dimms_linux(sudo_password_provider=lambda: "secret")

    assert commands == [
        (["dmidecode", "-t", "memory"], None, None),
        (["sudo", "-n", "dmidecode", "-t", "memory"], None, None),
        (["sudo", "-S", "-p", "", "dmidecode", "-t", "memory"], "secret\n", 30),
    ]
    assert dimms[0]["manufacturer"] == "Samsung"


def test_read_dram_dimms_linux_tries_non_interactive_sudo_before_password_provider(
    monkeypatch,
) -> None:
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
    """
    commands = []

    def fake_run_command(command):
        commands.append(command)
        if command == ["sudo", "-n", "dmidecode", "-t", "memory"]:
            return output
        return None

    def fail_password_provider():
        raise AssertionError("password provider should not be called")

    monkeypatch.setattr(static_info, "run_command", fake_run_command)

    dimms = _read_dram_dimms_linux(sudo_password_provider=fail_password_provider)

    assert commands == [
        ["dmidecode", "-t", "memory"],
        ["sudo", "-n", "dmidecode", "-t", "memory"],
    ]
    assert dimms[0]["manufacturer"] == "Samsung"


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

    hardware = cast(dict[str, object], info["hardware"])
    assert "pcie_devices" not in hardware
    npus = cast(list[dict[str, object]], hardware["npus"])
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


def test_get_pcie_static_info_reads_linux_revision_driver_and_known_names(
    monkeypatch, tmp_path
) -> None:
    sysfs = tmp_path / "pci"
    npu = sysfs / "0000_01_00.0"
    driver = tmp_path / "drivers" / "mblt_npu"
    npu.mkdir(parents=True)
    (driver / "module").mkdir(parents=True)

    _write(npu / "vendor", "0x209f\n")
    _write(npu / "device", "0x0000\n")
    _write(npu / "class", "0x078000\n")
    _write(npu / "revision", "0x02\n")
    _write(npu / "current_link_speed", "16.0 GT/s PCIe\n")
    _write(npu / "current_link_width", "8\n")
    _write(driver / "module" / "version", "1.8.1\n")
    (npu / "driver").symlink_to(driver)

    monkeypatch.setenv("MBLT_TRACKER_PCI_SYSFS", str(sysfs))
    monkeypatch.setattr(static_info, "run_command", lambda _command: None)

    info = get_pcie_static_info()

    hardware = cast(dict[str, object], info["hardware"])
    npu_info = cast(list[dict[str, object]], hardware["npus"])[0]
    assert npu_info["name"] == "MOBILINT NPU Accelerator"
    assert npu_info["manufacturer"] == "MOBILINT, Inc."
    assert npu_info["revision"] == "0x02"
    assert "driver_version" not in npu_info
    assert info["inference"]["npu_driver_version"] == "1.8.1"


def test_get_pcie_static_info_keeps_nested_firmware_metadata() -> None:
    info = get_pcie_static_info(
        devices=[
            {
                "bus_address": "PCI\\VEN_209F&DEV_0000",
                "vendor_id": "0x209f",
                "device_id": "0x0000",
                "name": "MOBILINT NPU Accelerator",
                "firmware": {"version": "2.0.3"},
            }
        ]
    )

    hardware = cast(dict[str, object], info["hardware"])
    npu_info = cast(list[dict[str, object]], hardware["npus"])[0]
    assert npu_info["firmware"] == {"version": "2.0.3"}


def test_get_pcie_static_info_preserves_original_npu_index_when_filtered() -> None:
    info = get_pcie_static_info(
        device_id="0000",
        devices=[
            {
                "bus_address": "0000:01:00.0",
                "vendor_id": "0x1ed5",
                "device_id": "0x0100",
                "class": "0x120000",
            },
            {
                "bus_address": "0000:02:00.0",
                "vendor_id": "0x209f",
                "device_id": "0x0000",
                "class": "0x120000",
            },
        ],
    )

    hardware = cast(dict[str, object], info["hardware"])
    npus = cast(list[dict[str, object]], hardware["npus"])
    assert npus == [
        {
            "dev_no": 1,
            "bus_address": "0000:02:00.0",
            "vendor_id": "0x209f",
            "device_id": "0x0000",
            "class": "0x120000",
        }
    ]


def test_lspci_metadata_parses_machine_readable_output(monkeypatch) -> None:
    output = '0000:01:00.0 "3D controller [0302]" "NVIDIA Corporation [10de]" "AD102 [GeForce RTX 4090] [2684]"\n'
    monkeypatch.setattr(static_info, "run_command", lambda _command: output)

    metadata = _read_lspci_device_metadata()

    assert metadata["0000:01:00.0"] == {
        "manufacturer": "NVIDIA Corporation",
        "name": "AD102 [GeForce RTX 4090]",
    }


def test_get_pcie_static_info_filters_intel_igpu_without_pcie_link(
    monkeypatch, tmp_path
) -> None:
    sysfs = tmp_path / "pci"
    igpu = sysfs / "0000_00_02.0"
    igpu.mkdir(parents=True)

    _write(igpu / "vendor", "0x8086\n")
    _write(igpu / "device", "0xa780\n")
    _write(igpu / "class", "0x030000\n")
    _write(igpu / "current_link_speed", "Unknown\n")
    _write(igpu / "current_link_width", "0\n")

    monkeypatch.setenv("MBLT_TRACKER_PCI_SYSFS", str(sysfs))
    monkeypatch.setattr(static_info, "run_command", lambda _command: None)

    info = get_pcie_static_info()

    assert info == {}


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

    hardware = cast(dict[str, object], info["hardware"])
    assert len(cast(list[dict[str, object]], hardware["pcie_devices"])) == 2


def test_get_pcie_static_info_includes_gpu_and_raw_devices_when_all_requested(
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

    hardware = cast(dict[str, object], info["hardware"])
    assert [
        device["bus_address"]
        for device in cast(list[dict[str, object]], hardware["pcie_devices"])
    ] == [
        "0000_01_00.0",
        "0000_02_00.0",
        "0000_03_00.0",
    ]
    gpus = cast(list[dict[str, object]], hardware["gpus"])
    assert gpus == [
        {
            "dev_no": 0,
            "bus_address": "0000_01_00.0",
            "vendor_id": "0x10de",
            "device_id": "0x2684",
            "class": "0x030000",
            "manufacturer": "NVIDIA",
        }
    ]
    assert "npus" not in hardware


def test_get_pcie_static_info_includes_gpu_but_omits_raw_devices_by_default(
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

    hardware = cast(dict[str, object], info["hardware"])
    assert "pcie_devices" not in hardware
    assert hardware["gpus"] == [
        {
            "dev_no": 0,
            "bus_address": "0000_01_00.0",
            "vendor_id": "0x10de",
            "device_id": "0x2684",
            "class": "0x030000",
            "manufacturer": "NVIDIA",
        }
    ]


def test_get_pcie_static_info_includes_amd_gpu_without_nvml() -> None:
    info = get_pcie_static_info(
        devices=[
            {
                "bus_address": "0000:05:00.0",
                "vendor_id": "0x1002",
                "device_id": "0x744c",
                "class": "0x030000",
                "name": "Navi 31 [Radeon RX 7900 XTX]",
                "manufacturer": "AMD",
                "current_link_speed": "16.0 GT/s PCIe",
                "current_link_width": "16",
            }
        ]
    )

    hardware = cast(dict[str, object], info["hardware"])
    assert hardware["gpus"] == [
        {
            "dev_no": 0,
            "bus_address": "0000:05:00.0",
            "vendor_id": "0x1002",
            "device_id": "0x744c",
            "class": "0x030000",
            "name": "Navi 31 [Radeon RX 7900 XTX]",
            "manufacturer": "AMD",
            "current_link_speed": "16.0 GT/s PCIe",
            "current_link_width": "16",
            "link_generation": "Gen4",
            "lane_width": "x16",
        }
    ]


def test_parse_windows_pci_id_reads_class_from_auxiliary_ids() -> None:
    device = _parse_windows_pci_id(
        "PCI\\VEN_209F&DEV_0000&SUBSYS_10930402&REV_02\\4&3691B449&0&0008",
        [
            "PCI\\VEN_209F&DEV_0000&SUBSYS_10930402&REV_02",
            "PCI\\VEN_209F&DEV_0000&CC_120000",
            "PCI\\VEN_209F&DEV_0000&CC_12",
        ],
    )

    assert device is not None
    assert device["vendor_id"] == "0x209f"
    assert device["device_id"] == "0x0000"
    assert device["class"] == "0x120000"
    assert device["revision"] == "0x02"


def test_get_pcie_static_info_filters_windows_devices_by_auxiliary_class(
    monkeypatch,
) -> None:
    output = json.dumps(
        [
            {
                "Name": "MOBILINT NPU Accelerator",
                "PNPDeviceID": "PCI\\VEN_209F&DEV_0000&SUBSYS_10930402&REV_02\\4&1",
                "HardwareID": [
                    "PCI\\VEN_209F&DEV_0000&SUBSYS_10930402&REV_02",
                    "PCI\\VEN_209F&DEV_0000&CC_120000",
                ],
                "CompatibleID": ["PCI\\VEN_209F&DEV_0000&CC_12"],
                "Manufacturer": "MOBILINT, Inc.",
                "Status": "OK",
            },
            {
                "Name": "Other Device",
                "PNPDeviceID": "PCI\\VEN_1234&DEV_ABCD\\4&2",
                "HardwareID": ["PCI\\VEN_1234&DEV_ABCD&CC_060400"],
                "Manufacturer": "Other",
                "Status": "OK",
            },
        ]
    )

    monkeypatch.setattr(static_info.platform, "system", lambda: "Windows")
    monkeypatch.setattr(static_info, "run_command", lambda _command: output)
    monkeypatch.setattr(
        static_info,
        "_read_windows_pci_link_properties",
        lambda _instance_ids: {},
    )

    info = get_pcie_static_info(class_filter="0x12")

    hardware = cast(dict[str, object], info["hardware"])
    npus = cast(list[dict[str, object]], hardware["npus"])
    assert npus == [
        {
            "dev_no": 0,
            "bus_address": "PCI\\VEN_209F&DEV_0000&SUBSYS_10930402&REV_02\\4&1",
            "vendor_id": "0x209f",
            "device_id": "0x0000",
            "subsystem_device_id": "0x1093",
            "subsystem_vendor_id": "0x0402",
            "class": "0x120000",
            "name": "MOBILINT NPU Accelerator",
            "manufacturer": "MOBILINT, Inc.",
            "status": "OK",
            "pnp_device_id": "PCI\\VEN_209F&DEV_0000&SUBSYS_10930402&REV_02\\4&1",
            "revision": "0x02",
        }
    ]


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
        "governor": None,
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
        "governor": None,
        "power_plan": "High performance",
        "min_processor_state_pct": 100,
        "max_processor_state_pct": 100,
    }


def test_get_cpu_power_policy_keeps_os_independent_shape(monkeypatch) -> None:
    monkeypatch.setattr(static_info.platform, "system", lambda: "Linux")
    monkeypatch.setattr(static_info, "get_cpu_governor", lambda: "powersave")

    assert get_cpu_power_policy() == {
        "governor": "powersave",
        "power_plan": None,
        "min_processor_state_pct": None,
        "max_processor_state_pct": None,
    }


def test_read_windows_pci_link_properties_includes_driver_and_firmware_metadata(
    monkeypatch,
) -> None:
    instance_id = "PCI\\VEN_209F&DEV_0000&SUBSYS_10930402&REV_02\\4&3691B449&0&0008"
    output = json.dumps(
        {
            "InstanceId": instance_id,
            "DEVPKEY_Device_DriverVersion": "1.8.1.1348",
            "DEVPKEY_Device_DriverDate": "/Date(1774828800000)/",
            "DEVPKEY_Device_DriverDesc": "MOBILINT NPU Accelerator",
            "DEVPKEY_Device_DriverProvider": "MOBILINT, Inc.",
            "DEVPKEY_Device_FirmwareVersion": None,
            "DEVPKEY_Device_FirmwareRevision": "2.0.3",
        }
    )

    monkeypatch.setattr(
        static_info,
        "run_command_with_timeout",
        lambda _command, timeout: output,
    )

    properties = _read_windows_pci_link_properties()

    device = properties[instance_id]
    assert device["driver_version"] == "1.8.1.1348"
    assert device["driver_date"] == "/Date(1774828800000)/"
    assert device["driver_description"] == "MOBILINT NPU Accelerator"
    assert device["driver_provider"] == "MOBILINT, Inc."
    assert device["firmware"] == {"version": "2.0.3"}
    assert "firmware_version" not in device
    assert "firmware_revision" not in device


def test_read_windows_pci_link_properties_enumerates_all_pci_without_instance_wildcard(
    monkeypatch,
) -> None:
    commands = []

    def fake_run_command_with_timeout(command, timeout):
        commands.append(command)
        return "[]"

    monkeypatch.setattr(
        static_info,
        "run_command_with_timeout",
        fake_run_command_with_timeout,
    )

    assert _read_windows_pci_link_properties(None) == {}

    powershell_command = commands[0][-1]
    assert "Get-PnpDevice -InstanceId 'PCI\\*'" not in powershell_command
    assert "Get-PnpDevice | Where-Object" in powershell_command
    assert "-like 'PCI\\*'" in powershell_command


def test_get_windows_npu_driver_firmware_info_uses_pnp_metadata(monkeypatch) -> None:
    monkeypatch.setattr(static_info.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        static_info,
        "get_pcie_static_info",
        lambda **_kwargs: {
            "hardware": {
                "npus": [
                    {
                        "dev_no": 0,
                        "vendor_id": "0x209f",
                        "name": "MOBILINT NPU Accelerator",
                        "pnp_device_id": "PCI\\VEN_209F&DEV_0000",
                    }
                ]
            }
            ,
            "inference": {"npu_driver_version": "1.8.1.1348"},
        },
    )

    info = get_windows_npu_driver_firmware_info()

    hardware = cast(dict[str, object], info["hardware"])
    assert hardware["npus"] == [
        {
            "dev_no": 0,
            "vendor_id": "0x209f",
            "name": "MOBILINT NPU Accelerator",
            "pnp_device_id": "PCI\\VEN_209F&DEV_0000",
        }
    ]
    assert info["inference"] == {"npu_driver_version": "1.8.1.1348"}


def test_get_windows_npu_driver_firmware_info_honors_pcie_filters(monkeypatch) -> None:
    monkeypatch.setattr(static_info.platform, "system", lambda: "Windows")
    captured: dict[str, object] = {}
    pcie_devices = [{"vendor_id": "0x1ed5", "device_id": "0x0100"}]

    def fake_get_pcie_static_info(**kwargs):
        captured.update(kwargs)
        return {"hardware": {"npus": [{"vendor_id": "0x1ed5"}]}}

    monkeypatch.setattr(static_info, "get_pcie_static_info", fake_get_pcie_static_info)

    get_windows_npu_driver_firmware_info(
        vendor_id="1ed5",
        device_id="0100",
        class_filter="0x12",
        devices=pcie_devices,
    )

    assert captured == {
        "vendor_id": "1ed5",
        "device_id": "0100",
        "class_filter": "0x12",
        "devices": pcie_devices,
    }


def test_get_linux_npu_driver_firmware_info_skips_when_filtered_npus_empty(
    monkeypatch,
) -> None:
    monkeypatch.setattr(static_info.platform, "system", lambda: "Linux")

    def fail_run_command(_command):
        raise AssertionError("mobilint-cli status should not be called")

    monkeypatch.setattr(static_info, "run_command", fail_run_command)

    assert get_linux_npu_driver_firmware_info(npu_devices=[]) == {}


def test_get_linux_npu_driver_firmware_info_limits_metadata_to_filtered_npus(
    monkeypatch,
) -> None:
    status_output = """
Drivers - Aries: 1.8.1 Regulus: 1.12.0 |
| 0 Aries(Board-A) | 42 C 2.0.1 |
| 0 42 C 2.0.1 |
| 1 Aries(Board-B) | 43 C 2.0.2 |
| 1 43 C 2.0.2 |
"""
    monkeypatch.setattr(static_info.platform, "system", lambda: "Linux")
    monkeypatch.setattr(static_info, "run_command", lambda _command: status_output)

    info = get_linux_npu_driver_firmware_info(npu_devices=[{"dev_no": 1}])

    hardware = cast(dict[str, object], info["hardware"])
    assert hardware["npus"] == [
        {"dev_no": 1, "board_name": "Board-B", "firmware": {"version": "2.0.2"}}
    ]
    assert info["inference"] == {
        "npu_driver_version": "1.8.1",
        "driver": {"aries_version": "1.8.1", "regulus_version": "1.12.0"},
    }


def test_parse_mobilint_status_query_static_info_assigns_mla400_card_ids_by_group() -> None:
    def mla400_device(dev_no: int) -> str:
        return f"""/dev/aries{dev_no}
    Product                   : Aries
    Firmware
        Version               : 1.2.5 (Rev: 0)
    Power
        Total                 : 0.00 W
        NPU                   : 4.00 W
        GOLDFINGER            : 0.00 W
    PCI Express
        Vendor ID             : 0x209F
        Device ID             : 0x0
        Sub Vendor ID         : 0x402
        Sub Device ID         : 0x108B
"""

    status_output = """Timestamp                     : 2026-05-07 15:29:48
Driver Version (Aries)        : 1.12.0 (Rev: 1)
Driver Version (Regulus)      : N/A
Connected NPUs                : 8
""" + "".join(mla400_device(dev_no) for dev_no in range(8))

    info = parse_mobilint_status_static_info(status_output)

    hardware = cast(dict[str, object], info["hardware"])
    npus = cast(list[dict[str, object]], hardware["npus"])
    assert [npu["dev_no"] for npu in npus] == list(range(8))
    assert [npu["card_model"] for npu in npus] == ["MLA400"] * 8
    assert [npu["card_id"] for npu in npus] == [0, 0, 0, 0, 1, 1, 1, 1]


def test_parse_mobilint_status_query_static_info_keeps_mla400_offset_card_id() -> None:
    def mla400_device(dev_no: int) -> str:
        return f"""/dev/aries{dev_no}
    Product                   : Aries
    Power
        Total                 : 0.00 W
        NPU                   : 4.00 W
        GOLDFINGER            : 0.00 W
    PCI Express
        Vendor ID             : 0x209F
        Device ID             : 0x0
        Sub Vendor ID         : 0x402
        Sub Device ID         : 0x108B
"""

    status_output = """Timestamp                     : 2026-05-07 15:29:48
Driver Version (Aries)        : 1.12.0 (Rev: 1)
Driver Version (Regulus)      : N/A
Connected NPUs                : 4
""" + "".join(mla400_device(dev_no) for dev_no in range(4, 8))

    info = parse_mobilint_status_static_info(status_output)

    hardware = cast(dict[str, object], info["hardware"])
    npus = cast(list[dict[str, object]], hardware["npus"])
    assert [npu["dev_no"] for npu in npus] == [4, 5, 6, 7]
    assert [npu["card_id"] for npu in npus] == [1, 1, 1, 1]
