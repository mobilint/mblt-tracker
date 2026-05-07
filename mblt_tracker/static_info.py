from __future__ import annotations

import json
import os
import platform
import re
import shlex
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import psutil

from ._types import (
    STATIC_INFO_CHILD_SCHEMAS,
    CollectOutput,
    CpuPowerPolicy,
    DimmInfo,
)


def get_host_static_info(
    sudo_password: str | None = None,
    sudo_password_provider: Callable[[], str] | None = None,
) -> CollectOutput:
    """Collect best-effort host CPU, DRAM, and OS static information."""
    virtual_memory = psutil.virtual_memory()
    info: CollectOutput = {
        "hardware": {
            "cpu": {
                "architecture": platform.machine(),
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "model_name": None,
                "vendor": None,
            },
            "dram": {
                "total_bytes": virtual_memory.total,
                "available_bytes": virtual_memory.available,
            },
        },
        "inference": {
            "cpu": {
                "governor": None,
                "power_plan": None,
                "min_processor_state_pct": None,
                "max_processor_state_pct": None,
            },
            "cuda": {"version": None},
            "os": {
                "name": platform.system(),
                "version": platform.version(),
                "kernel_version": platform.release(),
            },
            "qbcompiler": {"version": None},
            "qbruntime": {"version": None},
        },
    }

    if platform.system() == "Windows":
        model_name, vendor = _windows_cpu_identity()
    else:
        model_name, vendor = _linux_cpu_identity()
    if model_name is None:
        model_name = platform.processor() or platform.uname().processor
    cpu_info = info["hardware"]["cpu"]
    if model_name:
        cpu_info["model_name"] = model_name
    if vendor:
        cpu_info["vendor"] = vendor

    if platform.system() == "Windows":
        dimms = _read_dram_dimms_windows()
    elif platform.system() == "Linux":
        dimms = _read_dram_dimms_linux(
            sudo_password=sudo_password,
            sudo_password_provider=sudo_password_provider,
        )
    else:
        dimms = []
    if dimms:
        dram = info["hardware"]["dram"]
        dram["dimms"] = cast(list[DimmInfo], dimms)
        theoretical_bandwidth_gbps = _calculate_theoretical_bandwidth_gbps(dimms)
        if theoretical_bandwidth_gbps is not None:
            dram["theoretical_bandwidth_gbps"] = theoretical_bandwidth_gbps
    elif platform.system() == "Linux":
        info["hardware"]["dram"]["dimms_collection_note"] = (
            "Physical DIMM metadata requires dmidecode access. Run "
            "`mblt-tracker collect` in an interactive terminal and enter the "
            "sudo password, or install/configure dmidecode."
        )

    if platform.system() == "Linux":
        os_release = _read_os_release()
        pretty_name = os_release.get("PRETTY_NAME")
        if pretty_name:
            info["inference"]["os"]["version"] = pretty_name
        info["inference"]["cpu"] = get_cpu_power_policy()
    elif platform.system() == "Windows":
        info["inference"]["cpu"] = get_cpu_power_policy()

    info["inference"]["cuda"] = {"version": _get_cuda_version() or "not_found"}
    info["inference"]["qbruntime"] = {
        "version": _get_python_package_version("qbruntime") or "not_installed"
    }
    info["inference"]["qbcompiler"] = {
        "version": _get_python_package_version("qbcompiler") or "not_installed"
    }

    return cast(CollectOutput, _clean_typed_dict(info, CollectOutput))


def get_cpu_power_policy() -> CpuPowerPolicy:
    """Return OS-independent CPU power policy fields.

    Linux exposes CPU frequency governors while Windows exposes power plans and
    processor state limits. All fields are returned on every OS so type checking
    and JSON consumers can rely on the same shape.
    """
    policy: CpuPowerPolicy = {
        "governor": None,
        "power_plan": None,
        "min_processor_state_pct": None,
        "max_processor_state_pct": None,
    }
    if platform.system() == "Linux":
        policy["governor"] = get_cpu_governor()
    elif platform.system() == "Windows":
        policy.update(get_windows_power_policy())
    return policy


def _get_cuda_version() -> str | None:
    """Return the CUDA version visible to this Python environment.

    Prefer Python-level inspection via PyTorch when available. If PyTorch is not
    installed or does not report CUDA, fall back to the local ``nvcc`` command.
    """
    torch_cuda_version = _get_torch_cuda_version()
    if torch_cuda_version is not None:
        return torch_cuda_version

    nvcc_output = run_command(["nvcc", "--version"])
    if nvcc_output is None:
        return None
    return _parse_nvcc_cuda_version(nvcc_output)


def get_nvml_gpu_static_info(
    pcie_devices: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Collect NVIDIA GPU static information through NVML.

    NVML is available on both Linux and Windows when NVIDIA drivers are
    installed. NVML is the source of truth for the GPU list, while optional
    PCIe data enriches each NVML GPU entry with link and ID information.
    """
    try:
        import pynvml
    except ImportError:
        _warn_nvml_unavailable()
        return {}

    try:
        pynvml.nvmlInit()
    except Exception:
        _warn_nvml_unavailable()
        return {}

    try:
        device_count = pynvml.nvmlDeviceGetCount()
        driver_version = _decode_nvml_string(pynvml.nvmlSystemGetDriverVersion())
        cuda_driver_version = _format_nvml_cuda_driver_version(
            pynvml.nvmlSystemGetCudaDriverVersion()
        )
        pcie_gpu_candidates = [
            device for device in pcie_devices or [] if _is_likely_gpu_device(device)
        ]
        gpus = []
        for device_index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            name = _decode_nvml_string(pynvml.nvmlDeviceGetName(handle))
            bus_address = _get_nvml_pci_bus_address(pynvml, handle)
            nvml_static_metadata = _get_nvml_device_static_metadata(pynvml, handle)
            matched_pcie = _find_matching_pcie_gpu(
                pcie_gpu_candidates,
                bus_address,
                device_index,
            )
            gpu_entry = dict(matched_pcie) if matched_pcie is not None else {}
            _remove_os_pcie_link_fields(gpu_entry)
            gpu_entry["name"] = name
            gpu_entry["driver_version"] = driver_version
            gpu_entry.update(nvml_static_metadata)
            if bus_address is not None:
                gpu_entry["bus_address"] = bus_address
            gpus.append(_format_pcie_device(gpu_entry, device_index))
    except Exception:
        return {}
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    info: dict[str, object] = {
        "hardware": {
            "gpus": gpus,
        },
        "inference": {
            "gpu": {
                "driver": {"version": driver_version},
                "cuda_driver": {"version": cuda_driver_version},
            }
        },
    }
    return info


def _warn_nvml_unavailable() -> None:
    print(
        "Warning: NVML not available. GPU information will not be collected.",
        file=sys.stderr,
    )


def _get_nvml_device_static_metadata(
    pynvml_module: object,
    handle: object,
) -> dict[str, object]:
    """Return best-effort per-GPU NVML static metadata."""
    metadata: dict[str, object] = {}
    nvml = cast(Any, pynvml_module)

    try:
        memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
    except Exception:
        pass
    else:
        total = _to_int(getattr(memory_info, "total", None))
        if total is not None:
            metadata["memory_total_bytes"] = total

    try:
        architecture = nvml.nvmlDeviceGetArchitecture(handle)
    except Exception:
        pass
    else:
        architecture_name = _nvml_architecture_to_name(architecture)
        if architecture_name is not None:
            metadata["architecture"] = architecture_name

    try:
        current_link_generation = nvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
    except Exception:
        pass
    else:
        generation = _to_int(current_link_generation)
        if generation is not None and generation > 0:
            metadata["nvml_link_generation"] = f"Gen{generation}"

    try:
        current_link_width = nvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
    except Exception:
        pass
    else:
        width = _to_int(current_link_width)
        if width is not None and width > 0:
            metadata["nvml_lane_width"] = f"x{width}"

    return metadata


def _nvml_architecture_to_name(value: object) -> str | None:
    """Return a stable NVIDIA architecture name for a pynvml architecture enum."""
    architecture = _to_int(value)
    if architecture is None:
        return None
    architecture_names = {
        2: "Kepler",
        3: "Maxwell",
        4: "Pascal",
        5: "Volta",
        6: "Turing",
        7: "Ampere",
        8: "Ada Lovelace",
        9: "Hopper",
        10: "Blackwell",
    }
    return architecture_names.get(architecture, f"Unknown ({architecture})")


def _remove_os_pcie_link_fields(device: dict[str, object]) -> None:
    """Remove OS-level PCIe link fields when NVML is used as GPU source of truth."""
    for key in (
        "current_link_speed",
        "current_link_width",
        "max_link_speed",
        "max_link_width",
        "link_generation",
        "lane_width",
    ):
        device.pop(key, None)


def _find_matching_pcie_gpu(
    pcie_gpu_candidates: list[dict[str, object]],
    bus_address: str | None,
    device_index: int,
) -> dict[str, object] | None:
    if bus_address is not None:
        for candidate in pcie_gpu_candidates:
            candidate_bus_address = candidate.get("bus_address")
            if candidate_bus_address is None:
                continue
            if str(candidate_bus_address).lower() == bus_address.lower():
                return candidate

    nvidia_candidates = [
        candidate
        for candidate in pcie_gpu_candidates
        if _normalize_hex(str(candidate.get("vendor_id", ""))) == "10de"
        or str(candidate.get("bus_address", "")).upper().startswith("PCI\\VEN_10DE")
        or "nvidia" in str(candidate.get("manufacturer", "")).lower()
        or "nvidia" in str(candidate.get("name", "")).lower()
        or "nvidia" in str(candidate.get("pnp_device_id", "")).lower()
    ]
    if device_index < len(nvidia_candidates):
        return nvidia_candidates[device_index]
    return None


def _decode_nvml_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _format_nvml_cuda_driver_version(value: object) -> str | None:
    version = _to_int(value)
    if version is None or version <= 0:
        return None
    major = version // 1000
    minor = (version % 1000) // 10
    return f"{major}.{minor}"


def _get_nvml_pci_bus_address(pynvml_module: object, handle: object) -> str | None:
    try:
        pci_info = cast(Any, pynvml_module).nvmlDeviceGetPciInfo(handle)
    except Exception:
        return None
    bus_id = _decode_nvml_string(getattr(pci_info, "busId", "")).strip().lower()
    if not bus_id:
        return None
    # NVML often returns 8-digit PCI domains (e.g. 00000000:17:00.0), while
    # Linux sysfs commonly uses 4-digit domains (0000:17:00.0).
    match = re.match(r"([0-9a-f]{8}):(.*)", bus_id, re.IGNORECASE)
    if match is not None:
        return f"{match.group(1)[-4:]}:{match.group(2)}"
    return bus_id


def _get_torch_cuda_version() -> str | None:
    try:
        import torch
    except ImportError:
        return None

    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    if not isinstance(cuda_version, str):
        return None
    cuda_version = cuda_version.strip()
    return cuda_version or None


def _parse_nvcc_cuda_version(output: str) -> str | None:
    match = re.search(r"release\s+([0-9]+(?:\.[0-9]+)+)", output, re.IGNORECASE)
    if match is not None:
        return match.group(1)

    match = re.search(r"V([0-9]+(?:\.[0-9]+)+)", output)
    if match is not None:
        return match.group(1)
    return None


def _get_python_package_version(module_name: str) -> str | None:
    """Return a best-effort Python package version for ``module_name``.

    Some runtime packages expose ``__version__`` while others rely on installed
    package metadata, so inspect both without adding hard dependencies. Import
    failures can happen even when a distribution is installed, for example when
    native runtime dependencies are missing, so always fall back to metadata.
    """
    version: object = None
    try:
        module = __import__(module_name)
    except Exception:
        pass
    else:
        version = getattr(module, "__version__", None)
    if isinstance(version, str) and version.strip():
        return version.strip()

    package_names = {
        "qbruntime": "mobilint-qb-runtime",
        "qbcompiler": "qbcompiler",
    }
    try:
        return metadata.version(package_names.get(module_name, module_name))
    except metadata.PackageNotFoundError:
        return None


def get_cpu_governor() -> str | None:
    """Return CPU frequency governor values as a compact string on Linux."""
    governors = []
    cpu_root = Path("/sys/devices/system/cpu")
    for path in sorted(cpu_root.glob("cpu[0-9]*/cpufreq/scaling_governor")):
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            governors.append(value)
    if not governors:
        return None
    unique = sorted(set(governors))
    if len(unique) == 1:
        return unique[0]
    return ",".join(unique)


def get_windows_power_policy() -> CpuPowerPolicy:
    """Return active Windows power policy information.

    Windows does not expose Linux-style CPU frequency governors. Instead, the
    active power plan and processor state limits are returned using Windows
    native terminology.
    """
    active_scheme_output = run_command(["powercfg", "/getactivescheme"])
    if active_scheme_output is None:
        return {
            "governor": None,
            "power_plan": None,
            "min_processor_state_pct": None,
            "max_processor_state_pct": None,
        }

    scheme_guid, power_plan = _parse_windows_active_power_scheme(
        active_scheme_output
    )
    if scheme_guid is None and power_plan is None:
        return {
            "governor": None,
            "power_plan": None,
            "min_processor_state_pct": None,
            "max_processor_state_pct": None,
        }

    policy: CpuPowerPolicy = {
        "governor": None,
        "power_plan": None,
        "min_processor_state_pct": None,
        "max_processor_state_pct": None,
    }
    if power_plan is not None:
        policy["power_plan"] = _normalize_windows_power_plan_name(
            scheme_guid,
            power_plan,
        )

    if scheme_guid is not None:
        processor_states = _read_windows_processor_power_states(scheme_guid)
        if "min_processor_state_pct" in processor_states:
            policy["min_processor_state_pct"] = processor_states["min_processor_state_pct"]
        if "max_processor_state_pct" in processor_states:
            policy["max_processor_state_pct"] = processor_states["max_processor_state_pct"]

    return policy


def _normalize_windows_power_plan_name(
    scheme_guid: str | None,
    power_plan: str,
) -> str:
    """Return a stable English name for built-in Windows power plans.

    ``powercfg`` localizes plan names based on the OS display language. The
    built-in plan GUIDs are stable, so use them to keep tracker output
    language-independent while preserving custom plan names as a fallback.
    """
    if scheme_guid is None:
        return power_plan

    built_in_power_plans = {
        "381b4222-f694-41f0-9685-ff5bb260df2e": "Balanced",
        "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "High performance",
        "a1841308-3541-4fab-bc81-f71556f20b4a": "Power saver",
        "e9a42b02-d5df-448d-aa00-03f14749eb61": "Ultimate Performance",
    }
    return built_in_power_plans.get(scheme_guid.lower(), power_plan)


def _parse_windows_active_power_scheme(
    output: str,
) -> tuple[str | None, str | None]:
    """Parse ``powercfg /getactivescheme`` output.

    The surrounding text is localized by Windows, but the GUID and plan name in
    parentheses are stable enough to parse across locales.
    """
    guid_match = re.search(
        r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
        output,
        re.IGNORECASE,
    )
    name_match = re.search(r"\(([^()]*)\)\s*$", output.strip())

    scheme_guid = guid_match.group(1).lower() if guid_match is not None else None
    power_plan = None
    if name_match is not None:
        value = name_match.group(1).strip()
        power_plan = value or None
    return scheme_guid, power_plan


def _read_windows_processor_power_states(scheme_guid: str) -> dict[str, int]:
    processor_subgroup_guid = "54533251-82be-4824-96c1-47b60b740d00"
    min_processor_state_guid = "893dee8e-2bef-41e0-89c6-b55d0929964c"
    max_processor_state_guid = "bc5038f7-23e0-4960-96da-33abaf5935ec"

    values: dict[str, int] = {}
    min_value = _read_windows_power_setting_ac_value(
        scheme_guid,
        processor_subgroup_guid,
        min_processor_state_guid,
    )
    max_value = _read_windows_power_setting_ac_value(
        scheme_guid,
        processor_subgroup_guid,
        max_processor_state_guid,
    )
    if min_value is not None:
        values["min_processor_state_pct"] = min_value
    if max_value is not None:
        values["max_processor_state_pct"] = max_value
    return values


def _read_windows_power_setting_ac_value(
    scheme_guid: str,
    subgroup_guid: str,
    setting_guid: str,
) -> int | None:
    output = run_command(
        ["powercfg", "/query", scheme_guid, subgroup_guid, setting_guid]
    )
    if output is None:
        return None
    return _parse_windows_power_setting_ac_value(output)


def _parse_windows_power_setting_ac_value(output: str) -> int | None:
    """Parse an AC power setting value from ``powercfg /query`` output."""
    match = re.search(
        r"(?:Current AC Power Setting Index|현재\s*AC\s*전원\s*설정\s*인덱스)\s*:\s*0x([0-9a-f]+)",
        output,
        re.IGNORECASE,
    )
    if match is None:
        match = re.search(r"AC[^\n\r]*:\s*0x([0-9a-f]+)", output, re.IGNORECASE)
    if match is None:
        return None

    try:
        return int(match.group(1), 16)
    except ValueError:
        return None


def _read_dram_dimms_windows() -> list[dict[str, object]]:
    """Collect physical memory module information from Windows CIM."""
    output = run_command(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "Get-CimInstance Win32_PhysicalMemory | "
            "Select-Object Manufacturer,PartNumber,SerialNumber,Capacity,Speed,"
            "ConfiguredClockSpeed,DataWidth,TotalWidth,SMBIOSMemoryType | "
            "ConvertTo-Json -Depth 3",
        ]
    )
    if output is None:
        return []

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict):
        entries = [parsed]
    elif isinstance(parsed, list):
        entries = [entry for entry in parsed if isinstance(entry, dict)]
    else:
        return []

    dimms = []
    for entry in entries:
        dimm = {
            "manufacturer": _clean_dram_string(entry.get("Manufacturer")),
            "part_number": _clean_dram_string(entry.get("PartNumber")),
            "serial_number": _clean_dram_string(entry.get("SerialNumber")),
            "capacity_bytes": _to_int(entry.get("Capacity")),
            "speed_mhz": _to_int(entry.get("Speed")),
            "configured_speed_mhz": _to_int(entry.get("ConfiguredClockSpeed")),
            "data_width_bits": _to_int(entry.get("DataWidth")),
            "total_width_bits": _to_int(entry.get("TotalWidth")),
            "type": _windows_memory_type_to_text(entry.get("SMBIOSMemoryType")),
        }
        cleaned = {key: value for key, value in dimm.items() if value is not None}
        if cleaned:
            dimms.append(cleaned)
    return dimms


def _read_dram_dimms_linux(
    sudo_password: str | None = None,
    sudo_password_provider: Callable[[], str] | None = None,
) -> list[dict[str, object]]:
    """Collect physical memory module information from Linux dmidecode.

    ``dmidecode`` normally requires root privileges. Try the direct command
    first, then the best-effort non-interactive sudo fallback before asking for
    any interactive password. When a sudo password is supplied, pass it to
    ``sudo -S`` so callers can collect richer output without requiring users to
    run the whole CLI through sudo. If all attempts fail, callers add a
    permission note to the public output.
    """
    output = run_command(["dmidecode", "-t", "memory"])
    if output is None:
        output = run_command(["sudo", "-n", "dmidecode", "-t", "memory"])
    if output is None:
        if sudo_password is None and sudo_password_provider is not None:
            sudo_password = sudo_password_provider()
        if sudo_password is not None:
            output = run_command_with_input(
                ["sudo", "-S", "-p", "", "dmidecode", "-t", "memory"],
                input_text=f"{sudo_password}\n",
                timeout=30,
            )
    if output is None:
        return []
    return _parse_linux_dmidecode_memory(output)


def get_linux_npu_driver_firmware_info(
    npu_devices: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Collect Linux NPU driver/firmware metadata from mobilint-cli status."""
    if platform.system() != "Linux":
        return {}
    if npu_devices is not None and not npu_devices:
        return {}
    status_output = run_command(["mobilint-cli", "status"])
    if not status_output:
        return {}
    info = parse_mobilint_status_static_info(status_output)
    if npu_devices is not None:
        _filter_npu_metadata_to_selected_devices(info, npu_devices)
    return info


def _filter_npu_metadata_to_selected_devices(
    info: dict[str, object], selected_devices: list[dict[str, object]]
) -> None:
    hardware = info.get("hardware")
    if not isinstance(hardware, dict):
        return
    npus = hardware.get("npus")
    if not isinstance(npus, list):
        return
    selected_npus = [
        metadata
        for selected_device in selected_devices
        for metadata in npus
        if isinstance(metadata, dict)
        and _npu_metadata_matches_selected_device(metadata, selected_device)
    ]
    hardware["npus"] = selected_npus


def _npu_metadata_matches_selected_device(
    metadata: dict[str, object], selected_device: dict[str, object]
) -> bool:
    for key in ("dev_no", "bus_address", "pnp_device_id"):
        metadata_value = metadata.get(key)
        selected_value = selected_device.get(key)
        if metadata_value is None or selected_value is None:
            continue
        if key == "dev_no":
            if _to_int(metadata_value) == _to_int(selected_value):
                return True
        elif str(metadata_value).lower() == str(selected_value).lower():
            return True
    return False


def parse_mobilint_status_static_info(status_output: str) -> dict[str, object]:
    """Parse static NPU fields from ``mobilint-cli status`` table output."""
    info: dict[str, object] = {}
    driver_match = re.search(
        r"Drivers\s*-\s*Aries:\s*([^\s]+)\s+Regulus:\s*([^\s|]+)",
        status_output,
    )
    if driver_match is not None:
        inference = info.setdefault("inference", {})
        if isinstance(inference, dict):
            inference["npu_driver_version"] = driver_match.group(1)
            inference["driver"] = {
                "aries_version": driver_match.group(1),
                "regulus_version": driver_match.group(2),
            }
    device_matches = re.findall(
        r"\|\s*(\d+)\s+([A-Za-z0-9_-]+)\(([^)]+)\).*?\|", status_output
    )
    firmware_matches = re.findall(
        r"\|\s*\d+\s+[0-9]+(?:\.[0-9]+)?\s*C\s+([^\s|]+)", status_output,
    )
    if device_matches:
        devices = []
        for idx, (device_index, _product, board_name) in enumerate(device_matches):
            device: dict[str, object] = {
                "dev_no": int(device_index),
                "board_name": board_name,
            }
            if idx < len(firmware_matches):
                device["firmware"] = {"version": firmware_matches[idx]}
            devices.append(device)
        hardware: dict[str, object] = {}
        info["hardware"] = hardware
        hardware["npus"] = devices
    return info


def _parse_linux_dmidecode_memory(output: str) -> list[dict[str, object]]:
    dimms = []
    for section in re.split(r"\nHandle\s+", output):
        if "Memory Device" not in section:
            continue

        fields = _parse_dmidecode_section_fields(section)
        if fields.get("Size") in {None, "No Module Installed"}:
            continue

        dimm = {
            "manufacturer": _clean_dram_string(fields.get("Manufacturer")),
            "part_number": _clean_dram_string(fields.get("Part Number")),
            "serial_number": _clean_dram_string(fields.get("Serial Number")),
            "capacity_bytes": _parse_memory_size_to_bytes(fields.get("Size")),
            "speed_mhz": _parse_memory_speed_to_mhz(fields.get("Speed")),
            "configured_speed_mhz": _parse_memory_speed_to_mhz(
                fields.get("Configured Memory Speed")
            ),
            "data_width_bits": _parse_memory_width_to_bits(fields.get("Data Width")),
            "total_width_bits": _parse_memory_width_to_bits(fields.get("Total Width")),
            "type": _clean_dram_string(fields.get("Type")),
        }
        cleaned = {key: value for key, value in dimm.items() if value is not None}
        if cleaned:
            dimms.append(cleaned)
    return dimms


def _parse_dmidecode_section_fields(section: str) -> dict[str, str]:
    fields = {}
    for line in section.splitlines():
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if key:
            fields[key] = value
    return fields


def _calculate_theoretical_bandwidth_gbps(
    dimms: Sequence[Mapping[str, object]],
) -> float | None:
    """Estimate peak DRAM bandwidth in GB/s from DIMM speed and data width."""
    bandwidth_gbps = 0.0
    for dimm in dimms:
        speed = _to_int(dimm.get("configured_speed_mhz")) or _to_int(
            dimm.get("speed_mhz")
        )
        data_width_bits = _to_int(dimm.get("data_width_bits"))
        if speed is None or data_width_bits is None:
            continue
        bandwidth_gbps += speed * (data_width_bits / 8) / 1000

    if bandwidth_gbps <= 0:
        return None
    return round(bandwidth_gbps, 2)


def _windows_memory_type_to_text(value: object) -> str | None:
    memory_type = _to_int(value)
    if memory_type is None:
        return None
    memory_types = {
        20: "DDR",
        21: "DDR2",
        24: "DDR3",
        26: "DDR4",
        34: "DDR5",
    }
    return memory_types.get(memory_type)


def _parse_memory_size_to_bytes(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"([0-9]+)\s*([KMGT]B)", value, re.IGNORECASE)
    if match is None:
        return None
    amount = int(match.group(1))
    unit = match.group(2).upper()
    multipliers = {
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }
    return amount * multipliers[unit]


def _parse_memory_speed_to_mhz(value: str | None) -> int | None:
    if value is None or value.strip().lower() == "unknown":
        return None
    match = re.search(r"([0-9]+)", value)
    if match is None:
        return None
    return int(match.group(1))


def _parse_memory_width_to_bits(value: str | None) -> int | None:
    if value is None or value.strip().lower() == "unknown":
        return None
    match = re.search(r"([0-9]+)", value)
    if match is None:
        return None
    return int(match.group(1))


def _clean_dram_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value.lower() in {
        "unknown",
        "not specified",
        "not provided",
        "none",
        "to be filled by o.e.m.",
    }:
        return None
    return value


def _to_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None
    return None


def get_pcie_static_info(
    vendor_id: str | None = None,
    device_id: str | None = None,
    class_filter: str | None = None,
    include_all_devices: bool = False,
    devices: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Collect best-effort PCIe information.

    Args:
        vendor_id: Optional hexadecimal vendor id, with or without ``0x``.
        device_id: Optional hexadecimal device id, with or without ``0x``.
        class_filter: Optional PCI class prefix, e.g. ``0x12``.
        include_all_devices: Include all PCIe devices in ``hardware.pcie_devices``.
            When false, omit the raw device list and expose only categorized
            GPU/NPU lists.

    Returns:
        Nested dictionary containing categorized PCIe devices under top-level
        hardware keys. Raw ``hardware.pcie_devices`` is included only when
        requested.
    """
    if devices is None:
        devices = get_all_pcie_devices()
    hardware_info: dict[str, object] = {}
    if include_all_devices:
        hardware_info["pcie_devices"] = devices
    npus = _find_all_npu_devices(devices, vendor_id, device_id, class_filter)
    inference_info: dict[str, object] = {}
    if npus:
        npu_device_indices = _get_npu_device_indices(devices)
        formatted_npus = [
            _format_pcie_device(device, npu_device_indices.get(id(device), dev_no))
            for dev_no, device in enumerate(npus)
        ]
        npu_driver_version = _pop_npu_driver_version(formatted_npus)
        if npu_driver_version is not None:
            inference_info["npu_driver_version"] = npu_driver_version
        hardware_info["npus"] = formatted_npus
    info: dict[str, object] = {}
    if hardware_info:
        info["hardware"] = hardware_info
    if inference_info:
        info["inference"] = inference_info
    return info


def _get_npu_device_indices(devices: list[dict[str, object]]) -> dict[int, int]:
    """Return stable unfiltered NPU indices keyed by object identity."""
    return {
        id(device): index
        for index, device in enumerate(
            device for device in devices if _is_likely_npu_device(device)
        )
    }


def get_all_pcie_devices() -> list[dict[str, object]]:
    """Collect raw PCIe devices for NPU discovery and NVML GPU enrichment."""
    sysfs_override = os.environ.get("MBLT_TRACKER_PCI_SYSFS")
    if sysfs_override is not None:
        return _read_pcie_devices(Path(sysfs_override))
    if platform.system() == "Windows":
        return _read_pcie_devices_windows()
    return _read_pcie_devices(Path("/sys/bus/pci/devices"))


def _pop_npu_driver_version(npus: list[dict[str, object]]) -> str | None:
    npu_driver_version = None
    for npu in npus:
        driver_version = npu.pop("driver_version", None)
        if npu_driver_version is None and isinstance(driver_version, str):
            npu_driver_version = driver_version
    return npu_driver_version


def _read_pcie_devices_windows() -> list[dict[str, object]]:
    """Collect PCI device information from Windows PnP/CIM.

    Windows does not expose Linux-style PCI sysfs files. Instead, read PCI PnP
    entities through PowerShell/CIM and parse IDs such as
    ``PCI\\VEN_1ED5&DEV_0100&SUBSYS_...`` into the same normalized fields used
    by the Linux reader.
    """
    output = run_command(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { $_.PNPDeviceID -like 'PCI\\*' } | "
            "Select-Object Name,PNPDeviceID,DeviceID,HardwareID,CompatibleID,Manufacturer,Status | "
            "ConvertTo-Json -Depth 3",
        ]
    )
    if output is None:
        return []

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict):
        entities = [parsed]
    elif isinstance(parsed, list):
        entities = [entity for entity in parsed if isinstance(entity, dict)]
    else:
        return []

    devices = []
    for entity in entities:
        pnp_device_id = str(entity.get("PNPDeviceID") or entity.get("DeviceID") or "")
        device = _parse_windows_pci_id(
            pnp_device_id,
            _windows_pci_auxiliary_ids(entity),
        )
        if device is None:
            continue

        name = entity.get("Name")
        manufacturer = entity.get("Manufacturer")
        status = entity.get("Status")
        device.update(
            {
                "bus_address": pnp_device_id,
                "pnp_device_id": pnp_device_id,
                "name": name,
                "manufacturer": manufacturer,
                "status": status,
            }
        )
        devices.append({key: value for key, value in device.items() if value})

    relevant_instance_ids = [
        str(device["pnp_device_id"])
        for device in devices
        if device.get("pnp_device_id") is not None and _is_relevant_pcie_device(device)
    ]
    link_properties = _read_windows_pci_link_properties(relevant_instance_ids)
    for device in devices:
        pnp_device_id = str(device.get("pnp_device_id", ""))
        device.update(link_properties.get(pnp_device_id.upper(), {}))
    return devices


def _read_windows_pci_link_properties(
    instance_ids: list[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Read PCIe/PnP properties for selected Windows PCI devices.

    Querying every PCI device can be slow on some systems, so production code
    passes only relevant GPU/NPU/accelerator instance IDs. ``None`` keeps the
    previous all-PCI behavior for tests and fallback callers.
    """
    if instance_ids is None:
        device_expression = "Get-PnpDevice | Where-Object { $_.InstanceId -like 'PCI\\*' }"
    elif not instance_ids:
        return {}
    else:
        quoted_ids = ",".join(
            "'" + instance_id.replace("'", "''") + "'" for instance_id in instance_ids
        )
        device_expression = f"$instanceIds = @({quoted_ids}); $instanceIds | ForEach-Object {{ Get-PnpDevice -InstanceId $_ -ErrorAction SilentlyContinue }}"

    output = run_command_with_timeout(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "$keys = @("
            "'DEVPKEY_PciDevice_CurrentLinkSpeed',"
            "'DEVPKEY_PciDevice_CurrentLinkWidth',"
            "'DEVPKEY_PciDevice_MaxLinkSpeed',"
            "'DEVPKEY_PciDevice_MaxLinkWidth',"
            "'DEVPKEY_Device_DriverVersion',"
            "'DEVPKEY_Device_DriverDate',"
            "'DEVPKEY_Device_DriverDesc',"
            "'DEVPKEY_Device_DriverProvider',"
            "'DEVPKEY_Device_FirmwareVersion',"
            "'DEVPKEY_Device_FirmwareRevision'"
            "); "
            f"{device_expression} | "
            "ForEach-Object { "
            "$item = [ordered]@{ InstanceId = $_.InstanceId }; "
            "foreach ($key in $keys) { "
            "$prop = Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName $key "
            "-ErrorAction SilentlyContinue; "
            "if ($null -ne $prop) { $item[$key] = $prop.Data } "
            "}; "
            "[pscustomobject]$item "
            "} | ConvertTo-Json -Depth 3",
        ],
        timeout=20,
    )
    if output is None:
        return {}

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return {}

    if isinstance(parsed, dict):
        entries = [parsed]
    elif isinstance(parsed, list):
        entries = [entry for entry in parsed if isinstance(entry, dict)]
    else:
        return {}

    properties: dict[str, dict[str, object]] = {}
    for entry in entries:
        instance_id = entry.get("InstanceId")
        if not isinstance(instance_id, str):
            continue

        device_properties: dict[str, object] = {}
        current_speed = _windows_link_speed_to_text(
            entry.get("DEVPKEY_PciDevice_CurrentLinkSpeed")
        )
        max_speed = _windows_link_speed_to_text(
            entry.get("DEVPKEY_PciDevice_MaxLinkSpeed")
        )
        current_width = entry.get("DEVPKEY_PciDevice_CurrentLinkWidth")
        max_width = entry.get("DEVPKEY_PciDevice_MaxLinkWidth")

        if current_speed is not None:
            device_properties["current_link_speed"] = current_speed
        if max_speed is not None:
            device_properties["max_link_speed"] = max_speed
        if current_width is not None:
            device_properties["current_link_width"] = str(current_width)
        if max_width is not None:
            device_properties["max_link_width"] = str(max_width)

        driver_version = _clean_windows_device_property(
            entry.get("DEVPKEY_Device_DriverVersion")
        )
        driver_date = _clean_windows_device_property(
            entry.get("DEVPKEY_Device_DriverDate")
        )
        driver_description = _clean_windows_device_property(
            entry.get("DEVPKEY_Device_DriverDesc")
        )
        driver_provider = _clean_windows_device_property(
            entry.get("DEVPKEY_Device_DriverProvider")
        )
        firmware_version = _clean_windows_device_property(
            entry.get("DEVPKEY_Device_FirmwareVersion")
        )
        firmware_revision = _clean_windows_device_property(
            entry.get("DEVPKEY_Device_FirmwareRevision")
        )

        if driver_version is not None:
            device_properties["driver_version"] = driver_version
        if driver_date is not None:
            device_properties["driver_date"] = driver_date
        if driver_description is not None:
            device_properties["driver_description"] = driver_description
        if driver_provider is not None:
            device_properties["driver_provider"] = driver_provider
        firmware: dict[str, object] = {}
        if firmware_version is not None:
            firmware["version"] = firmware_version
        elif firmware_revision is not None:
            firmware["version"] = firmware_revision
        if firmware:
            device_properties["firmware"] = firmware

        if device_properties:
            properties[instance_id.upper()] = device_properties
    return properties


def _windows_pci_auxiliary_ids(entity: Mapping[str, object]) -> list[str]:
    """Return auxiliary Windows PCI IDs that may contain class code metadata."""
    auxiliary_ids = []
    for key in ("HardwareID", "CompatibleID"):
        value = entity.get(key)
        if isinstance(value, str):
            auxiliary_ids.append(value)
        elif isinstance(value, list):
            auxiliary_ids.extend(item for item in value if isinstance(item, str))
    return auxiliary_ids


def get_windows_npu_driver_firmware_info(
    vendor_ids: tuple[str, ...] = ("1ed5", "209f"),
    vendor_id: str | None = None,
    device_id: str | None = None,
    class_filter: str | None = None,
    devices: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Collect Windows NPU driver metadata from PnP properties.

    This avoids calling ``mobilint-cli``. Driver metadata is exposed by Windows
    through standard PnP device properties and is normalized to
    ``inference.npu_driver_version``.
    """
    if platform.system() != "Windows":
        return {}

    pcie_info = get_pcie_static_info(
        vendor_id=vendor_id,
        device_id=device_id,
        class_filter=class_filter,
        devices=devices,
    )
    hardware = pcie_info.get("hardware", {})
    if not isinstance(hardware, dict):
        return {}
    npus = hardware.get("npus", [])
    if not isinstance(npus, list):
        return {}

    normalized_vendor_ids = {_normalize_hex(vendor_id) for vendor_id in vendor_ids}
    devices = []
    for npu in npus:
        if not isinstance(npu, dict):
            continue
        vendor_id = _normalize_hex(str(npu.get("vendor_id", "")))
        if vendor_id not in normalized_vendor_ids:
            continue

        device = dict(npu)
        if device:
            devices.append(device)

    info: dict[str, object] = {}
    pcie_inference = pcie_info.get("inference", {})
    if isinstance(pcie_inference, dict):
        npu_driver_version = pcie_inference.get("npu_driver_version")
        if isinstance(npu_driver_version, str):
            info.setdefault("inference", {})["npu_driver_version"] = npu_driver_version
    if devices:
        info.setdefault("hardware", {})["npus"] = devices
        npu_driver_version = _pop_npu_driver_version(devices)
        if npu_driver_version is not None:
            info.setdefault("inference", {})["npu_driver_version"] = npu_driver_version
    return info


def _clean_windows_device_property(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return str(value)


def _windows_link_speed_to_text(value: object) -> str | None:
    if not isinstance(value, (int, str)):
        return None
    try:
        speed = int(value)
    except (TypeError, ValueError):
        return None

    speeds = {
        1: "2.5 GT/s PCIe",
        2: "5.0 GT/s PCIe",
        3: "8.0 GT/s PCIe",
        4: "16.0 GT/s PCIe",
        5: "32.0 GT/s PCIe",
        6: "64.0 GT/s PCIe",
    }
    return speeds.get(speed)


def _parse_windows_pci_id(
    pnp_device_id: str,
    auxiliary_ids: Sequence[str] | None = None,
) -> dict[str, object] | None:
    if not pnp_device_id.upper().startswith("PCI\\"):
        return None

    pci_ids = [pnp_device_id, *(auxiliary_ids or [])]

    def match_hex(pattern: str, values: Sequence[str] = pci_ids) -> str | None:
        for value in values:
            match = re.search(pattern, value, re.IGNORECASE)
            if match is not None:
                return f"0x{match.group(1).lower()}"
        return None

    vendor_id = match_hex(r"VEN_([0-9A-F]{4})")
    device_id = match_hex(r"DEV_([0-9A-F]{4})")
    if vendor_id is None or device_id is None:
        return None

    device: dict[str, object] = {
        "vendor_id": vendor_id,
        "device_id": device_id,
    }

    subsys = re.search(r"SUBSYS_([0-9A-F]{8})", pnp_device_id, re.IGNORECASE)
    if subsys is not None:
        value = subsys.group(1).lower()
        device["subsystem_device_id"] = f"0x{value[:4]}"
        device["subsystem_vendor_id"] = f"0x{value[4:]}"

    class_code = _match_windows_pci_class_code(pci_ids)
    if class_code is not None:
        device["class"] = class_code
    revision = match_hex(r"REV_([0-9A-F]{2})")
    if revision is not None:
        device["revision"] = revision

    return device


def _match_windows_pci_class_code(pci_ids: Sequence[str]) -> str | None:
    class_codes = []
    for pci_id in pci_ids:
        for match in re.finditer(r"CC_([0-9A-F]{2,6})", pci_id, re.IGNORECASE):
            class_codes.append(match.group(1).lower())
    if not class_codes:
        return None
    return f"0x{max(class_codes, key=len)}"


def _read_pcie_devices(sysfs_root: Path) -> list[dict[str, object]]:
    devices = []
    if not sysfs_root.exists():
        return devices
    lspci_metadata = _read_lspci_device_metadata()
    for device_dir in sorted(path for path in sysfs_root.iterdir() if path.is_dir()):
        device = {
            "bus_address": device_dir.name,
            "vendor_id": _read_first_line(device_dir / "vendor"),
            "device_id": _read_first_line(device_dir / "device"),
            "subsystem_vendor_id": _read_first_line(device_dir / "subsystem_vendor"),
            "subsystem_device_id": _read_first_line(device_dir / "subsystem_device"),
            "class": _read_first_line(device_dir / "class"),
            "revision": _read_first_line(device_dir / "revision"),
            "current_link_speed": _read_first_line(device_dir / "current_link_speed"),
            "current_link_width": _read_first_line(device_dir / "current_link_width"),
            "max_link_speed": _read_first_line(device_dir / "max_link_speed"),
            "max_link_width": _read_first_line(device_dir / "max_link_width"),
        }
        device.update(_read_linux_pcie_driver_metadata(device_dir))
        device.update(lspci_metadata.get(device_dir.name, {}))
        device.update(_linux_known_pcie_metadata(device))
        devices.append({key: value for key, value in device.items() if value is not None})
    return devices


def _read_linux_pcie_driver_metadata(device_dir: Path) -> dict[str, object]:
    metadata: dict[str, object] = {}
    driver_path = device_dir / "driver"
    if not driver_path.exists():
        return metadata
    try:
        driver_module_path = driver_path.resolve() / "module" / "version"
    except OSError:
        return metadata
    version = _read_first_line(driver_module_path)
    if version is not None:
        metadata["driver_version"] = version
    return metadata


def _read_lspci_device_metadata() -> dict[str, dict[str, object]]:
    output = run_command(["lspci", "-Dmmnn"])
    if output is None:
        return {}

    metadata: dict[str, dict[str, object]] = {}
    for line in output.splitlines():
        try:
            fields = shlex.split(line)
        except ValueError:
            continue
        if len(fields) < 4:
            continue
        bus_address = fields[0]
        vendor = _strip_lspci_numeric_suffix(fields[2])
        name = _strip_lspci_numeric_suffix(fields[3])
        values = {}
        if vendor:
            values["manufacturer"] = vendor
        if name:
            values["name"] = name
        if values:
            metadata[bus_address] = values
    return metadata


def _strip_lspci_numeric_suffix(value: str) -> str | None:
    value = re.sub(r"\s*\[[0-9a-fA-F]{4,6}\]\s*$", "", value).strip()
    return value or None


def _linux_known_pcie_metadata(device: dict[str, object]) -> dict[str, object]:
    vendor = _normalize_hex(str(device.get("vendor_id", "")))
    device_id = _normalize_hex(str(device.get("device_id", "")))
    known: dict[tuple[str, str | None], dict[str, object]] = {
        ("209f", None): {
            "manufacturer": "MOBILINT, Inc.",
            "name": "MOBILINT NPU Accelerator",
        },
        ("1ed5", None): {
            "manufacturer": "MOBILINT, Inc.",
            "name": "MOBILINT NPU Accelerator",
        },
        ("10de", None): {"manufacturer": "NVIDIA"},
        ("8086", None): {"manufacturer": "Intel"},
        ("1002", None): {"manufacturer": "AMD"},
    }
    values = dict(known.get((vendor or "", None), {}))
    values.update(known.get((vendor or "", device_id), {}))
    resolved = {}
    for key, value in values.items():
        existing = device.get(key)
        if existing is None or _is_generic_lspci_label(str(existing)):
            resolved[key] = value
    return resolved


def _is_generic_lspci_label(value: str) -> bool:
    return value.strip().lower() in {"device", "vendor"}


def _find_all_npu_devices(
    devices: list[dict[str, object]],
    vendor_id: str | None,
    device_id: str | None,
    class_filter: str | None,
) -> list[dict[str, object]]:
    normalized_vendor_id = _normalize_hex(vendor_id)
    normalized_device_id = _normalize_hex(device_id)
    normalized_class_filter = _normalize_hex(class_filter)

    if (
        normalized_vendor_id is None
        and normalized_device_id is None
        and normalized_class_filter is None
    ):
        return [device for device in devices if _is_likely_npu_device(device)]

    matched = []
    for device in devices:
        if normalized_vendor_id is not None and _normalize_hex(
            str(device.get("vendor_id", ""))
        ) != normalized_vendor_id:
            continue
        if normalized_device_id is not None and _normalize_hex(
            str(device.get("device_id", ""))
        ) != normalized_device_id:
            continue
        device_class = _normalize_hex(str(device.get("class", "")))
        if (
            normalized_class_filter is not None
            and (device_class is None or not device_class.startswith(normalized_class_filter))
        ):
            continue
        matched.append(device)
    return matched


def _find_all_gpu_devices(devices: list[dict[str, object]]) -> list[dict[str, object]]:
    return [device for device in devices if _is_likely_gpu_device(device)]


def _format_pcie_device(device: dict[str, object], dev_no: int) -> dict[str, object]:
    formatted: dict[str, object] = {"dev_no": dev_no}
    for key in (
        "bus_address",
        "vendor_id",
        "device_id",
        "subsystem_vendor_id",
        "subsystem_device_id",
        "class",
        "name",
        "manufacturer",
        "status",
        "pnp_device_id",
        "revision",
        "driver_version",
        "driver_name",
        "driver_date",
        "driver_description",
        "driver_provider",
        "firmware",
        "firmware_version",
        "firmware_revision",
        "current_link_speed",
        "current_link_width",
        "max_link_speed",
        "max_link_width",
        "memory_total_bytes",
        "architecture",
    ):
        if device.get(key) is not None:
            formatted[key] = device[key]

    if device.get("nvml_link_generation") is not None:
        formatted["link_generation"] = device["nvml_link_generation"]
    elif device.get("current_link_speed") is not None:
        generation = _link_speed_to_generation(str(device["current_link_speed"]))
        if generation is not None:
            formatted["link_generation"] = generation
    if device.get("nvml_lane_width") is not None:
        formatted["lane_width"] = device["nvml_lane_width"]
    elif device.get("current_link_width") is not None:
        formatted["lane_width"] = f"x{device['current_link_width']}"
    return formatted


def _filter_relevant_pcie_devices(
    devices: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [device for device in devices if _is_relevant_pcie_device(device)]


def _is_relevant_pcie_device(device: dict[str, object]) -> bool:
    """Return true for PCIe devices that are useful in tracker output.

    The default collection output should avoid noisy platform devices such as
    bridges and root ports while keeping accelerators users typically care
    about: GPUs, NPUs, and processing accelerators.
    """
    if _is_likely_npu_device(device):
        return True

    if _is_likely_gpu_device(device):
        return True

    class_code = _normalize_hex(str(device.get("class", "")))
    if class_code is not None and class_code.startswith("12"):
        return True

    text = " ".join(
        str(device.get(key, "")) for key in ("name", "manufacturer", "pnp_device_id")
    ).lower()
    gpu_npu_keywords = (
        "amd",
        "geforce",
        "gpu",
        "npu",
        "nvidia",
        "radeon",
        "tesla",
    )
    return any(keyword in text for keyword in gpu_npu_keywords)


def _is_likely_gpu_device(device: dict[str, object]) -> bool:
    if _is_integrated_gpu_without_pcie_link(device):
        return False

    if _is_gpu_companion_device(device):
        return False

    vendor = _normalize_hex(str(device.get("vendor_id", "")))
    if vendor in {"10de", "1002"}:
        return True

    class_code = _normalize_hex(str(device.get("class", "")))
    if class_code is not None and class_code.startswith("03"):
        return True

    text = " ".join(
        str(device.get(key, "")) for key in ("name", "manufacturer", "pnp_device_id")
    ).lower()
    gpu_keywords = ("amd", "geforce", "gpu", "nvidia", "radeon", "tesla")
    return any(keyword in text for keyword in gpu_keywords)


def _is_gpu_companion_device(device: dict[str, object]) -> bool:
    """Return true for non-GPU functions exposed by a discrete GPU card.

    NVIDIA/AMD PCIe devices often expose companion functions such as HD audio,
    USB-C controllers, or bridges under the same vendor ID as the display
    controller. These devices must not be used to enrich NVML GPU entries.
    """
    vendor = _normalize_hex(str(device.get("vendor_id", "")))
    if vendor not in {"10de", "1002"}:
        return False

    class_code = _normalize_hex(str(device.get("class", "")))
    if class_code is not None and class_code.startswith(("04", "0c", "06")):
        return True

    text = " ".join(
        str(device.get(key, ""))
        for key in (
            "name",
            "manufacturer",
            "driver_description",
            "driver_provider",
            "pnp_device_id",
        )
    ).lower()
    companion_keywords = (
        "audio",
        "high definition audio",
        "usb",
        "usb-c",
        "type-c",
        "bridge",
    )
    return any(keyword in text for keyword in companion_keywords)


def _is_integrated_gpu_without_pcie_link(device: dict[str, object]) -> bool:
    """Return true for on-die display controllers that are not PCIe GPUs."""
    class_code = _normalize_hex(str(device.get("class", "")))
    if class_code is None or not class_code.startswith("03"):
        return False

    bus_address = str(device.get("bus_address", ""))
    current_link_speed = str(device.get("current_link_speed", "")).strip().lower()
    current_link_width = str(device.get("current_link_width", "")).strip()
    vendor = _normalize_hex(str(device.get("vendor_id", "")))

    is_root_bus_display = bus_address.startswith(("0000:00:", "0000_00_"))
    has_no_pcie_link = current_link_speed in {"", "unknown"} or current_link_width in {
        "",
        "0",
    }
    return vendor == "8086" and is_root_bus_display and has_no_pcie_link


def _is_likely_npu_device(device: dict[str, object]) -> bool:
    vendor = _normalize_hex(str(device.get("vendor_id", "")))
    if vendor in {"1ed5", "209f"}:
        return True

    text = " ".join(
        str(device.get(key, "")) for key in ("name", "manufacturer", "pnp_device_id")
    ).lower()
    return "mobilint" in text or "npu" in text


def _read_first_line(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    return value or None


def _normalize_hex(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().lower()
    if not value:
        return None
    if value.startswith("0x"):
        value = value[2:]
    return value


def _deep_merge(
    base: dict[str, object] | object, overlay: dict[str, object] | object
) -> dict[str, object] | object:
    """Recursively merge nested dictionaries and return ``base``."""
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return base
    for key, value in overlay.items():
        existing = base.get(key)
        if key in {"npus", "gpus", "pcie_devices"} and isinstance(existing, list) and isinstance(value, list):
            base[key] = _merge_device_lists(existing, value)
        elif isinstance(existing, dict) and isinstance(value, dict):
            _deep_merge(existing, value)
        else:
            base[key] = value
    return base


def _merge_device_lists(
    base: list[object], overlay: list[object]
) -> list[object]:
    merged = [dict(item) if isinstance(item, dict) else item for item in base]
    for overlay_index, overlay_item in enumerate(overlay):
        if not isinstance(overlay_item, dict):
            merged.append(overlay_item)
            continue
        overlay_key = overlay_item.get("dev_no", overlay_index)
        overlay_has_identity = _has_device_identity(overlay_item)
        matched = False
        for base_index, base_item in enumerate(merged):
            if not isinstance(base_item, dict):
                continue
            if _device_identity_matches(base_item, overlay_item):
                _deep_merge(base_item, overlay_item)
                matched = True
                break
            if overlay_has_identity:
                continue
            base_key = base_item.get("dev_no", base_index)
            if base_key == overlay_key:
                _deep_merge(base_item, overlay_item)
                matched = True
                break
        if not matched:
            merged.append(dict(overlay_item))
    return merged


def _has_device_identity(item: dict[str, object]) -> bool:
    return any(item.get(key) is not None for key in ("bus_address", "pnp_device_id"))


def _device_identity_matches(
    base_item: dict[str, object], overlay_item: dict[str, object]
) -> bool:
    for key in ("bus_address", "pnp_device_id"):
        base_value = base_item.get(key)
        overlay_value = overlay_item.get(key)
        if base_value is not None and overlay_value is not None:
            if str(base_value).lower() == str(overlay_value).lower():
                return True
    return False


def _clean_typed_dict(value: object, schema: object | None = None) -> object:
    """Remove None from optional fields while preserving required schema keys."""
    if isinstance(value, dict):
        required_keys = getattr(schema, "__required_keys__", frozenset())
        child_schemas = (
            STATIC_INFO_CHILD_SCHEMAS.get(schema, {}) if isinstance(schema, type) else {}
        )
        cleaned: dict[str, object] = {}
        for key, item in value.items():
            child_schema = child_schemas.get(key)
            cleaned_item = _clean_typed_dict(item, child_schema)
            if cleaned_item is not None or key in required_keys:
                cleaned[key] = cleaned_item
        return cleaned
    if isinstance(value, list):
        item_schema = schema[0] if isinstance(schema, list) and schema else None
        return [_clean_typed_dict(item, item_schema) for item in value]
    return value


def _remove_none_values(value: object) -> object:
    """Backward-compatible wrapper for callers/tests that clean untyped data."""
    return _clean_typed_dict(value)


def _link_speed_to_generation(link_speed: str) -> str | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*GT/s", link_speed, re.IGNORECASE)
    if match is None:
        return None
    speed = float(match.group(1))
    if speed >= 63.0:
        return "Gen6"
    if speed >= 31.0:
        return "Gen5"
    if speed >= 15.0:
        return "Gen4"
    if speed >= 7.0:
        return "Gen3"
    if speed >= 4.0:
        return "Gen2"
    if speed >= 2.0:
        return "Gen1"
    return None


def _linux_cpu_identity() -> tuple[str | None, str | None]:
    cpuinfo = Path("/proc/cpuinfo")
    try:
        lines = cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None, None
    model_name = None
    vendor = None
    for line in lines:
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if key == "model name" and model_name is None:
            model_name = value
        elif key == "vendor_id" and vendor is None:
            vendor = value
        if model_name is not None and vendor is not None:
            break
    return model_name, vendor


def _windows_cpu_identity() -> tuple[str | None, str | None]:
    """Return Windows CPU brand string and vendor from the registry.

    ``platform.processor()`` often returns a low-level CPUID descriptor such as
    ``Intel64 Family 6 Model 191 Stepping 2, GenuineIntel`` on Windows. The
    registry value below contains the user-facing processor brand string, e.g.
    ``13th Gen Intel(R) Core(TM) i5-13500``.
    """
    try:
        import winreg
    except ImportError:
        return None, None
    winreg_module = cast(Any, winreg)

    try:
        with winreg_module.OpenKey(
            winreg_module.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            model_name = _read_windows_registry_string(
                winreg_module,
                key,
                "ProcessorNameString",
            )
            vendor = _read_windows_registry_string(
                winreg_module,
                key,
                "VendorIdentifier",
            )
    except OSError:
        return None, None
    return model_name, vendor


def _read_windows_registry_string(winreg_module: object, key: object, name: str) -> str | None:
    try:
        value, _value_type = cast(Any, winreg_module).QueryValueEx(key, name)
    except OSError:
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _read_os_release() -> dict[str, str]:
    try:
        output = Path("/etc/os-release").read_text(encoding="utf-8")
    except OSError:
        return {}
    values = {}
    for line in output.splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def run_command(command: list[str]) -> str | None:
    """Run a command and return stdout on success."""
    return run_command_with_timeout(command, timeout=5)


def run_command_with_timeout(command: list[str], timeout: int) -> str | None:
    """Run a command and return stdout on success."""
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout or None


def run_command_with_input(
    command: list[str],
    input_text: str,
    timeout: int,
) -> str | None:
    """Run a command with stdin input and return stdout on success."""
    try:
        result = subprocess.run(
            command,
            input=input_text,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout or None
