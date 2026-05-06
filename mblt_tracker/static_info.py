from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Optional

import psutil


def get_host_static_info() -> dict[str, object]:
    """Collect best-effort host CPU, DRAM, and OS static information."""
    info: dict[str, object] = {
        "hardware": {
            "host": {
                "cpu": {
                    "architecture": platform.machine(),
                    "physical_cores": psutil.cpu_count(logical=False),
                    "logical_cores": psutil.cpu_count(logical=True),
                },
                "dram": {
                    "total_bytes": psutil.virtual_memory().total,
                    "available_bytes": psutil.virtual_memory().available,
                },
            }
        },
        "inference": {
            "os": {
                "name": platform.system(),
                "version": platform.version(),
                "kernel_version": platform.release(),
            }
        },
    }

    model_name, vendor = _linux_cpu_identity()
    if model_name is None:
        model_name = platform.processor() or platform.uname().processor
    if model_name:
        info["hardware"]["host"]["cpu"]["model_name"] = model_name
    if vendor:
        info["hardware"]["host"]["cpu"]["vendor"] = vendor

    if platform.system() == "Linux":
        os_release = _read_os_release()
        pretty_name = os_release.get("PRETTY_NAME")
        if pretty_name:
            info["inference"]["os"]["version"] = pretty_name
        governor = get_cpu_governor()
        if governor is not None:
            info["inference"]["cpu"] = {"governor": governor}

    return _remove_none_values(info)


def get_cpu_governor() -> Optional[str]:
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


def get_pcie_static_info(
    vendor_id: Optional[str] = None,
    device_id: Optional[str] = None,
    class_filter: Optional[str] = None,
    include_all_devices: bool = False,
) -> dict[str, object]:
    """Collect best-effort PCIe information.

    Args:
        vendor_id: Optional hexadecimal vendor id, with or without ``0x``.
        device_id: Optional hexadecimal device id, with or without ``0x``.
        class_filter: Optional PCI class prefix, e.g. ``0x12``.
        include_all_devices: Include all PCIe devices in ``hardware.pcie.devices``.
            When false, omit the raw device list and expose only categorized
            GPU/NPU lists.

    Returns:
        Nested dictionary containing categorized PCIe devices. Raw
        ``hardware.pcie.devices`` is included only when requested.
    """
    sysfs_override = os.environ.get("MBLT_TRACKER_PCI_SYSFS")
    if sysfs_override is not None:
        devices = _read_pcie_devices(Path(sysfs_override))
    elif platform.system() == "Windows":
        devices = _read_pcie_devices_windows()
    else:
        devices = _read_pcie_devices(Path("/sys/bus/pci/devices"))
    pcie_info: dict[str, object] = {}
    if include_all_devices:
        pcie_info["devices"] = devices
    npus = _find_all_npu_devices(devices, vendor_id, device_id, class_filter)
    if npus:
        pcie_info["npus"] = [
            _format_pcie_device(device, dev_no) for dev_no, device in enumerate(npus)
        ]
    gpus = _find_all_gpu_devices(devices)
    if gpus:
        pcie_info["gpus"] = [
            _format_pcie_device(device, dev_no) for dev_no, device in enumerate(gpus)
        ]
    return {"hardware": {"pcie": pcie_info}} if pcie_info else {}


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
            "Select-Object Name,PNPDeviceID,DeviceID,Manufacturer,Status | "
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

    link_properties = _read_windows_pci_link_properties()
    devices = []
    for entity in entities:
        pnp_device_id = str(entity.get("PNPDeviceID") or entity.get("DeviceID") or "")
        device = _parse_windows_pci_id(pnp_device_id)
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
        device.update(link_properties.get(pnp_device_id.upper(), {}))
        devices.append({key: value for key, value in device.items() if value})
    return devices


def _read_windows_pci_link_properties() -> dict[str, dict[str, object]]:
    """Read PCIe link properties for all Windows PCI devices in one call."""
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
            "'DEVPKEY_PciDevice_MaxLinkWidth'"
            "); "
            r"Get-PnpDevice -InstanceId 'PCI\*' | "
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

        if device_properties:
            properties[instance_id.upper()] = device_properties
    return properties


def _windows_link_speed_to_text(value: object) -> Optional[str]:
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


def _parse_windows_pci_id(pnp_device_id: str) -> Optional[dict[str, object]]:
    if not pnp_device_id.upper().startswith("PCI\\"):
        return None

    def match_hex(pattern: str) -> Optional[str]:
        match = re.search(pattern, pnp_device_id, re.IGNORECASE)
        if match is None:
            return None
        return f"0x{match.group(1).lower()}"

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

    class_code = match_hex(r"CC_([0-9A-F]{2,6})")
    if class_code is not None:
        device["class"] = class_code
    revision = match_hex(r"REV_([0-9A-F]{2})")
    if revision is not None:
        device["revision"] = revision

    return device


def _read_pcie_devices(sysfs_root: Path) -> list[dict[str, object]]:
    devices = []
    if not sysfs_root.exists():
        return devices
    for device_dir in sorted(path for path in sysfs_root.iterdir() if path.is_dir()):
        device = {
            "bus_address": device_dir.name,
            "vendor_id": _read_first_line(device_dir / "vendor"),
            "device_id": _read_first_line(device_dir / "device"),
            "subsystem_vendor_id": _read_first_line(device_dir / "subsystem_vendor"),
            "subsystem_device_id": _read_first_line(device_dir / "subsystem_device"),
            "class": _read_first_line(device_dir / "class"),
            "current_link_speed": _read_first_line(device_dir / "current_link_speed"),
            "current_link_width": _read_first_line(device_dir / "current_link_width"),
            "max_link_speed": _read_first_line(device_dir / "max_link_speed"),
            "max_link_width": _read_first_line(device_dir / "max_link_width"),
        }
        devices.append({key: value for key, value in device.items() if value is not None})
    return devices


def _find_all_npu_devices(
    devices: list[dict[str, object]],
    vendor_id: Optional[str],
    device_id: Optional[str],
    class_filter: Optional[str],
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
        if normalized_class_filter is not None and not _normalize_hex(
            str(device.get("class", ""))
        ).startswith(normalized_class_filter):
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
        "current_link_speed",
        "current_link_width",
        "max_link_speed",
        "max_link_width",
    ):
        if device.get(key) is not None:
            formatted[key] = device[key]

    if device.get("current_link_speed") is not None:
        generation = _link_speed_to_generation(str(device["current_link_speed"]))
        if generation is not None:
            formatted["link_generation"] = generation
    if device.get("current_link_width") is not None:
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
    class_code = _normalize_hex(str(device.get("class", "")))
    if class_code is not None and class_code.startswith("03"):
        return True

    text = " ".join(
        str(device.get(key, "")) for key in ("name", "manufacturer", "pnp_device_id")
    ).lower()
    gpu_keywords = ("amd", "geforce", "gpu", "nvidia", "radeon", "tesla")
    return any(keyword in text for keyword in gpu_keywords)


def _is_likely_npu_device(device: dict[str, object]) -> bool:
    vendor = _normalize_hex(str(device.get("vendor_id", "")))
    if vendor in {"1ed5", "209f"}:
        return True

    text = " ".join(
        str(device.get(key, "")) for key in ("name", "manufacturer", "pnp_device_id")
    ).lower()
    return "mobilint" in text or "npu" in text


def _read_first_line(path: Path) -> Optional[str]:
    try:
        value = path.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    return value or None


def _normalize_hex(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip().lower()
    if not value:
        return None
    if value.startswith("0x"):
        value = value[2:]
    return value


def _deep_merge(
    base: dict[str, object], overlay: dict[str, object]
) -> dict[str, object]:
    """Recursively merge nested dictionaries and return ``base``."""
    for key, value in overlay.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _deep_merge(existing, value)
        else:
            base[key] = value
    return base


def _remove_none_values(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := _remove_none_values(item)) is not None
        }
    if isinstance(value, list):
        return [_remove_none_values(item) for item in value]
    return value


def _link_speed_to_generation(link_speed: str) -> Optional[str]:
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


def _linux_cpu_identity() -> tuple[Optional[str], Optional[str]]:
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


def run_command(command: list[str]) -> Optional[str]:
    """Run a command and return stdout on success."""
    return run_command_with_timeout(command, timeout=5)


def run_command_with_timeout(command: list[str], timeout: int) -> Optional[str]:
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