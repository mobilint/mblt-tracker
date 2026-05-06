from __future__ import annotations

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
        "hardware.host.cpu.architecture": platform.machine(),
        "hardware.host.cpu.physical_cores": psutil.cpu_count(logical=False),
        "hardware.host.cpu.logical_cores": psutil.cpu_count(logical=True),
        "hardware.host.dram.total_bytes": psutil.virtual_memory().total,
        "hardware.host.dram.available_bytes": psutil.virtual_memory().available,
        "inference.os.name": platform.system(),
        "inference.os.version": platform.version(),
        "inference.os.kernel_version": platform.release(),
    }

    model_name, vendor = _linux_cpu_identity()
    if model_name is None:
        model_name = platform.processor() or platform.uname().processor
    if model_name:
        info["hardware.host.cpu.model_name"] = model_name
    if vendor:
        info["hardware.host.cpu.vendor"] = vendor

    if platform.system() == "Linux":
        os_release = _read_os_release()
        pretty_name = os_release.get("PRETTY_NAME")
        if pretty_name:
            info["inference.os.version"] = pretty_name
        governor = get_cpu_governor()
        if governor is not None:
            info["inference.cpu.governor"] = governor

    return info


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
) -> dict[str, object]:
    """Collect best-effort PCIe information from Linux sysfs.

    Args:
        vendor_id: Optional hexadecimal vendor id, with or without ``0x``.
        device_id: Optional hexadecimal device id, with or without ``0x``.
        class_filter: Optional PCI class prefix, e.g. ``0x12``.

    Returns:
        Dictionary containing all PCIe devices and, when filters match, the first
        matching device under ``hardware.pcie.npu.*`` keys.
    """
    sysfs_root = Path(os.environ.get("MBLT_TRACKER_PCI_SYSFS", "/sys/bus/pci/devices"))
    devices = _read_pcie_devices(sysfs_root)
    info: dict[str, object] = {"hardware.pcie.devices": devices}
    matched = _find_pcie_device(devices, vendor_id, device_id, class_filter)
    if matched is not None:
        info.update(
            {
                "hardware.pcie.npu.bus_address": matched.get("bus_address"),
                "hardware.pcie.npu.vendor_id": matched.get("vendor_id"),
                "hardware.pcie.npu.device_id": matched.get("device_id"),
            }
        )
        if matched.get("current_link_speed") is not None:
            info["hardware.pcie.npu.link_speed"] = matched["current_link_speed"]
            generation = _link_speed_to_generation(str(matched["current_link_speed"]))
            if generation is not None:
                info["hardware.pcie.npu.link_generation"] = generation
        if matched.get("current_link_width") is not None:
            info["hardware.pcie.npu.lane_width"] = f"x{matched['current_link_width']}"
    return {key: value for key, value in info.items() if value is not None}


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


def _find_pcie_device(
    devices: list[dict[str, object]],
    vendor_id: Optional[str],
    device_id: Optional[str],
    class_filter: Optional[str],
) -> Optional[dict[str, object]]:
    normalized_vendor_id = _normalize_hex(vendor_id)
    normalized_device_id = _normalize_hex(device_id)
    normalized_class_filter = _normalize_hex(class_filter)
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
        return device
    return None


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
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout or None