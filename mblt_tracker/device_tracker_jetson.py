import glob
import os
import time

import numpy as np

from .device_tracker import BaseDeviceTracker

# Jetson modules expose their on-module INA3221 power monitors through hwmon.
_INA3221_HWMON_GLOB = "sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"
# GPU load in 0.1% units; the platform path differs between L4T releases.
_GPU_LOAD_GLOBS = (
    "sys/devices/platform/bus@0/*.gpu/load",
    "sys/devices/platform/*.gpu/load",
    "sys/devices/gpu.0/load",
)
_THERMAL_ZONE_GLOB = "sys/class/thermal/thermal_zone*"
_GPU_THERMAL_ZONE_TYPE = "gpu-thermal"
_DEVICE_TREE_MODEL = "proc/device-tree/model"
_DEVICE_TREE_COMPATIBLE = "proc/device-tree/compatible"
_L4T_RELEASE = "etc/nv_tegra_release"
_MEMINFO = "proc/meminfo"


class JetsonDeviceTracker(BaseDeviceTracker):
    """Track NVIDIA Jetson power, GPU load, memory, and temperature through sysfs.

    Jetson GPUs do not report power or utilization through NVML. This tracker
    reads the on-module INA3221 power rails, the integrated GPU load, the GPU
    thermal zone, and system memory (CPU and GPU share it). Total power is the
    sum of all labeled INA3221 rails. No root privileges are required.
    """

    def __init__(
        self,
        interval: float = 0.1,
        rails: list[str] | None = None,
        sysfs_root: str = "/",
    ):
        """Initialize the Jetson device tracker.

        Args:
            interval (float): The interval in seconds at which the device should
                be polled.
            rails (list[str] | None): INA3221 rail labels to sum into total
                power, such as ``["VDD_GPU_SOC", "VDD_CPU_CV"]``. If None, all
                labeled rails are used.
            sysfs_root (str): Filesystem root to read from; tests point it at a
                fake tree.

        Raises:
            ValueError: If no INA3221 rails are found or an unknown rail is
                requested.
        """
        super().__init__(interval=interval)
        self._root = sysfs_root
        available_rails = _discover_rails(sysfs_root)
        if not available_rails:
            raise ValueError("No Jetson INA3221 power rails found")
        if rails is None:
            rails = list(available_rails)
        else:
            unknown = [rail for rail in rails if rail not in available_rails]
            if unknown:
                raise ValueError(
                    f"Unknown Jetson power rail(s): {unknown}; "
                    f"available: {list(available_rails)}"
                )
        self._rail_paths = {rail: available_rails[rail] for rail in rails}
        self._gpu_load_path = _first_existing(sysfs_root, _GPU_LOAD_GLOBS)
        self._gpu_temp_path = _find_thermal_zone(sysfs_root, _GPU_THERMAL_ZONE_TYPE)
        self._job_id = "jetson_device_track"
        self.reset()

    @staticmethod
    def is_available(sysfs_root: str = "/") -> bool:
        """Return True when running on a Jetson with readable INA3221 rails."""
        compatible = _read_text(os.path.join(sysfs_root, _DEVICE_TREE_COMPATIBLE))
        if compatible is None or "nvidia,tegra" not in compatible:
            return False
        return bool(_discover_rails(sysfs_root))

    @property
    def rails(self) -> list[str]:
        """Return the INA3221 rail labels summed into total power."""
        return list(self._rail_paths)

    def rail_power(self) -> dict[str, float | None]:
        """Get instantaneous power in Watts for each tracked rail.

        Returns:
            dict[str, float | None]: Rail label to power in Watts, or None when
                the rail could not be read.
        """
        powers: dict[str, float | None] = {}
        for rail, (voltage_path, current_path) in self._rail_paths.items():
            voltage_mv = _read_float(voltage_path)
            current_ma = _read_float(current_path)
            if voltage_mv is None or current_ma is None:
                powers[rail] = None
            else:
                powers[rail] = voltage_mv * current_ma / 1e6
        return powers

    def gpu_utilization(self) -> float | None:
        """Get the current integrated GPU load in percent, or None if unavailable."""
        if self._gpu_load_path is None:
            return None
        load = _read_float(self._gpu_load_path)
        return None if load is None else load / 10.0

    def gpu_temperature(self) -> float | None:
        """Get the current GPU temperature in Celsius, or None if unavailable."""
        if self._gpu_temp_path is None:
            return None
        temp_mc = _read_float(self._gpu_temp_path)
        return None if temp_mc is None else temp_mc / 1000.0

    def memory_info(self) -> tuple[float, float] | None:
        """Get (used_mb, total_mb) of the system memory shared by CPU and GPU."""
        meminfo = _read_text(os.path.join(self._root, _MEMINFO))
        if meminfo is None:
            return None
        fields: dict[str, float] = {}
        for line in meminfo.splitlines():
            key, _, value = line.partition(":")
            parts = value.split()
            if parts:
                try:
                    fields[key] = float(parts[0])
                except ValueError:
                    continue
        total_kb = fields.get("MemTotal")
        available_kb = fields.get("MemAvailable")
        if total_kb is None or available_kb is None:
            return None
        return (total_kb - available_kb) / 1024.0, total_kb / 1024.0

    def _func_for_sched(self) -> None:
        """Sample rail power, GPU load, memory usage, and temperature."""
        powers = self.rail_power()
        gpu_util_pct = self.gpu_utilization()
        memory = self.memory_info()
        temp_c = self.gpu_temperature()
        ts = time.time()

        valid_powers = [power for power in powers.values() if power is not None]
        for rail, power in powers.items():
            if power is not None:
                self._rail_power_glance[rail].append(power)
        if valid_powers:
            self._power_trace.append((ts, float(sum(valid_powers))))
        if gpu_util_pct is not None:
            self._gpu_util_trace.append((ts, gpu_util_pct))
        if memory is not None:
            used_mb, total_mb = memory
            self._mem_total_mb = total_mb
            self._mem_used_trace.append((ts, used_mb))
            self._mem_used_pct_trace.append(
                (ts, used_mb / total_mb * 100.0 if total_mb > 0 else 0.0)
            )
        if temp_c is not None:
            self._temp_trace.append((ts, temp_c))

    def get_metric(self) -> dict[str, object]:
        """Return summarized Jetson metrics since start or last reset.

        Returns:
            Dict[str, object]: Average, p99, and peak total power, GPU load,
                shared memory, and GPU temperature, plus per-rail power under
                ``"rails"``.
        """
        power_samples = [p for _, p in self._power_trace]
        gpu_util_samples = [u for _, u in self._gpu_util_trace]
        mem_used_samples = [m for _, m in self._mem_used_trace]
        mem_used_pct_samples = [m for _, m in self._mem_used_pct_trace]
        temp_samples = [t for _, t in self._temp_trace]
        avg_gpu_util = _mean_or_none(gpu_util_samples)
        p99_gpu_util = _p99_or_none(gpu_util_samples)
        max_gpu_util = _max_or_none(gpu_util_samples)
        return {
            "avg_power_w": _mean_or_none(power_samples),
            "p99_power_w": _p99_or_none(power_samples),
            "max_power_w": _max_or_none(power_samples),
            "avg_gpu_util_pct": avg_gpu_util,
            "p99_gpu_util_pct": p99_gpu_util,
            "max_gpu_util_pct": max_gpu_util,
            # Generic names for cross-device consumers.
            "avg_utilization_pct": avg_gpu_util,
            "p99_utilization_pct": p99_gpu_util,
            "max_utilization_pct": max_gpu_util,
            "avg_memory_used_mb": _mean_or_none(mem_used_samples),
            "p99_memory_used_mb": _p99_or_none(mem_used_samples),
            "max_memory_used_mb": _max_or_none(mem_used_samples),
            "total_memory_mb": self._mem_total_mb,
            "avg_memory_used_pct": _mean_or_none(mem_used_pct_samples),
            "p99_memory_used_pct": _p99_or_none(mem_used_pct_samples),
            "max_memory_used_pct": _max_or_none(mem_used_pct_samples),
            "avg_temperature_c": _mean_or_none(temp_samples),
            "p99_temperature_c": _p99_or_none(temp_samples),
            "max_temperature_c": _max_or_none(temp_samples),
            "samples": max(
                len(power_samples),
                len(gpu_util_samples),
                len(mem_used_samples),
                len(temp_samples),
            ),
            "util_samples": len(gpu_util_samples),
            "rails": {
                rail: {
                    "avg_power_w": _mean_or_none(samples),
                    "p99_power_w": _p99_or_none(samples),
                    "max_power_w": _max_or_none(samples),
                }
                for rail, samples in self._rail_power_glance.items()
            },
        }

    def get_trace(self) -> list[tuple[float, float]]:
        """Return a time-series trace of total rail power.

        Returns:
            list[tuple[float, float]]: List of (timestamp, total_power_w) pairs.
        """
        return list(self._power_trace)

    def get_util_trace(self) -> list[tuple[float, float]]:
        """Return a time-series trace of GPU load.

        Returns:
            list[tuple[float, float]]: List of (timestamp, gpu_util_pct) pairs.
        """
        return list(self._gpu_util_trace)

    def get_temp_trace(self) -> list[tuple[float, float]]:
        """Return a time-series trace of GPU temperature."""
        return list(self._temp_trace)

    def get_static_info(self) -> dict[str, object]:
        """Return best-effort Jetson model, L4T release, and power-rail labels."""
        model = _read_text(os.path.join(self._root, _DEVICE_TREE_MODEL))
        release = _read_text(os.path.join(self._root, _L4T_RELEASE))
        return {
            "hardware": {
                "gpu": {
                    "device_count": 1,
                    "devices": [
                        {
                            "device_index": 0,
                            "name": model.strip("\x00").strip() if model else None,
                        }
                    ],
                    "power_rails": self.rails,
                }
            },
            "inference": {
                "gpu": {
                    "l4t": {"version": _parse_l4t_release(release)},
                }
            },
        }

    def reset(self) -> None:
        """Reset all collected Jetson metrics and traces."""
        self._rail_power_glance: dict[str, list[float]] = {
            rail: [] for rail in self._rail_paths
        }
        self._mem_total_mb: float | None = None
        self._power_trace: list[tuple[float, float]] = []
        self._gpu_util_trace: list[tuple[float, float]] = []
        self._mem_used_trace: list[tuple[float, float]] = []
        self._mem_used_pct_trace: list[tuple[float, float]] = []
        self._temp_trace: list[tuple[float, float]] = []


def _discover_rails(sysfs_root: str) -> dict[str, tuple[str, str]]:
    """Map each labeled INA3221 channel to its (voltage_mv, current_ma) paths."""
    rails: dict[str, tuple[str, str]] = {}
    for hwmon in sorted(glob.glob(os.path.join(sysfs_root, _INA3221_HWMON_GLOB))):
        for label_path in sorted(glob.glob(os.path.join(hwmon, "in*_label"))):
            channel = os.path.basename(label_path)[len("in") : -len("_label")]
            label = _read_text(label_path)
            if not label or label.strip().lower().startswith("sum of"):
                continue
            enable = _read_float(os.path.join(hwmon, f"in{channel}_enable"))
            if enable == 0:
                continue
            voltage_path = os.path.join(hwmon, f"in{channel}_input")
            current_path = os.path.join(hwmon, f"curr{channel}_input")
            if os.path.exists(voltage_path) and os.path.exists(current_path):
                rails[label.strip()] = (voltage_path, current_path)
    return rails


def _first_existing(sysfs_root: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(sysfs_root, pattern)))
        if matches:
            return matches[0]
    return None


def _find_thermal_zone(sysfs_root: str, zone_type: str) -> str | None:
    for zone in sorted(glob.glob(os.path.join(sysfs_root, _THERMAL_ZONE_GLOB))):
        if (_read_text(os.path.join(zone, "type")) or "").strip() == zone_type:
            return os.path.join(zone, "temp")
    return None


def _parse_l4t_release(text: str | None) -> str | None:
    """Turn ``# R36 (release), REVISION: 5.2, ...`` into ``36.5.2``."""
    if not text:
        return None
    first_line = text.splitlines()[0]
    major = first_line.split("(")[0].strip("# ").lstrip("R").strip()
    revision = first_line.partition("REVISION:")[2].split(",")[0].strip()
    if not major or not revision:
        return None
    return f"{major}.{revision}"


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def _read_float(path: str) -> float | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return float(text.strip())
    except ValueError:
        return None


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _p99_or_none(values: list[float]) -> float | None:
    return float(np.percentile(values, 99)) if values else None


def _max_or_none(values: list[float]) -> float | None:
    return float(np.max(values)) if values else None
