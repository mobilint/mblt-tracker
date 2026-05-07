import json
import os
import platform
import re
import shlex
import subprocess
import time
from typing import Optional

import numpy as np

from .device_tracker import BaseDeviceTracker
from .static_info import (
    _deep_merge,
    _filter_npu_metadata_to_selected_devices,
    get_all_pcie_devices,
    get_pcie_static_info,
    get_windows_npu_driver_firmware_info,
    parse_mobilint_status_query_output,
    parse_mobilint_status_static_info,
    run_command,
)


class NPUDeviceTracker(BaseDeviceTracker):
    """Track NPU power and utilization by polling `mobilint-cli status`."""

    def __init__(self, interval: float = 0.5, status_cmd: Optional[str] = None):
        """Initialize the NPU device tracker.

        Args:
            interval (float): The interval in seconds at which the NPU should be polled.
            status_cmd (Optional[str]): Custom command to fetch NPU status.
                Defaults to `mobilint-cli status -q`.

        Raises:
            RuntimeError: If the operating system is not Linux.
        """
        super().__init__(interval=interval)
        if platform.system() != "Linux":
            raise RuntimeError("NPUDeviceTracker currently supports Linux only")
        self._status_cmd = (
            status_cmd
            if status_cmd is not None
            else "mobilint-cli status -q"
        )
        self._job_id = "npu_device_track"
        self._npu_power_glance: list[float] = []
        self._ddr_power_glance: list[float] = []
        self._pmic_power_glance: list[float] = []
        self._total_power_glance: list[float] = []
        self._npu_util_glance: list[float] = []
        self._npu_mem_used_mb_glance: list[float] = []
        self._npu_mem_used_pct_glance: list[float] = []
        self._npu_temp_glance: list[float] = []
        self._npu_mem_total_mb: Optional[float] = None
        self._power_trace: list[tuple[float, float]] = []
        self._npu_power_trace: list[tuple[float, float]] = []
        self._ddr_power_trace: list[tuple[float, float]] = []
        self._pmic_power_trace: list[tuple[float, float]] = []
        self._util_trace: list[tuple[float, float]] = []
        self._mem_used_trace: list[tuple[float, float]] = []
        self._mem_used_pct_trace: list[tuple[float, float]] = []
        self._temp_trace: list[tuple[float, float]] = []

    def _fetch_metrics(
        self,
    ) -> Optional[
        tuple[
            float,
            float,
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
        ]
    ]:
        """Execute the status command and parse NPU metric output.

        The default path uses ``mobilint-cli status -q`` and parses the
        indentation-based query format directly. For backward compatibility,
        custom commands that return the legacy JSON payload are still accepted.

        Returns:
            Optional[tuple]: A tuple containing (npu_power_w, total_power_w,
                ddr_power_w, pmic_power_w, npu_util_pct, npu_mem_used_mb,
                npu_mem_total_mb, npu_mem_used_pct, npu_temp_c) if successful,
                else None.
        """
        try:
            result = subprocess.run(
                shlex.split(self._status_cmd),
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None

        if result.returncode != 0 or not result.stdout:
            return None

        output = result.stdout.strip()
        metrics = _parse_mobilint_status_query_metrics(output)
        if metrics is not None:
            return metrics

        try:
            payload = json.loads(output)
        except Exception:
            return None

        if not payload.get("ok", False):
            return None
        if "npu_power_w" not in payload or "total_power_w" not in payload:
            return None

        npu_power_w = float(payload["npu_power_w"])
        total_power_w = float(payload["total_power_w"])
        ddr_power_w = _get_optional_payload_float(
            payload,
            "ddr_power_w",
            "npu_ddr_power_w",
            "ram_power_w",
        )
        pmic_power_w = _get_optional_payload_float(payload, "pmic_power_w")
        npu_util_pct = payload.get("npu_util_pct")
        if npu_util_pct is not None:
            npu_util_pct = float(npu_util_pct)
        npu_mem_used_mb = payload.get("npu_mem_used_mb")
        if npu_mem_used_mb is not None:
            npu_mem_used_mb = float(npu_mem_used_mb)
        npu_mem_total_mb = payload.get("npu_mem_total_mb")
        if npu_mem_total_mb is not None:
            npu_mem_total_mb = float(npu_mem_total_mb)
        npu_mem_used_pct = payload.get("npu_mem_used_pct")
        if (
            npu_mem_used_pct is None
            and npu_mem_used_mb is not None
            and npu_mem_total_mb is not None
            and npu_mem_total_mb != 0.0
        ):
            npu_mem_used_pct = (npu_mem_used_mb / npu_mem_total_mb) * 100.0
        elif npu_mem_used_pct is not None:
            npu_mem_used_pct = float(npu_mem_used_pct)
        npu_temp_c = payload.get("npu_temp_c")
        if npu_temp_c is not None:
            npu_temp_c = float(npu_temp_c)
        return (
            npu_power_w,
            total_power_w,
            ddr_power_w,
            pmic_power_w,
            npu_util_pct,
            npu_mem_used_mb,
            npu_mem_total_mb,
            npu_mem_used_pct,
            npu_temp_c,
        )

    def _func_for_sched(self) -> None:
        """Sample NPU metrics via the background scheduler."""
        metrics = self._fetch_metrics()
        if metrics is None:
            return
        (
            npu_power_w,
            total_power_w,
            ddr_power_w,
            pmic_power_w,
            npu_util_pct,
            npu_mem_used_mb,
            npu_mem_total_mb,
            npu_mem_used_pct,
            npu_temp_c,
        ) = metrics
        ts = time.time()
        self._npu_power_glance.append(npu_power_w)
        self._total_power_glance.append(total_power_w)
        self._power_trace.append((ts, total_power_w))
        self._npu_power_trace.append((ts, npu_power_w))
        if ddr_power_w is not None:
            self._ddr_power_glance.append(ddr_power_w)
            self._ddr_power_trace.append((ts, ddr_power_w))
        if pmic_power_w is not None:
            self._pmic_power_glance.append(pmic_power_w)
            self._pmic_power_trace.append((ts, pmic_power_w))
        if npu_util_pct is not None:
            self._npu_util_glance.append(npu_util_pct)
            self._util_trace.append((ts, npu_util_pct))
        if npu_mem_used_mb is not None:
            self._npu_mem_used_mb_glance.append(npu_mem_used_mb)
            self._mem_used_trace.append((ts, npu_mem_used_mb))
        if npu_mem_total_mb is not None:
            self._npu_mem_total_mb = npu_mem_total_mb
        if npu_mem_used_pct is not None:
            self._npu_mem_used_pct_glance.append(npu_mem_used_pct)
            self._mem_used_pct_trace.append((ts, npu_mem_used_pct))
        if npu_temp_c is not None:
            self._npu_temp_glance.append(npu_temp_c)
            self._temp_trace.append((ts, npu_temp_c))

    def get_metric(self) -> dict[str, Optional[float]]:
        """Return summarized NPU metrics since start or last reset.

        Returns:
            Dict[str, Optional[float]]: A dictionary containing average and peak
                power, utilization, and memory statistics.
        """
        npu_avg = (
            float(np.mean(self._npu_power_glance)) if self._npu_power_glance else None
        )
        ddr_avg = (
            float(np.mean(self._ddr_power_glance)) if self._ddr_power_glance else None
        )
        pmic_avg = (
            float(np.mean(self._pmic_power_glance))
            if self._pmic_power_glance
            else None
        )
        total_avg = (
            float(np.mean(self._total_power_glance))
            if self._total_power_glance
            else None
        )
        npu_p99 = (
            float(np.percentile(self._npu_power_glance, 99))
            if self._npu_power_glance
            else None
        )
        npu_max = (
            float(np.max(self._npu_power_glance)) if self._npu_power_glance else None
        )
        ddr_p99 = (
            float(np.percentile(self._ddr_power_glance, 99))
            if self._ddr_power_glance
            else None
        )
        ddr_max = (
            float(np.max(self._ddr_power_glance)) if self._ddr_power_glance else None
        )
        pmic_p99 = (
            float(np.percentile(self._pmic_power_glance, 99))
            if self._pmic_power_glance
            else None
        )
        pmic_max = (
            float(np.max(self._pmic_power_glance))
            if self._pmic_power_glance
            else None
        )
        total_p99 = (
            float(np.percentile(self._total_power_glance, 99))
            if self._total_power_glance
            else None
        )
        total_max = (
            float(np.max(self._total_power_glance))
            if self._total_power_glance
            else None
        )
        npu_util_avg = (
            float(np.mean(self._npu_util_glance)) if self._npu_util_glance else None
        )
        npu_util_p99 = (
            float(np.percentile(self._npu_util_glance, 99))
            if self._npu_util_glance
            else None
        )
        npu_util_max = (
            float(np.max(self._npu_util_glance)) if self._npu_util_glance else None
        )
        npu_mem_used_avg = (
            float(np.mean(self._npu_mem_used_mb_glance))
            if self._npu_mem_used_mb_glance
            else None
        )
        npu_mem_used_p99 = (
            float(np.percentile(self._npu_mem_used_mb_glance, 99))
            if self._npu_mem_used_mb_glance
            else None
        )
        npu_mem_used_max = (
            float(np.max(self._npu_mem_used_mb_glance))
            if self._npu_mem_used_mb_glance
            else None
        )
        npu_mem_used_pct_avg = (
            float(np.mean(self._npu_mem_used_pct_glance))
            if self._npu_mem_used_pct_glance
            else None
        )
        npu_mem_used_pct_p99 = (
            float(np.percentile(self._npu_mem_used_pct_glance, 99))
            if self._npu_mem_used_pct_glance
            else None
        )
        npu_mem_used_pct_max = (
            float(np.max(self._npu_mem_used_pct_glance))
            if self._npu_mem_used_pct_glance
            else None
        )
        npu_temp_avg = (
            float(np.mean(self._npu_temp_glance)) if self._npu_temp_glance else None
        )
        npu_temp_p99 = (
            float(np.percentile(self._npu_temp_glance, 99))
            if self._npu_temp_glance
            else None
        )
        npu_temp_max = (
            float(np.max(self._npu_temp_glance)) if self._npu_temp_glance else None
        )
        return {
            "avg_power_w": total_avg,
            "p99_power_w": total_p99,
            "max_power_w": total_max,
            "avg_npu_power_w": npu_avg,
            "p99_npu_power_w": npu_p99,
            "max_npu_power_w": npu_max,
            "avg_ddr_power_w": ddr_avg,
            "p99_ddr_power_w": ddr_p99,
            "max_ddr_power_w": ddr_max,
            "avg_pmic_power_w": pmic_avg,
            "p99_pmic_power_w": pmic_p99,
            "max_pmic_power_w": pmic_max,
            "avg_total_power_w": total_avg,
            "p99_total_power_w": total_p99,
            "max_total_power_w": total_max,
            "avg_npu_util_pct": npu_util_avg,
            "p99_npu_util_pct": npu_util_p99,
            "max_npu_util_pct": npu_util_max,
            # Generic names for cross-device consumers.
            "avg_utilization_pct": npu_util_avg,
            "p99_utilization_pct": npu_util_p99,
            "max_utilization_pct": npu_util_max,
            "avg_memory_used_mb": npu_mem_used_avg,
            "p99_memory_used_mb": npu_mem_used_p99,
            "max_memory_used_mb": npu_mem_used_max,
            "total_memory_mb": self._npu_mem_total_mb,
            "avg_memory_used_pct": npu_mem_used_pct_avg,
            "p99_memory_used_pct": npu_mem_used_pct_p99,
            "max_memory_used_pct": npu_mem_used_pct_max,
            "avg_temperature_c": npu_temp_avg,
            "p99_temperature_c": npu_temp_p99,
            "max_temperature_c": npu_temp_max,
            "samples": len(self._power_trace),
            "npu_power_samples": len(self._npu_power_trace),
            "ddr_power_samples": len(self._ddr_power_trace),
            "pmic_power_samples": len(self._pmic_power_trace),
            "util_samples": len(self._util_trace),
        }

    def get_trace(self) -> list[tuple[float, float]]:
        """Return a time-series trace of total system power.

        Returns:
            list[tuple[float, float]]: List of (timestamp, total_power_w) pairs.
        """
        return list(self._power_trace)

    def get_static_info(self) -> dict[str, object]:
        """Return best-effort NPU static information.

        The PCIe section is collected from Linux sysfs when available. Firmware,
        driver, product, and form-factor fields are parsed from ``mobilint-cli
        status`` on a best-effort basis.
        """
        pcie_vendor_id = os.environ.get("MBLT_TRACKER_NPU_PCI_VENDOR_ID")
        pcie_device_id = os.environ.get("MBLT_TRACKER_NPU_PCI_DEVICE_ID")
        pcie_class_filter = os.environ.get("MBLT_TRACKER_NPU_PCI_CLASS_FILTER")
        has_pcie_filter = any((pcie_vendor_id, pcie_device_id, pcie_class_filter))
        pcie_devices = get_all_pcie_devices()
        info = get_pcie_static_info(
            vendor_id=pcie_vendor_id,
            device_id=pcie_device_id,
            class_filter=pcie_class_filter,
            devices=pcie_devices,
        )
        hardware = info.get("hardware", {})
        filtered_npus = []
        if isinstance(hardware, dict) and isinstance(hardware.get("npus"), list):
            filtered_npus = [npu for npu in hardware["npus"] if isinstance(npu, dict)]
        if platform.system() == "Windows":
            _deep_merge(
                info,
                get_windows_npu_driver_firmware_info(
                    vendor_id=pcie_vendor_id,
                    device_id=pcie_device_id,
                    class_filter=pcie_class_filter,
                    devices=pcie_devices,
                ),
            )
        else:
            status_output = run_command(["mobilint-cli", "status", "-q"])
            if not status_output:
                status_output = run_command(["mobilint-cli", "status"])
            if status_output:
                status_info = _parse_mobilint_status_static_info(status_output)
                if has_pcie_filter:
                    _filter_npu_metadata_to_selected_devices(status_info, filtered_npus)
                _deep_merge(info, status_info)
        return info

    def get_util_trace(self) -> list[tuple[float, float]]:
        """Return a time-series trace of NPU utilization.

        Returns:
            list[tuple[float, float]]: List of (timestamp, npu_util_pct) pairs.
        """
        return list(self._util_trace)

    def get_npu_power_trace(self) -> list[tuple[float, float]]:
        """Return a time-series trace of NPU core power."""
        return list(self._npu_power_trace)

    def get_ddr_power_trace(self) -> list[tuple[float, float]]:
        """Return a time-series trace of on-board NPU DDR power."""
        return list(self._ddr_power_trace)

    def get_pmic_power_trace(self) -> list[tuple[float, float]]:
        """Return a time-series trace of NPU PMIC power."""
        return list(self._pmic_power_trace)

    def get_temp_trace(self) -> list[tuple[float, float]]:
        """Return a time-series trace of NPU temperature."""
        return list(self._temp_trace)

    def reset(self) -> None:
        """Reset all collected NPU metrics and traces."""
        self._npu_power_glance = []
        self._ddr_power_glance = []
        self._pmic_power_glance = []
        self._total_power_glance = []
        self._npu_util_glance = []
        self._npu_mem_used_mb_glance = []
        self._npu_mem_used_pct_glance = []
        self._npu_temp_glance = []
        self._npu_mem_total_mb = None
        self._power_trace = []
        self._npu_power_trace = []
        self._ddr_power_trace = []
        self._pmic_power_trace = []
        self._util_trace = []
        self._mem_used_trace = []
        self._mem_used_pct_trace = []
        self._temp_trace = []


def _parse_mobilint_status_static_info(status_output: str) -> dict[str, object]:
    """Parse static NPU fields from ``mobilint-cli status`` table output."""
    return parse_mobilint_status_static_info(status_output)


def _parse_mobilint_status_query_metrics(
    status_output: str,
) -> Optional[
    tuple[
        float,
        float,
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
    ]
]:
    parsed = parse_mobilint_status_query_output(status_output)
    devices = parsed.get("devices")
    if not isinstance(devices, list) or not devices:
        return None
    first_device = devices[0]
    if not isinstance(first_device, dict):
        return None

    power = first_device.get("Power")
    if not isinstance(power, dict):
        return None
    npu_power_w = _parse_status_number(power.get("NPU"))
    total_power_w = _parse_status_number(power.get("Total"))
    if npu_power_w is None or total_power_w is None:
        return None
    ddr_power_w = _parse_status_number(power.get("DDR"))
    pmic_power_w = _parse_status_number(power.get("PMIC"))

    utilization = first_device.get("Utilization")
    npu_util_pct = None
    if isinstance(utilization, dict):
        npu_util_pct = _parse_status_number(utilization.get("Total"))

    memory = first_device.get("Memory")
    npu_mem_used_mb = None
    npu_mem_total_mb = None
    npu_mem_used_pct = None
    if isinstance(memory, dict):
        npu_mem_used_mb = _parse_status_number(memory.get("Usage"))
        npu_mem_total_mb = _parse_status_number(memory.get("Total"))
        if npu_mem_used_mb is not None and npu_mem_total_mb not in (None, 0.0):
            npu_mem_used_pct = (npu_mem_used_mb / npu_mem_total_mb) * 100.0

    npu_temp_c = _parse_status_number(first_device.get("Temperature"))
    return (
        npu_power_w,
        total_power_w,
        ddr_power_w,
        pmic_power_w,
        npu_util_pct,
        npu_mem_used_mb,
        npu_mem_total_mb,
        npu_mem_used_pct,
        npu_temp_c,
    )


def _get_optional_payload_float(
    payload: dict[str, object],
    *keys: str,
) -> Optional[float]:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return float(value)
    return None


def _parse_status_number(value: object) -> Optional[float]:
    if not isinstance(value, str):
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match is not None else None
