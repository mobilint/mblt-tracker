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
    get_pcie_static_info,
    get_windows_npu_driver_firmware_info,
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
                Defaults to using the internal `device_tracker_npu.sh` script.

        Raises:
            RuntimeError: If the operating system is not Linux.
        """
        super().__init__(interval=interval)
        if platform.system() != "Linux":
            raise RuntimeError("NPUDeviceTracker currently supports Linux only")
        script_path = os.path.join(os.path.dirname(__file__), "device_tracker_npu.sh")
        self._status_cmd = (
            status_cmd
            if status_cmd is not None
            else f"bash {script_path} --sample-once --json"
        )
        self._job_id = "npu_device_track"
        self._npu_power_glance: list[float] = []
        self._total_power_glance: list[float] = []
        self._npu_util_glance: list[float] = []
        self._npu_mem_used_mb_glance: list[float] = []
        self._npu_mem_used_pct_glance: list[float] = []
        self._npu_temp_glance: list[float] = []
        self._npu_mem_total_mb: Optional[float] = None
        self._power_trace: list[tuple[float, float]] = []
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
        ]
    ]:
        """Execute the status command and parse the JSON output.

        Returns:
            Optional[tuple]: A tuple containing (npu_power_w, total_power_w,
                npu_util_pct, npu_mem_used_mb, npu_mem_total_mb,
                npu_mem_used_pct, npu_temp_c) if successful, else None.
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

        try:
            payload = json.loads(result.stdout.strip())
        except Exception:
            return None

        if not payload.get("ok", False):
            return None
        if "npu_power_w" not in payload or "total_power_w" not in payload:
            return None

        npu_power_w = float(payload["npu_power_w"])
        total_power_w = float(payload["total_power_w"])
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
        info = get_pcie_static_info(
            vendor_id=os.environ.get("MBLT_TRACKER_NPU_PCI_VENDOR_ID", "1ed5"),
            device_id=os.environ.get("MBLT_TRACKER_NPU_PCI_DEVICE_ID"),
            class_filter=os.environ.get("MBLT_TRACKER_NPU_PCI_CLASS_FILTER"),
        )
        if platform.system() == "Windows":
            _deep_merge(info, get_windows_npu_driver_firmware_info())
        else:
            status_output = run_command(["mobilint-cli", "status"])
            if status_output:
                _deep_merge(info, _parse_mobilint_status_static_info(status_output))
        return info

    def get_util_trace(self) -> list[tuple[float, float]]:
        """Return a time-series trace of NPU utilization.

        Returns:
            list[tuple[float, float]]: List of (timestamp, npu_util_pct) pairs.
        """
        return list(self._util_trace)

    def get_temp_trace(self) -> list[tuple[float, float]]:
        """Return a time-series trace of NPU temperature."""
        return list(self._temp_trace)

    def reset(self) -> None:
        """Reset all collected NPU metrics and traces."""
        self._npu_power_glance = []
        self._total_power_glance = []
        self._npu_util_glance = []
        self._npu_mem_used_mb_glance = []
        self._npu_mem_used_pct_glance = []
        self._npu_temp_glance = []
        self._npu_mem_total_mb = None
        self._power_trace = []
        self._util_trace = []
        self._mem_used_trace = []
        self._mem_used_pct_trace = []
        self._temp_trace = []


def _parse_mobilint_status_static_info(status_output: str) -> dict[str, object]:
    """Parse static NPU fields from ``mobilint-cli status`` table output."""
    return parse_mobilint_status_static_info(status_output)
