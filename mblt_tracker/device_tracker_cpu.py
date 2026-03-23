import logging
import time
from typing import Optional, Union

import numpy as np
import psutil
import pyRAPL

from .device_tracker import BaseDeviceTracker

logging.getLogger().setLevel(logging.ERROR)


class CPUDeviceTracker(BaseDeviceTracker):
    """Track CPU power and utilization through RAPL and psutil."""

    def __init__(
        self, interval: float = 0.1, cpu_id: Union[int, list[int], None] = None
    ):
        """Initialize the CPU device tracker.

        Args:
            interval (float): The interval in seconds at which the CPU should be polled.
            cpu_id (Union[int, List[int], None]): Specific CPU socket IDs to track.
                If None, all detected sockets are tracked.

        Raises:
            ValueError: If no CPU sockets are found or if an invalid CPU ID is provided.
        """
        super().__init__(interval=interval)
        pyRAPL.setup(devices=[pyRAPL.Device.PKG])
        self.num_cpus = self.cpu_num()
        if self.num_cpus == 0:
            raise ValueError("No CPU sockets found")

        if cpu_id is None:
            cpu_id = list(range(self.num_cpus))
        elif isinstance(cpu_id, int):
            if cpu_id < 0 or cpu_id >= self.num_cpus:
                raise ValueError(f"Invalid CPU ID: {cpu_id}")
            cpu_id = [cpu_id]
        else:
            for i in cpu_id:
                if i < 0 or i >= self.num_cpus:
                    raise ValueError(f"Invalid CPU ID: {i}")
        self._cpu_id = cpu_id
        self._job_id = "cpu_device_track"
        self._power_glance = {cpu: [] for cpu in self._cpu_id}
        self._cpu_util_glance = {cpu: [] for cpu in self._cpu_id}
        self._mem_util_glance = {cpu: [] for cpu in self._cpu_id}
        self._mem_used_glance = {cpu: [] for cpu in self._cpu_id}
        self._mem_used_pct_glance = {cpu: [] for cpu in self._cpu_id}
        self._mem_total_mb: Optional[float] = None
        self._power_trace: list[tuple[float, float]] = []
        self._cpu_util_trace: list[tuple[float, float]] = []
        self._mem_util_trace: list[tuple[float, float]] = []
        self._mem_used_trace: list[tuple[float, float]] = []
        self._mem_used_pct_trace: list[tuple[float, float]] = []
        self._meter: Optional[pyRAPL.Measurement] = None

    def cpu_num(self) -> int:
        """Get the number of detected CPU sockets.

        Returns:
            int: Number of sockets.
        """
        # pylint: disable=protected-access
        return len(pyRAPL._sensor._socket_ids)

    def _func_for_sched(self) -> None:
        """Sample CPU power, utilization, and memory usage."""
        ts = time.time()

        # CPU Power via pyRAPL
        meter = self._meter
        if meter is not None:
            try:
                meter.end()
                duration_s = meter.result.duration / 1_000_000.0
                if duration_s > 0:
                    total_power_w = 0.0
                    for socket_id in self._cpu_id:
                        # Assuming socket_id corresponds to index in pkg
                        energy_uj = meter.result.pkg[socket_id]
                        power_w = (energy_uj / 1_000_000.0) / duration_s
                        self._power_glance[socket_id].append(power_w)
                        total_power_w += power_w
                    self._power_trace.append((ts, total_power_w))
            except Exception:
                pass

        self._meter = pyRAPL.Measurement("cpu")
        self._meter.begin()

        # CPU Utilization via psutil
        cpu_util_pct = psutil.cpu_percent(interval=None)
        self._cpu_util_trace.append((ts, float(cpu_util_pct)))
        for socket_id in self._cpu_id:
            self._cpu_util_glance[socket_id].append(float(cpu_util_pct))

        # Memory usage via psutil
        mem = psutil.virtual_memory()
        mem_used_mb = mem.used / (1024 * 1024)
        mem_util_pct = mem.percent
        self._mem_total_mb = float(mem.total) / (1024 * 1024)

        self._mem_used_trace.append((ts, float(mem_used_mb)))
        self._mem_used_pct_trace.append((ts, float(mem_util_pct)))
        for socket_id in self._cpu_id:
            self._mem_used_glance[socket_id].append(float(mem_used_mb))
            self._mem_used_pct_glance[socket_id].append(float(mem_util_pct))

    def get_metric(self) -> dict[str, object]:
        """Return summarized CPU metrics since start or last reset.

        Returns:
            Dict[str, object]: A dictionary containing average and peak power,
                utilization, and memory statistics.
        """
        cpu_stats = {}
        for cpu in self._cpu_id:
            power_samples = self._power_glance[cpu]
            util_samples = self._cpu_util_glance[cpu]
            mem_used_samples = self._mem_used_glance[cpu]
            mem_used_pct_samples = self._mem_used_pct_glance[cpu]
            cpu_stats[cpu] = {
                "avg_power_w": float(np.mean(power_samples)) if power_samples else None,
                "p99_power_w": (
                    float(np.percentile(power_samples, 99)) if power_samples else None
                ),
                "avg_util_pct": float(np.mean(util_samples)) if util_samples else None,
                "p99_util_pct": (
                    float(np.percentile(util_samples, 99)) if util_samples else None
                ),
                "max_util_pct": float(np.max(util_samples)) if util_samples else None,
                "max_power_w": float(np.max(power_samples)) if power_samples else None,
                "avg_memory_used_mb": (
                    float(np.mean(mem_used_samples)) if mem_used_samples else None
                ),
                "p99_memory_used_mb": (
                    float(np.percentile(mem_used_samples, 99))
                    if mem_used_samples
                    else None
                ),
                "max_memory_used_mb": (
                    float(np.max(mem_used_samples)) if mem_used_samples else None
                ),
                "total_memory_mb": self._mem_total_mb,
                "avg_memory_used_pct": (
                    float(np.mean(mem_used_pct_samples))
                    if mem_used_pct_samples
                    else None
                ),
                "p99_memory_used_pct": (
                    float(np.percentile(mem_used_pct_samples, 99))
                    if mem_used_pct_samples
                    else None
                ),
                "max_memory_used_pct": (
                    float(np.max(mem_used_pct_samples))
                    if mem_used_pct_samples
                    else None
                ),
            }

        total_power_samples = [p for _, p in self._power_trace]
        total_util_samples = [u for _, u in self._cpu_util_trace]
        total_mem_used_samples = [m for _, m in self._mem_used_trace]
        total_mem_used_pct_samples = [m for _, m in self._mem_used_pct_trace]

        avg_util_pct = (
            float(np.mean(total_util_samples)) if total_util_samples else None
        )
        p99_util_pct = (
            float(np.percentile(total_util_samples, 99)) if total_util_samples else None
        )
        max_util_pct = float(np.max(total_util_samples)) if total_util_samples else None

        return {
            "avg_power_w": (
                float(np.mean(total_power_samples)) if total_power_samples else None
            ),
            "p99_power_w": (
                float(np.percentile(total_power_samples, 99))
                if total_power_samples
                else None
            ),
            "max_power_w": (
                float(np.max(total_power_samples)) if total_power_samples else None
            ),
            "avg_utilization_pct": avg_util_pct,
            "p99_utilization_pct": p99_util_pct,
            "max_utilization_pct": max_util_pct,
            "avg_memory_used_mb": (
                float(np.mean(total_mem_used_samples))
                if total_mem_used_samples
                else None
            ),
            "p99_memory_used_mb": (
                float(np.percentile(total_mem_used_samples, 99))
                if total_mem_used_samples
                else None
            ),
            "max_memory_used_mb": (
                float(np.max(total_mem_used_samples))
                if total_mem_used_samples
                else None
            ),
            "total_memory_mb": self._mem_total_mb,
            "avg_memory_used_pct": (
                float(np.mean(total_mem_used_pct_samples))
                if total_mem_used_pct_samples
                else None
            ),
            "p99_memory_used_pct": (
                float(np.percentile(total_mem_used_pct_samples, 99))
                if total_mem_used_pct_samples
                else None
            ),
            "max_memory_used_pct": (
                float(np.max(total_mem_used_pct_samples))
                if total_mem_used_pct_samples
                else None
            ),
            "samples": len(total_power_samples),
            "util_samples": len(total_util_samples),
            "cpu": cpu_stats,
        }

    def get_trace(self) -> list[tuple[float, float]]:
        """Return a time-series trace of total CPU power.

        Returns:
            list[tuple[float, float]]: List of (timestamp, total_power_w) pairs.
        """
        return list(self._power_trace)

    def get_util_trace(self) -> list[tuple[float, float]]:
        """Return a time-series trace of total CPU utilization.

        Returns:
            list[tuple[float, float]]: List of (timestamp, util_pct) pairs.
        """
        return list(self._cpu_util_trace)

    def reset(self) -> None:
        """Reset all collected CPU metrics and traces."""
        self._power_glance = {cpu: [] for cpu in self._cpu_id}
        self._cpu_util_glance = {cpu: [] for cpu in self._cpu_id}
        self._mem_util_glance = {cpu: [] for cpu in self._cpu_id}
        self._mem_used_glance = {cpu: [] for cpu in self._cpu_id}
        self._mem_used_pct_glance = {cpu: [] for cpu in self._cpu_id}
        self._mem_total_mb = None
        self._power_trace = []
        self._cpu_util_trace = []
        self._mem_util_trace = []
        self._mem_used_trace = []
        self._mem_used_pct_trace = []
