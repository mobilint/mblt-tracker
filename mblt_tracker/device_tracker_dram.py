import time
from typing import Optional, Union

import numpy as np

from ._logging import suppress_pyrapl_optional_output_warnings
from .device_tracker import BaseDeviceTracker
from .static_info import get_host_static_info

with suppress_pyrapl_optional_output_warnings():
    import pyRAPL


class DRAMDeviceTracker(BaseDeviceTracker):
    """Track host DRAM power through the Intel RAPL DRAM domain."""

    def __init__(
        self,
        interval: float = 0.1,
        socket_id: Union[int, list[int], None] = None,
    ):
        """Initialize the DRAM device tracker.

        Args:
            interval (float): The interval in seconds at which DRAM should be polled.
            socket_id (Union[int, list[int], None]): Specific CPU socket IDs whose
                DRAM domains should be tracked. If None, all detected sockets are
                tracked.

        Raises:
            RuntimeError: If the host does not expose a RAPL DRAM energy domain.
            ValueError: If no sockets are found or if an invalid socket ID is provided.
        """
        super().__init__(interval=interval)
        try:
            pyRAPL.setup(devices=[pyRAPL.Device.DRAM])
        except Exception as exc:
            raise RuntimeError(
                "DRAM power tracking is not supported on this system"
            ) from exc

        self.num_sockets = self.socket_num()
        if self.num_sockets == 0:
            raise ValueError("No CPU sockets found")

        if socket_id is None:
            socket_id = list(range(self.num_sockets))
        elif isinstance(socket_id, int):
            if socket_id < 0 or socket_id >= self.num_sockets:
                raise ValueError(f"Invalid socket ID: {socket_id}")
            socket_id = [socket_id]
        else:
            for i in socket_id:
                if i < 0 or i >= self.num_sockets:
                    raise ValueError(f"Invalid socket ID: {i}")

        self._socket_id = socket_id
        self._job_id = "dram_device_track"
        self._power_glance = {socket: [] for socket in self._socket_id}
        self._power_trace: list[tuple[float, float]] = []
        self._meter: Optional[pyRAPL.Measurement] = None

    def socket_num(self) -> int:
        """Get the number of detected CPU sockets."""
        # pylint: disable=protected-access
        return len(pyRAPL._sensor._socket_ids)

    def _func_for_sched(self) -> None:
        """Sample host DRAM power via pyRAPL."""
        ts = time.time()
        meter = self._meter
        if meter is not None:
            try:
                meter.end()
                duration_s = meter.result.duration / 1_000_000.0
                dram_energy_uj = meter.result.dram
                if duration_s > 0 and dram_energy_uj is not None:
                    total_power_w = 0.0
                    power_samples = 0
                    for socket_id in self._socket_id:
                        if socket_id >= len(dram_energy_uj):
                            continue
                        energy_uj = dram_energy_uj[socket_id]
                        if energy_uj is None or energy_uj < 0:
                            continue
                        power_w = (energy_uj / 1_000_000.0) / duration_s
                        self._power_glance[socket_id].append(power_w)
                        total_power_w += power_w
                        power_samples += 1
                    if power_samples:
                        self._power_trace.append((ts, total_power_w))
            except Exception:
                pass

        self._meter = pyRAPL.Measurement("dram")
        self._meter.begin()

    def get_metric(self) -> dict[str, object]:
        """Return summarized host DRAM power metrics since start or last reset."""
        dram_stats = {}
        for socket in self._socket_id:
            power_samples = self._power_glance[socket]
            dram_stats[socket] = {
                "avg_power_w": float(np.mean(power_samples)) if power_samples else None,
                "p99_power_w": (
                    float(np.percentile(power_samples, 99)) if power_samples else None
                ),
                "max_power_w": float(np.max(power_samples)) if power_samples else None,
            }

        total_power_samples = [p for _, p in self._power_trace]
        avg_power = (
            float(np.mean(total_power_samples)) if total_power_samples else None
        )
        p99_power = (
            float(np.percentile(total_power_samples, 99))
            if total_power_samples
            else None
        )
        max_power = float(np.max(total_power_samples)) if total_power_samples else None

        return {
            "avg_power_w": avg_power,
            "p99_power_w": p99_power,
            "max_power_w": max_power,
            "avg_dram_power_w": avg_power,
            "p99_dram_power_w": p99_power,
            "max_dram_power_w": max_power,
            "samples": len(total_power_samples),
            "dram": dram_stats,
        }

    def get_trace(self) -> list[tuple[float, float]]:
        """Return a time-series trace of total host DRAM power."""
        return list(self._power_trace)

    def get_static_info(self) -> dict[str, object]:
        """Return shared host static info with DRAM units and motherboard metadata."""
        return get_host_static_info()

    def reset(self) -> None:
        """Reset all collected DRAM metrics and traces."""
        self._power_glance = {socket: [] for socket in self._socket_id}
        self._power_trace = []
        self._meter = None
