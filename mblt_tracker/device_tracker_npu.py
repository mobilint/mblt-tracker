import json
import os
import platform
import re
import shlex
import subprocess
import time
from typing import Optional, Union

import numpy as np

from .device_tracker import BaseDeviceTracker
from .static_info import (
    _deep_merge,
    _filter_npu_metadata_to_selected_devices,
    _mla400_static_card_id,
    get_all_pcie_devices,
    get_pcie_static_info,
    get_windows_npu_driver_firmware_info,
    parse_mobilint_status_query_output,
    parse_mobilint_status_static_info,
    run_command,
)

_DEFAULT_STATUS_CMD = "mobilint-cli status -q"
_MLA400_SUBSYSTEM_VENDOR_ID = "0x402"
_MLA400_SUBSYSTEM_DEVICE_ID = "0x108b"
_MLA100_SUBSYSTEM_VENDOR_ID = "0x401"
_MLA100_SUBSYSTEM_DEVICE_ID = "0x1093"

_MetricTuple = tuple[
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
_MetricSample = dict[str, object]


class NPUDeviceTracker(BaseDeviceTracker):
    """Track NPU power and utilization by polling `mobilint-cli status`."""

    def __init__(
        self,
        interval: float = 0.5,
        status_cmd: Optional[str] = None,
        npu_id: Union[int, list[int], None] = None,
    ):
        """Initialize the NPU device tracker.

        Args:
            interval (float): The interval in seconds at which the NPU should be polled.
            status_cmd (Optional[str]): Custom command to fetch NPU status.
                Defaults to `mobilint-cli status -q`.
            npu_id (Union[int, list[int], None]): Logical NPU card indices to track.
                If None, all detected logical cards are tracked. MLA400 boards are
                grouped as one logical card when `mobilint-cli status -q` exposes
                the GOLDFINGER power rail.

        Raises:
            RuntimeError: If the operating system is not Linux.
        """
        super().__init__(interval=interval)
        if platform.system() != "Linux":
            raise RuntimeError("NPUDeviceTracker currently supports Linux only")
        self._status_cmd = status_cmd if status_cmd is not None else _DEFAULT_STATUS_CMD
        if isinstance(npu_id, int):
            if npu_id < 0:
                raise ValueError(f"Invalid NPU ID: {npu_id}")
            npu_id = [npu_id]
        elif npu_id is not None:
            for i in npu_id:
                if i < 0:
                    raise ValueError(f"Invalid NPU ID: {i}")
        self._npu_id = npu_id
        self._job_id = "npu_device_track"
        self._npu_power_glance: list[float] = []
        self._ddr_power_glance: list[float] = []
        self._pmic_power_glance: list[float] = []
        self._goldfinger_power_glance: list[float] = []
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
        self._goldfinger_power_trace: list[tuple[float, float]] = []
        self._util_trace: list[tuple[float, float]] = []
        self._mem_used_trace: list[tuple[float, float]] = []
        self._mem_used_pct_trace: list[tuple[float, float]] = []
        self._temp_trace: list[tuple[float, float]] = []
        self._npu_metric_glance: dict[int, dict[str, list[float]]] = {}
        self._npu_memory_total_mb: dict[int, float] = {}
        self._npu_card_model: dict[int, str] = {}

    def _fetch_metrics(
        self,
    ) -> Optional[
        _MetricTuple
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
        output = _run_status_command(shlex.split(self._status_cmd))
        metrics = _parse_mobilint_status_metrics(output) if output is not None else None
        if metrics is not None:
            return metrics

        if self._status_cmd != _DEFAULT_STATUS_CMD:
            return None

        fallback_output = _run_status_command(_legacy_status_json_command())
        if fallback_output is None:
            return None
        return _parse_mobilint_status_metrics(fallback_output)

    def _fetch_metric_samples(self) -> Optional[list[_MetricSample]]:
        """Fetch logical NPU card samples.

        `mobilint-cli status -q` samples are grouped into logical cards. MLA400
        is detected by the GOLDFINGER power rail and represented as one card;
        MLA100 devices remain one card per device.
        """
        has_fetch_metrics_override = (
            "_fetch_metrics" in self.__dict__
            or type(self)._fetch_metrics is not NPUDeviceTracker._fetch_metrics
        )
        if has_fetch_metrics_override:
            metrics = self._fetch_metrics()
            return (
                _filter_metric_samples(
                    _metric_tuple_to_samples(metrics),
                    getattr(self, "_npu_id", None),
                )
                if metrics is not None
                else None
            )

        output = _run_status_command(shlex.split(self._status_cmd))
        samples = (
            _parse_mobilint_status_query_metric_samples(output)
            if output is not None
            else None
        )
        if samples:
            return _filter_metric_samples(samples, getattr(self, "_npu_id", None))

        metrics = _parse_mobilint_status_json_metrics(output) if output is not None else None
        if metrics is not None:
            return _filter_metric_samples(
                _metric_tuple_to_samples(metrics),
                getattr(self, "_npu_id", None),
            )

        if self._status_cmd != _DEFAULT_STATUS_CMD:
            return None

        fallback_output = _run_status_command(_legacy_status_json_command())
        if fallback_output is None:
            return None
        metrics = _parse_mobilint_status_metrics(fallback_output)
        return (
            _filter_metric_samples(
                _metric_tuple_to_samples(metrics),
                getattr(self, "_npu_id", None),
            )
            if metrics is not None
            else None
        )

    def _func_for_sched(self) -> None:
        """Sample NPU metrics via the background scheduler."""
        samples = self._fetch_metric_samples()
        if not samples:
            return
        ts = time.time()
        total_power_w = _sum_sample_key(samples, "total_power_w")
        npu_power_w = _sum_sample_key(samples, "npu_power_w")
        ddr_power_w = _sum_optional_sample_key(samples, "ddr_power_w")
        pmic_power_w = _sum_optional_sample_key(samples, "pmic_power_w")
        goldfinger_power_w = _sum_optional_sample_key(samples, "goldfinger_power_w")
        npu_util_pct = _avg_optional_sample_key(samples, "npu_util_pct")
        npu_mem_used_mb = _sum_optional_sample_key(samples, "npu_mem_used_mb")
        npu_mem_total_mb = _sum_optional_sample_key(samples, "npu_mem_total_mb")
        npu_mem_used_pct = (
            (npu_mem_used_mb / npu_mem_total_mb) * 100.0
            if npu_mem_used_mb is not None and npu_mem_total_mb not in (None, 0.0)
            else _avg_optional_sample_key(samples, "npu_mem_used_pct")
        )
        npu_temp_c = _avg_optional_sample_key(samples, "npu_temp_c")

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
        if goldfinger_power_w is not None:
            self._goldfinger_power_glance.append(goldfinger_power_w)
            self._goldfinger_power_trace.append((ts, goldfinger_power_w))
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
        for sample in samples:
            _record_per_npu_sample(self, sample)

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
        goldfinger_samples = getattr(self, "_goldfinger_power_glance", [])
        goldfinger_avg = float(np.mean(goldfinger_samples)) if goldfinger_samples else None
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
        goldfinger_p99 = (
            float(np.percentile(goldfinger_samples, 99)) if goldfinger_samples else None
        )
        goldfinger_max = float(np.max(goldfinger_samples)) if goldfinger_samples else None
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
            "avg_goldfinger_power_w": goldfinger_avg,
            "p99_goldfinger_power_w": goldfinger_p99,
            "max_goldfinger_power_w": goldfinger_max,
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
            "goldfinger_power_samples": len(
                getattr(self, "_goldfinger_power_trace", [])
            ),
            "util_samples": len(self._util_trace),
            "npu": _summarize_per_npu_metrics(self),
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

    def get_goldfinger_power_trace(self) -> list[tuple[float, float]]:
        """Return a time-series trace of MLA400 GOLDFINGER input power."""
        return list(getattr(self, "_goldfinger_power_trace", []))

    def get_temp_trace(self) -> list[tuple[float, float]]:
        """Return a time-series trace of NPU temperature."""
        return list(self._temp_trace)

    def reset(self) -> None:
        """Reset all collected NPU metrics and traces."""
        self._npu_power_glance = []
        self._ddr_power_glance = []
        self._pmic_power_glance = []
        self._goldfinger_power_glance = []
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
        self._goldfinger_power_trace = []
        self._util_trace = []
        self._mem_used_trace = []
        self._mem_used_pct_trace = []
        self._temp_trace = []
        self._npu_metric_glance = {}
        self._npu_memory_total_mb = {}
        self._npu_card_model = {}


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


def _parse_mobilint_status_query_metric_samples(
    status_output: str,
) -> Optional[list[_MetricSample]]:
    parsed = parse_mobilint_status_query_output(status_output)
    devices = parsed.get("devices")
    if not isinstance(devices, list) or not devices:
        return None
    raw_devices = [device for device in devices if isinstance(device, dict)]
    device_samples = [_query_device_to_metric_sample(device) for device in raw_devices]
    device_samples = [sample for sample in device_samples if sample is not None]
    if not device_samples:
        return None

    return _group_query_metric_samples(device_samples)


def _group_query_metric_samples(samples: list[_MetricSample]) -> list[_MetricSample]:
    grouped_samples: list[_MetricSample] = []
    mla400_groups: dict[int, list[_MetricSample]] = {}

    for sample in samples:
        sample["card_model"] = _classify_sample_card_model(sample)
        if sample["card_model"] == "MLA400":
            group_id = _mla400_group_id(sample, len(mla400_groups))
            mla400_groups.setdefault(group_id, []).append(sample)
        else:
            grouped_samples.append(sample)

    for group_id, group_samples in sorted(mla400_groups.items()):
        grouped_samples.append(_aggregate_mla400_samples(group_samples, group_id))

    grouped_samples.sort(key=_sample_sort_key)
    return grouped_samples


def _mla400_group_id(sample: _MetricSample, fallback_group_id: int) -> int:
    dev_no = _sample_int(sample, "dev_no")
    return _mla400_static_card_id(dev_no, fallback_group_id * 4)


def _sample_sort_key(sample: _MetricSample) -> tuple[int, int]:
    devices = sample.get("devices")
    if isinstance(devices, list):
        dev_nos = [
            _sample_int(device, "dev_no")
            for device in devices
            if isinstance(device, dict)
        ]
        dev_nos = [dev_no for dev_no in dev_nos if dev_no is not None]
        if dev_nos:
            return (min(dev_nos), 0)

    dev_no = _sample_int(sample, "dev_no")
    if dev_no is not None:
        return (dev_no, 1)
    return (10**9, 1)


def _query_device_to_metric_sample(device: dict[str, object]) -> Optional[_MetricSample]:
    power = device.get("Power")
    if not isinstance(power, dict):
        return None
    npu_power_w = _parse_status_number(power.get("NPU"))
    total_power_w = _parse_status_number(power.get("Total"))
    if npu_power_w is None or total_power_w is None:
        return None

    memory = device.get("Memory")
    npu_mem_used_mb = None
    npu_mem_total_mb = None
    npu_mem_used_pct = None
    if isinstance(memory, dict):
        npu_mem_used_mb = _parse_status_number(memory.get("Usage"))
        npu_mem_total_mb = _parse_status_number(memory.get("Total"))
        if npu_mem_used_mb is not None and npu_mem_total_mb not in (None, 0.0):
            npu_mem_used_pct = (npu_mem_used_mb / npu_mem_total_mb) * 100.0

    utilization = device.get("Utilization")
    npu_util_pct = None
    if isinstance(utilization, dict):
        npu_util_pct = _parse_status_number(utilization.get("Total"))

    sample: _MetricSample = {
        "card_id": _parse_device_index_from_path(_get_status_str(device.get("path"))),
        "dev_no": _parse_device_index_from_path(_get_status_str(device.get("path"))),
        "board_name": os.path.basename(_get_status_str(device.get("path")) or ""),
        "npu_power_w": npu_power_w,
        "total_power_w": total_power_w,
        "ddr_power_w": _parse_status_number(power.get("DDR")),
        "pmic_power_w": _parse_status_number(power.get("PMIC")),
        "goldfinger_power_w": _parse_status_number(power.get("GOLDFINGER")),
        "npu_util_pct": npu_util_pct,
        "npu_mem_used_mb": npu_mem_used_mb,
        "npu_mem_total_mb": npu_mem_total_mb,
        "npu_mem_used_pct": npu_mem_used_pct,
        "npu_temp_c": _parse_status_number(device.get("Temperature")),
    }
    pcie = device.get("PCI Express")
    if isinstance(pcie, dict):
        sample["subsystem_vendor_id"] = _get_status_str(pcie.get("Sub Vendor ID"))
        sample["subsystem_device_id"] = _get_status_str(pcie.get("Sub Device ID"))
    return sample


def _aggregate_mla400_samples(
    samples: list[_MetricSample], card_id: int
) -> _MetricSample:
    total_power_values = [_sample_float(sample, "total_power_w") for sample in samples]
    non_zero_total_power = [value for value in total_power_values if value is not None and value > 0]
    npu_mem_used_mb = _sum_optional_sample_key(samples, "npu_mem_used_mb")
    npu_mem_total_mb = _sum_optional_sample_key(samples, "npu_mem_total_mb")
    return {
        "card_id": card_id,
        "card_model": "MLA400",
        "chip_count": len(samples),
        "npu_power_w": _sum_sample_key(samples, "npu_power_w"),
        "total_power_w": sum(non_zero_total_power)
        if non_zero_total_power
        else _sum_sample_key(samples, "total_power_w"),
        "ddr_power_w": _sum_optional_sample_key(samples, "ddr_power_w"),
        "pmic_power_w": _sum_optional_sample_key(samples, "pmic_power_w"),
        "goldfinger_power_w": _sum_optional_sample_key(samples, "goldfinger_power_w"),
        "npu_util_pct": _avg_optional_sample_key(samples, "npu_util_pct"),
        "npu_mem_used_mb": npu_mem_used_mb,
        "npu_mem_total_mb": npu_mem_total_mb,
        "npu_mem_used_pct": (
            (npu_mem_used_mb / npu_mem_total_mb) * 100.0
            if npu_mem_used_mb is not None and npu_mem_total_mb not in (None, 0.0)
            else None
        ),
        "npu_temp_c": _avg_optional_sample_key(samples, "npu_temp_c"),
        "devices": samples,
    }


def _filter_metric_samples(
    samples: list[_MetricSample], npu_id: Optional[list[int]]
) -> list[_MetricSample]:
    if npu_id is None:
        return samples
    selected_ids = set(npu_id)
    return [
        sample
        for sample in samples
        if _sample_int(sample, "card_id") in selected_ids
        or _sample_int(sample, "dev_no") in selected_ids
    ]


def _metric_tuple_to_samples(metrics: _MetricTuple) -> list[_MetricSample]:
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
    return [
        {
            "card_id": 0,
            "card_model": "unknown",
            "npu_power_w": npu_power_w,
            "total_power_w": total_power_w,
            "ddr_power_w": ddr_power_w,
            "pmic_power_w": pmic_power_w,
            "goldfinger_power_w": None,
            "npu_util_pct": npu_util_pct,
            "npu_mem_used_mb": npu_mem_used_mb,
            "npu_mem_total_mb": npu_mem_total_mb,
            "npu_mem_used_pct": npu_mem_used_pct,
            "npu_temp_c": npu_temp_c,
        }
    ]


def _classify_sample_card_model(sample: _MetricSample) -> str:
    if sample.get("goldfinger_power_w") is not None:
        return "MLA400"
    subsystem_vendor_id = _normalize_status_hex(sample.get("subsystem_vendor_id"))
    subsystem_device_id = _normalize_status_hex(sample.get("subsystem_device_id"))
    if (
        subsystem_vendor_id == _MLA400_SUBSYSTEM_VENDOR_ID.lower()
        and subsystem_device_id == _MLA400_SUBSYSTEM_DEVICE_ID.lower()
    ):
        return "MLA400"
    if (
        subsystem_vendor_id == _MLA100_SUBSYSTEM_VENDOR_ID.lower()
        and subsystem_device_id == _MLA100_SUBSYSTEM_DEVICE_ID.lower()
    ):
        return "MLA100"
    return "unknown"


def _record_per_npu_sample(tracker: NPUDeviceTracker, sample: _MetricSample) -> None:
    card_id = _sample_int(sample, "card_id")
    if card_id is None:
        return
    glance = tracker._npu_metric_glance.setdefault(
        card_id,
        {
            "npu_power_w": [],
            "total_power_w": [],
            "ddr_power_w": [],
            "pmic_power_w": [],
            "goldfinger_power_w": [],
            "npu_util_pct": [],
            "npu_mem_used_mb": [],
            "npu_mem_used_pct": [],
            "npu_temp_c": [],
        },
    )
    for key in glance:
        value = _sample_float(sample, key)
        if value is not None:
            glance[key].append(value)
    mem_total = _sample_float(sample, "npu_mem_total_mb")
    if mem_total is not None:
        tracker._npu_memory_total_mb[card_id] = mem_total
    card_model = sample.get("card_model")
    if isinstance(card_model, str):
        tracker._npu_card_model[card_id] = card_model


def _summarize_per_npu_metrics(tracker: NPUDeviceTracker) -> dict[int, dict[str, object]]:
    stats: dict[int, dict[str, object]] = {}
    for card_id, glance in sorted(getattr(tracker, "_npu_metric_glance", {}).items()):
        stats[card_id] = {
            "card_model": getattr(tracker, "_npu_card_model", {}).get(card_id),
            "avg_power_w": _mean_or_none(glance["total_power_w"]),
            "p99_power_w": _p99_or_none(glance["total_power_w"]),
            "max_power_w": _max_or_none(glance["total_power_w"]),
            "avg_npu_power_w": _mean_or_none(glance["npu_power_w"]),
            "p99_npu_power_w": _p99_or_none(glance["npu_power_w"]),
            "max_npu_power_w": _max_or_none(glance["npu_power_w"]),
            "avg_ddr_power_w": _mean_or_none(glance["ddr_power_w"]),
            "avg_pmic_power_w": _mean_or_none(glance["pmic_power_w"]),
            "avg_goldfinger_power_w": _mean_or_none(glance["goldfinger_power_w"]),
            "avg_utilization_pct": _mean_or_none(glance["npu_util_pct"]),
            "avg_memory_used_mb": _mean_or_none(glance["npu_mem_used_mb"]),
            "total_memory_mb": getattr(tracker, "_npu_memory_total_mb", {}).get(card_id),
            "avg_memory_used_pct": _mean_or_none(glance["npu_mem_used_pct"]),
            "avg_temperature_c": _mean_or_none(glance["npu_temp_c"]),
        }
    return stats


def _sum_sample_key(samples: list[_MetricSample], key: str) -> float:
    return sum(_sample_float(sample, key) or 0.0 for sample in samples)


def _sum_optional_sample_key(samples: list[_MetricSample], key: str) -> Optional[float]:
    values = [_sample_float(sample, key) for sample in samples]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _avg_optional_sample_key(samples: list[_MetricSample], key: str) -> Optional[float]:
    values = [_sample_float(sample, key) for sample in samples]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _sample_float(sample: _MetricSample, key: str) -> Optional[float]:
    value = sample.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _sample_int(sample: _MetricSample, key: str) -> Optional[int]:
    value = sample.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _mean_or_none(values: list[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _p99_or_none(values: list[float]) -> Optional[float]:
    return float(np.percentile(values, 99)) if values else None


def _max_or_none(values: list[float]) -> Optional[float]:
    return float(np.max(values)) if values else None


def _get_status_str(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _parse_device_index_from_path(path: Optional[str]) -> Optional[int]:
    if path is None:
        return None
    match = re.search(r"(\d+)$", path)
    return int(match.group(1)) if match is not None else None


def _normalize_status_hex(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if not value:
        return None
    hex_digits = value[2:] if value.startswith("0x") else value
    hex_digits = hex_digits.lstrip("0") or "0"
    return f"0x{hex_digits}"


def _parse_mobilint_status_metrics(status_output: str):
    metrics = _parse_mobilint_status_query_metrics(status_output)
    if metrics is not None:
        return metrics
    return _parse_mobilint_status_json_metrics(status_output)


def _parse_mobilint_status_json_metrics(status_output: str):
    try:
        payload = json.loads(status_output)
    except Exception:
        return None

    if not isinstance(payload, dict) or not payload.get("ok", False):
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


def _run_status_command(command: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    if result.returncode != 0 or not result.stdout:
        return None
    return result.stdout.strip()


def _legacy_status_json_command() -> list[str]:
    script_path = os.path.join(os.path.dirname(__file__), "device_tracker_npu.sh")
    return ["bash", script_path, "--sample-once", "--json"]


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
