from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable
from typing import Any

import numpy as np

from .device_tracker import BaseDeviceTracker
from .static_info import (
    _deep_merge,
    _sanitize_static_info_for_public_output,
    get_all_pcie_devices,
    get_pcie_static_info,
)

try:  # pragma: no cover - exercised through tests with a fake module
    import mbltml
except Exception:  # pragma: no cover
    mbltml = None  # type: ignore[assignment]


_LOGGER = logging.getLogger(__name__)
_MB_TO_BYTES = 1024 * 1024
_EXTRA_RAIL_REFRESH_PERIOD_S = 1.0
_DEFAULT_RAIL_METRICS = "npu"
_RAIL_ORDER = ("npu", "ddr", "pmic", "goldfinger")


class NPUDeviceTracker(BaseDeviceTracker):
    """Track Mobilint NPU metrics through the OS-independent ``mbltml`` API.

    The default path is intentionally low-latency: it reads total power, the
    default NPU extra-PMIC rail, utilization, memory, and temperature without
    changing the firmware rail selection register.

    Non-NPU extra rails (DDR, PMIC, and GoldFinger) share the same firmware
    register selection as the NPU rail. Changing that selection with
    ``mbltmlSetExtraPmicID`` is reflected by firmware only after its refresh
    cycle, which can take up to about 1 second. For this reason non-NPU rails
    are opt-in via ``rail_metrics`` and are sampled by a non-blocking state
    machine; their effective sampling rate can be lower than ``interval``.
    """

    def __init__(
        self,
        interval: float = 0.5,
        npu_id: int | list[int] | None = None,
        rail_metrics: str | Iterable[str] = _DEFAULT_RAIL_METRICS,
    ):
        """Initialize the NPU device tracker.

        Args:
            interval: Polling interval in seconds.
            npu_id: Physical ``mbltml`` device indices to track. If ``None``,
                all detected devices are tracked.
            rail_metrics: Extra PMIC rails to monitor. ``"npu"`` is the
                default and does not change the firmware rail selection.
                ``"all"`` enables ``npu``, ``ddr``, ``pmic``, and
                ``goldfinger``. Non-NPU rails require a firmware register
                selection change and become valid only after approximately one
                firmware refresh period (1 second).
        """
        super().__init__(interval=interval)
        _ensure_mbltml_available()
        _initialize_mbltml()

        self._npu_id = _normalize_npu_ids(npu_id)
        self._rail_metrics = _normalize_rail_metrics(rail_metrics)
        self._job_id = "npu_device_track"
        self._device_count = _safe_call(mbltml.mbltmlGetDeviceCount, default=0)
        if self._device_count <= 0:
            raise RuntimeError("No Mobilint NPU devices were detected by mbltml")
        invalid_ids = [i for i in self._npu_id or [] if i >= self._device_count]
        if invalid_ids:
            raise ValueError(f"Invalid NPU ID(s): {invalid_ids}")

        self._selected_rail: dict[int, str] = dict.fromkeys(range(self._device_count), "npu")
        self._rail_selected_at: dict[int, float] = dict.fromkeys(range(self._device_count), 0.0)
        self._next_extra_rail_index = 0
        self.reset()

    def _func_for_sched(self) -> None:
        samples = self._fetch_metric_samples()
        if not samples:
            return
        ts = time.time()
        self._record_samples(ts, samples)

    def _fetch_metric_samples(self) -> list[dict[str, Any]]:
        selected = self._selected_device_indices()
        now = time.time()
        extra_rail_to_read = self._advance_extra_rail_state(now) if self._has_extra_rails else None
        samples = []
        for dev_no in selected:
            sample = self._read_device_sample(dev_no, extra_rail_to_read, now)
            if sample is not None:
                samples.append(sample)
        return samples

    @property
    def _has_extra_rails(self) -> bool:
        return any(rail != "npu" for rail in self._rail_metrics)

    def _advance_extra_rail_state(self, now: float) -> str | None:
        extra_rails = [rail for rail in self._rail_metrics if rail != "npu"]
        if not extra_rails:
            return None
        rail = extra_rails[self._next_extra_rail_index % len(extra_rails)]
        self._next_extra_rail_index += 1
        for dev_no in self._selected_device_indices():
            if self._selected_rail.get(dev_no) != rail:
                if _set_extra_rail(dev_no, rail):
                    self._selected_rail[dev_no] = rail
                    self._rail_selected_at[dev_no] = now
        return rail

    def _read_device_sample(
        self, dev_no: int, extra_rail_to_read: str | None, now: float
    ) -> dict[str, Any] | None:
        total_power_w = _safe_call(mbltml.mbltmlGetTotalPower, dev_no)
        if total_power_w is None:
            return None
        memory_usage_bytes = _safe_call(mbltml.mbltmlGetMemoryUsage, dev_no)
        memory_total_bytes = _safe_call(mbltml.mbltmlGetMemoryTotal, dev_no)
        memory_usage_mb = _bytes_to_mb(memory_usage_bytes)
        memory_total_mb = _bytes_to_mb(memory_total_bytes)
        sample: dict[str, Any] = {
            "dev_no": dev_no,
            "total_power_w": float(total_power_w),
            "total_current_a": _safe_call(mbltml.mbltmlGetTotalCurrent, dev_no),
            "total_voltage_v": _safe_call(mbltml.mbltmlGetTotalVoltage, dev_no),
            "total_utilization_pct": _normalize_utilization(
                _safe_call(mbltml.mbltmlGetTotalUtilization, dev_no)
            ),
            "memory_usage_mb": memory_usage_mb,
            "memory_total_mb": memory_total_mb,
            "memory_usage_pct": _usage_pct(memory_usage_mb, memory_total_mb),
            "temperature_c": _safe_call(mbltml.mbltmlGetTemperature, dev_no),
            "node_name": _safe_call(mbltml.mbltmlGetNodeName, dev_no),
            "hardware_version": _hardware_version_name(
                _safe_call(mbltml.mbltmlGetHardwareVersion, dev_no)
            ),
        }
        if extra_rail_to_read is not None and self._selected_rail.get(dev_no) == extra_rail_to_read:
            selected_at = self._rail_selected_at.get(dev_no, 0.0)
            if now - selected_at >= _EXTRA_RAIL_REFRESH_PERIOD_S:
                sample.update(_read_rail_values(dev_no, extra_rail_to_read))
                if "npu" in self._rail_metrics and _set_extra_rail(dev_no, "npu"):
                    self._selected_rail[dev_no] = "npu"
                    self._rail_selected_at[dev_no] = now
        elif "npu" in self._rail_metrics:
            if self._selected_rail.get(dev_no) == "npu":
                sample.update(_read_rail_values(dev_no, "npu"))
            elif _set_extra_rail(dev_no, "npu"):
                self._selected_rail[dev_no] = "npu"
                self._rail_selected_at[dev_no] = now
        return sample

    def _record_samples(self, ts: float, samples: list[dict[str, Any]]) -> None:
        self._append("total_power_w", _sum(samples, "total_power_w"), ts)
        self._append("total_current_a", _sum_optional(samples, "total_current_a"), ts)
        self._append("total_voltage_v", _avg_optional(samples, "total_voltage_v"), ts)
        self._append("total_utilization_pct", _avg_optional(samples, "total_utilization_pct"), ts)
        self._append("memory_usage_mb", _sum_optional(samples, "memory_usage_mb"), ts)
        memory_total_mb = _sum_optional(samples, "memory_total_mb")
        if memory_total_mb is not None:
            self._memory_total_mb = memory_total_mb
        self._append("memory_usage_pct", _avg_optional(samples, "memory_usage_pct"), ts)
        self._append("temperature_c", _avg_optional(samples, "temperature_c"), ts)
        for rail in _RAIL_ORDER:
            self._append(f"{rail}_rail_power_w", _sum_optional(samples, f"{rail}_rail_power_w"), ts)
            self._append(f"{rail}_rail_current_a", _sum_optional(samples, f"{rail}_rail_current_a"), ts)
            self._append(f"{rail}_rail_voltage_v", _avg_optional(samples, f"{rail}_rail_voltage_v"), ts)
        for sample in samples:
            self._record_device_sample(sample)

    def _append(self, key: str, value: float | None, ts: float) -> None:
        if value is None:
            return
        self._glance[key].append(float(value))
        self._traces[key].append((ts, float(value)))

    def _record_device_sample(self, sample: dict[str, Any]) -> None:
        dev_no = sample["dev_no"]
        glance = self._device_glance.setdefault(dev_no, {key: [] for key in self._glance})
        for key in glance:
            value = _sample_float(sample, key)
            if value is not None:
                glance[key].append(value)
        self._device_metadata[dev_no] = {
            "node_name": sample.get("node_name"),
            "hardware_version": sample.get("hardware_version"),
        }
        memory_total_mb = _sample_float(sample, "memory_total_mb")
        if memory_total_mb is not None:
            self._device_memory_total_mb[dev_no] = memory_total_mb

    def _selected_device_indices(self) -> list[int]:
        if self._npu_id is None:
            return list(range(self._device_count))
        return list(self._npu_id)

    def get_metric(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for key, values in self._glance.items():
            metrics[f"avg_{key}"] = _mean_or_none(values)
            metrics[f"p99_{key}"] = _p99_or_none(values)
            metrics[f"max_{key}"] = _max_or_none(values)
        metrics["memory_total_mb"] = self._memory_total_mb
        metrics["samples"] = len(self._traces["total_power_w"])
        for key, trace in self._traces.items():
            metrics[f"{key}_samples"] = len(trace)
        metrics["rail_metrics"] = {
            "selected": list(self._rail_metrics),
            "firmware_refresh_period_s": _EXTRA_RAIL_REFRESH_PERIOD_S,
            "extra_rail_requires_selection_delay": True,
        }
        metrics["devices"] = self._summarize_devices()
        return metrics

    def _summarize_devices(self) -> dict[int, dict[str, Any]]:
        devices: dict[int, dict[str, Any]] = {}
        for dev_no, glance in sorted(self._device_glance.items()):
            summary: dict[str, Any] = dict(self._device_metadata.get(dev_no, {}))
            for key, values in glance.items():
                summary[f"avg_{key}"] = _mean_or_none(values)
                summary[f"p99_{key}"] = _p99_or_none(values)
                summary[f"max_{key}"] = _max_or_none(values)
            summary["memory_total_mb"] = self._device_memory_total_mb.get(dev_no)
            devices[dev_no] = summary
        return devices

    def get_trace(self) -> list[tuple[float, float]]:
        """Return total power trace as ``(timestamp, total_power_w)`` pairs."""
        return self.get_total_power_trace()

    def get_total_power_trace(self) -> list[tuple[float, float]]:
        return list(self._traces["total_power_w"])

    def get_total_utilization_trace(self) -> list[tuple[float, float]]:
        return list(self._traces["total_utilization_pct"])

    def get_util_trace(self) -> list[tuple[float, float]]:
        return self.get_total_utilization_trace()

    def get_temperature_trace(self) -> list[tuple[float, float]]:
        return list(self._traces["temperature_c"])

    def get_temp_trace(self) -> list[tuple[float, float]]:
        return self.get_temperature_trace()

    def get_npu_rail_power_trace(self) -> list[tuple[float, float]]:
        return list(self._traces["npu_rail_power_w"])

    def get_ddr_rail_power_trace(self) -> list[tuple[float, float]]:
        return list(self._traces["ddr_rail_power_w"])

    def get_pmic_rail_power_trace(self) -> list[tuple[float, float]]:
        return list(self._traces["pmic_rail_power_w"])

    def get_goldfinger_rail_power_trace(self) -> list[tuple[float, float]]:
        return list(self._traces["goldfinger_rail_power_w"])

    def get_npu_power_trace(self) -> list[tuple[float, float]]:
        return self.get_npu_rail_power_trace()

    def get_ddr_power_trace(self) -> list[tuple[float, float]]:
        return self.get_ddr_rail_power_trace()

    def get_pmic_power_trace(self) -> list[tuple[float, float]]:
        return self.get_pmic_rail_power_trace()

    def get_goldfinger_power_trace(self) -> list[tuple[float, float]]:
        return self.get_goldfinger_rail_power_trace()

    def get_static_info(self) -> dict[str, object]:
        pcie_vendor_id = os.environ.get("MBLT_TRACKER_NPU_PCI_VENDOR_ID")
        pcie_device_id = os.environ.get("MBLT_TRACKER_NPU_PCI_DEVICE_ID")
        pcie_class_filter = os.environ.get("MBLT_TRACKER_NPU_PCI_CLASS_FILTER")
        pcie_devices = get_all_pcie_devices()
        info = get_pcie_static_info(
            vendor_id=pcie_vendor_id,
            device_id=pcie_device_id,
            class_filter=pcie_class_filter,
            devices=pcie_devices,
            include_private_identifiers=True,
        )
        _deep_merge(info, _get_mbltml_static_info())
        return _sanitize_static_info_for_public_output(info)

    def reset(self) -> None:
        metric_keys = [
            "total_power_w",
            "total_current_a",
            "total_voltage_v",
            "total_utilization_pct",
            "memory_usage_mb",
            "memory_usage_pct",
            "temperature_c",
        ]
        for rail in _RAIL_ORDER:
            metric_keys.extend(
                [
                    f"{rail}_rail_power_w",
                    f"{rail}_rail_current_a",
                    f"{rail}_rail_voltage_v",
                ]
            )
        self._glance: dict[str, list[float]] = {key: [] for key in metric_keys}
        self._traces: dict[str, list[tuple[float, float]]] = {key: [] for key in metric_keys}
        self._memory_total_mb: float | None = None
        self._device_glance: dict[int, dict[str, list[float]]] = {}
        self._device_metadata: dict[int, dict[str, Any]] = {}
        self._device_memory_total_mb: dict[int, float] = {}


def _ensure_mbltml_available() -> None:
    if mbltml is None:
        raise RuntimeError("NPU tracking requires the mbltml package")


def _initialize_mbltml() -> None:
    try:
        mbltml.mbltmlInitDevices({mbltml.MBLTML_DEVICE_ARIES})
    except AttributeError:
        mbltml.mbltmlInit()


def _normalize_npu_ids(npu_id: int | list[int] | None) -> list[int] | None:
    if npu_id is None:
        return None
    ids = [npu_id] if isinstance(npu_id, int) else list(npu_id)
    for i in ids:
        if i < 0:
            raise ValueError(f"Invalid NPU ID: {i}")
    return ids


def _normalize_rail_metrics(rail_metrics: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(rail_metrics, str):
        rails = list(_RAIL_ORDER) if rail_metrics == "all" else [rail_metrics]
    else:
        rails = list(rail_metrics)
    normalized = []
    for rail in rails:
        rail = rail.lower()
        if rail not in _RAIL_ORDER:
            raise ValueError(f"Invalid rail metric: {rail}")
        if rail not in normalized:
            normalized.append(rail)
    return tuple(normalized or ["npu"])


def _set_extra_rail(dev_no: int, rail: str) -> bool:
    rail_id = getattr(mbltml, f"MBLTML_EXTRA_PMIC_ID_{rail.upper()}")
    try:
        mbltml.mbltmlSetExtraPmicID(dev_no, rail_id)
    except Exception as exc:
        _LOGGER.debug("Unable to select %s rail for NPU %s: %s", rail, dev_no, exc)
        return False
    return True


def _read_rail_values(dev_no: int, rail: str) -> dict[str, float | None]:
    return {
        f"{rail}_rail_power_w": _safe_call(mbltml.mbltmlGetExtraPmicPower, dev_no),
        f"{rail}_rail_current_a": _safe_call(mbltml.mbltmlGetExtraPmicCurrent, dev_no),
        f"{rail}_rail_voltage_v": _safe_call(mbltml.mbltmlGetExtraPmicVoltage, dev_no),
    }


def _safe_call(func, *args, default=None):
    try:
        return func(*args)
    except Exception as exc:
        _LOGGER.debug("mbltml call failed: %s", exc)
        return default


def _bytes_to_mb(value: int | None) -> float | None:
    return float(value) / _MB_TO_BYTES if value is not None else None


def _usage_pct(usage: float | None, total: float | None) -> float | None:
    return (usage / total) * 100.0 if usage is not None and total not in (None, 0.0) else None


def _normalize_utilization(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value * 100.0 if 0.0 <= value <= 1.0 else value


def _sum(samples: list[dict[str, Any]], key: str) -> float:
    return sum(_sample_float(sample, key) or 0.0 for sample in samples)


def _sum_optional(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [_sample_float(sample, key) for sample in samples]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _avg_optional(samples: list[dict[str, Any]], key: str) -> float | None:
    values = [_sample_float(sample, key) for sample in samples]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _sample_float(sample: dict[str, Any], key: str) -> float | None:
    value = sample.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _p99_or_none(values: list[float]) -> float | None:
    return float(np.percentile(values, 99)) if values else None


def _max_or_none(values: list[float]) -> float | None:
    return float(np.max(values)) if values else None


def _hex_or_none(value: int | None) -> str | None:
    return f"0x{value:x}" if isinstance(value, int) else None


def _device_type_name(value: int | None) -> str | None:
    mapping = {
        getattr(mbltml, "MBLTML_DEVICE_ARIES", object()): "Aries",
        getattr(mbltml, "MBLTML_DEVICE_REGULUS", object()): "Regulus",
        getattr(mbltml, "MBLTML_DEVICE_REGULUS_USB", object()): "Regulus USB",
    }
    return mapping.get(value)


def _hardware_version_name(value: int | None) -> str | None:
    mapping = {
        getattr(mbltml, "MBLTML_HARDWARE_VERSION_ARIES", object()): "Aries",
        getattr(mbltml, "MBLTML_HARDWARE_VERSION_ARIES2", object()): "Aries2",
        getattr(mbltml, "MBLTML_HARDWARE_VERSION_REGULUS", object()): "Regulus",
        getattr(mbltml, "MBLTML_HARDWARE_VERSION_REGULUS2", object()): "Regulus2",
    }
    return mapping.get(value)


def _get_mbltml_static_info() -> dict[str, object]:
    _ensure_mbltml_available()
    try:
        _initialize_mbltml()
    except Exception:
        return {}
    device_count = _safe_call(mbltml.mbltmlGetDeviceCount, default=0)
    npus = [_get_mbltml_device_static_info(dev_no) for dev_no in range(device_count)]
    info: dict[str, object] = {"hardware": {"npus": [npu for npu in npus if npu]}}
    aries_driver = _safe_call(mbltml.mbltmlGetDriverVersion, mbltml.MBLTML_DEVICE_ARIES)
    if aries_driver is not None:
        info["inference"] = {"npu_driver_version": aries_driver, "driver": {"aries_version": aries_driver}}
    return info


def _get_mbltml_device_static_info(dev_no: int) -> dict[str, object]:
    firmware_version = _safe_call(mbltml.mbltmlGetFirmwareVersion, dev_no)
    firmware_revision = _safe_call(mbltml.mbltmlGetFirmwareRevision, dev_no)
    return {
        key: value
        for key, value in {
            "dev_no": dev_no,
            "node_name": _safe_call(mbltml.mbltmlGetNodeName, dev_no),
            "device_type": _device_type_name(_safe_call(mbltml.mbltmlGetDeviceType, dev_no)),
            "hardware_version": _hardware_version_name(_safe_call(mbltml.mbltmlGetHardwareVersion, dev_no)),
            "firmware": {
                "version": firmware_version,
                "revision": str(firmware_revision) if firmware_revision is not None else None,
            },
            "vendor_id": _hex_or_none(_safe_call(mbltml.mbltmlGetVendorId, dev_no)),
            "device_id": _hex_or_none(_safe_call(mbltml.mbltmlGetDeviceId, dev_no)),
            "subsystem_vendor_id": _hex_or_none(_safe_call(mbltml.mbltmlGetSubVendorId, dev_no)),
            "subsystem_device_id": _hex_or_none(_safe_call(mbltml.mbltmlGetSubDeviceId, dev_no)),
            "link_generation": _safe_call(mbltml.mbltmlGetPcieGen, dev_no),
            "lane_width": _safe_call(mbltml.mbltmlGetPcieLanes, dev_no),
            "revision": _hex_or_none(_safe_call(mbltml.mbltmlGetPcieRev, dev_no)),
            "class": _hex_or_none(_safe_call(mbltml.mbltmlGetPcieClassCode, dev_no)),
            "memory_total_bytes": _safe_call(mbltml.mbltmlGetMemoryTotal, dev_no),
        }.items()
        if value is not None and value != {"version": None, "revision": None}
    }
