from __future__ import annotations

import atexit
import logging
import math
import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import mbltml
import numpy as np

from .device_tracker import BaseDeviceTracker
from .static_info import (
    _sanitize_static_info_for_public_output,
    enrich_mbltml_npus_with_pcie_info,
    get_all_pcie_devices,
    get_pcie_static_info,
)

_LOGGER = logging.getLogger(__name__)
_MB_TO_BYTES = 1024 * 1024
_EXTRA_RAIL_REFRESH_PERIOD_S = 1.0
_DEFAULT_RAIL_METRICS = "npu"
_RAIL_ORDER = ("npu", "ddr", "pmic", "goldfinger")

# mobilint-ml 1.5 addresses a device by (target device type, index within that
# type). Each entry maps a target-device-type constant to its public name and to
# the hardware-version label of the pre-1.5 ``mbltml`` API (ARIES == aries-ra,
# ARIES2 == aries-rb, REGULUS == regulus-ra, REGULUS2 == regulus-rb).
_TARGET_DEVICE_TYPES = (
    ("MBLTML_TARGET_DEVICE_ARIES_RA", "aries-ra", "Aries"),
    ("MBLTML_TARGET_DEVICE_ARIES_RB", "aries-rb", "Aries2"),
    ("MBLTML_TARGET_DEVICE_REGULUS_RA", "regulus-ra", "Regulus"),
    ("MBLTML_TARGET_DEVICE_REGULUS_RB", "regulus-rb", "Regulus2"),
    ("MBLTML_TARGET_DEVICE_REGULUS_RA_USB", "regulus-ra-usb", "Regulus"),
    ("MBLTML_TARGET_DEVICE_REGULUS_RB_USB", "regulus-rb-usb", "Regulus2"),
)
# Node families in the order mbltml opens them. mbltml numbers its system-wide
# device list in this order, then by node number within a family.
_NODE_FAMILY_ORDER = ("/dev/aries", "/dev/regulus-npu", "/dev/regulus-usb")
_NODE_NAME_PATTERN = re.compile(r"^(.*?)(\d+)$")
_INIT_LOCK = threading.Lock()
_initialized_module: object | None = None


@dataclass(frozen=True)
class _NpuHandle:
    """Address of one NPU in the mobilint-ml per-target-device-type API."""

    target_device_type: int
    target_device_type_name: str
    index: int
    node_name: str | None
    hardware_version: str
    type_rank: int

    def call(self, func: Callable[..., Any], *args: Any) -> Any:
        return _safe_call(func, self.target_device_type, self.index, *args)


class NPUDeviceTracker(BaseDeviceTracker):
    """Track Mobilint NPU metrics through the OS-independent ``mbltml`` API.

    ``mbltml`` is provided by the ``mobilint-ml`` package, whose API addresses a
    device by target device type plus an index within that type. The tracker
    keeps one system-wide physical ``dev_no`` namespace instead: devices are
    numbered in mbltml's own enumeration order (node family, then node number),
    which is the device index of the pre-1.5 ``mbltml`` API.

    The default path reads total power, utilization, memory, and temperature on
    every tick. The NPU extra-PMIC rail is selected explicitly on the first tick
    and is recorded only after the firmware refresh period has elapsed, because
    the tracker cannot assume the firmware selection left by a previous process.

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
            npu_id: Physical NPU device indices (``dev_no``) to track. If
                ``None``, all detected devices are tracked.
            rail_metrics: Extra PMIC rails to monitor. ``"npu"`` is the
                default. ``"all"`` enables ``npu``, ``ddr``, ``pmic``, and
                ``goldfinger``. Extra PMIC rails require a firmware register
                selection change and become valid only after approximately one
                firmware refresh period (1 second).
        """
        super().__init__(interval=interval)
        _initialize_mbltml()

        self._npu_id = _normalize_npu_ids(npu_id)
        self._rail_metrics = _normalize_rail_metrics(rail_metrics)
        self._job_id = "npu_device_track"
        self._devices = _discover_devices()
        self._device_count = len(self._devices)
        if self._device_count <= 0:
            raise RuntimeError("No Mobilint NPU devices were detected by mbltml")
        invalid_ids = [i for i in self._npu_id or [] if i >= self._device_count]
        if invalid_ids:
            raise ValueError(f"Invalid NPU ID(s): {invalid_ids}")

        self._selected_rail: dict[int, str | None] = dict.fromkeys(
            range(self._device_count), None
        )
        self._rail_selected_at: dict[int, float] = dict.fromkeys(range(self._device_count), 0.0)
        self._pending_extra_rail: str | None = None
        self._next_extra_rail_index = 0
        self._unavailable_extra_rails: set[str] = set()
        self.reset()

    def stop(self) -> None:
        """Stop tracking and restore firmware Extra PMIC rail selection to NPU."""
        try:
            super().stop()
        finally:
            self._restore_npu_rail_selection()

    def _func_for_sched(self) -> None:
        samples = self._fetch_metric_samples()
        if not samples:
            return
        ts = time.time()
        self._record_samples(ts, samples)

    def _fetch_metric_samples(self) -> list[dict[str, Any]]:
        selected = self._selected_device_indices()
        now = time.time()
        extra_rail_to_read = self._advance_extra_rail_state(now)
        samples = []
        for dev_no in selected:
            sample = self._read_device_sample(dev_no, extra_rail_to_read, now)
            if sample is not None:
                samples.append(sample)
        if extra_rail_to_read is not None:
            self._complete_extra_rail_sample(extra_rail_to_read)
        return samples

    def _advance_extra_rail_state(self, now: float) -> str | None:
        rail_cycle = list(self._rail_metrics)
        if not rail_cycle:
            return None
        selected_devices = self._selected_device_indices()
        if self._pending_extra_rail is None:
            for _ in range(len(rail_cycle)):
                rail = rail_cycle[self._next_extra_rail_index % len(rail_cycle)]
                if rail in self._unavailable_extra_rails:
                    self._next_extra_rail_index += 1
                    continue
                if self._select_extra_rail_for_devices(rail, selected_devices, now):
                    self._pending_extra_rail = rail
                    break
                self._unavailable_extra_rails.add(rail)
                self._next_extra_rail_index += 1
            else:
                return None

        rail = self._pending_extra_rail
        if rail is None:
            return None
        if self._selected_devices_ready_for_rail(rail, selected_devices, now):
            return rail
        return None

    def _selected_devices_ready_for_rail(
        self, rail: str, selected_devices: list[int], now: float
    ) -> bool:
        return all(
            self._selected_rail.get(dev_no) == rail
            and now - self._rail_selected_at.get(dev_no, 0.0)
            >= _EXTRA_RAIL_REFRESH_PERIOD_S
            for dev_no in selected_devices
        )

    def _select_extra_rail_for_devices(
        self, rail: str, selected_devices: list[int], now: float
    ) -> bool:
        changed_devices: list[int] = []
        for dev_no in selected_devices:
            if self._selected_rail.get(dev_no) != rail:
                if _set_extra_rail(self._devices[dev_no], dev_no, rail):
                    self._selected_rail[dev_no] = rail
                    self._rail_selected_at[dev_no] = now
                    changed_devices.append(dev_no)
                else:
                    self._rollback_devices_to_npu_rail(changed_devices, now)
                    return False
        return True

    def _rollback_devices_to_npu_rail(self, dev_nos: list[int], now: float) -> None:
        for dev_no in dev_nos:
            if _set_extra_rail(self._devices[dev_no], dev_no, "npu"):
                self._selected_rail[dev_no] = "npu"
                self._rail_selected_at[dev_no] = now

    def _complete_extra_rail_sample(self, rail: str) -> None:
        if self._pending_extra_rail != rail:
            return
        self._pending_extra_rail = None
        self._next_extra_rail_index += 1

    def _read_device_sample(
        self, dev_no: int, extra_rail_to_read: str | None, now: float
    ) -> dict[str, Any] | None:
        device = self._devices[dev_no]
        total_power_w = device.call(mbltml.mbltmlGetTotalPower)
        if total_power_w is None:
            return None
        memory_usage_mb = _bytes_to_mb(device.call(mbltml.mbltmlGetMemoryUsage))
        memory_total_mb = _bytes_to_mb(device.call(mbltml.mbltmlGetMemoryTotal))
        sample: dict[str, Any] = {
            "dev_no": dev_no,
            "total_power_w": float(total_power_w),
            "total_current_a": device.call(mbltml.mbltmlGetTotalCurrent),
            "total_voltage_v": device.call(mbltml.mbltmlGetTotalVoltage),
            "total_utilization_pct": _normalize_utilization(
                device.call(mbltml.mbltmlGetTotalUtilization)
            ),
            "memory_usage_mb": memory_usage_mb,
            "memory_total_mb": memory_total_mb,
            "memory_usage_pct": _usage_pct(memory_usage_mb, memory_total_mb),
            "temperature_c": device.call(mbltml.mbltmlGetTemperature),
            "node_name": device.node_name,
            "hardware_version": device.hardware_version,
            "target_device_type": device.target_device_type_name,
        }
        if extra_rail_to_read is not None and self._selected_rail.get(dev_no) == extra_rail_to_read:
            selected_at = self._rail_selected_at.get(dev_no, 0.0)
            if now - selected_at >= _EXTRA_RAIL_REFRESH_PERIOD_S:
                sample.update(_read_rail_values(device, extra_rail_to_read))
        return sample

    def _restore_npu_rail_selection(self) -> None:
        now = time.time()
        for dev_no in self._selected_device_indices():
            if _set_extra_rail(self._devices[dev_no], dev_no, "npu"):
                self._selected_rail[dev_no] = "npu"
                self._rail_selected_at[dev_no] = now
        self._pending_extra_rail = None

    def _record_samples(self, ts: float, samples: list[dict[str, Any]]) -> None:
        self._append("total_power_w", _sum(samples, "total_power_w"), ts)
        self._append("total_current_a", _sum_optional(samples, "total_current_a"), ts)
        self._append("total_voltage_v", _avg_optional(samples, "total_voltage_v"), ts)
        self._append("total_utilization_pct", _avg_optional(samples, "total_utilization_pct"), ts)
        memory_usage_mb = _sum_optional(samples, "memory_usage_mb")
        self._append("memory_usage_mb", memory_usage_mb, ts)
        memory_total_mb = _sum_optional(samples, "memory_total_mb")
        if memory_total_mb is not None:
            self._memory_total_mb = memory_total_mb
        self._append("memory_usage_pct", _usage_pct(memory_usage_mb, memory_total_mb), ts)
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
            "target_device_type": sample.get("target_device_type"),
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
        _add_generic_metric_aliases(metrics)
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
        pcie_devices = get_all_pcie_devices()
        pcie_info = get_pcie_static_info(
            devices=pcie_devices,
            include_private_identifiers=True,
            include_npus=True,
        )
        mbltml_info = _get_mbltml_static_info()
        enrich_mbltml_npus_with_pcie_info(mbltml_info, pcie_info)
        _filter_static_info_to_device_indices(
            mbltml_info,
            set(self._selected_device_indices()),
        )
        return _sanitize_static_info_for_public_output(mbltml_info)

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


def _initialize_mbltml() -> None:
    """Initialize mbltml once per process.

    ``mbltmlInit()`` closes and reopens every device node, so calling it again
    while another tracker samples on its scheduler thread is unsafe. An explicit
    ``mbltmlShutdown()`` at exit avoids the mobilint-ml 1.5.3 Windows crash in
    its static destructors when the library is still initialized (A2-364).
    """
    global _initialized_module
    with _INIT_LOCK:
        if _initialized_module is mbltml:
            return
        if not hasattr(mbltml, "mbltmlGetTargetDeviceCount"):
            raise RuntimeError(
                "The installed mbltml module predates the target-device-type API. "
                "Uninstall the legacy 'mbltml' package and install 'mobilint-ml'."
            )
        mbltml.mbltmlInit()
        _initialized_module = mbltml
        atexit.register(_shutdown_mbltml, mbltml)


def _shutdown_mbltml(module: Any) -> None:
    global _initialized_module
    with _INIT_LOCK:
        if _initialized_module is not module:
            return
        _initialized_module = None
        if hasattr(module, "mbltmlShutdown"):
            _safe_call(module.mbltmlShutdown)


def _discover_devices() -> list[_NpuHandle]:
    """Return every addressable NPU, ordered by its physical ``dev_no``.

    Devices are collected per target device type and ordered the way mbltml
    enumerates its system-wide device list: node family (ARIES, REGULUS, then
    REGULUS USB) and node number. The list position is the physical
    ``dev_no``, so a host with one target device type keeps mbltml's per-type
    index. Nodes mbltml cannot classify are not addressable and are skipped.
    """
    devices = []
    for type_rank, (attr, type_name, hw_version) in enumerate(_TARGET_DEVICE_TYPES):
        tdt = getattr(mbltml, attr, None)
        if not isinstance(tdt, int):
            continue
        count = _safe_call(mbltml.mbltmlGetTargetDeviceCount, tdt, default=0)
        for index in range(count or 0):
            devices.append(
                _NpuHandle(
                    target_device_type=tdt,
                    target_device_type_name=type_name,
                    index=index,
                    node_name=_safe_call(mbltml.mbltmlGetNodeName, tdt, index),
                    hardware_version=hw_version,
                    type_rank=type_rank,
                )
            )
    return sorted(devices, key=_enumeration_key)


def _enumeration_key(device: _NpuHandle) -> tuple[int, float, int, int]:
    match = _NODE_NAME_PATTERN.match(device.node_name or "")
    if match is None or match.group(1) not in _NODE_FAMILY_ORDER:
        return len(_NODE_FAMILY_ORDER), math.inf, device.type_rank, device.index
    family_rank = _NODE_FAMILY_ORDER.index(match.group(1))
    return family_rank, float(match.group(2)), device.type_rank, device.index


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


def _set_extra_rail(device: _NpuHandle, dev_no: int, rail: str) -> bool:
    try:
        rail_id = getattr(mbltml, f"MBLTML_EXTRA_PMIC_ID_{rail.upper()}")
        mbltml.mbltmlSetExtraPmicID(device.target_device_type, device.index, rail_id)
    except Exception as exc:
        _LOGGER.debug("Unable to select %s rail for NPU %s: %s", rail, dev_no, exc)
        return False
    return True


def _read_rail_values(device: _NpuHandle, rail: str) -> dict[str, float | None]:
    return {
        f"{rail}_rail_power_w": device.call(mbltml.mbltmlGetExtraPmicPower),
        f"{rail}_rail_current_a": device.call(mbltml.mbltmlGetExtraPmicCurrent),
        f"{rail}_rail_voltage_v": device.call(mbltml.mbltmlGetExtraPmicVoltage),
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


def _add_generic_metric_aliases(metrics: dict[str, Any]) -> None:
    """Add CPU/GPU-compatible metric names while preserving NPU-specific keys."""
    aliases = {
        "avg_power_w": "avg_total_power_w",
        "p99_power_w": "p99_total_power_w",
        "max_power_w": "max_total_power_w",
        "avg_utilization_pct": "avg_total_utilization_pct",
        "p99_utilization_pct": "p99_total_utilization_pct",
        "max_utilization_pct": "max_total_utilization_pct",
        "avg_memory_used_mb": "avg_memory_usage_mb",
        "p99_memory_used_mb": "p99_memory_usage_mb",
        "max_memory_used_mb": "max_memory_usage_mb",
        "total_memory_mb": "memory_total_mb",
        "avg_memory_used_pct": "avg_memory_usage_pct",
        "p99_memory_used_pct": "p99_memory_usage_pct",
        "max_memory_used_pct": "max_memory_usage_pct",
    }
    for alias, source in aliases.items():
        metrics[alias] = metrics.get(source)


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _p99_or_none(values: list[float]) -> float | None:
    return float(np.percentile(values, 99)) if values else None


def _max_or_none(values: list[float]) -> float | None:
    return float(np.max(values)) if values else None


def _hex_or_none(value: int | None) -> str | None:
    return f"0x{value:x}" if isinstance(value, int) else None


def _driver_type_name(value: int | None) -> str | None:
    mapping = {
        getattr(mbltml, "MBLTML_DRIVER_ARIES", object()): "Aries",
        getattr(mbltml, "MBLTML_DRIVER_REGULUS", object()): "Regulus",
        getattr(mbltml, "MBLTML_DRIVER_REGULUS_USB", object()): "Regulus USB",
    }
    return mapping.get(value)


def _get_mbltml_static_info() -> dict[str, object]:
    try:
        _initialize_mbltml()
    except Exception:
        return {}
    npus = [
        _get_mbltml_device_static_info(dev_no, device)
        for dev_no, device in enumerate(_discover_devices())
    ]
    info: dict[str, object] = {"hardware": {"npus": [npu for npu in npus if npu]}}
    driver_versions = _get_mbltml_driver_versions()
    if driver_versions:
        info["inference"] = {
            "npu_driver_version": driver_versions.get("aries_version")
            or next(iter(driver_versions.values())),
            "driver": driver_versions,
        }
    return info


def _get_mbltml_driver_versions() -> dict[str, str]:
    driver_types = {
        "aries_version": getattr(mbltml, "MBLTML_DRIVER_ARIES", None),
        "regulus_version": getattr(mbltml, "MBLTML_DRIVER_REGULUS", None),
        "regulus_usb_version": getattr(mbltml, "MBLTML_DRIVER_REGULUS_USB", None),
    }
    versions = {}
    for key, driver_type in driver_types.items():
        if driver_type is None:
            continue
        version = _safe_call(mbltml.mbltmlGetDriverVersion, driver_type)
        if version is not None:
            versions[key] = version
    return versions


def _filter_static_info_to_device_indices(
    info: dict[str, object], selected_indices: set[int]
) -> None:
    hardware = info.get("hardware")
    if not isinstance(hardware, dict):
        return
    npus = hardware.get("npus")
    if not isinstance(npus, list):
        return
    hardware["npus"] = [
        npu
        for npu in npus
        if isinstance(npu, dict) and npu.get("dev_no") in selected_indices
    ]


def _get_mbltml_device_static_info(
    dev_no: int, device: _NpuHandle
) -> dict[str, object]:
    firmware_version = device.call(mbltml.mbltmlGetFirmwareVersion)
    firmware_revision = device.call(mbltml.mbltmlGetFirmwareRevision)
    return {
        key: value
        for key, value in {
            "dev_no": dev_no,
            "node_name": device.node_name,
            "device_type": _driver_type_name(device.call(mbltml.mbltmlGetDriverType)),
            "target_device_type": device.target_device_type_name,
            "hardware_version": device.hardware_version,
            "firmware": {
                "version": firmware_version,
                "revision": str(firmware_revision) if firmware_revision is not None else None,
            },
            "vendor_id": _hex_or_none(device.call(mbltml.mbltmlGetVendorId)),
            "device_id": _hex_or_none(device.call(mbltml.mbltmlGetDeviceId)),
            "subsystem_vendor_id": _hex_or_none(
                device.call(mbltml.mbltmlGetSubVendorId)
            ),
            "subsystem_device_id": _hex_or_none(
                device.call(mbltml.mbltmlGetSubDeviceId)
            ),
            "link_generation": device.call(mbltml.mbltmlGetPcieGen),
            "lane_width": device.call(mbltml.mbltmlGetPcieLanes),
            "revision": _hex_or_none(device.call(mbltml.mbltmlGetPcieRev)),
            "class": _hex_or_none(device.call(mbltml.mbltmlGetPcieClassCode)),
            "memory_total_bytes": device.call(mbltml.mbltmlGetMemoryTotal),
        }.items()
        if value is not None and value != {"version": None, "revision": None}
    }
