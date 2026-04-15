from __future__ import annotations

from mblt_tracker.device_tracker_gpu import GPUDeviceTracker


def _make_tracker() -> GPUDeviceTracker:
    tracker = object.__new__(GPUDeviceTracker)
    tracker._gpu_id = [0]
    tracker._power_glance = {0: []}
    tracker._gpu_util_glance = {0: []}
    tracker._mem_util_glance = {0: []}
    tracker._mem_used_glance = {0: []}
    tracker._mem_used_pct_glance = {0: []}
    tracker._temp_glance = {0: []}
    tracker._mem_total_mb = {}
    tracker._power_trace = []
    tracker._gpu_util_trace = []
    tracker._mem_util_trace = []
    tracker._mem_used_trace = []
    tracker._mem_used_pct_trace = []
    tracker._temp_trace = []
    return tracker


def test_sampling_keeps_util_and_memory_when_temperature_fails(monkeypatch) -> None:
    tracker = _make_tracker()

    monkeypatch.setattr(tracker, "gpu_utilization", lambda: [(61.0, 17.0)])
    monkeypatch.setattr(tracker, "gpu_memory_info", lambda: [(256.0, 1024.0)])
    monkeypatch.setattr(tracker, "_safe_gpu_power", lambda gpu: 42.5)
    monkeypatch.setattr(tracker, "_safe_gpu_temperature", lambda gpu: None)
    monkeypatch.setattr("mblt_tracker.device_tracker_gpu.time.time", lambda: 123.0)

    tracker._func_for_sched()

    assert tracker._gpu_util_trace == [(123.0, 61.0)]
    assert tracker._mem_util_trace == [(123.0, 17.0)]
    assert tracker._mem_used_trace == [(123.0, 256.0)]
    assert tracker._mem_used_pct_trace == [(123.0, 25.0)]
    assert tracker._power_trace == [(123.0, 42.5)]
    assert tracker._temp_trace == []
    assert tracker._temp_glance[0] == []


def test_sampling_keeps_util_and_memory_when_power_fails(monkeypatch) -> None:
    tracker = _make_tracker()

    monkeypatch.setattr(tracker, "gpu_utilization", lambda: [(88.0, 33.0)])
    monkeypatch.setattr(tracker, "gpu_memory_info", lambda: [(512.0, 2048.0)])
    monkeypatch.setattr(tracker, "_safe_gpu_power", lambda gpu: None)
    monkeypatch.setattr(tracker, "_safe_gpu_temperature", lambda gpu: 67.0)
    monkeypatch.setattr("mblt_tracker.device_tracker_gpu.time.time", lambda: 456.0)

    tracker._func_for_sched()

    assert tracker._gpu_util_trace == [(456.0, 88.0)]
    assert tracker._mem_util_trace == [(456.0, 33.0)]
    assert tracker._mem_used_trace == [(456.0, 512.0)]
    assert tracker._mem_used_pct_trace == [(456.0, 25.0)]
    assert tracker._power_trace == []
    assert tracker._temp_trace == [(456.0, 67.0)]
    assert tracker._power_glance[0] == []
