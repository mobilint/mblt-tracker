from __future__ import annotations

from typing import Optional

import pytest

from mblt_tracker.device_tracker_dram import DRAMDeviceTracker


class _FakeResult:
    def __init__(self, duration: float, dram: Optional[list[float]]):
        self.duration = duration
        self.dram = dram


class _FakeMeter:
    def __init__(self, result: _FakeResult):
        self.result = result
        self.begin_called = False
        self.end_called = False

    def begin(self) -> None:
        self.begin_called = True

    def end(self) -> None:
        self.end_called = True


def _make_tracker() -> DRAMDeviceTracker:
    tracker = object.__new__(DRAMDeviceTracker)
    tracker._socket_id = [0, 1]
    tracker._power_glance = {0: [], 1: []}
    tracker._power_trace = []
    tracker._meter = None
    return tracker


def test_sampling_records_total_and_per_socket_dram_power(monkeypatch) -> None:
    tracker = _make_tracker()
    previous_meter = _FakeMeter(_FakeResult(duration=100_000.0, dram=[200_000.0, 300_000.0]))
    next_meter = _FakeMeter(_FakeResult(duration=0.0, dram=None))
    tracker._meter = previous_meter

    monkeypatch.setattr(
        "mblt_tracker.device_tracker_dram.pyRAPL.Measurement",
        lambda label: next_meter,
    )
    monkeypatch.setattr("mblt_tracker.device_tracker_dram.time.time", lambda: 123.0)

    tracker._func_for_sched()

    assert previous_meter.end_called is True
    assert next_meter.begin_called is True
    assert tracker._power_glance[0] == pytest.approx([2.0])
    assert tracker._power_glance[1] == pytest.approx([3.0])
    trace = tracker.get_trace()
    assert trace[0][0] == 123.0
    assert trace[0][1] == pytest.approx(5.0)


def test_get_metric_summarizes_dram_power() -> None:
    tracker = _make_tracker()
    tracker._power_glance = {0: [1.0, 3.0], 1: [2.0, 4.0]}
    tracker._power_trace = [(1.0, 3.0), (2.0, 7.0)]

    metrics = tracker.get_metric()

    assert metrics["avg_power_w"] == 5.0
    assert metrics["avg_dram_power_w"] == 5.0
    assert metrics["max_power_w"] == 7.0
    assert metrics["max_dram_power_w"] == 7.0
    assert metrics["samples"] == 2
    assert metrics["dram"][0]["avg_power_w"] == 2.0
    assert metrics["dram"][1]["max_power_w"] == 4.0


def test_sampling_skips_when_dram_result_is_missing(monkeypatch) -> None:
    tracker = _make_tracker()
    tracker._meter = _FakeMeter(_FakeResult(duration=100_000.0, dram=None))
    next_meter = _FakeMeter(_FakeResult(duration=0.0, dram=None))

    monkeypatch.setattr(
        "mblt_tracker.device_tracker_dram.pyRAPL.Measurement",
        lambda label: next_meter,
    )

    tracker._func_for_sched()

    assert tracker.get_trace() == []
    assert tracker.get_metric()["samples"] == 0


def test_reset_clears_collected_dram_power() -> None:
    tracker = _make_tracker()
    tracker._power_glance = {0: [1.0], 1: [2.0]}
    tracker._power_trace = [(1.0, 3.0)]
    tracker._meter = _FakeMeter(_FakeResult(duration=100_000.0, dram=[100_000.0, 200_000.0]))

    tracker.reset()

    assert tracker._power_glance == {0: [], 1: []}
    assert tracker.get_trace() == []
    assert tracker._meter is None


def test_invalid_socket_id_raises_value_error(monkeypatch) -> None:
    monkeypatch.setattr("mblt_tracker.device_tracker_dram.pyRAPL.setup", lambda devices: None)
    monkeypatch.setattr(DRAMDeviceTracker, "socket_num", lambda self: 1)

    with pytest.raises(ValueError, match="Invalid socket ID"):
        DRAMDeviceTracker(socket_id=1)