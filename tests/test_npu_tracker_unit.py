# ruff: noqa: N802
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import mblt_tracker.device_tracker_npu as npu_module
from mblt_tracker.device_tracker_npu import NPUDeviceTracker


class FakeMbltmlError(Exception):
    pass


@dataclass
class FakeMbltml:
    """mobilint-ml 1.5 API fake: every device call takes (tdt, dev_no) first.

    The default host has two aries-rb devices. ``set_calls`` records
    ``(index_within_type, rail_id)`` and ``selected_rail`` is keyed by the same
    per-type index, which equals the physical ``dev_no`` on a one-type host.
    """

    MBLTML_DRIVER_ARIES: int = 1
    MBLTML_DRIVER_REGULUS: int = 2
    MBLTML_DRIVER_REGULUS_USB: int = 4
    MBLTML_TARGET_DEVICE_ARIES_RA: int = 1
    MBLTML_TARGET_DEVICE_REGULUS_RA: int = 2
    MBLTML_TARGET_DEVICE_ARIES_RB: int = 3
    MBLTML_TARGET_DEVICE_REGULUS_RB: int = 4
    MBLTML_TARGET_DEVICE_REGULUS_RA_USB: int = 5
    MBLTML_TARGET_DEVICE_REGULUS_RB_USB: int = 6
    MBLTML_EXTRA_PMIC_ID_NPU: int = 0
    MBLTML_EXTRA_PMIC_ID_DDR: int = 1
    MBLTML_EXTRA_PMIC_ID_PMIC: int = 2
    MBLTML_EXTRA_PMIC_ID_GOLDFINGER: int = 3
    init_calls: int = 0
    shutdown_calls: int = 0
    devices: dict[int, list[str]] = field(
        default_factory=lambda: {3: ["/dev/aries0", "/dev/aries1"]}
    )
    selected_rail: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    set_calls: list[tuple[int, int]] = field(default_factory=list)

    def _check(self, tdt, dev_no):
        if dev_no < 0 or dev_no >= len(self.devices.get(tdt, [])):
            raise FakeMbltmlError(f"no device ({tdt}, {dev_no})")

    def mbltmlInit(self):
        self.init_calls += 1

    def mbltmlShutdown(self):
        self.shutdown_calls += 1

    def mbltmlGetDeviceCount(self):
        return sum(len(nodes) for nodes in self.devices.values())

    def mbltmlGetTargetDeviceCount(self, tdt):
        return len(self.devices.get(tdt, []))

    def mbltmlGetNodeName(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return self.devices[tdt][dev_no]

    def mbltmlGetDriverType(self, tdt, dev_no):
        self._check(tdt, dev_no)
        if tdt in (1, 3):  # aries-ra, aries-rb
            return self.MBLTML_DRIVER_ARIES
        if tdt in (2, 4):  # regulus-ra, regulus-rb
            return self.MBLTML_DRIVER_REGULUS
        return self.MBLTML_DRIVER_REGULUS_USB

    def mbltmlGetTotalPower(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return 10.0 + dev_no

    def mbltmlGetTotalCurrent(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return 1.0 + dev_no

    def mbltmlGetTotalVoltage(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return 12.0

    def mbltmlGetTotalUtilization(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return 0.25 + (0.25 * dev_no)

    def mbltmlGetMemoryUsage(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return (256 + 256 * dev_no) * 1024 * 1024

    def mbltmlGetMemoryTotal(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return 1024 * 1024 * 1024

    def mbltmlGetTemperature(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return 40 + dev_no

    def mbltmlSetExtraPmicID(self, tdt, dev_no, rail_id):
        self._check(tdt, dev_no)
        self.selected_rail[dev_no] = rail_id
        self.set_calls.append((dev_no, rail_id))

    def mbltmlGetExtraPmicPower(self, tdt, dev_no):
        self._check(tdt, dev_no)
        rail = self.selected_rail[dev_no]
        return {0: 2.0, 1: 3.0, 2: 4.0, 3: 5.0}[rail] + dev_no

    def mbltmlGetExtraPmicCurrent(self, tdt, dev_no):
        self._check(tdt, dev_no)
        rail = self.selected_rail[dev_no]
        return {0: 0.2, 1: 0.3, 2: 0.4, 3: 0.5}[rail] + dev_no

    def mbltmlGetExtraPmicVoltage(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return 12.0

    def mbltmlGetDriverVersion(self, driver_type):
        return {
            self.MBLTML_DRIVER_ARIES: "1.2.3",
            self.MBLTML_DRIVER_REGULUS: "4.5.6",
            self.MBLTML_DRIVER_REGULUS_USB: "7.8.9",
        }[driver_type]

    def mbltmlGetFirmwareVersion(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return "2.0.1"

    def mbltmlGetFirmwareRevision(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return 7

    def mbltmlGetVendorId(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return 0x209F

    def mbltmlGetDeviceId(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return 0

    def mbltmlGetSubVendorId(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return 0x402

    def mbltmlGetSubDeviceId(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return 0x108B

    def mbltmlGetPcieGen(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return 4

    def mbltmlGetPcieLanes(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return 8

    def mbltmlGetPcieRev(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return 2

    def mbltmlGetPcieClassCode(self, tdt, dev_no):
        self._check(tdt, dev_no)
        return 0x7800002


@pytest.fixture()
def fake_mbltml(monkeypatch):
    fake = FakeMbltml()
    monkeypatch.setattr(npu_module, "mbltml", fake)
    return fake


def test_npu_default_sampling_selects_npu_and_waits_for_refresh(
    fake_mbltml, monkeypatch
):
    tracker = NPUDeviceTracker(interval=0.1)
    monkeypatch.setattr(npu_module.time, "time", lambda: 100.0)

    tracker._func_for_sched()

    assert fake_mbltml.init_calls == 1
    assert fake_mbltml.set_calls == [
        (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU),
        (1, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU),
    ]
    assert tracker.get_trace() == [(100.0, 21.0)]
    assert tracker.get_npu_rail_power_trace() == []

    monkeypatch.setattr(npu_module.time, "time", lambda: 101.1)
    tracker._func_for_sched()

    assert tracker.get_npu_rail_power_trace() == [(101.1, 5.0)]
    metrics = tracker.get_metric()
    assert metrics["avg_total_power_w"] == 21.0
    assert metrics["avg_power_w"] == 21.0
    assert metrics["avg_npu_rail_power_w"] == 5.0
    assert metrics["avg_total_utilization_pct"] == pytest.approx(37.5)
    assert metrics["avg_utilization_pct"] == pytest.approx(37.5)
    assert metrics["avg_memory_used_mb"] == 768.0
    assert metrics["avg_memory_used_pct"] == 37.5
    assert metrics["memory_total_mb"] == 2048.0
    assert metrics["total_memory_mb"] == 2048.0
    assert metrics["devices"][0]["node_name"] == "/dev/aries0"
    assert metrics["devices"][0]["hardware_version"] == "Aries2"
    assert metrics["devices"][0]["target_device_type"] == "aries-rb"
    assert metrics["rail_metrics"] == {
        "selected": ["npu"],
        "firmware_refresh_period_s": 1.0,
        "extra_rail_requires_selection_delay": True,
    }


def test_npu_sampling_does_not_label_stale_goldfinger_as_npu(
    fake_mbltml, monkeypatch
):
    fake_mbltml.selected_rail = {
        0: fake_mbltml.MBLTML_EXTRA_PMIC_ID_GOLDFINGER,
        1: fake_mbltml.MBLTML_EXTRA_PMIC_ID_GOLDFINGER,
    }
    tracker = NPUDeviceTracker(interval=0.1)
    monkeypatch.setattr(npu_module.time, "time", lambda: 100.0)

    tracker._func_for_sched()

    assert fake_mbltml.set_calls == [
        (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU),
        (1, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU),
    ]
    assert tracker.get_npu_rail_power_trace() == []

    monkeypatch.setattr(npu_module.time, "time", lambda: 101.1)
    tracker._func_for_sched()

    assert tracker.get_npu_rail_power_trace() == [(101.1, 5.0)]


def test_npu_aggregate_memory_usage_pct_uses_summed_usage_and_capacity(
    fake_mbltml, monkeypatch
) -> None:
    memory_usage_mb = {0: 1024, 1: 0}
    memory_total_mb = {0: 1024, 1: 15360}
    monkeypatch.setattr(
        fake_mbltml,
        "mbltmlGetMemoryUsage",
        lambda tdt, dev_no: memory_usage_mb[dev_no] * 1024 * 1024,
    )
    monkeypatch.setattr(
        fake_mbltml,
        "mbltmlGetMemoryTotal",
        lambda tdt, dev_no: memory_total_mb[dev_no] * 1024 * 1024,
    )
    monkeypatch.setattr(npu_module.time, "time", lambda: 100.0)
    tracker = NPUDeviceTracker(interval=0.1)

    tracker._func_for_sched()

    metrics = tracker.get_metric()
    assert metrics["avg_memory_usage_mb"] == 1024.0
    assert metrics["memory_total_mb"] == 16384.0
    assert metrics["avg_memory_usage_pct"] == pytest.approx(6.25)
    assert metrics["avg_memory_usage_pct"] != pytest.approx(50.0)


def test_npu_id_filters_mbltml_devices(fake_mbltml, monkeypatch):
    tracker = NPUDeviceTracker(interval=0.1, npu_id=1)
    monkeypatch.setattr(npu_module.time, "time", lambda: 100.0)

    tracker._func_for_sched()

    assert tracker.get_trace() == [(100.0, 11.0)]
    assert tracker.get_npu_rail_power_trace() == []

    monkeypatch.setattr(npu_module.time, "time", lambda: 101.1)
    tracker._func_for_sched()

    metrics = tracker.get_metric()
    assert tracker.get_npu_rail_power_trace() == [(101.1, 3.0)]
    assert list(metrics["devices"]) == [1]


def test_npu_metric_generic_aliases_preserve_specific_keys(fake_mbltml, monkeypatch):
    tracker = NPUDeviceTracker(interval=0.1, npu_id=0)
    monkeypatch.setattr(npu_module.time, "time", lambda: 100.0)

    tracker._func_for_sched()
    metrics = tracker.get_metric()

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
        assert metrics[alias] == metrics[source]


def test_extra_rail_samples_wait_for_firmware_refresh(fake_mbltml, monkeypatch):
    now = 100.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker = NPUDeviceTracker(interval=0.1, npu_id=0, rail_metrics=["npu", "ddr"])

    tracker._func_for_sched()
    assert fake_mbltml.set_calls == [(0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU)]
    assert tracker.get_npu_rail_power_trace() == []
    assert tracker.get_ddr_rail_power_trace() == []

    now = 101.1
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_npu_rail_power_trace() == [(101.1, 2.0)]
    assert tracker.get_ddr_rail_power_trace() == []

    now = 101.2
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert fake_mbltml.set_calls[-1] == (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_DDR)

    now = 102.2
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()

    assert tracker.get_ddr_rail_power_trace() == [(102.2, 3.0)]
    assert tracker.get_metric()["ddr_rail_power_w_samples"] == 1


def test_failed_extra_rail_selection_does_not_leave_pending_state(
    fake_mbltml, monkeypatch
):
    now = 100.0
    original_set_extra_pmic_id = fake_mbltml.mbltmlSetExtraPmicID

    def fail_ddr_on_second_device(tdt, dev_no, rail_id):
        if dev_no == 1 and rail_id == fake_mbltml.MBLTML_EXTRA_PMIC_ID_DDR:
            raise RuntimeError("unsupported rail")
        original_set_extra_pmic_id(tdt, dev_no, rail_id)

    monkeypatch.setattr(fake_mbltml, "mbltmlSetExtraPmicID", fail_ddr_on_second_device)
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker = NPUDeviceTracker(interval=0.1, rail_metrics=["npu", "ddr"])

    tracker._func_for_sched()
    assert tracker._pending_extra_rail == "npu"
    assert tracker._selected_rail == {0: "npu", 1: "npu"}
    assert fake_mbltml.selected_rail == {
        0: fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU,
        1: fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU,
    }
    assert fake_mbltml.set_calls == [
        (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU),
        (1, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU),
    ]
    assert tracker.get_ddr_rail_power_trace() == []
    assert tracker.get_npu_rail_power_trace() == []

    now = 101.1
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()

    assert tracker._pending_extra_rail is None
    assert tracker._selected_rail == {0: "npu", 1: "npu"}
    assert tracker._unavailable_extra_rails == set()
    assert fake_mbltml.set_calls.count((0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_DDR)) == 0
    assert fake_mbltml.set_calls.count((0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU)) == 1
    assert fake_mbltml.set_calls.count((1, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU)) == 1
    assert tracker.get_ddr_rail_power_trace() == []
    assert tracker.get_npu_rail_power_trace()[-1] == (101.1, 5.0)

    now = 101.2
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()

    assert tracker._pending_extra_rail == "npu"
    assert tracker._selected_rail == {0: "npu", 1: "npu"}
    assert tracker._unavailable_extra_rails == {"ddr"}
    assert fake_mbltml.set_calls.count((0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_DDR)) == 1


def test_missing_extra_rail_constant_does_not_crash(monkeypatch):
    class MissingDdrMbltml(FakeMbltml):
        def __getattribute__(self, name):
            if name == "MBLTML_EXTRA_PMIC_ID_DDR":
                raise AttributeError(name)
            return super().__getattribute__(name)

    fake = MissingDdrMbltml()
    monkeypatch.setattr(npu_module, "mbltml", fake)
    monkeypatch.setattr(npu_module.time, "time", lambda: 100.0)
    tracker = NPUDeviceTracker(interval=0.1, rail_metrics="all")

    tracker._func_for_sched()

    assert tracker._pending_extra_rail == "npu"
    assert tracker._unavailable_extra_rails == set()
    assert tracker.get_ddr_rail_power_trace() == []
    assert tracker.get_npu_rail_power_trace() == []


def test_multiple_extra_rails_wait_before_advancing(fake_mbltml, monkeypatch):
    now = 100.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker = NPUDeviceTracker(
        interval=0.1,
        npu_id=0,
        rail_metrics=["npu", "ddr", "pmic"],
    )

    tracker._func_for_sched()
    assert fake_mbltml.set_calls == [(0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU)]
    assert tracker.get_npu_rail_power_trace() == []
    assert tracker.get_ddr_rail_power_trace() == []

    now = 100.5
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert fake_mbltml.set_calls == [(0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU)]
    assert tracker.get_npu_rail_power_trace() == []
    assert tracker.get_ddr_rail_power_trace() == []

    now = 101.1
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_npu_rail_power_trace() == [(101.1, 2.0)]
    assert tracker.get_ddr_rail_power_trace() == []

    now = 101.2
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert fake_mbltml.set_calls[-1] == (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_DDR)

    now = 101.3
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_ddr_rail_power_trace() == []
    assert fake_mbltml.set_calls[-1] == (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_DDR)

    now = 102.1
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_ddr_rail_power_trace() == []
    assert fake_mbltml.set_calls[-1] == (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_DDR)

    now = 102.2
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_ddr_rail_power_trace() == [(102.2, 3.0)]

    now = 103.2
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert fake_mbltml.set_calls[-1] == (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_PMIC)

    now = 103.3
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_pmic_rail_power_trace() == []


def test_all_rail_sampling_cycles_selected_rails_in_order(
    fake_mbltml, monkeypatch
):
    now = 100.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker = NPUDeviceTracker(interval=1.0, npu_id=0, rail_metrics="all")

    tracker._func_for_sched()
    assert fake_mbltml.set_calls == [(0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU)]
    assert tracker.get_npu_rail_power_trace() == []
    assert tracker.get_ddr_rail_power_trace() == []

    now = 101.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_npu_rail_power_trace() == [(101.0, 2.0)]
    assert tracker.get_ddr_rail_power_trace() == []

    now = 102.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert fake_mbltml.set_calls[-1] == (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_DDR)
    assert tracker.get_ddr_rail_power_trace() == []

    now = 103.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_ddr_rail_power_trace() == [(103.0, 3.0)]

    now = 104.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert fake_mbltml.set_calls[-1] == (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_PMIC)

    now = 105.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_pmic_rail_power_trace() == [(105.0, 4.0)]

    now = 106.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert fake_mbltml.set_calls[-1] == (
        0,
        fake_mbltml.MBLTML_EXTRA_PMIC_ID_GOLDFINGER,
    )
    assert tracker.get_goldfinger_rail_power_trace() == []

    now = 107.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_goldfinger_rail_power_trace() == [(107.0, 5.0)]

    metrics = tracker.get_metric()
    for rail in metrics["rail_metrics"]["selected"]:
        assert metrics[f"{rail}_rail_power_w_samples"] > 0


def test_all_rail_sampling_skips_unavailable_goldfinger(
    fake_mbltml, monkeypatch
):
    original_set_extra_pmic_id = fake_mbltml.mbltmlSetExtraPmicID

    def fail_goldfinger(tdt, dev_no, rail_id):
        if rail_id == fake_mbltml.MBLTML_EXTRA_PMIC_ID_GOLDFINGER:
            raise RuntimeError("unsupported rail")
        original_set_extra_pmic_id(tdt, dev_no, rail_id)

    monkeypatch.setattr(fake_mbltml, "mbltmlSetExtraPmicID", fail_goldfinger)
    now = 100.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker = NPUDeviceTracker(interval=1.0, npu_id=0, rail_metrics="all")

    for timestamp in range(100, 112):
        now = float(timestamp)
        monkeypatch.setattr(npu_module.time, "time", lambda now=now: now)
        tracker._func_for_sched()

    metrics = tracker.get_metric()
    assert tracker._unavailable_extra_rails == {"goldfinger"}
    assert metrics["ddr_rail_power_w_samples"] > 1
    assert metrics["pmic_rail_power_w_samples"] > 1
    assert metrics["goldfinger_rail_power_w_samples"] == 0
    assert tracker.get_ddr_rail_power_trace()[-1][0] > 106.0
    assert tracker.get_pmic_rail_power_trace()[-1][0] > 106.0


def test_stop_restores_npu_rail_selection(fake_mbltml, monkeypatch):
    now = 100.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker = NPUDeviceTracker(interval=0.1, npu_id=0, rail_metrics=["npu", "ddr"])

    tracker._func_for_sched()
    assert fake_mbltml.set_calls == [(0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU)]

    now = 100.5
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker.stop()

    assert fake_mbltml.set_calls[-1] == (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU)
    assert tracker._selected_rail == {0: "npu", 1: None}
    assert tracker._rail_selected_at[0] == 100.5
    assert tracker._pending_extra_rail is None


def test_invalid_npu_id_and_rail_are_rejected(fake_mbltml):
    with pytest.raises(ValueError, match="Invalid NPU ID"):
        NPUDeviceTracker(npu_id=-1)
    with pytest.raises(ValueError, match="Invalid NPU ID"):
        NPUDeviceTracker(npu_id=2)
    with pytest.raises(ValueError, match="Invalid rail metric"):
        NPUDeviceTracker(rail_metrics="bad")


def test_get_static_info_merges_mbltml_metadata(fake_mbltml, monkeypatch):
    monkeypatch.setattr(npu_module, "get_all_pcie_devices", lambda: [])
    monkeypatch.setattr(
        npu_module,
        "get_pcie_static_info",
        lambda **kwargs: {"hardware": {"npus": [{"dev_no": 0, "vendor_id": "0x209f"}]}},
    )

    tracker = NPUDeviceTracker(npu_id=0)
    info = tracker.get_static_info()

    assert info["inference"] == {
        "npu_driver_version": "1.2.3",
        "driver": {
            "aries_version": "1.2.3",
            "regulus_version": "4.5.6",
            "regulus_usb_version": "7.8.9",
        },
    }
    assert info["hardware"]["npus"][0] == {
        "dev_no": 0,
        "vendor_id": "0x209f",
        "node_name": "/dev/aries0",
        "device_type": "Aries",
        "target_device_type": "aries-rb",
        "hardware_version": "Aries2",
        "firmware": {"version": "2.0.1", "revision": "7"},
        "device_id": "0x0",
        "subsystem_vendor_id": "0x402",
        "subsystem_device_id": "0x108b",
        "link_generation": 4,
        "lane_width": 8,
        "revision": "0x2",
        "class": "0x7800002",
        "memory_total_bytes": 1073741824,
    }


def test_get_static_info_enriches_mbltml_npus_with_pcie_metadata(
    fake_mbltml, monkeypatch
) -> None:
    monkeypatch.setattr(npu_module, "get_all_pcie_devices", lambda: [])
    monkeypatch.setattr(
        npu_module,
        "get_pcie_static_info",
        lambda **kwargs: {
            "hardware": {
                "npus": [
                    {
                        "dev_no": 0,
                        "vendor_id": "0x1ed5",
                        "device_id": "0x0100",
                        "current_link_speed": "16.0 GT/s PCIe",
                        "current_link_width": "8",
                        "max_link_speed": "16.0 GT/s PCIe",
                        "max_link_width": "8",
                        "status": "OK",
                    }
                ]
            }
        },
    )

    tracker = NPUDeviceTracker(npu_id=0)
    npu = tracker.get_static_info()["hardware"]["npus"][0]

    assert npu["vendor_id"] == "0x209f"
    assert npu["device_id"] == "0x0"
    assert npu["current_link_speed"] == "16.0 GT/s PCIe"
    assert npu["current_link_width"] == "8"
    assert npu["max_link_speed"] == "16.0 GT/s PCIe"
    assert npu["max_link_width"] == "8"
    assert npu["max_link_generation"] == "Gen4"
    assert npu["max_lane_width"] == "x8"
    assert npu["status"] == "OK"


def test_get_static_info_enriches_before_selected_npu_filter(
    fake_mbltml, monkeypatch
) -> None:
    monkeypatch.setattr(npu_module, "get_all_pcie_devices", lambda: [])
    monkeypatch.setattr(
        npu_module,
        "get_pcie_static_info",
        lambda **kwargs: {
            "hardware": {
                "npus": [
                    {
                        "dev_no": 0,
                        "vendor_id": "0x209f",
                        "device_id": "0x0",
                        "current_link_speed": "8.0 GT/s PCIe",
                        "current_link_width": "4",
                    },
                    {
                        "dev_no": 1,
                        "vendor_id": "0x209f",
                        "device_id": "0x0",
                        "current_link_speed": "16.0 GT/s PCIe",
                        "current_link_width": "8",
                    },
                ]
            }
        },
    )

    tracker = NPUDeviceTracker(npu_id=1)
    npu = tracker.get_static_info()["hardware"]["npus"][0]

    assert npu["dev_no"] == 1
    assert npu["node_name"] == "/dev/aries1"
    assert npu["current_link_speed"] == "16.0 GT/s PCIe"
    assert npu["current_link_width"] == "8"


def test_get_static_info_limits_mbltml_metadata_to_selected_npu(
    fake_mbltml, monkeypatch
) -> None:
    monkeypatch.setattr(npu_module, "get_all_pcie_devices", lambda: [])
    monkeypatch.setattr(npu_module, "get_pcie_static_info", lambda **kwargs: {})

    tracker = NPUDeviceTracker(npu_id=1)

    info = tracker.get_static_info()

    assert info["hardware"]["npus"] == [
        {
            "dev_no": 1,
            "node_name": "/dev/aries1",
            "device_type": "Aries",
            "target_device_type": "aries-rb",
            "hardware_version": "Aries2",
            "firmware": {"version": "2.0.1", "revision": "7"},
            "vendor_id": "0x209f",
            "device_id": "0x0",
            "subsystem_vendor_id": "0x402",
            "subsystem_device_id": "0x108b",
            "link_generation": 4,
            "lane_width": 8,
            "revision": "0x2",
            "class": "0x7800002",
            "memory_total_bytes": 1073741824,
        }
    ]


def test_get_static_info_ignores_legacy_pcie_filter_env_vars(
    fake_mbltml, monkeypatch
) -> None:
    monkeypatch.setenv("MBLT_TRACKER_NPU_PCI_VENDOR_ID", "209f")
    monkeypatch.setattr(npu_module, "get_all_pcie_devices", lambda: [])
    monkeypatch.setattr(npu_module, "get_pcie_static_info", lambda **kwargs: {})

    tracker = NPUDeviceTracker()

    info = tracker.get_static_info()

    assert [npu["dev_no"] for npu in info["hardware"]["npus"]] == [0, 1]


def test_physical_dev_no_follows_mbltml_enumeration_across_target_types(
    monkeypatch,
) -> None:
    fake = FakeMbltml(
        devices={
            # Per-type lists, deliberately not in system enumeration order.
            6: ["/dev/regulus-usb0"],
            2: ["/dev/regulus-npu0"],
            1: ["/dev/aries1"],
            3: ["/dev/aries0", "/dev/aries2"],
        },
        selected_rail={0: 0, 1: 0},
    )
    monkeypatch.setattr(npu_module, "mbltml", fake)

    devices = npu_module._discover_devices()

    assert [(d.node_name, d.target_device_type, d.index) for d in devices] == [
        ("/dev/aries0", 3, 0),
        ("/dev/aries1", 1, 0),
        ("/dev/aries2", 3, 1),
        ("/dev/regulus-npu0", 2, 0),
        ("/dev/regulus-usb0", 6, 0),
    ]
    assert [d.hardware_version for d in devices] == [
        "Aries2",
        "Aries",
        "Aries2",
        "Regulus",
        "Regulus2",
    ]


def test_selected_dev_no_routes_calls_to_its_target_type_and_index(
    monkeypatch,
) -> None:
    fake = FakeMbltml(
        devices={1: ["/dev/aries1"], 3: ["/dev/aries0", "/dev/aries2"]},
        selected_rail={0: 0, 1: 0},
    )
    set_calls: list[tuple[int, int, int]] = []

    def record_set(tdt, dev_no, rail_id):
        set_calls.append((tdt, dev_no, rail_id))

    monkeypatch.setattr(fake, "mbltmlSetExtraPmicID", record_set)
    monkeypatch.setattr(npu_module, "mbltml", fake)
    monkeypatch.setattr(npu_module.time, "time", lambda: 100.0)

    # Physical dev_no 2 is /dev/aries2, the second aries-rb device.
    tracker = NPUDeviceTracker(interval=0.1, npu_id=2)
    tracker._func_for_sched()
    tracker._restore_npu_rail_selection()

    assert tracker.get_trace() == [(100.0, 11.0)]
    assert list(tracker.get_metric()["devices"]) == [2]
    assert tracker.get_metric()["devices"][2]["node_name"] == "/dev/aries2"
    assert set_calls == [
        (fake.MBLTML_TARGET_DEVICE_ARIES_RB, 1, fake.MBLTML_EXTRA_PMIC_ID_NPU),
        (fake.MBLTML_TARGET_DEVICE_ARIES_RB, 1, fake.MBLTML_EXTRA_PMIC_ID_NPU),
    ]
    with pytest.raises(ValueError, match="Invalid NPU ID"):
        NPUDeviceTracker(npu_id=3)


def test_static_info_numbers_mixed_target_types_by_physical_dev_no(
    monkeypatch,
) -> None:
    fake = FakeMbltml(devices={1: ["/dev/aries1"], 3: ["/dev/aries0"]})
    monkeypatch.setattr(npu_module, "mbltml", fake)

    npus = npu_module._get_mbltml_static_info()["hardware"]["npus"]

    assert [
        (npu["dev_no"], npu["node_name"], npu["target_device_type"]) for npu in npus
    ] == [(0, "/dev/aries0", "aries-rb"), (1, "/dev/aries1", "aries-ra")]


def test_mbltml_is_initialized_once_per_module(fake_mbltml) -> None:
    NPUDeviceTracker(interval=0.1)
    NPUDeviceTracker(interval=0.1, npu_id=1).get_static_info()

    assert fake_mbltml.init_calls == 1


def test_legacy_mbltml_api_is_rejected_with_install_hint(monkeypatch) -> None:
    class LegacyMbltml:
        def mbltmlInit(self):
            raise AssertionError("legacy module must not be initialized")

        def mbltmlGetDeviceCount(self):
            return 1

    monkeypatch.setattr(npu_module, "mbltml", LegacyMbltml())

    with pytest.raises(RuntimeError, match="mobilint-ml"):
        NPUDeviceTracker(interval=0.1)
    assert npu_module._get_mbltml_static_info() == {}


def test_no_addressable_devices_raises(monkeypatch) -> None:
    monkeypatch.setattr(npu_module, "mbltml", FakeMbltml(devices={}))

    with pytest.raises(RuntimeError, match="No Mobilint NPU devices"):
        NPUDeviceTracker(interval=0.1)


def test_mbltml_shutdown_is_registered_for_interpreter_exit(monkeypatch) -> None:
    fake = FakeMbltml()
    registered = []
    monkeypatch.setattr(npu_module, "mbltml", fake)
    monkeypatch.setattr(
        npu_module.atexit, "register", lambda *args: registered.append(args)
    )

    NPUDeviceTracker(interval=0.1)
    NPUDeviceTracker(interval=0.1)

    assert registered == [(npu_module._shutdown_mbltml, fake)]
    npu_module._shutdown_mbltml(fake)
    npu_module._shutdown_mbltml(fake)
    assert fake.shutdown_calls == 1

    NPUDeviceTracker(interval=0.1)
    assert fake.init_calls == 2
