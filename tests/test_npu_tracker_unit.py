# ruff: noqa: N802
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import mblt_tracker.device_tracker_npu as npu_module
from mblt_tracker.device_tracker_npu import NPUDeviceTracker


@dataclass
class FakeMbltml:
    MBLTML_DEVICE_ARIES: int = 1
    MBLTML_DEVICE_REGULUS: int = 2
    MBLTML_DEVICE_REGULUS_USB: int = 4
    MBLTML_HARDWARE_VERSION_ARIES: int = 1
    MBLTML_HARDWARE_VERSION_ARIES2: int = 3
    MBLTML_HARDWARE_VERSION_REGULUS: int = 2
    MBLTML_HARDWARE_VERSION_REGULUS2: int = 4
    MBLTML_EXTRA_PMIC_ID_NPU: int = 0
    MBLTML_EXTRA_PMIC_ID_DDR: int = 1
    MBLTML_EXTRA_PMIC_ID_PMIC: int = 2
    MBLTML_EXTRA_PMIC_ID_GOLDFINGER: int = 3
    init_device_types: set[int] = field(default_factory=set)
    selected_rail: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    set_calls: list[tuple[int, int]] = field(default_factory=list)

    def mbltmlInitDevices(self, device_types):
        self.init_device_types = device_types

    def mbltmlGetDeviceCount(self):
        return 2

    def mbltmlGetTotalPower(self, dev_no):
        return 10.0 + dev_no

    def mbltmlGetTotalCurrent(self, dev_no):
        return 1.0 + dev_no

    def mbltmlGetTotalVoltage(self, dev_no):
        return 12.0

    def mbltmlGetTotalUtilization(self, dev_no):
        return 0.25 + (0.25 * dev_no)

    def mbltmlGetMemoryUsage(self, dev_no):
        return (256 + 256 * dev_no) * 1024 * 1024

    def mbltmlGetMemoryTotal(self, dev_no):
        return 1024 * 1024 * 1024

    def mbltmlGetTemperature(self, dev_no):
        return 40 + dev_no

    def mbltmlGetNodeName(self, dev_no):
        return f"aries{dev_no}"

    def mbltmlGetHardwareVersion(self, dev_no):
        return self.MBLTML_HARDWARE_VERSION_ARIES2

    def mbltmlGetDeviceType(self, dev_no):
        return self.MBLTML_DEVICE_ARIES

    def mbltmlSetExtraPmicID(self, dev_no, rail_id):
        self.selected_rail[dev_no] = rail_id
        self.set_calls.append((dev_no, rail_id))

    def mbltmlGetExtraPmicPower(self, dev_no):
        rail = self.selected_rail[dev_no]
        return {0: 2.0, 1: 3.0, 2: 4.0, 3: 5.0}[rail] + dev_no

    def mbltmlGetExtraPmicCurrent(self, dev_no):
        rail = self.selected_rail[dev_no]
        return {0: 0.2, 1: 0.3, 2: 0.4, 3: 0.5}[rail] + dev_no

    def mbltmlGetExtraPmicVoltage(self, dev_no):
        return 12.0

    def mbltmlGetDriverVersion(self, device_type):
        return {
            self.MBLTML_DEVICE_ARIES: "1.2.3",
            self.MBLTML_DEVICE_REGULUS: "4.5.6",
            self.MBLTML_DEVICE_REGULUS_USB: "7.8.9",
        }[device_type]

    def mbltmlGetFirmwareVersion(self, dev_no):
        return "2.0.1"

    def mbltmlGetFirmwareRevision(self, dev_no):
        return 7

    def mbltmlGetVendorId(self, dev_no):
        return 0x209F

    def mbltmlGetDeviceId(self, dev_no):
        return 0

    def mbltmlGetSubVendorId(self, dev_no):
        return 0x402

    def mbltmlGetSubDeviceId(self, dev_no):
        return 0x108B

    def mbltmlGetPcieGen(self, dev_no):
        return 4

    def mbltmlGetPcieLanes(self, dev_no):
        return 8

    def mbltmlGetPcieRev(self, dev_no):
        return 2

    def mbltmlGetPcieClassCode(self, dev_no):
        return 0x7800002


@pytest.fixture()
def fake_mbltml(monkeypatch):
    fake = FakeMbltml()
    monkeypatch.setattr(npu_module, "mbltml", fake)
    return fake


def test_npu_default_sampling_uses_mbltml_without_rail_selection(fake_mbltml, monkeypatch):
    tracker = NPUDeviceTracker(interval=0.1)
    monkeypatch.setattr(npu_module.time, "time", lambda: 100.0)

    tracker._func_for_sched()

    assert fake_mbltml.init_device_types == {
        fake_mbltml.MBLTML_DEVICE_ARIES,
        fake_mbltml.MBLTML_DEVICE_REGULUS,
        fake_mbltml.MBLTML_DEVICE_REGULUS_USB,
    }
    assert fake_mbltml.set_calls == []
    assert tracker.get_trace() == [(100.0, 21.0)]
    assert tracker.get_npu_rail_power_trace() == [(100.0, 5.0)]
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
    assert metrics["devices"][0]["node_name"] == "aries0"
    assert metrics["rail_metrics"] == {
        "selected": ["npu"],
        "firmware_refresh_period_s": 1.0,
        "extra_rail_requires_selection_delay": True,
    }


def test_npu_aggregate_memory_usage_pct_uses_summed_usage_and_capacity(
    fake_mbltml, monkeypatch
) -> None:
    memory_usage_mb = {0: 1024, 1: 0}
    memory_total_mb = {0: 1024, 1: 15360}
    monkeypatch.setattr(
        fake_mbltml,
        "mbltmlGetMemoryUsage",
        lambda dev_no: memory_usage_mb[dev_no] * 1024 * 1024,
    )
    monkeypatch.setattr(
        fake_mbltml,
        "mbltmlGetMemoryTotal",
        lambda dev_no: memory_total_mb[dev_no] * 1024 * 1024,
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

    metrics = tracker.get_metric()
    assert tracker.get_trace() == [(100.0, 11.0)]
    assert tracker.get_npu_rail_power_trace() == [(100.0, 3.0)]
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
    assert fake_mbltml.set_calls == [(0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_DDR)]
    assert tracker.get_ddr_rail_power_trace() == []

    now = 101.1
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()

    assert tracker.get_ddr_rail_power_trace() == [(101.1, 3.0)]
    assert tracker.get_metric()["ddr_rail_power_w_samples"] == 1


def test_failed_extra_rail_selection_does_not_leave_pending_state(
    fake_mbltml, monkeypatch
):
    now = 100.0
    original_set_extra_pmic_id = fake_mbltml.mbltmlSetExtraPmicID

    def fail_ddr_on_second_device(dev_no, rail_id):
        if dev_no == 1 and rail_id == fake_mbltml.MBLTML_EXTRA_PMIC_ID_DDR:
            raise RuntimeError("unsupported rail")
        original_set_extra_pmic_id(dev_no, rail_id)

    monkeypatch.setattr(fake_mbltml, "mbltmlSetExtraPmicID", fail_ddr_on_second_device)
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker = NPUDeviceTracker(interval=0.1, rail_metrics=["npu", "ddr"])

    tracker._func_for_sched()
    assert tracker._pending_extra_rail is None
    assert tracker._selected_rail == {0: "npu", 1: "npu"}
    assert fake_mbltml.selected_rail == {
        0: fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU,
        1: fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU,
    }
    assert fake_mbltml.set_calls == [
        (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_DDR),
        (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU),
    ]
    assert tracker.get_ddr_rail_power_trace() == []
    assert tracker.get_npu_rail_power_trace() == [(100.0, 5.0)]

    now = 101.1
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()

    assert tracker._pending_extra_rail is None
    assert tracker._selected_rail == {0: "npu", 1: "npu"}
    assert tracker._unavailable_extra_rails == {"ddr"}
    assert fake_mbltml.set_calls.count((0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_DDR)) == 1
    assert fake_mbltml.set_calls.count((0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU)) == 1
    assert tracker.get_ddr_rail_power_trace() == []
    assert tracker.get_npu_rail_power_trace()[-1] == (101.1, 5.0)


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

    assert tracker._pending_extra_rail == "pmic"
    assert tracker._unavailable_extra_rails == {"ddr"}
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
    assert fake_mbltml.set_calls == [(0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_DDR)]
    assert tracker.get_ddr_rail_power_trace() == []

    now = 100.5
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert fake_mbltml.set_calls == [(0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_DDR)]
    assert tracker.get_ddr_rail_power_trace() == []

    now = 101.1
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_ddr_rail_power_trace() == [(101.1, 3.0)]

    now = 101.2
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_npu_rail_power_trace() == []
    assert fake_mbltml.set_calls[-1] == (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU)

    now = 101.3
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_npu_rail_power_trace() == []
    assert fake_mbltml.set_calls[-1] == (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU)

    now = 102.1
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_npu_rail_power_trace() == [(102.1, 2.0)]
    assert fake_mbltml.set_calls[-1] == (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU)

    now = 102.2
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert fake_mbltml.set_calls[-1] == (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_PMIC)

    now = 103.2
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_pmic_rail_power_trace() == [(103.2, 4.0)]

    now = 103.3
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_npu_rail_power_trace() == [(102.1, 2.0)]


def test_all_rail_sampling_inserts_npu_slot_between_extra_rails(
    fake_mbltml, monkeypatch
):
    now = 100.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker = NPUDeviceTracker(interval=1.0, npu_id=0, rail_metrics="all")

    tracker._func_for_sched()
    assert fake_mbltml.set_calls == [(0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_DDR)]
    assert tracker.get_npu_rail_power_trace() == []
    assert tracker.get_ddr_rail_power_trace() == []

    now = 101.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_ddr_rail_power_trace() == [(101.0, 3.0)]
    assert tracker.get_npu_rail_power_trace() == []
    assert fake_mbltml.set_calls[-1] == (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU)

    now = 102.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_npu_rail_power_trace() == [(102.0, 2.0)]
    assert fake_mbltml.set_calls[-1] == (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU)

    now = 103.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert fake_mbltml.set_calls[-1] == (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_PMIC)
    assert tracker.get_pmic_rail_power_trace() == []

    now = 104.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_pmic_rail_power_trace() == [(104.0, 4.0)]
    assert fake_mbltml.set_calls[-1] == (0, fake_mbltml.MBLTML_EXTRA_PMIC_ID_NPU)

    now = 105.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker._func_for_sched()
    assert tracker.get_npu_rail_power_trace() == [(102.0, 2.0), (105.0, 2.0)]

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

    def fail_goldfinger(dev_no, rail_id):
        if rail_id == fake_mbltml.MBLTML_EXTRA_PMIC_ID_GOLDFINGER:
            raise RuntimeError("unsupported rail")
        original_set_extra_pmic_id(dev_no, rail_id)

    monkeypatch.setattr(fake_mbltml, "mbltmlSetExtraPmicID", fail_goldfinger)
    now = 100.0
    monkeypatch.setattr(npu_module.time, "time", lambda: now)
    tracker = NPUDeviceTracker(interval=1.0, npu_id=0, rail_metrics="all")

    for timestamp in range(100, 111):
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
        "node_name": "aries0",
        "device_type": "Aries",
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
    assert npu["node_name"] == "aries1"
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
            "node_name": "aries1",
            "device_type": "Aries",
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
