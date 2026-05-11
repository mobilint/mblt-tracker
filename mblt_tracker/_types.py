from __future__ import annotations

from typing import TypedDict


class CpuHardwareInfo(TypedDict):
    architecture: str
    physical_cores: int | None
    logical_cores: int | None
    model_name: str | None
    vendor: str | None


class DramInfo(TypedDict):
    total_bytes: int
    available_bytes: int


class DramInfoOptional(DramInfo, total=False):
    ram_type: str
    speed_mhz: int
    configured_speed_mhz: int
    theoretical_bandwidth_gbps: float


class CpuPowerPolicy(TypedDict):
    governor: str | None
    power_plan: str | None
    min_processor_state_pct: int | None
    max_processor_state_pct: int | None


class VersionInfo(TypedDict):
    version: str | None


class OsInfo(TypedDict):
    name: str
    version: str
    kernel_version: str


class NpuDriverInfo(TypedDict):
    aries_version: str | None
    regulus_version: str | None


class NpuFirmwareInfo(TypedDict):
    version: str | None


class GpuStaticDeviceInfo(TypedDict):
    device_index: int
    name: str


class GpuHardwareInfo(TypedDict):
    device_count: int | None


class GpuHardwareInfoOptional(GpuHardwareInfo, total=False):
    devices: list[GpuStaticDeviceInfo]


class GpuInferenceInfo(TypedDict):
    driver: VersionInfo
    cuda_driver: VersionInfo


PcieDeviceInfo = TypedDict(
    "PcieDeviceInfo",
    {
        "dev_no": int,
        "vendor_id": str,
        "device_id": str,
        "subsystem_vendor_id": str,
        "subsystem_device_id": str,
        "class": str,
        "name": str,
        "manufacturer": str,
        "status": str,
        "revision": str,
        "driver_version": str,
        "driver_name": str,
        "driver_date": str,
        "driver_description": str,
        "driver_provider": str,
        "firmware_version": str,
        "firmware_revision": str,
        "current_link_speed": str,
        "current_link_width": str,
        "max_link_speed": str,
        "max_link_width": str,
        "link_generation": str,
        "lane_width": str,
        "memory_total_bytes": int,
        "architecture": str,
    },
    total=False,
)


class NpuDeviceInfo(PcieDeviceInfo, total=False):
    board_name: str
    card_id: int
    card_model: str
    firmware: NpuFirmwareInfo


class _HardwareInfoRequired(TypedDict):
    cpu: CpuHardwareInfo
    dram: DramInfoOptional


class HardwareInfo(_HardwareInfoRequired, total=False):
    gpu: GpuHardwareInfoOptional
    gpus: list[PcieDeviceInfo]
    npus: list[NpuDeviceInfo]
    pcie_devices: list[PcieDeviceInfo]


class _InferenceInfoRequired(TypedDict):
    cpu: CpuPowerPolicy
    cuda: VersionInfo
    npu_driver_version: str
    os: OsInfo
    qbcompiler: VersionInfo
    qbruntime: VersionInfo


class InferenceInfo(_InferenceInfoRequired, total=False):
    driver: NpuDriverInfo
    gpu: GpuInferenceInfo


class CollectOutput(TypedDict):
    hardware: HardwareInfo
    inference: InferenceInfo


STATIC_INFO_CHILD_SCHEMAS: dict[type, dict[str, object]] = {
    CollectOutput: {"hardware": HardwareInfo, "inference": InferenceInfo},
    HardwareInfo: {
        "cpu": CpuHardwareInfo,
        "dram": DramInfoOptional,
        "gpu": GpuHardwareInfoOptional,
        "gpus": [PcieDeviceInfo],
        "npus": [NpuDeviceInfo],
        "pcie_devices": [PcieDeviceInfo],
    },
    GpuHardwareInfoOptional: {"devices": [GpuStaticDeviceInfo]},
    InferenceInfo: {
        "cpu": CpuPowerPolicy,
        "cuda": VersionInfo,
        "driver": NpuDriverInfo,
        "gpu": GpuInferenceInfo,
        "os": OsInfo,
        "qbcompiler": VersionInfo,
        "qbruntime": VersionInfo,
    },
    GpuInferenceInfo: {"driver": VersionInfo, "cuda_driver": VersionInfo},
    NpuDeviceInfo: {"firmware": NpuFirmwareInfo},
}
