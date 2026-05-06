from __future__ import annotations

from typing import TypedDict


class CpuHardwareInfo(TypedDict):
    architecture: str
    physical_cores: int | None
    logical_cores: int | None
    model_name: str | None
    vendor: str | None


class DimmInfo(TypedDict, total=False):
    manufacturer: str
    part_number: str
    serial_number: str
    capacity_bytes: int
    speed_mhz: int
    configured_speed_mhz: int
    data_width_bits: int
    total_width_bits: int
    type: str


class DramInfo(TypedDict):
    total_bytes: int
    available_bytes: int


class DramInfoOptional(DramInfo, total=False):
    dimms: list[DimmInfo]
    theoretical_bandwidth_gbps: float
    dimms_collection_note: str


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
        "bus_address": str,
        "vendor_id": str,
        "device_id": str,
        "subsystem_vendor_id": str,
        "subsystem_device_id": str,
        "class": str,
        "name": str,
        "manufacturer": str,
        "status": str,
        "pnp_device_id": str,
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
    gpu: GpuInferenceInfo
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
    DramInfoOptional: {"dimms": [DimmInfo]},
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