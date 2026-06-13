from __future__ import annotations

from typing import TypedDict


class _CpuHardwareInfoRequired(TypedDict):
    architecture: str
    model_name: str | None
    vendor: str | None


class CpuHardwareInfo(_CpuHardwareInfoRequired):
    pass


class DramInfo(TypedDict):
    total_bytes: int
    available_bytes: int


class DramModuleInfo(TypedDict, total=False):
    capacity_bytes: int
    capacity_mb: float
    capacity_gb: float
    ram_type: str
    speed_mhz: int
    configured_speed_mhz: int
    data_width_bits: int
    total_width_bits: int
    theoretical_bandwidth_gbps: float


class DramInfoOptional(DramInfo, total=False):
    total_mb: float
    total_gb: float
    available_mb: float
    available_gb: float
    ram_type: str
    speed_mhz: int
    configured_speed_mhz: int
    theoretical_bandwidth_gbps: float
    module_count: int
    modules: list[DramModuleInfo]


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
    regulus_usb_version: str | None


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
        "max_link_generation": str,
        "max_lane_width": str,
        "memory_total_bytes": int,
        "architecture": str,
    },
    total=False,
)


class NpuDeviceInfo(PcieDeviceInfo, total=False):
    firmware: NpuFirmwareInfo


class MotherboardPcieInfo(TypedDict, total=False):
    max_link_generation: str
    max_link_speed: str
    max_lane_width: str


class MotherboardInfo(TypedDict, total=False):
    manufacturer: str
    model_name: str
    version: str
    chipset: str
    pcie: MotherboardPcieInfo


class _HardwareInfoRequired(TypedDict):
    cpu: CpuHardwareInfo
    dram: DramInfoOptional


class HardwareInfo(_HardwareInfoRequired, total=False):
    gpu: GpuHardwareInfoOptional
    gpus: list[PcieDeviceInfo]
    motherboard: MotherboardInfo
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
        "motherboard": MotherboardInfo,
        "npus": [NpuDeviceInfo],
        "pcie_devices": [PcieDeviceInfo],
    },
    DramInfoOptional: {"modules": [DramModuleInfo]},
    MotherboardInfo: {"pcie": MotherboardPcieInfo},
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
