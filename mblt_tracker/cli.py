from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, TextIO, cast

from ._types import CollectOutput
from .static_info import (
    _clean_typed_dict,
    _deep_merge,
    _remove_os_pcie_link_fields,
    _sanitize_static_info_for_public_output,
    get_all_pcie_devices,
    get_host_static_info,
    get_nvml_gpu_static_info,
    get_pcie_static_info,
)


def collect_static_info(
    pcie_vendor_id: str | None = None,
    pcie_device_id: str | None = None,
    pcie_class_filter: str | None = None,
    all_pcie_devices: bool = False,
    sudo_password: str | None = None,
    sudo_password_provider: Callable[[], str] | None = None,
) -> CollectOutput:
    """Collect best-effort static host and PCIe information."""
    pcie_devices = get_all_pcie_devices()
    info = cast(
        dict[str, object],
        get_host_static_info(
            sudo_password=sudo_password,
            sudo_password_provider=sudo_password_provider,
            pcie_devices=pcie_devices,
        ),
    )
    pcie_info = get_pcie_static_info(
        vendor_id=pcie_vendor_id,
        device_id=pcie_device_id,
        class_filter=pcie_class_filter,
        include_all_devices=all_pcie_devices,
        devices=pcie_devices,
        include_private_identifiers=True,
    )
    npu_metadata_filter = (
        _extract_hardware_npus(pcie_info)
        if _has_pcie_filter(pcie_vendor_id, pcie_device_id, pcie_class_filter)
        else None
    )
    _deep_merge(info, pcie_info)
    nvml_gpu_info = get_nvml_gpu_static_info(
        pcie_devices=pcie_devices,
        include_private_identifiers=True,
    )
    _remove_os_link_fields_for_nvml_gpu_matches(info, nvml_gpu_info)
    _deep_merge(
        info,
        nvml_gpu_info,
    )
    if npu_metadata_filter != []:
        _deep_merge(info, _collect_mbltml_npu_metadata(filtered_npus=npu_metadata_filter))
    public_info = _sanitize_static_info_for_public_output(info)
    return cast(CollectOutput, _clean_typed_dict(public_info, CollectOutput))


def _has_pcie_filter(
    vendor_id: str | None, device_id: str | None, class_filter: str | None
) -> bool:
    return any(value is not None for value in (vendor_id, device_id, class_filter))


def _extract_hardware_npus(info: Mapping[str, object]) -> list[dict[str, object]]:
    hardware = info.get("hardware")
    if not isinstance(hardware, Mapping):
        return []
    npus = hardware.get("npus")
    if not isinstance(npus, list):
        return []
    return [dict(npu) for npu in npus if isinstance(npu, dict)]


def _remove_os_link_fields_for_nvml_gpu_matches(
    base_info: dict[str, object],
    nvml_info: Mapping[str, object],
) -> None:
    """Drop stale OS PCIe link fields before merging matching NVML GPU entries."""
    base_gpus = _extract_hardware_gpus(base_info)
    nvml_gpus = _extract_hardware_gpus(nvml_info)
    for overlay_index, nvml_gpu in enumerate(nvml_gpus):
        matched_gpu = _find_matching_base_gpu_for_overlay(
            base_gpus,
            nvml_gpu,
            overlay_index,
        )
        if matched_gpu is not None:
            _remove_os_pcie_link_fields(matched_gpu)


def _extract_hardware_gpus(info: Mapping[str, object]) -> list[dict[str, object]]:
    hardware = info.get("hardware")
    if not isinstance(hardware, Mapping):
        return []
    gpus = hardware.get("gpus")
    if not isinstance(gpus, list):
        return []
    return [gpu for gpu in gpus if isinstance(gpu, dict)]


def _find_matching_base_gpu_for_overlay(
    base_gpus: list[dict[str, object]],
    overlay_gpu: dict[str, object],
    overlay_index: int,
) -> dict[str, object] | None:
    overlay_has_identity = _has_device_identity(overlay_gpu)
    overlay_key = overlay_gpu.get("dev_no", overlay_index)
    for base_index, base_gpu in enumerate(base_gpus):
        if _device_identity_matches(base_gpu, overlay_gpu):
            return base_gpu
        if overlay_has_identity:
            continue
        base_key = base_gpu.get("dev_no", base_index)
        if base_key == overlay_key:
            return base_gpu
    return None


def _has_device_identity(device: Mapping[str, object]) -> bool:
    return any(device.get(key) is not None for key in ("bus_address", "pnp_device_id"))


def _device_identity_matches(
    base_device: Mapping[str, object],
    overlay_device: Mapping[str, object],
) -> bool:
    for key in ("bus_address", "pnp_device_id"):
        base_value = base_device.get(key)
        overlay_value = overlay_device.get(key)
        if base_value is not None and overlay_value is not None:
            if str(base_value).lower() == str(overlay_value).lower():
                return True
    return False


def _collect_mbltml_npu_metadata(
    filtered_npus: list[dict[str, object]] | None,
) -> dict[str, object]:
    if filtered_npus == []:
        return {}
    try:
        from .device_tracker_npu import _get_mbltml_static_info
    except Exception:
        return {}
    info = _get_mbltml_static_info()
    if filtered_npus is not None:
        _filter_npu_metadata_to_selected_devices(info, filtered_npus)
    return info


def _filter_npu_metadata_to_selected_devices(
    info: dict[str, object], selected_devices: list[dict[str, object]]
) -> None:
    hardware = info.get("hardware")
    if not isinstance(hardware, dict):
        return
    npus = hardware.get("npus")
    if not isinstance(npus, list):
        return
    selected = []
    for metadata in npus:
        if not isinstance(metadata, dict):
            continue
        for selected_device in selected_devices:
            if metadata.get("dev_no") == selected_device.get("dev_no"):
                selected.append(metadata)
                break
            if str(metadata.get("vendor_id", "")).lower() == str(
                selected_device.get("vendor_id", "")
            ).lower():
                selected.append(metadata)
                break
    hardware["npus"] = selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mblt-tracker",
        description="Mobilint device tracker command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect static host and PCIe information.",
    )
    collect_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write collected static information to a JSON file.",
    )
    collect_parser.add_argument(
        "--pcie-vendor-id",
        help="Optional PCIe vendor ID filter, with or without 0x prefix.",
    )
    collect_parser.add_argument(
        "--pcie-device-id",
        help="Optional PCIe device ID filter, with or without 0x prefix.",
    )
    collect_parser.add_argument(
        "--pcie-class-filter",
        help="Optional PCIe class prefix filter, e.g. 0x12.",
    )
    collect_parser.add_argument(
        "--all-pcie-devices",
        action="store_true",
        help=(
            "Include all PCIe devices instead of only GPU/NPU-related devices; "
            "private identifiers are still omitted from output."
        ),
    )

    return parser


def _write_json(info: Mapping[str, object], output: Path | None, stdout: TextIO) -> None:
    text = json.dumps(cast(Any, info), indent=2, sort_keys=True) + "\n"
    if output is None:
        stdout.write(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "collect":
        info = collect_static_info(
            pcie_vendor_id=args.pcie_vendor_id,
            pcie_device_id=args.pcie_device_id,
            pcie_class_filter=args.pcie_class_filter,
            all_pcie_devices=args.all_pcie_devices,
            sudo_password_provider=None,
        )
        _write_json(info, args.output, sys.stdout)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
