from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence, TextIO

from .static_info import (
    _deep_merge,
    get_host_static_info,
    get_pcie_static_info,
    get_windows_npu_driver_firmware_info,
)


def collect_static_info(
    pcie_vendor_id: Optional[str] = None,
    pcie_device_id: Optional[str] = None,
    pcie_class_filter: Optional[str] = None,
    all_pcie_devices: bool = False,
) -> dict[str, object]:
    """Collect best-effort static host and PCIe information."""
    info = get_host_static_info()
    _deep_merge(
        info,
        get_pcie_static_info(
            vendor_id=pcie_vendor_id,
            device_id=pcie_device_id,
            class_filter=pcie_class_filter,
            include_all_devices=all_pcie_devices,
        ),
    )
    _deep_merge(info, get_windows_npu_driver_firmware_info())
    return info


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
        help="Include all PCIe devices instead of only GPU/NPU-related devices.",
    )

    return parser


def _write_json(info: dict[str, object], output: Optional[Path], stdout: TextIO) -> None:
    text = json.dumps(info, indent=2, sort_keys=True) + "\n"
    if output is None:
        stdout.write(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "collect":
        info = collect_static_info(
            pcie_vendor_id=args.pcie_vendor_id,
            pcie_device_id=args.pcie_device_id,
            pcie_class_filter=args.pcie_class_filter,
            all_pcie_devices=args.all_pcie_devices,
        )
        _write_json(info, args.output, sys.stdout)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())