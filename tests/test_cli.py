from __future__ import annotations

import json
import sys

import pytest

from mblt_tracker import cli


def test_collect_prints_static_info_as_json(monkeypatch, capsys) -> None:
    def fake_collect_static_info(**kwargs):
        assert kwargs == {
            "all_pcie_devices": False,
            "sudo_password_provider": None,
        }
        return {"hardware": {"cpu": {"architecture": "x86_64"}}}

    monkeypatch.setattr(cli, "collect_static_info", fake_collect_static_info)

    exit_code = cli.main(["collect"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "hardware": {"cpu": {"architecture": "x86_64"}}
    }


def test_collect_writes_static_info_to_output_file(
    monkeypatch, tmp_path, capsys
) -> None:
    def fake_collect_static_info(**kwargs):
        assert kwargs == {
            "all_pcie_devices": True,
            "sudo_password_provider": None,
        }
        return {"hardware": {"npus": [{"vendor_id": "0x1ed5"}]}}

    monkeypatch.setattr(cli, "collect_static_info", fake_collect_static_info)
    output = tmp_path / "nested" / "static_info.json"

    exit_code = cli.main(
        [
            "collect",
            "--output",
            str(output),
            "--all-pcie-devices",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == ""
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "hardware": {"npus": [{"vendor_id": "0x1ed5"}]}
    }


def test_collect_never_creates_sudo_prompt_provider(monkeypatch, capsys) -> None:
    def fake_collect_static_info(**kwargs):
        assert kwargs == {
            "all_pcie_devices": False,
            "sudo_password_provider": None,
        }
        return {"hardware": {"cpu": {"architecture": "x86_64"}}}

    monkeypatch.setattr(cli, "collect_static_info", fake_collect_static_info)

    exit_code = cli.main(["collect"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "hardware": {"cpu": {"architecture": "x86_64"}}
    }


def test_collect_static_info_merges_mbltml_npu_driver_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "get_host_static_info",
        lambda sudo_password=None, sudo_password_provider=None, pcie_devices=None: {
            "hardware": {"cpu": {"architecture": "AMD64"}}
        },
    )
    monkeypatch.setattr(
        cli,
        "get_pcie_static_info",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        cli,
        "_collect_mbltml_npu_metadata",
        lambda **_kwargs: {
            "hardware": {"npus": [{"vendor_id": "0x209f"}]},
            "inference": {"npu_driver_version": "1.8.1.1348"},
        },
    )
    monkeypatch.setattr(cli, "get_all_pcie_devices", lambda: [])
    monkeypatch.setattr(cli, "get_nvml_gpu_static_info", lambda **_kwargs: {})

    info = cli.collect_static_info()

    assert info == {
        "hardware": {
            "cpu": {"architecture": "AMD64"},
            "npus": [{"vendor_id": "0x209f"}],
        },
        "inference": {"npu_driver_version": "1.8.1.1348"},
    }


def test_collect_mbltml_npu_metadata_import_errors_are_not_suppressed(
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "mblt_tracker.device_tracker_npu", None)

    with pytest.raises(ModuleNotFoundError):
        cli._collect_mbltml_npu_metadata()


def test_collect_static_info_merges_mbltml_npu_firmware_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "get_host_static_info",
        lambda sudo_password=None, sudo_password_provider=None, pcie_devices=None: {
            "hardware": {"cpu": {"architecture": "x86_64"}}
        },
    )
    monkeypatch.setattr(
        cli,
        "get_pcie_static_info",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(cli, "get_all_pcie_devices", lambda: [])
    monkeypatch.setattr(cli, "get_nvml_gpu_static_info", lambda **_kwargs: {})
    monkeypatch.setattr(
        cli,
        "_collect_mbltml_npu_metadata",
        lambda **_kwargs: {
            "hardware": {"npus": [{"dev_no": 0, "firmware": {"version": "1.2.4"}}]},
            "inference": {"npu_driver_version": "1.12.0"},
        },
    )

    info = cli.collect_static_info()

    assert info == {
        "hardware": {
            "cpu": {"architecture": "x86_64"},
            "npus": [
                {"dev_no": 0, "firmware": {"version": "1.2.4"}}
            ],
        },
        "inference": {"npu_driver_version": "1.12.0"},
    }


def test_collect_static_info_removes_os_link_fields_for_nvml_gpu_match(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "get_host_static_info",
        lambda sudo_password=None, sudo_password_provider=None, pcie_devices=None: {
            "hardware": {}
        },
    )
    monkeypatch.setattr(cli, "get_all_pcie_devices", lambda: [])
    monkeypatch.setattr(
        cli,
        "get_pcie_static_info",
        lambda **_kwargs: {
            "hardware": {
                "gpus": [
                    {
                        "dev_no": 0,
                        "bus_address": "0000:03:00.0",
                        "vendor_id": "0x10de",
                        "device_id": "0x2204",
                        "current_link_speed": "8.0 GT/s PCIe",
                        "current_link_width": "4",
                        "max_link_speed": "16.0 GT/s PCIe",
                        "max_link_width": "16",
                        "link_generation": "Gen3",
                        "lane_width": "x4",
                        "max_link_generation": "Gen4",
                        "max_lane_width": "x16",
                    }
                ]
            }
        },
    )
    monkeypatch.setattr(
        cli,
        "get_nvml_gpu_static_info",
        lambda **_kwargs: {
            "hardware": {
                "gpus": [
                    {
                        "dev_no": 0,
                        "bus_address": "0000:03:00.0",
                        "vendor_id": "0x10de",
                        "device_id": "0x2204",
                        "driver_version": "595.97",
                        "link_generation": "Gen2",
                        "lane_width": "x4",
                        "max_link_generation": "Gen4",
                        "max_lane_width": "x16",
                        "name": "NVIDIA GeForce RTX 3090",
                    }
                ]
            }
        },
    )
    monkeypatch.setattr(cli, "_collect_mbltml_npu_metadata", lambda **_kwargs: {})

    info = cli.collect_static_info()

    assert info == {
        "hardware": {
            "gpus": [
                {
                    "dev_no": 0,
                    "vendor_id": "0x10de",
                    "device_id": "0x2204",
                    "driver_version": "595.97",
                    "link_generation": "Gen2",
                    "lane_width": "x4",
                    "max_link_generation": "Gen4",
                    "max_lane_width": "x16",
                    "name": "NVIDIA GeForce RTX 3090",
                }
            ]
        }
    }


def test_collect_static_info_enriches_mbltml_npus_with_pcie_metadata(
    monkeypatch,
) -> None:
    pcie_devices = [
        {
            "vendor_id": "0x209f",
            "device_id": "0x0000",
            "class": "0x120000",
            "current_link_speed": "16.0 GT/s PCIe",
            "current_link_width": "8",
            "max_link_speed": "16.0 GT/s PCIe",
            "max_link_width": "8",
            "status": "OK",
        },
    ]

    monkeypatch.setattr(
        cli,
        "get_host_static_info",
        lambda sudo_password=None, sudo_password_provider=None, pcie_devices=None: {
            "hardware": {}
        },
    )
    monkeypatch.setattr(cli, "get_all_pcie_devices", lambda: pcie_devices)
    monkeypatch.setattr(
        cli,
        "get_pcie_static_info",
        lambda **kwargs: {
            "hardware": {"npus": [pcie_devices[0]]}
        }
        if kwargs.get("include_npus")
        else {},
    )
    monkeypatch.setattr(cli, "get_nvml_gpu_static_info", lambda **_kwargs: {})
    monkeypatch.setattr(
        cli,
        "_collect_mbltml_npu_metadata",
        lambda: {"hardware": {"npus": [{"dev_no": 0, "vendor_id": "0x209f", "device_id": "0x0"}]}},
    )

    info = cli.collect_static_info()

    assert info["hardware"]["npus"] == [
        {
            "dev_no": 0,
            "vendor_id": "0x209f",
            "device_id": "0x0",
            "current_link_speed": "16.0 GT/s PCIe",
            "current_link_width": "8",
            "max_link_speed": "16.0 GT/s PCIe",
            "max_link_width": "8",
            "max_link_generation": "Gen4",
            "max_lane_width": "x8",
            "status": "OK",
        }
    ]


def test_collect_static_info_always_collects_mbltml_npu_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "get_host_static_info",
        lambda sudo_password=None, sudo_password_provider=None, pcie_devices=None: {
            "hardware": {}
        },
    )
    monkeypatch.setattr(cli, "get_all_pcie_devices", lambda: [])
    monkeypatch.setattr(cli, "get_pcie_static_info", lambda **_kwargs: {})
    monkeypatch.setattr(cli, "get_nvml_gpu_static_info", lambda **_kwargs: {})
    monkeypatch.setattr(
        cli,
        "_collect_mbltml_npu_metadata",
        lambda: {"hardware": {"npus": [{"dev_no": 0}]}},
    )

    info = cli.collect_static_info()

    assert info == {"hardware": {"npus": [{"dev_no": 0}]}}


def test_collect_static_info_does_not_create_npus_from_pcie_only(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "get_host_static_info",
        lambda sudo_password=None, sudo_password_provider=None, pcie_devices=None: {
            "hardware": {}
        },
    )
    monkeypatch.setattr(cli, "get_all_pcie_devices", lambda: [])
    monkeypatch.setattr(
        cli,
        "get_pcie_static_info",
        lambda **kwargs: {
            "hardware": {
                "npus": [
                    {
                        "dev_no": 0,
                        "vendor_id": "0x209f",
                        "device_id": "0x0000",
                        "current_link_speed": "16.0 GT/s PCIe",
                    }
                ]
            }
        }
        if kwargs.get("include_npus")
        else {},
    )
    monkeypatch.setattr(cli, "get_nvml_gpu_static_info", lambda **_kwargs: {})
    monkeypatch.setattr(cli, "_collect_mbltml_npu_metadata", lambda: {})

    info = cli.collect_static_info()

    assert "npus" not in info.get("hardware", {})


def test_collect_static_info_enriches_four_mbltml_npus_by_pcie_order(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "get_host_static_info",
        lambda sudo_password=None, sudo_password_provider=None, pcie_devices=None: {
            "hardware": {}
        },
    )
    monkeypatch.setattr(cli, "get_all_pcie_devices", lambda: [])
    monkeypatch.setattr(
        cli,
        "get_pcie_static_info",
        lambda **kwargs: {
            "hardware": {
                "npus": [
                    {
                        "dev_no": index,
                        "vendor_id": "0x209f",
                        "device_id": "0x0",
                        "current_link_speed": f"speed-{index}",
                        "current_link_width": str(index + 1),
                        "status": "OK",
                    }
                    for index in range(4)
                ]
            }
        }
        if kwargs.get("include_npus")
        else {},
    )
    monkeypatch.setattr(cli, "get_nvml_gpu_static_info", lambda **_kwargs: {})
    monkeypatch.setattr(
        cli,
        "_collect_mbltml_npu_metadata",
        lambda: {
            "hardware": {
                "npus": [
                    {
                        "dev_no": index,
                        "node_name": f"aries{index}",
                        "vendor_id": "0x209f",
                        "device_id": "0x0",
                    }
                    for index in range(4)
                ]
            }
        },
    )

    info = cli.collect_static_info()

    assert [npu["dev_no"] for npu in info["hardware"]["npus"]] == [0, 1, 2, 3]
    assert [npu["node_name"] for npu in info["hardware"]["npus"]] == [
        "aries0",
        "aries1",
        "aries2",
        "aries3",
    ]
    assert [npu["current_link_speed"] for npu in info["hardware"]["npus"]] == [
        "speed-0",
        "speed-1",
        "speed-2",
        "speed-3",
    ]


def test_collect_parser_rejects_removed_pcie_npu_filter_option() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["collect", "--npu-pci-vendor-id", "209f"])
