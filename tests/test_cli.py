from __future__ import annotations

import json

from mblt_tracker import cli


def test_collect_prints_static_info_as_json(monkeypatch, capsys) -> None:
    def fake_collect_static_info(**kwargs):
        assert kwargs == {
            "pcie_vendor_id": None,
            "pcie_device_id": None,
            "pcie_class_filter": None,
            "all_pcie_devices": False,
            "sudo_password": "secret",
        }
        return {"hardware": {"cpu": {"architecture": "x86_64"}}}

    monkeypatch.setattr(cli, "collect_static_info", fake_collect_static_info)
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: "secret")

    exit_code = cli.main(["collect"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "hardware": {"cpu": {"architecture": "x86_64"}}
    }


def test_collect_writes_static_info_to_output_file(monkeypatch, tmp_path, capsys) -> None:
    def fake_collect_static_info(**kwargs):
        assert kwargs == {
            "pcie_vendor_id": "1ed5",
            "pcie_device_id": "0100",
            "pcie_class_filter": "0x12",
            "all_pcie_devices": True,
            "sudo_password": "secret",
        }
        return {"hardware": {"npus": [{"vendor_id": "0x1ed5"}]}}

    monkeypatch.setattr(cli, "collect_static_info", fake_collect_static_info)
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.getpass, "getpass", lambda _prompt: "secret")
    output = tmp_path / "nested" / "static_info.json"

    exit_code = cli.main(
        [
            "collect",
            "--output",
            str(output),
            "--pcie-vendor-id",
            "1ed5",
            "--pcie-device-id",
            "0100",
            "--pcie-class-filter",
            "0x12",
            "--all-pcie-devices",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out == ""
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "hardware": {"npus": [{"vendor_id": "0x1ed5"}]}
    }


def test_collect_static_info_merges_windows_npu_driver_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "get_host_static_info",
        lambda sudo_password=None: {"hardware": {"cpu": {"architecture": "AMD64"}}},
    )
    monkeypatch.setattr(
        cli,
        "get_pcie_static_info",
        lambda **_kwargs: {
            "hardware": {"npus": [{"vendor_id": "0x209f"}]}
        },
    )
    monkeypatch.setattr(
        cli,
        "get_windows_npu_driver_firmware_info",
        lambda: {
            "hardware": {
                "npus": [
                    {"vendor_id": "0x209f", "driver_version": "1.8.1.1348"}
                ]
            }
        },
    )
    monkeypatch.setattr(cli, "get_linux_npu_driver_firmware_info", lambda: {})

    info = cli.collect_static_info()

    assert info == {
        "hardware": {
            "cpu": {"architecture": "AMD64"},
            "npus": [{"vendor_id": "0x209f", "driver_version": "1.8.1.1348"}],
        },
    }


def test_collect_static_info_merges_linux_npu_driver_firmware_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "get_host_static_info",
        lambda sudo_password=None: {"hardware": {"cpu": {"architecture": "x86_64"}}},
    )
    monkeypatch.setattr(
        cli,
        "get_pcie_static_info",
        lambda **_kwargs: {"hardware": {"npus": [{"vendor_id": "0x209f"}]}},
    )
    monkeypatch.setattr(cli, "get_windows_npu_driver_firmware_info", lambda: {})
    monkeypatch.setattr(
        cli,
        "get_linux_npu_driver_firmware_info",
        lambda: {
            "hardware": {"npus": [{"dev_no": 0, "firmware": {"version": "1.2.4"}}]},
            "inference": {"driver": {"aries_version": "1.12.0"}},
        },
    )

    info = cli.collect_static_info()

    assert info == {
        "hardware": {
            "cpu": {"architecture": "x86_64"},
            "npus": [
                {"vendor_id": "0x209f", "dev_no": 0, "firmware": {"version": "1.2.4"}}
            ],
        },
        "inference": {"driver": {"aries_version": "1.12.0"}},
    }