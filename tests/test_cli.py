from __future__ import annotations

import json

from mblt_tracker import cli


class FakeStdin:
    def __init__(self, is_tty: bool) -> None:
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


def test_collect_prints_static_info_as_json(monkeypatch, capsys) -> None:
    def fake_collect_static_info(**kwargs):
        sudo_password_provider = kwargs.pop("sudo_password_provider")
        assert callable(sudo_password_provider)
        assert kwargs == {
            "pcie_vendor_id": None,
            "pcie_device_id": None,
            "pcie_class_filter": None,
            "all_pcie_devices": False,
        }
        return {"hardware": {"cpu": {"architecture": "x86_64"}}}

    def fail_getpass(_prompt):
        raise AssertionError("getpass should be deferred until dmidecode fails")

    monkeypatch.setattr(cli, "collect_static_info", fake_collect_static_info)
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.sys, "stdin", FakeStdin(is_tty=True))
    monkeypatch.setattr(cli.getpass, "getpass", fail_getpass)

    exit_code = cli.main(["collect"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "hardware": {"cpu": {"architecture": "x86_64"}}
    }


def test_collect_writes_static_info_to_output_file(monkeypatch, tmp_path, capsys) -> None:
    def fake_collect_static_info(**kwargs):
        sudo_password_provider = kwargs.pop("sudo_password_provider")
        assert callable(sudo_password_provider)
        assert kwargs == {
            "pcie_vendor_id": "1ed5",
            "pcie_device_id": "0100",
            "pcie_class_filter": "0x12",
            "all_pcie_devices": True,
        }
        return {"hardware": {"npus": [{"vendor_id": "0x1ed5"}]}}

    def fail_getpass(_prompt):
        raise AssertionError("getpass should be deferred until dmidecode fails")

    monkeypatch.setattr(cli, "collect_static_info", fake_collect_static_info)
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.sys, "stdin", FakeStdin(is_tty=True))
    monkeypatch.setattr(cli.getpass, "getpass", fail_getpass)
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


def test_collect_skips_sudo_prompt_when_non_interactive(monkeypatch, capsys) -> None:
    def fake_collect_static_info(**kwargs):
        assert kwargs == {
            "pcie_vendor_id": None,
            "pcie_device_id": None,
            "pcie_class_filter": None,
            "all_pcie_devices": False,
            "sudo_password_provider": None,
        }
        return {"hardware": {"cpu": {"architecture": "x86_64"}}}

    def fail_getpass(_prompt):
        raise AssertionError("getpass should not be called for non-interactive stdin")

    monkeypatch.setattr(cli, "collect_static_info", fake_collect_static_info)
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.sys, "stdin", FakeStdin(is_tty=False))
    monkeypatch.setattr(cli.getpass, "getpass", fail_getpass)

    exit_code = cli.main(["collect"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "hardware": {"cpu": {"architecture": "x86_64"}}
    }


def test_collect_static_info_merges_windows_npu_driver_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "get_host_static_info",
        lambda sudo_password=None, sudo_password_provider=None: {
            "hardware": {"cpu": {"architecture": "AMD64"}}
        },
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
        lambda **_kwargs: {
            "hardware": {"npus": [{"vendor_id": "0x209f"}]},
            "inference": {"npu_driver_version": "1.8.1.1348"},
        },
    )
    monkeypatch.setattr(cli, "get_linux_npu_driver_firmware_info", lambda **_kwargs: {})
    monkeypatch.setattr(cli, "get_all_pcie_devices", lambda: [])
    monkeypatch.setattr(cli, "get_nvml_gpu_static_info", lambda pcie_devices: {})

    info = cli.collect_static_info()

    assert info == {
        "hardware": {
            "cpu": {"architecture": "AMD64"},
            "npus": [{"vendor_id": "0x209f"}],
        },
        "inference": {"npu_driver_version": "1.8.1.1348"},
    }


def test_collect_static_info_merges_linux_npu_driver_firmware_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "get_host_static_info",
        lambda sudo_password=None, sudo_password_provider=None: {
            "hardware": {"cpu": {"architecture": "x86_64"}}
        },
    )
    monkeypatch.setattr(
        cli,
        "get_pcie_static_info",
        lambda **_kwargs: {"hardware": {"npus": [{"vendor_id": "0x209f"}]}},
    )
    monkeypatch.setattr(cli, "get_windows_npu_driver_firmware_info", lambda **_kwargs: {})
    monkeypatch.setattr(cli, "get_all_pcie_devices", lambda: [])
    monkeypatch.setattr(cli, "get_nvml_gpu_static_info", lambda pcie_devices: {})
    monkeypatch.setattr(
        cli,
        "get_linux_npu_driver_firmware_info",
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
                {"vendor_id": "0x209f", "dev_no": 0, "firmware": {"version": "1.2.4"}}
            ],
        },
        "inference": {"npu_driver_version": "1.12.0"},
    }


def test_collect_static_info_passes_pcie_filters_to_npu_metadata_helpers(
    monkeypatch,
) -> None:
    pcie_devices = [
        {"vendor_id": "0x1ed5", "device_id": "0x0100", "class": "0x120000"},
        {"vendor_id": "0x209f", "device_id": "0x0000", "class": "0x120000"},
    ]
    pcie_info = {"hardware": {"npus": [{"dev_no": 0, "vendor_id": "0x1ed5"}]}}
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "get_host_static_info",
        lambda sudo_password=None, sudo_password_provider=None: {"hardware": {}},
    )
    monkeypatch.setattr(cli, "get_all_pcie_devices", lambda: pcie_devices)
    monkeypatch.setattr(cli, "get_pcie_static_info", lambda **_kwargs: pcie_info)
    monkeypatch.setattr(cli, "get_nvml_gpu_static_info", lambda pcie_devices: {})

    def fake_windows_metadata(**kwargs):
        calls["windows"] = kwargs
        return {}

    def fake_linux_metadata(**kwargs):
        calls["linux"] = kwargs
        return {}

    monkeypatch.setattr(cli, "get_windows_npu_driver_firmware_info", fake_windows_metadata)
    monkeypatch.setattr(cli, "get_linux_npu_driver_firmware_info", fake_linux_metadata)

    cli.collect_static_info(
        pcie_vendor_id="1ed5",
        pcie_device_id="0100",
        pcie_class_filter="0x12",
    )

    assert calls["windows"] == {
        "vendor_id": "1ed5",
        "device_id": "0100",
        "class_filter": "0x12",
        "devices": pcie_devices,
    }
    assert calls["linux"] == {"npu_devices": [{"dev_no": 0, "vendor_id": "0x1ed5"}]}


def test_collect_static_info_does_not_limit_npu_metadata_without_pcie_filter(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "get_host_static_info",
        lambda sudo_password=None, sudo_password_provider=None: {"hardware": {}},
    )
    monkeypatch.setattr(cli, "get_all_pcie_devices", lambda: [])
    monkeypatch.setattr(cli, "get_pcie_static_info", lambda **_kwargs: {})
    monkeypatch.setattr(cli, "get_nvml_gpu_static_info", lambda pcie_devices: {})
    monkeypatch.setattr(cli, "get_windows_npu_driver_firmware_info", lambda **_kwargs: {})

    def fake_linux_metadata(**kwargs):
        calls.update(kwargs)
        return {}

    monkeypatch.setattr(cli, "get_linux_npu_driver_firmware_info", fake_linux_metadata)

    cli.collect_static_info()

    assert calls == {"npu_devices": None}


def test_collect_static_info_does_not_add_unfiltered_npu_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "get_host_static_info",
        lambda sudo_password=None, sudo_password_provider=None: {"hardware": {}},
    )
    monkeypatch.setattr(cli, "get_all_pcie_devices", lambda: [])
    monkeypatch.setattr(cli, "get_pcie_static_info", lambda **_kwargs: {})
    monkeypatch.setattr(cli, "get_nvml_gpu_static_info", lambda pcie_devices: {})
    monkeypatch.setattr(
        cli,
        "get_windows_npu_driver_firmware_info",
        lambda **_kwargs: {
            "hardware": {"npus": [{"vendor_id": "0x209f"}]},
            "inference": {"npu_driver_version": "1.8.1.1348"},
        },
    )
    monkeypatch.setattr(
        cli,
        "get_linux_npu_driver_firmware_info",
        lambda **_kwargs: {
            "hardware": {"npus": [{"dev_no": 0, "firmware": {"version": "1.2.4"}}]},
            "inference": {"npu_driver_version": "1.12.0"},
        },
    )

    info = cli.collect_static_info(pcie_vendor_id="1ed5")

    assert info == {"hardware": {}}