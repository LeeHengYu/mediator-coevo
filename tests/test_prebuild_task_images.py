from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_prebuild_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "utils" / "prebuild_task_images.py"
    spec = importlib.util.spec_from_file_location("prebuild_task_images", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prebuilder_tags_repo_base_for_legacy_task_dockerfile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_prebuild_module()
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        f"FROM {module.LEGACY_HARBOR_BASE_IMAGE}\nWORKDIR /root\n",
        encoding="utf-8",
    )
    local_images = {module.SKILLFLOW_HARBOR_BASE_IMAGE}
    calls: list[tuple[list[str], bool]] = []

    def fake_run(command: list[str], check: bool = True) -> SimpleNamespace:
        calls.append((command, check))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "run_cmd", fake_run)

    assert module.ensure_legacy_base_image_alias(
        dockerfile=dockerfile,
        local_images=local_images,
        dry_run=False,
    )

    assert calls == [
        (
            [
                "docker",
                "tag",
                module.SKILLFLOW_HARBOR_BASE_IMAGE,
                module.LEGACY_HARBOR_BASE_IMAGE,
            ],
            False,
        )
    ]
    assert module.LEGACY_HARBOR_BASE_IMAGE in local_images


def test_prebuilder_does_not_tag_when_legacy_base_is_unused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_prebuild_module()
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        f"FROM {module.SKILLFLOW_HARBOR_BASE_IMAGE}\n",
        encoding="utf-8",
    )

    def fake_run(command: list[str], check: bool = True) -> SimpleNamespace:
        del command, check
        raise AssertionError("docker tag should not run")

    monkeypatch.setattr(module, "run_cmd", fake_run)

    assert module.ensure_legacy_base_image_alias(
        dockerfile=dockerfile,
        local_images={module.SKILLFLOW_HARBOR_BASE_IMAGE},
        dry_run=False,
    )
