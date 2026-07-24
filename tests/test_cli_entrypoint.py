from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

from typer import Typer


def test_console_script_targets_the_packaged_typer_app() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    target = pyproject["project"]["scripts"]["medcoevo"]

    assert target == "mediated_coevo.main:app"

    module_name, attr_name = target.split(":", maxsplit=1)
    app = getattr(importlib.import_module(module_name), attr_name)

    assert isinstance(app, Typer)
