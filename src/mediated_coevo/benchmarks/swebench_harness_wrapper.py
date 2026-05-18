"""Project-owned entrypoint for the official SWE-bench harness."""

from __future__ import annotations

import runpy

STALE_SYMPY_REPO = "sympy/sympy"
STALE_SYMPY_BASE_COMMIT = "cffd4e0f86fefd4802349a9f9b19ed70934ea354"
STALE_SYMPY_BRANCH = "1.7"
OFFICIAL_SWEBENCH_RUNNER = "swebench.harness.run_evaluation"


def main() -> None:
    """Patch stale upstream constants, then delegate to the official CLI."""
    _remove_stale_sympy_branch_mapping()
    _patch_stale_sympy_clone_scope()
    runpy.run_module(OFFICIAL_SWEBENCH_RUNNER, run_name="__main__")


def _remove_stale_sympy_branch_mapping() -> None:
    from swebench.harness import constants  # type: ignore[import-untyped]

    repo_branch_map = constants.REPO_BASE_COMMIT_BRANCH.get(STALE_SYMPY_REPO)
    if repo_branch_map is None:
        return
    if repo_branch_map.get(STALE_SYMPY_BASE_COMMIT) != STALE_SYMPY_BRANCH:
        return

    del repo_branch_map[STALE_SYMPY_BASE_COMMIT]


def _patch_stale_sympy_clone_scope() -> None:
    from swebench.harness.test_spec import create_scripts  # type: ignore[import-untyped]
    from swebench.harness.test_spec import python as test_spec_python  # type: ignore[import-untyped]

    original_make_repo_script_list_py = test_spec_python.make_repo_script_list_py

    def make_repo_script_list_py(
        specs,
        repo,
        repo_directory,
        base_commit,
        env_name,
    ) -> list:
        script = original_make_repo_script_list_py(
            specs,
            repo,
            repo_directory,
            base_commit,
            env_name,
        )
        if repo != STALE_SYMPY_REPO or base_commit != STALE_SYMPY_BASE_COMMIT:
            return script

        return [
            command.replace(" --single-branch ", " ", 1)
            if command.startswith("git clone ")
            else command
            for command in script
        ]

    test_spec_python.make_repo_script_list_py = make_repo_script_list_py
    create_scripts.make_repo_script_list_py = make_repo_script_list_py


if __name__ == "__main__":
    main()
