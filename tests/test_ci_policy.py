"""Regression checks for the release suite's no-skip policy."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "source",
    [
        'import pytest\npytest.skip("collection", allow_module_level=True)\n',
        'import pytest\ndef test_skip():\n    pytest.skip("runtime")\n',
        'import pytest\npytest.importorskip("bmk_intentionally_missing_dependency")\n',
    ],
)
def test_required_suite_rejects_skips(tmp_path, source):
    (tmp_path / "conftest.py").write_text(
        Path(__file__).with_name("conftest.py").read_text(), encoding="utf-8"
    )
    (tmp_path / "test_example.py").write_text(source, encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    for required in (False, True):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *(["--fail-on-skip"] if required else []),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if required:
            assert result.returncode in (1, 2), result.stdout + result.stderr
            assert "Required test" in result.stdout
        else:
            assert result.returncode in (0, 5), result.stdout + result.stderr
            assert "1 skipped" in result.stdout
