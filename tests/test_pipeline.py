from pathlib import Path
import subprocess
import sys

import pytest

from src.pipeline import validate_input


def test_missing_input_file_raises_clear_error(tmp_path) -> None:
    missing = tmp_path / "missing.mp4"

    with pytest.raises(FileNotFoundError, match="Входной файл не найден"):
        validate_input(missing)


def test_existing_input_file_is_valid(tmp_path) -> None:
    video = tmp_path / "video.mp4"
    video.touch()

    validate_input(video)


def test_pipeline_supports_direct_script_invocation() -> None:
    result = subprocess.run(
        [sys.executable, "src/pipeline.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--device" in result.stdout
