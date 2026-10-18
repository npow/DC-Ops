"""Reuse the known-good AppKit configuration without changing the camera build."""
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent.parent
destination = root / "translation/board/RTE"
if not destination.exists():
    shutil.copytree(root / "board/DevKit-E8/RTE", destination)
