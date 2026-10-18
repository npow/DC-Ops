"""Create an isolated copy of the confirmed AppKit pin/panel/camera setup."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[2]
destination = root / "translation/live/RTE"
if not destination.exists():
    shutil.copytree(root / "translation/board/RTE", destination)
