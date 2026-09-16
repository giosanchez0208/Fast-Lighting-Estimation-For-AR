"""Checkpoint writes that a killed process cannot corrupt.

`torch.save` writes straight to its target. A process killed mid-write leaves a
truncated file where the last good checkpoint was, `--resume` then has nothing
to load, and the whole run is lost. Training here gets interrupted routinely,
by a laptop that died mid-run and by deliberate pauses.

Writing to a sibling temporary file and renaming it over the target closes the
window. `os.replace` is atomic when both paths are on the same volume, on
Windows as on POSIX, so a reader sees the old checkpoint or the new one and
never a partial write. A kill during the temporary write leaves a stale `.tmp`
that the next save overwrites and that resume never reads.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def save_atomic(obj, path: str | Path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
