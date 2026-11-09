"""Download helpers. Files land in data/raw/<dataset>/ and are only ever parsed as data."""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path

import httpx


def fetch(url: str, path: Path) -> Path:
    """Download url to path unless it is already there."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as response:
        response.raise_for_status()
        with partial.open("wb") as file:
            for block in response.iter_bytes(1 << 20):
                file.write(block)
    partial.rename(path)
    return path


def extract(archive: Path, into: Path) -> Path:
    """Extract a .zip or .tgz once, refusing members that would land outside `into`."""
    marker = into / f".extracted-{archive.name}"
    if marker.exists():
        return into
    into.mkdir(parents=True, exist_ok=True)
    root = into.resolve()
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if not (root / name).resolve().is_relative_to(root):
                    raise ValueError(f"{archive}: unsafe member {name}")
            z.extractall(into)
    else:
        with tarfile.open(archive) as t:
            t.extractall(into, filter="data")
    marker.touch()
    return into


def hf_files(repo: str, patterns: list[str], into: Path) -> Path:
    """Download matching files of a Hugging Face dataset repo into `into`."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns, local_dir=into)
    return into


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)
