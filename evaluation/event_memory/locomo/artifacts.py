"""Artifact name resolution for the canonical (renamed) layout.

Files on disk carry canonical, config-derived names. Scripts keep writing the
legacy tag strings they always did; wrap any artifact filename in `A(...)`:

    db  = A(f"locomo-{tag}.sqlite")
    out = A(f"search-{stag}.json")

`A` returns the canonical path. It works for artifacts that already exist
(exact map, built at rename time) and for runs that do not exist yet (the
name is derived from the same decoder, so it is stateless -- no counter, no
manifest lookup required).

Round-trip:  legacy_name -> A() -> canonical    |    canonical -> legacy() -> legacy_name
"""
from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAP_PATH = os.path.join(_HERE, "artifact_map.json")

_KINDS = ("eval", "search", "locomo", "ingest", "log")
_EXTS = (".vec.sqlite", ".sqlite", ".json", ".out", ".txt")

_fwd: dict[str, str] = {}
_rev: dict[str, str] = {}
if os.path.exists(_MAP_PATH):
    with open(_MAP_PATH) as _f:
        _fwd = json.load(_f)
    _rev = {v: k for k, v in _fwd.items()}


def _derive(name: str) -> str:
    """Canonical name for an artifact that is not in the map (a new run)."""
    try:
        from decode import decode
        from manifest import EVAL_SUFFIX, canonical, tag_hash, tag_of
    except Exception:
        return name  # decoder unavailable -> behave as a no-op
    tag, kind = tag_of(name)
    ext = next((e for e in _EXTS if name.endswith(e)), "")
    stem = name[: -len(ext)] if ext else name
    suffix = [t for t in stem.split("-") if t in EVAL_SUFFIX]
    sfx = ("__" + "-".join(suffix)) if suffix else ""
    base = canonical(tag_hash(tag), decode(tag))
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "", f"{base}__{kind}{sfx}{ext}")


def A(name: str) -> str:
    """Legacy artifact name -> canonical name on disk.

    `artifact_map.json` is the signal that the rename has been applied. If it
    is absent the directory is in the legacy layout, so this is a pass-through
    and every call site keeps working unchanged -- that is what makes the
    rename revertible without touching the scripts.
    """
    if not _fwd:
        return name
    base = os.path.basename(name)
    d = os.path.dirname(name)
    out = _fwd.get(base) or _derive(base)
    return os.path.join(d, out) if d else out


def legacy(name: str) -> str:
    """Canonical name -> the original legacy name (for reading old notes)."""
    base = os.path.basename(name)
    return _rev.get(base, base)


def exists(name: str) -> bool:
    """True if the artifact exists under its canonical name."""
    return os.path.exists(os.path.join(_HERE, A(name)))


def sibling(name: str, kind: str, ext: str | None = None) -> str | None:
    """Given any canonical artifact, return the same run's artifact for
    another stage. Canonical names are `{base}__{kind}[__{suffix}]{ext}`, so
    the run identity is everything before the LAST `__{kind}` marker.

    Replaces the legacy string-surgery (strip '-mini-mb-c14', prepend
    'search-') which no longer applies once names are canonical.

        sibling("ab12__seg__eval__seg-mini-mb-c14.json", "search")
        -> "ab12__seg__search.json"
    """
    base = os.path.basename(name)
    cur_ext = next((e for e in _EXTS if base.endswith(e)), "")
    stem = base[: -len(cur_ext)] if cur_ext else base
    parts = stem.split("__")
    idx = next((i for i, p in enumerate(parts)
                if p in ("eval", "search", "locomo", "ingest",
                         "logeval", "logsearch", "logingest")), None)
    if idx is None:
        return None
    run = "__".join(parts[:idx])
    want_ext = ext if ext is not None else cur_ext
    # The target may carry its own stage suffix (e.g. `__search__seg.json`),
    # so match by glob rather than reconstructing an exact name.
    import glob as _glob
    hits = sorted(_glob.glob(os.path.join(_HERE, f"{run}__{kind}*{want_ext}")))
    if hits:
        return os.path.basename(hits[0])
    exact = os.path.join(_HERE, f"{run}__{kind}{want_ext}")
    return os.path.basename(exact) if os.path.exists(exact) else None


def pattern(kind: str, ext: str = "") -> str:
    """Glob pattern matching all canonical artifacts of a stage.

    Canonical names end with `__{kind}[__{eval suffix}]{ext}`, so a legacy
    pattern like "eval-*.json" becomes pattern("eval", ".json").
    Do NOT pass globs to A() -- it resolves concrete names only.
    """
    return f"*__{kind}*{ext}" if ext else f"*__{kind}*"
