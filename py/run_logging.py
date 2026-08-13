# run_logging.py
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log_command(run_id: str, run_results_dir: Path, extra: str = "") -> None:
    """
    Append current command to:
      <run_results_dir>/run_commands.txt
    """
    run_results_dir.mkdir(parents=True, exist_ok=True)
    cmd_path = run_results_dir / "run_commands.txt"

    py = sys.executable
    cwd = os.getcwd()
    argv = " ".join([_quote(a) for a in sys.argv])

    line = f"[{_now()}] python={py}  cwd={cwd}  argv={argv}"
    if extra.strip():
        line += f"  | {extra.strip()}"
    with cmd_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def update_manifest(manifest_path: Path, patch: Dict[str, Any]) -> None:
    """
    Merge patch into run_manifest.json (top-level merge).
    Safe for repeated calls.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        try:
            obj = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                obj = {}
        except Exception:
            obj = {}
    else:
        obj = {}

    def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                _deep_merge(dst[key], value)
            else:
                dst[key] = value

    obj["_updated_at"] = _now()
    _deep_merge(obj, patch)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{manifest_path.name}.", suffix=".tmp", dir=str(manifest_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, manifest_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def snapshot_file(
    src: Path,
    run_results_dir: Path,
    category: str,
    rename_to: Optional[str] = None,
) -> Optional[Path]:
    """
    Copy src into:
      <run_results_dir>/inputs_snapshot/<category>/<filename>
    Return destination path, or None if src doesn't exist.
    """
    if src is None:
        return None
    src = Path(src)
    if not src.exists():
        return None

    snap_dir = run_results_dir / "inputs_snapshot" / category
    snap_dir.mkdir(parents=True, exist_ok=True)

    dst_name = rename_to if rename_to else src.name
    dst = snap_dir / dst_name

    # avoid silent overwrite: if exists and identical size/time, keep; otherwise create suffix
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return dst
        dst = snap_dir / f"{dst.stem}__{datetime.now().strftime('%H%M%S')}{dst.suffix}"

    shutil.copy2(src, dst)
    return dst


def _quote(s: str) -> str:
    # make Windows-friendly quoting for spaces
    if " " in s or "\t" in s:
        return f'"{s}"'
    return s
