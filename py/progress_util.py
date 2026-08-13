# progress_util.py
from __future__ import annotations

from typing import Iterable, Iterator, Optional, TypeVar
import sys

T = TypeVar("T")

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


def progress(it: Iterable[T], total: Optional[int] = None, desc: str = "") -> Iterator[T]:
    """
    tqdm if available; otherwise a lightweight percent printer.
    IMPORTANT (Windows PowerShell): tqdm default writes to stderr which may be treated as errors.
    We force tqdm to write to stdout to avoid NativeCommandError.
    """
    if tqdm is not None:
        yield from tqdm(it, total=total, desc=desc, ncols=90, file=sys.stdout)
        return

    # fallback: print percent occasionally
    if total is None:
        try:
            total = len(it)  # type: ignore
        except Exception:
            total = None

    if total is None or total <= 0:
        for x in it:
            yield x
        return

    step = max(1, total // 50)  # print ~50 updates
    for i, x in enumerate(it, start=1):
        if i == 1 or i % step == 0 or i == total:
            pct = 100.0 * i / total
            print(f"[progress] {desc} {i}/{total} ({pct:.1f}%)", end="\r")
        yield x
    print()