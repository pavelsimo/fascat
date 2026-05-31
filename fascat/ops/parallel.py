from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")


def worker_count(jobs: int, item_count: int) -> int:
    if jobs < 1:
        raise ValueError("jobs must be greater than or equal to 1")
    if item_count <= 1:
        return 1
    return min(jobs, item_count)


def parallel_map(items: Sequence[_T], worker: Callable[[_T], _R], *, jobs: int) -> list[_R]:
    workers = worker_count(jobs, len(items))
    if workers <= 1:
        return [worker(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fascat-part") as executor:
        return list(executor.map(worker, items))
