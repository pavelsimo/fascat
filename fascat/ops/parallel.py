from __future__ import annotations

import multiprocessing
import pickle
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Literal, TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")

ExecutorKind = Literal["thread", "process"]


def worker_count(jobs: int, item_count: int) -> int:
    if jobs < 1:
        raise ValueError("jobs must be greater than or equal to 1")
    if item_count <= 1:
        return 1
    return min(jobs, item_count)


def _process_pool_usable(worker: Callable[[_T], _R], items: Sequence[_T]) -> bool:
    # Preflight instead of catching errors from the pool: exceptions raised by
    # the worker itself must propagate unmasked, so eligibility is decided
    # before any work is submitted. Items are homogeneous payloads — checking
    # the first one is representative.
    try:
        pickle.dumps(worker)
        pickle.dumps(items[0])
    except Exception:
        return False
    return True


def parallel_map(
    items: Sequence[_T],
    worker: Callable[[_T], _R],
    *,
    jobs: int,
    executor: ExecutorKind = "thread",
) -> list[_R]:
    """Map ``worker`` over ``items``, preserving input order.

    With ``executor="process"`` the worker must be a module-level function and
    every item picklable; payloads must never carry ``Part.source_shape``
    (OCCT handles cannot cross process boundaries) — strip parts with
    ``part.copy(keep_source=False)`` first. Unpicklable workers or items fall
    back to threads; worker exceptions always propagate. Environments that
    cannot spawn processes should run with ``jobs=1``.
    """
    workers = worker_count(jobs, len(items))
    if workers <= 1:
        return [worker(item) for item in items]
    if executor == "process" and _process_pool_usable(worker, items):
        # spawn is the only fork-safe start method alongside OCCT/native
        # threads, and the platform default on macOS and Windows.
        context = multiprocessing.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                chunksize = max(1, len(items) // (workers * 4))
                return list(pool.map(worker, items, chunksize=chunksize))
        except BrokenProcessPool as exc:
            raise RuntimeError("parallel worker process crashed; retry with jobs=1 to run in-process") from exc
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fascat-part") as pool:
        return list(pool.map(worker, items))
