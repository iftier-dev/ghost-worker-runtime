"""
pipeline_cache.py — in-process, VRAM-resident model/pipeline cache.

Without this, every single task would reload its model from disk into
VRAM from scratch (10-30+ seconds for a diffusion model) -- at scale
this would make the whole distributed network's effective latency worse
than a single non-cached local install. This keeps loaded pipeline
objects resident in the CURRENT process's VRAM across calls, evicting
only the least-recently-used one when a NEW model needs VRAM room and
none is free (measured via torch, not guessed).

Also provides per-loader-key locks: exactly one heavy GPU job runs at a
time per model family in this process, serializing concurrent requests
onto the same GPU instead of racing two generations into the same VRAM
pool (a real OOM/crash risk on a single-GPU volunteer node). Cheap
(non-GPU) task types like embedding are exempt via `exclusive=False`.
"""
from __future__ import annotations
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("worker_runtime.pipeline_cache")

_PIPELINE_IDLE_TTL_SECONDS = 600  # evict a loaded pipeline after 10 min unused

_pipelines: Dict[str, Any] = {}
_pipeline_last_used: Dict[str, float] = {}
_pipeline_cache_lock = threading.Lock()  # guards the dicts above only

_loader_locks: Dict[str, threading.Lock] = {}
_loader_locks_guard = threading.Lock()  # guards creation of entries in _loader_locks


_REAPER_INTERVAL_SECONDS = 60  # active idle-sweep tick -- independent
# of whether any new task ever arrives. Without this, an idle pipeline
# left resident after the last task would burn a volunteer's GPU
# power/VRAM indefinitely with zero new work ever triggering its
# eviction (the lazy sweep inside get_or_load() only runs when SOMETHING
# calls get_or_load() again). This background thread guarantees eviction
# happens on a wall-clock schedule regardless of future task arrival.
_reaper_started = False
_reaper_started_lock = threading.Lock()


def _reaper_tick() -> None:
    with _pipeline_cache_lock:
        now = time.time()
        stale = [k for k, last in _pipeline_last_used.items() if now - last > _PIPELINE_IDLE_TTL_SECONDS]
        for k in stale:
            _evict_locked(k)
        if stale:
            logger.info("[pipeline_cache] reaper evicted %d idle pipeline(s): %s", len(stale), stale)


def _reaper_loop() -> None:
    while True:
        time.sleep(_REAPER_INTERVAL_SECONDS)
        try:
            _reaper_tick()
        except Exception as e:
            logger.warning("[pipeline_cache] reaper tick failed (non-fatal): %s", e)


def _start_reaper_thread() -> None:
    """Idempotent -- safe to call from multiple import sites, starts the
    background reaper exactly once per process."""
    global _reaper_started
    with _reaper_started_lock:
        if _reaper_started:
            return
        t = threading.Thread(target=_reaper_loop, daemon=True, name="pipeline-cache-reaper")
        t.start()
        _reaper_started = True
        logger.info("[pipeline_cache] idle-pipeline reaper thread started (interval=%ds, ttl=%ds)",
                    _REAPER_INTERVAL_SECONDS, _PIPELINE_IDLE_TTL_SECONDS)


_start_reaper_thread()


def get_loader_lock(loader_key: str) -> threading.Lock:
    """One lock per loader_key (e.g. 'image_diffusers', 'video_diffusers')
    -- shared across every model within that family, since they all
    compete for the same GPU's VRAM in this process."""
    with _loader_locks_guard:
        if loader_key not in _loader_locks:
            _loader_locks[loader_key] = threading.Lock()
        return _loader_locks[loader_key]


def _free_vram_gb() -> Optional[float]:
    """Real, measured free VRAM via torch -- returns None if torch/CUDA
    isn't available (e.g. this loader doesn't need a GPU at all)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        return free_bytes / (1024 ** 3)
    except Exception:
        return None


def get_or_load(cache_key: str, loader_fn: Callable[[], Any], required_vram_gb: float = 0.0) -> Any:
    """Returns a cached pipeline object for cache_key, loading it via
    loader_fn() only on a cache miss. Evicts the least-recently-used
    OTHER cached pipeline first if measured free VRAM is below
    required_vram_gb, so a big model swap doesn't fail with an avoidable
    OOM when an old, unused model is still resident. Never evicts
    cache_key itself, and never evicts if it's the only entry."""
    with _pipeline_cache_lock:
        if cache_key in _pipelines:
            _pipeline_last_used[cache_key] = time.time()
            return _pipelines[cache_key]

        # TTL sweep -- drop anything idle past the TTL before considering
        # VRAM-pressure eviction, cheap and keeps the cache tidy over time.
        now = time.time()
        stale = [k for k, last in _pipeline_last_used.items() if now - last > _PIPELINE_IDLE_TTL_SECONDS]
        for k in stale:
            _evict_locked(k)

        free_gb = _free_vram_gb()
        if free_gb is not None and required_vram_gb > 0 and free_gb < required_vram_gb and _pipelines:
            # Evict the single least-recently-used entry and re-check --
            # repeat until enough room or nothing left to evict. Bounded
            # by len(_pipelines) so this can never loop forever.
            for _ in range(len(_pipelines)):
                if not _pipelines:
                    break
                lru_key = min(_pipeline_last_used, key=_pipeline_last_used.get)
                _evict_locked(lru_key)
                free_gb = _free_vram_gb()
                if free_gb is None or free_gb >= required_vram_gb:
                    break

        logger.info("[pipeline_cache] loading model for cache_key=%s (cold start)", cache_key)
        pipeline = loader_fn()
        _pipelines[cache_key] = pipeline
        _pipeline_last_used[cache_key] = time.time()
        return pipeline


def _evict_locked(cache_key: str) -> None:
    """Caller must hold _pipeline_cache_lock. Drops the Python reference
    and asks torch to release the freed VRAM back to the driver."""
    pipeline = _pipelines.pop(cache_key, None)
    _pipeline_last_used.pop(cache_key, None)
    if pipeline is not None:
        try:
            del pipeline
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("[pipeline_cache] evicted cache_key=%s", cache_key)
