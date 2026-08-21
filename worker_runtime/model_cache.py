"""
model_cache.py — disk-quota-aware LRU model cache, mirrors ghost-agent-go-repo's
gpu_worker.go tieredCacheCapBytes()/enforceCacheSizeCap() design: size-based
LRU eviction as the PRIMARY mechanism, TTL as a secondary safety net. Every
provider (Colab, local sidecar, RunPod, Vast.ai) shares this same cache
directory logic so a model downloaded once by one task doesn't re-download
for the next task on the same machine.
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Optional

MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/tmp/ghost-model-cache")
MODEL_CACHE_TTL_SECONDS = int(os.environ.get("MODEL_CACHE_TTL_SECONDS", str(7 * 24 * 3600)))  # 7 days


def _tiered_cache_cap_bytes(vram_free_gb: float) -> int:
    """Same 4-tier thresholds as Go's tieredCacheCapBytes()."""
    if vram_free_gb > 24:
        return 200 * 1024 ** 3
    if vram_free_gb > 16:
        return 80 * 1024 ** 3
    if vram_free_gb > 8:
        return 40 * 1024 ** 3
    return 15 * 1024 ** 3


def ensure_cache_dir() -> str:
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    return MODEL_CACHE_DIR


def enforce_cache_cap(vram_free_gb: float) -> None:
    """Size-based LRU eviction (PRIMARY) -- walks the cache dir, evicts
    least-recently-used entries (by mtime) until under the VRAM-tiered
    cap. Also drops anything past MODEL_CACHE_TTL_SECONDS regardless of
    size (SECONDARY safety net), matching the Go agent's dual-mechanism
    design."""
    cache_dir = ensure_cache_dir()
    cap_bytes = _tiered_cache_cap_bytes(vram_free_gb)
    now = time.time()

    entries = []
    total_size = 0
    for root, _, files in os.walk(cache_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                st = os.stat(fpath)
            except FileNotFoundError:
                continue
            entries.append((fpath, st.st_size, st.st_mtime))
            total_size += st.st_size

    # TTL sweep first (secondary safety net)
    for fpath, size, mtime in list(entries):
        if now - mtime > MODEL_CACHE_TTL_SECONDS:
            try:
                os.remove(fpath)
                total_size -= size
                entries.remove((fpath, size, mtime))
            except Exception:
                pass

    # Size-based LRU eviction (primary) -- oldest mtime first
    if total_size > cap_bytes:
        entries.sort(key=lambda e: e[2])  # oldest first
        for fpath, size, _mtime in entries:
            if total_size <= cap_bytes:
                break
            try:
                os.remove(fpath)
                total_size -= size
            except Exception:
                pass


def model_cache_path(repo: str) -> str:
    """Deterministic on-disk path for a given HF repo id / model name --
    used as HF_HOME/TRANSFORMERS_CACHE style redirect so every loader
    downloads into the same shared, size-capped directory instead of
    each library's own default (unbounded) cache location."""
    safe = repo.replace("/", "__") if repo else "unknown"
    return os.path.join(ensure_cache_dir(), safe)
