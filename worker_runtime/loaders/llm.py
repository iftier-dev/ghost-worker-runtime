"""
llm.py — llm_inference (via Ollama, already-running local daemon assumed
-- same OLLAMA_BASE pattern as api-key's litellm_config_generator.py) and
embedding (via sentence-transformers/HF, VRAM-resident pipeline cache).
"""
from __future__ import annotations
import os
from typing import Any, Dict

from ..pipeline_cache import get_or_load

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


def _run_llm_inference(payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    import httpx
    prompt = payload.get("prompt", "")
    if not prompt:
        return {"success": False, "error": "prompt is required for llm_inference"}
    repo = model_cfg.get("repo") or "llama3.1:8b"
    try:
        with httpx.Client(timeout=180) as client:
            r = client.post(f"{OLLAMA_BASE}/api/generate", json={
                "model": repo, "prompt": prompt, "stream": False,
            })
            r.raise_for_status()
            data = r.json()
        return {"output_text": data.get("response", "")}
    except Exception as e:
        return {"success": False, "error": f"ollama request failed: {e}"}


def _get_embedding_model(repo: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(repo)


def _run_embedding(payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    text = payload.get("text") or payload.get("prompt", "")
    if not text:
        return {"success": False, "error": "text is required for embedding"}
    repo = model_cfg.get("repo") or "BAAI/bge-large-en-v1.5"
    model = get_or_load(
        cache_key=f"embedding:{repo}",
        loader_fn=lambda: _get_embedding_model(repo),
        required_vram_gb=model_cfg.get("min_vram_gb", 3.0),
    )
    vector = model.encode(text).tolist()
    return {"embedding": vector, "dimensions": len(vector)}


def run(task_type: str, payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if task_type == "llm_inference":
        return _run_llm_inference(payload, model_cfg)
    if task_type == "embedding":
        return _run_embedding(payload, model_cfg)
    return {"success": False, "error": f"llm.py loader does not handle task_type={task_type!r}"}
