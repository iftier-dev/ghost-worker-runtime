"""
image.py — image_generation (text-to-image), image_to_image, inpainting,
image_upscale. Uses diffusers pipelines, VRAM-resident via pipeline_cache
so repeated calls to the same model never re-pay the disk-load cost.
Output is a real file written to the shared cache dir and returned as a
local path/URL under output_url -- consistent with what
gpu_manager.py's _verify_result_wellformed() and batch_coordinator.py
expect for IMAGE_GEN task types.
"""
from __future__ import annotations
import base64
import io
import os
import time
import uuid
from typing import Any, Dict

from ..pipeline_cache import get_or_load
from ..model_cache import model_cache_path

OUTPUT_DIR = os.environ.get("WORKER_OUTPUT_DIR", "/tmp/ghost-worker-output")


def _ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def _save_image_and_url(image, prefix: str) -> str:
    out_dir = _ensure_output_dir()
    fname = f"{prefix}_{uuid.uuid4().hex}.png"
    fpath = os.path.join(out_dir, fname)
    image.save(fpath)
    # Real deployments should replace this with an upload to the same
    # storage backend batch_coordinator.py's STITCH_UPLOAD_URL points at
    # (or a dedicated image upload endpoint) -- local path works for
    # same-host consumption (red-gpu's own aiohttp session downloads
    # from wherever output_url points, so any reachable URL is fine).
    upload_url = os.environ.get("IMAGE_UPLOAD_URL", "")
    if upload_url:
        try:
            import httpx
            with open(fpath, "rb") as f, httpx.Client(timeout=60) as client:
                resp = client.post(upload_url, files={"file": (fname, f, "image/png")})
                resp.raise_for_status()
                remote_url = resp.json().get("url")
                if remote_url:
                    return remote_url
        except Exception:
            pass  # fall through to local path
    return fpath


def _load_txt2img_pipeline(repo: str, pipeline_class: str):
    import torch
    import diffusers
    cls = getattr(diffusers, pipeline_class)
    pipe = cls.from_pretrained(
        repo, torch_dtype=torch.float16, cache_dir=model_cache_path(repo),
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe


def _run_image_generation(payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    prompt = payload.get("prompt", "")
    if not prompt:
        return {"success": False, "error": "prompt is required for image_generation"}
    repo = model_cfg.get("repo")
    pipeline_class = model_cfg.get("pipeline_class", "StableDiffusionXLPipeline")
    pipe = get_or_load(
        cache_key=f"image_gen:{repo}",
        loader_fn=lambda: _load_txt2img_pipeline(repo, pipeline_class),
        required_vram_gb=model_cfg.get("min_vram_gb", 10.0),
    )
    negative_prompt = payload.get("negative_prompt")
    result = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=payload.get("steps", 30))
    image = result.images[0]
    url = _save_image_and_url(image, "img")
    return {"output_url": url}


def _run_image_to_image(payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    prompt = payload.get("prompt", "")
    image_url = payload.get("image_url")
    if not prompt or not image_url:
        return {"success": False, "error": "prompt and image_url are required for image_to_image"}
    repo = model_cfg.get("repo")
    pipeline_class = model_cfg.get("pipeline_class", "StableDiffusionXLImg2ImgPipeline")
    pipe = get_or_load(
        cache_key=f"image_img2img:{repo}",
        loader_fn=lambda: _load_txt2img_pipeline(repo, pipeline_class),
        required_vram_gb=model_cfg.get("min_vram_gb", 10.0),
    )
    from PIL import Image
    import httpx
    with httpx.Client(timeout=60) as client:
        resp = client.get(image_url)
        resp.raise_for_status()
        src_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    strength = payload.get("strength", 0.7)
    result = pipe(prompt=prompt, image=src_image, strength=strength)
    image = result.images[0]
    url = _save_image_and_url(image, "img2img")
    return {"output_url": url}


def _run_inpainting(payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    prompt = payload.get("prompt", "")
    image_url = payload.get("image_url")
    mask_url = payload.get("mask_url")
    if not prompt or not image_url or not mask_url:
        return {"success": False, "error": "prompt, image_url, and mask_url are required for inpainting"}
    repo = model_cfg.get("repo")
    pipeline_class = model_cfg.get("pipeline_class", "StableDiffusionXLInpaintPipeline")
    pipe = get_or_load(
        cache_key=f"image_inpaint:{repo}",
        loader_fn=lambda: _load_txt2img_pipeline(repo, pipeline_class),
        required_vram_gb=model_cfg.get("min_vram_gb", 10.0),
    )
    from PIL import Image
    import httpx
    with httpx.Client(timeout=60) as client:
        img_resp = client.get(image_url); img_resp.raise_for_status()
        mask_resp = client.get(mask_url); mask_resp.raise_for_status()
    src_image = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
    mask_image = Image.open(io.BytesIO(mask_resp.content)).convert("RGB")
    result = pipe(prompt=prompt, image=src_image, mask_image=mask_image)
    image = result.images[0]
    url = _save_image_and_url(image, "inpaint")
    return {"output_url": url}


def _load_upscale_pipeline(repo: str):
    # Real-ESRGAN via the realesrgan pip package -- lazy import, only
    # paid when an image_upscale task actually runs on this machine.
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(scale=4, model_path=repo, model=model_arch, half=True)


def _run_image_upscale(payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    image_url = payload.get("image_url")
    if not image_url:
        return {"success": False, "error": "image_url is required for image_upscale"}
    repo = model_cfg.get("repo")
    upscaler = get_or_load(
        cache_key=f"image_upscale:{repo}",
        loader_fn=lambda: _load_upscale_pipeline(repo),
        required_vram_gb=model_cfg.get("min_vram_gb", 4.0),
    )
    import numpy as np
    from PIL import Image
    import httpx
    with httpx.Client(timeout=60) as client:
        resp = client.get(image_url); resp.raise_for_status()
    src_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    output_np, _ = upscaler.enhance(np.array(src_image), outscale=4)
    out_image = Image.fromarray(output_np)
    url = _save_image_and_url(out_image, "upscaled")
    return {"output_url": url}


def run(task_type: str, payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if task_type == "image_generation":
        return _run_image_generation(payload, model_cfg)
    if task_type == "image_to_image":
        return _run_image_to_image(payload, model_cfg)
    if task_type == "inpainting":
        return _run_inpainting(payload, model_cfg)
    if task_type == "image_upscale":
        return _run_image_upscale(payload, model_cfg)
    return {"success": False, "error": f"image.py loader does not handle task_type={task_type!r}"}
