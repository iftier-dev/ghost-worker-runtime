"""
video.py — video_generation (text-to-video), image_to_video, video_upscale.
Same pipeline_cache pattern as image.py: model stays VRAM-resident across
calls, single exclusive lock per loader family (enforced by executor.py,
not here) so two video jobs never race onto the same GPU's VRAM.
"""
from __future__ import annotations
import io
import os
import uuid
from typing import Any, Dict

from ..pipeline_cache import get_or_load
from ..model_cache import model_cache_path

OUTPUT_DIR = os.environ.get("WORKER_OUTPUT_DIR", "/tmp/ghost-worker-output")


def _ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def _save_video_and_url(frames, fps: int, prefix: str) -> str:
    """Encodes a list of PIL frames to mp4 via diffusers' export_to_video
    helper, saves to the shared output dir, returns a URL/path the same
    way image.py's _save_image_and_url does (VIDEO_UPLOAD_URL env, else
    local path)."""
    from diffusers.utils import export_to_video
    out_dir = _ensure_output_dir()
    fname = f"{prefix}_{uuid.uuid4().hex}.mp4"
    fpath = os.path.join(out_dir, fname)
    export_to_video(frames, fpath, fps=fps)

    upload_url = os.environ.get("VIDEO_UPLOAD_URL", "")
    if upload_url:
        try:
            import httpx
            with open(fpath, "rb") as f, httpx.Client(timeout=120) as client:
                resp = client.post(upload_url, files={"file": (fname, f, "video/mp4")})
                resp.raise_for_status()
                remote_url = resp.json().get("url")
                if remote_url:
                    return remote_url
        except Exception:
            pass  # fall through to local path
    return fpath


def _load_video_pipeline(repo: str, pipeline_class: str):
    import torch
    import diffusers
    cls = getattr(diffusers, pipeline_class)
    pipe = cls.from_pretrained(
        repo, torch_dtype=torch.float16, cache_dir=model_cache_path(repo),
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe


def _run_video_generation(payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    prompt = payload.get("prompt", "")
    if not prompt:
        return {"success": False, "error": "prompt is required for video_generation"}
    repo = model_cfg.get("repo")
    pipeline_class = model_cfg.get("pipeline_class", "AnimateDiffPipeline")
    pipe = get_or_load(
        cache_key=f"video_gen:{repo}",
        loader_fn=lambda: _load_video_pipeline(repo, pipeline_class),
        required_vram_gb=model_cfg.get("min_vram_gb", 14.0),
    )
    num_frames = payload.get("duration", 5) * payload.get("fps", 8)
    result = pipe(prompt=prompt, num_frames=num_frames, num_inference_steps=payload.get("steps", 25))
    frames = result.frames[0] if hasattr(result, "frames") else result.images
    url = _save_video_and_url(frames, fps=payload.get("fps", 8), prefix="vid")
    return {"output_url": url}


def _run_image_to_video(payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    image_url = payload.get("image_url")
    if not image_url:
        return {"success": False, "error": "image_url is required for image_to_video"}
    repo = model_cfg.get("repo")
    pipeline_class = model_cfg.get("pipeline_class", "StableVideoDiffusionPipeline")
    pipe = get_or_load(
        cache_key=f"video_img2vid:{repo}",
        loader_fn=lambda: _load_video_pipeline(repo, pipeline_class),
        required_vram_gb=model_cfg.get("min_vram_gb", 18.0),
    )
    from PIL import Image
    import httpx
    with httpx.Client(timeout=60) as client:
        resp = client.get(image_url)
        resp.raise_for_status()
        src_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
    result = pipe(image=src_image, num_frames=payload.get("num_frames", 25))
    frames = result.frames[0]
    url = _save_video_and_url(frames, fps=payload.get("fps", 7), prefix="img2vid")
    return {"output_url": url}


def _load_video_upscaler(repo: str):
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(scale=4, model_path=repo, model=model_arch, half=True)


def _run_video_upscale(payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Frame-by-frame upscale via the same Real-ESRGAN backend as
    image_upscale, then re-encoded to mp4 -- avoids a second, separate
    video-specific upscale model just for resolution bump."""
    video_url = payload.get("video_url")
    if not video_url:
        return {"success": False, "error": "video_url is required for video_upscale"}
    repo = model_cfg.get("repo")
    upscaler = get_or_load(
        cache_key=f"video_upscale:{repo}",
        loader_fn=lambda: _load_video_upscaler(repo),
        required_vram_gb=model_cfg.get("min_vram_gb", 8.0),
    )
    import numpy as np
    import httpx
    import imageio
    out_dir = _ensure_output_dir()
    src_path = os.path.join(out_dir, f"src_{uuid.uuid4().hex}.mp4")
    with httpx.Client(timeout=120) as client:
        resp = client.get(video_url)
        resp.raise_for_status()
    with open(src_path, "wb") as f:
        f.write(resp.content)

    reader = imageio.get_reader(src_path)
    fps = reader.get_meta_data().get("fps", 24)
    upscaled_frames = []
    for frame in reader:
        out_np, _ = upscaler.enhance(np.array(frame), outscale=4)
        upscaled_frames.append(out_np)
    reader.close()
    os.remove(src_path)

    fname = f"video_upscaled_{uuid.uuid4().hex}.mp4"
    fpath = os.path.join(out_dir, fname)
    imageio.mimsave(fpath, upscaled_frames, fps=fps)

    upload_url = os.environ.get("VIDEO_UPLOAD_URL", "")
    if upload_url:
        try:
            with open(fpath, "rb") as f, httpx.Client(timeout=120) as client:
                resp = client.post(upload_url, files={"file": (fname, f, "video/mp4")})
                resp.raise_for_status()
                remote_url = resp.json().get("url")
                if remote_url:
                    return {"output_url": remote_url}
        except Exception:
            pass
    return {"output_url": fpath}


def run(task_type: str, payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if task_type == "video_generation":
        return _run_video_generation(payload, model_cfg)
    if task_type == "image_to_video":
        return _run_image_to_video(payload, model_cfg)
    if task_type == "video_upscale":
        return _run_video_upscale(payload, model_cfg)
    return {"success": False, "error": f"video.py loader does not handle task_type={task_type!r}"}
