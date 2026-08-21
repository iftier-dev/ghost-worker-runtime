"""
vision.py — vision_analyze task type: image + text question in, text
answer out. Used by the agent-internal gpu_first_llm.py helper for
screenshot/image reasoning (NOT user-facing vision content generation --
there is no such product surface; this is purely an internal-reasoning
task type, same category as llm_inference).
"""
from __future__ import annotations
import base64
import io
import re
from typing import Any, Dict

from ..pipeline_cache import get_or_load
from ..model_cache import model_cache_path


def _decode_data_url(data_url: str):
    from PIL import Image
    match = re.match(r"^data:image/\w+;base64,(.+)$", data_url)
    if not match:
        raise ValueError("image_data_url must be a data: URL (data:image/...;base64,...)")
    raw = base64.b64decode(match.group(1))
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _load_qwen_vl(repo: str):
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        repo, torch_dtype=torch.float16, cache_dir=model_cache_path(repo),
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    processor = AutoProcessor.from_pretrained(repo, cache_dir=model_cache_path(repo))
    return {"model": model, "processor": processor}


def _run_qwen_vl(payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    image_data_url = payload.get("image_data_url")
    question = payload.get("question", "Describe this image in detail.")
    if not image_data_url:
        return {"success": False, "error": "image_data_url is required for vision_analyze"}

    repo = model_cfg.get("repo")
    bundle = get_or_load(
        cache_key=f"vision_qwen:{repo}",
        loader_fn=lambda: _load_qwen_vl(repo),
        required_vram_gb=model_cfg.get("min_vram_gb", 10.0),
    )
    model, processor = bundle["model"], bundle["processor"]

    image = _decode_data_url(image_data_url)
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}],
    }]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=payload.get("max_tokens", 800))
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    tokens_used = int(generated.shape[1])
    return {"output_text": answer, "tokens_used": tokens_used}


def _run_visual_click_grounding(payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Last-resort click-target locator: given a screenshot and a plain-
    English description of what to click ("the blue Download button",
    "the Sign In link in the top right"), returns pixel coordinates
    {x, y} for the browser tool to click directly -- used ONLY when
    DOM-selector detection (_llm_detect_selector, CSS-based, faster and
    far more reliable) has already failed, e.g. canvas-rendered UI, an
    image-only button with no accessible DOM node, or a page that
    deliberately obfuscates its markup. This is NOT the primary click
    mechanism anywhere in this codebase -- DOM selectors remain first
    choice everywhere; this exists so a page that defeats DOM detection
    doesn't leave the agent with zero options."""
    import torch
    image_data_url = payload.get("image_data_url")
    target_description = payload.get("target_description", "")
    image_width = payload.get("image_width")
    image_height = payload.get("image_height")
    if not image_data_url or not target_description:
        return {"success": False, "error": "image_data_url and target_description are required for visual_click_grounding"}

    repo = model_cfg.get("repo")
    bundle = get_or_load(
        cache_key=f"vision_grounding:{repo}",
        loader_fn=lambda: _load_qwen_vl(repo),
        required_vram_gb=model_cfg.get("min_vram_gb", 10.0),
    )
    model, processor = bundle["model"], bundle["processor"]

    image = _decode_data_url(image_data_url)
    actual_width, actual_height = image.size

    grounding_prompt = (
        f"Locate the UI element described as: \"{target_description}\". "
        f"Respond with ONLY a JSON object in the exact format "
        f'{{"x": <pixel_x>, "y": <pixel_y>, "confidence": <0.0-1.0>}}, '
        f"where x and y are the CENTER pixel coordinates of that element "
        f"in this {actual_width}x{actual_height} image. If the element is "
        f'not visible, respond with {{"x": null, "y": null, "confidence": 0.0}}.'
    )
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": grounding_prompt}],
    }]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=100)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    import json as _json, re as _re
    match = _re.search(r"\{[^{}]*\}", raw_answer)
    if not match:
        return {"success": False, "error": f"model did not return parseable coordinates: {raw_answer[:200]}"}
    try:
        coords = _json.loads(match.group(0))
    except Exception:
        return {"success": False, "error": f"could not parse coordinate JSON: {raw_answer[:200]}"}

    x, y = coords.get("x"), coords.get("y")
    if x is None or y is None:
        return {"success": True, "found": False, "confidence": coords.get("confidence", 0.0)}

    # Scale coordinates if the caller's actual browser viewport differs
    # from the image we analyzed (e.g. image was downscaled before
    # sending) -- image_width/image_height in payload are the caller's
    # REAL viewport dimensions; scale our model's pixel-space answer
    # into that space so the click lands in the right place.
    if image_width and image_height and (image_width != actual_width or image_height != actual_height):
        x = round(x * (image_width / actual_width))
        y = round(y * (image_height / actual_height))

    return {
        "success": True, "found": True, "x": x, "y": y,
        "confidence": coords.get("confidence", 0.5),
        "tokens_used": int(generated.shape[1]),
    }


def run(task_type: str, payload: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if task_type == "vision_analyze":
        loader = model_cfg.get("loader", "vision_qwen")
        if loader == "vision_qwen":
            return _run_qwen_vl(payload, model_cfg)
        return {"success": False, "error": f"vision.py loader does not support loader={loader!r} yet"}
    if task_type == "visual_click_grounding":
        return _run_visual_click_grounding(payload, model_cfg)
    return {"success": False, "error": f"vision.py loader does not handle task_type={task_type!r}"}
