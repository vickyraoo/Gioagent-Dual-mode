"""Optional high-resolution imagery tools. Install requirements-vision.txt first."""
from __future__ import annotations
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import gc

_OUTPUT_DIR_SETTING = Path(os.getenv("GEOAGENT_OUTPUT_DIR", "outputs"))
OUTPUT_DIR = _OUTPUT_DIR_SETTING if _OUTPUT_DIR_SETTING.is_absolute() else Path(__file__).resolve().parent / _OUTPUT_DIR_SETTING
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DETECTOR = os.getenv("HF_DETECTOR_MODEL_ID", "google/owlv2-base-patch16-ensemble")
DEFAULT_SEGMENTER = os.getenv("HF_SEGMENTER_MODEL_ID", "CIDAS/clipseg-rd64-refined")


def _get_detector():
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("High-res detection requires the optional dependencies in requirements-vision.txt") from exc
    return pipeline("zero-shot-object-detection", model=DEFAULT_DETECTOR)


@lru_cache(maxsize=1)
def _cached_detector():
    return _get_detector()


@lru_cache(maxsize=1)
def _cached_segmenter():
    """Load CLIPSeg once per process instead of on every click/agent call."""
    try:
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    except ImportError as exc:
        raise RuntimeError("Text segmentation requires the optional dependencies in requirements-vision.txt") from exc
    processor = CLIPSegProcessor.from_pretrained(DEFAULT_SEGMENTER)
    model = CLIPSegForImageSegmentation.from_pretrained(DEFAULT_SEGMENTER)
    model.eval()
    return processor, model


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:40] or "prompt"


def _tiles(width: int, height: int, tile_size: int, overlap: float):
    stride = max(1, int(tile_size * (1.0 - overlap)))
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    if not xs or xs[-1] != max(0, width - tile_size):
        xs.append(max(0, width - tile_size))
    if not ys or ys[-1] != max(0, height - tile_size):
        ys.append(max(0, height - tile_size))
    return [(x, y, min(x + tile_size, width), min(y + tile_size, height)) for y in sorted(set(ys)) for x in sorted(set(xs))]


def _iou_and_ios(a, b) -> tuple[float, float]:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    iou = inter / max(area_a + area_b - inter, 1e-9)
    ios = inter / max(min(area_a, area_b), 1e-9)
    return iou, ios

CROSS_LABEL_IOU = 0.7
def _nms(
    detections: list[dict[str, Any]],
    iou_threshold: float = 0.5,
    cross_label_iou: float = CROSS_LABEL_IOU,
    ios_threshold: float = 0.75,
):
    """Per-label NMS (with IoS for tile-edge fragments) followed by cross-label deduplication."""
    kept = []
    for label in sorted({d["label"] for d in detections}):
        group = sorted((d for d in detections if d["label"] == label), key=lambda d: d["score"], reverse=True)
        selected = []
        while group:
            best = group.pop(0)
            selected.append(best)
            group = [
                d for d in group
                if _iou_and_ios(best["box"], d["box"])[0] < iou_threshold
                and _iou_and_ios(best["box"], d["box"])[1] < ios_threshold
            ]
        kept.extend(selected)
    unique: list[dict[str, Any]] = []
    for det in sorted(kept, key=lambda d: d["score"], reverse=True):
        if all(
            _iou_and_ios(det["box"], other["box"])[0] < cross_label_iou
            and _iou_and_ios(det["box"], other["box"])[1] < ios_threshold
            for other in unique
        ):
            unique.append(det)
    return sorted(unique, key=lambda d: (d["label"], -d["score"]))


def run_object_detection(image_path: str, object_queries: list[str], score_threshold: float = 0.15, tile_size: int = 768, overlap: float = 0.2) -> dict[str, Any]:
    """Run an open-vocabulary detector on tiled high-resolution imagery and merge boxes."""
    image = Image.open(image_path).convert("RGB")
    if not object_queries or not any(q.strip() for q in object_queries):
        raise ValueError("Enter one or more object prompts such as 'ship', 'airplane', or 'storage tank'.")
    if tile_size < 256 or tile_size > 1280:
        raise ValueError("tile_size must be between 256 and 1280 pixels")
    overlap = float(np.clip(overlap, 0.0, 0.6))
    _cached_segmenter.cache_clear()
    gc.collect()
    detector = _cached_detector()
    detections: list[dict[str, Any]] = []
    for x1, y1, x2, y2 in _tiles(image.width, image.height, int(tile_size), overlap):
        tile = image.crop((x1, y1, x2, y2))
        tile_w, tile_h = tile.size
        side = max(tile_w, tile_h)
        if tile_w != tile_h:
            # OWLv2 pads inputs to a square, but the pipeline rescales boxes by the unpadded height/width,
            # which shifts boxes on the short axis. Pad to a square ourselves (neutral grey) and clip afterwards.
            canvas = Image.new("RGB", (side, side), (128, 128, 128))
            canvas.paste(tile, (0, 0))
            tile = canvas
        outputs = detector(tile, candidate_labels=[q.strip() for q in object_queries if q.strip()], threshold=float(score_threshold))
        for row in outputs:
            box = row.get("box", {})
            bx1 = min(max(float(box.get("xmin", 0)), 0.0), tile_w)
            by1 = min(max(float(box.get("ymin", 0)), 0.0), tile_h)
            bx2 = min(max(float(box.get("xmax", 0)), 0.0), tile_w)
            by2 = min(max(float(box.get("ymax", 0)), 0.0), tile_h)
            if bx2 - bx1 < 2 or by2 - by1 < 2:
                continue  # lies in the padding, or clipped to nothing
            detections.append({
                "label": str(row.get("label", "object")),
                "score": float(row.get("score", 0.0)),
                "box": [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
            })
    detections = _nms(detections)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1 + 2, max(0, y1 - 15)), f"{det['label']} {det['score']:.2f}", fill="red", stroke_width=1, stroke_fill="white")
    query_slug = _slug("_".join(q.strip() for q in object_queries if q.strip()))
    output_path = OUTPUT_DIR / f"detections_{Path(image_path).stem}_{query_slug}.png"
    overlay.save(output_path)
    return {
        "ok": True,
        "model_id": DEFAULT_DETECTOR,
        "image_width_px": image.width,
        "image_height_px": image.height,
        "queries": object_queries,
        "object_count": len(detections),
        "counts_by_label": {label: sum(d["label"] == label for d in detections) for label in sorted({d["label"] for d in detections})},
        "detections": [{"label": d["label"], "score": round(d["score"], 4), "bbox_xyxy_px": [round(v, 1) for v in d["box"]]} for d in detections],
        "saved_overlay_path": str(output_path),
        "warning": "Open-vocabulary detector predictions are not ground truth. Small, crowded, rotated, or low-contrast objects can be missed or double-counted; validate against labeled tiles before reporting accuracy.",
    }


def run_text_segmentation(image_path: str, text_prompt: str, threshold: float = 0.5, meters_per_pixel: float | None = None) -> dict[str, Any]:
    """Use CLIPSeg text-conditioned segmentation; report coverage, not an uncalibrated object area."""
    try:
        import torch
        import torch.nn.functional as F
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    except ImportError as exc:
        raise RuntimeError("Text segmentation requires the optional dependencies in requirements-vision.txt") from exc
    if not text_prompt.strip():
        raise ValueError("Enter a segmentation prompt, e.g. 'water' or 'vegetation'.")
    _cached_detector.cache_clear()
    gc.collect()
    processor, model = _cached_segmenter()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        logits = F.interpolate(logits.unsqueeze(1), size=(image.height, image.width), mode="bilinear", align_corners=False).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
    mask = probs >= float(np.clip(threshold, 0.0, 1.0))
    coverage = float(mask.mean() * 100.0)
    tag = f"{Path(image_path).stem}_{_slug(text_prompt)}_t{int(round(float(threshold) * 100))}"
    mask_path = OUTPUT_DIR / f"mask_{tag}.png"
    overlay_path = OUTPUT_DIR / f"mask_overlay_{tag}.png"
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
    overlay = image.copy().convert("RGBA")
    colored = Image.new("RGBA", image.size, (0, 0, 0, 0))
    alpha = Image.fromarray((mask * 100).astype(np.uint8))
    color_layer = Image.new("RGBA", image.size, (0, 180, 255, 0))
    color_layer.putalpha(alpha)
    overlay = Image.alpha_composite(overlay, color_layer)
    overlay.convert("RGB").save(overlay_path)
    area = None
    if meters_per_pixel is not None and float(meters_per_pixel) > 0:
        area = float(mask.sum() * float(meters_per_pixel) ** 2)
    return {
        "ok": True,
        "model_id": DEFAULT_SEGMENTER,
        "text_prompt": text_prompt,
        "threshold": float(threshold),
        "mask_pixels": int(mask.sum()),
        "total_pixels": int(mask.size),
        "pixel_coverage_percent": round(coverage, 4),
        "estimated_area_m2": round(area, 3) if area is not None else None,
        "meters_per_pixel_assumption": meters_per_pixel if area is not None else None,
        "saved_mask_path": str(mask_path),
        "saved_overlay_path": str(overlay_path),
        "warning": "CLIPSeg output is a model-generated semantic mask. Pixel coverage is measured from the thresholded mask; ground area is only estimated if a trustworthy meters-per-pixel value is supplied.",
    }


# Lazy wrappers registered with smolagents. Optional model packages load only on invocation.
from smolagents import tool


@tool
def detect_and_count_objects(image_path: str, object_queries: list[str], score_threshold: float = 0.15, tile_size: int = 768, overlap: float = 0.2) -> dict[str, Any]:
    """Detect and count prompted object classes in a high-resolution uploaded image.

    Args:
        image_path: Local path to a user-uploaded aerial image (PNG/JPG/WEBP).
        object_queries: Labels such as ['ship', 'airplane', 'storage tank'].
        score_threshold: Detection score threshold between 0 and 1.
        tile_size: Square tile size in pixels, between 256 and 1280.
        overlap: Tile overlap fraction between 0 and 0.6 to reduce edge misses.

    Returns:
        A compact dictionary with approximate counts, bounding boxes, and a saved overlay path.
    """
    try:
        return run_object_detection(image_path, object_queries, score_threshold, tile_size, overlap)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@tool
def segment_and_measure(image_path: str, text_prompt: str, threshold: float = 0.5, meters_per_pixel: float = 0.0) -> dict[str, Any]:
    """Create a text-conditioned CLIPSeg mask for an uploaded high-resolution image.

    Args:
        image_path: Local path to a user-uploaded aerial image (PNG/JPG/WEBP).
        text_prompt: Concept to segment, for example 'water', 'vegetation', or 'runway'.
        threshold: Probability threshold between 0 and 1 for mask pixels.
        meters_per_pixel: Optional known ground sample distance; use 0 when unknown.

    Returns:
        A compact dictionary with mask coverage and output paths. Ground area is omitted if GSD is unknown.
    """
    try:
        mpp = None if float(meters_per_pixel) <= 0 else float(meters_per_pixel)
        return run_text_segmentation(image_path, text_prompt, threshold, mpp)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def vqa_describe_scene(image: Image.Image, question: str, hf_token: str, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> str:
    """Qualitative-only image Q&A. Must never be used for counts, areas, or spectral indices."""
    from huggingface_hub import InferenceClient
    import base64
    import io
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    client = InferenceClient(token=hf_token, timeout=120)
    messages = [
        {"role": "system", "content": (
            "You are a qualitative remote-sensing image assistant. Describe visible context only. "
            "Do NOT count objects, estimate areas, draw exact boundaries, calculate spectral indices, "
            "or invent location/date/sensor metadata. If asked for measurements, state that deterministic "
            "raster or detection/segmentation tools are required. Mention ambiguity from resolution and color composites."
        )},
        {"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
        ]},
    ]
    result = client.chat_completion(model=model_id, messages=messages, max_tokens=500, temperature=0.2)
    content = result.choices[0].message.content
    return content if isinstance(content, str) else str(content)

@tool
def describe_scene_context(image_path: str, question: str) -> dict[str, Any]:
    """Answer qualitative visual questions about an uploaded image (e.g., land-use type, layout, visible features).
    Never use this tool to count objects, measure areas, or compute spectral indices.

    Args:
        image_path: Local path to a user-uploaded aerial/satellite image.
        question: Natural-language question about the visible scene context.

    Returns:
        A dictionary containing the qualitative description from the vision-language model.
    """
    try:
        token = os.getenv("HF_TOKEN", "").strip()
        model_id = os.getenv("HF_VISION_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
        answer = vqa_describe_scene(Image.open(image_path), question, token, model_id)
        return {"ok": True, "model_id": model_id, "question": question, "description": answer}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}