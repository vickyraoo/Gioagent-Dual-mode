"""Hugging Face-backed smolagents planner with a fixed geospatial toolbelt."""
from __future__ import annotations
import os
from smolagents import CodeAgent, InferenceClientModel
from geo_tools import fetch_sentinel2_stac, compute_spectral_index, detect_change
from vision_tools import detect_and_count_objects, segment_and_measure, describe_scene_context

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

SYSTEM_INSTRUCTIONS = """You are GeoAgent, a careful remote-sensing tool planner.
Use tools for every numeric/spatial result. Do not invent scene IDs, asset URLs, counts, hectares,
coordinates, or measurements. Tool outputs are structured dictionaries; read their fields carefully.

Routing rules:
- For live Sentinel-2 searches, call fetch_sentinel2_stac first.
- For NDVI/NDWI/NBR, call compute_spectral_index using an exact scene_id and bbox_wsen from search output.
- For vegetation change across two dates, search for scenes and call detect_change on two scene IDs over the same bbox.
- For object counts in high-resolution uploaded images, call detect_and_count_objects.
- For semantic masks/coverage on high-resolution images, call segment_and_measure.
- Never estimate object counts, hectares, area, NDVI, or boundaries from visual descriptions.
- Do not claim Sentinel-2 can count small cars, aircraft, or ships: its multispectral bands are too coarse for these small objects.
- If required values are missing, ask one concise follow-up rather than inventing them.
- Explain cloud masking, thresholds, acquisition dates, spatial resolution, and uncertainty when relevant.
- Do not execute unrelated code, browse files outside user-provided image paths, or reveal secrets.
"""


def build_agent() -> CodeAgent:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Add your Hugging Face access token to .env.")
    model_id = os.getenv("HF_MODEL_ID", DEFAULT_MODEL).strip()
    provider = os.getenv("HF_PROVIDER", "").strip() or None
    model = InferenceClientModel(
        model_id=model_id,
        token=token,
        provider=provider,
        temperature=0.1,
        max_tokens=1800,
    )
    return CodeAgent(
        tools=[fetch_sentinel2_stac, compute_spectral_index, detect_change, detect_and_count_objects, segment_and_measure, describe_scene_context],
        model=model,
        max_steps=6,
        add_base_tools=False,
        instructions=SYSTEM_INSTRUCTIONS,
        additional_authorized_imports=[],
    )
