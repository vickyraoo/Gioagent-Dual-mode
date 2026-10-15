"""Deterministic remote-sensing tools for the Dual-Mode GeoAgent.

Agent-facing functions return JSON-serializable dictionaries. Raster arrays remain in Python;
only scalar summaries and saved artifact paths are sent back to the LLM.
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from geopy.geocoders import Nominatim
from pystac_client import Client
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from index_math import ndvi, ndwi_green_nir, nbr, area_ha

STAC_URL = os.getenv("STAC_URL", "https://earth-search.aws.element84.com/v1")
COLLECTION = "sentinel-2-l2a"
_OUTPUT_DIR_SETTING = Path(os.getenv("GEOAGENT_OUTPUT_DIR", "outputs"))
OUTPUT_DIR = _OUTPUT_DIR_SETTING if _OUTPUT_DIR_SETTING.is_absolute() else Path(__file__).resolve().parent / _OUTPUT_DIR_SETTING
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_WINDOW_SIDE = 4096
CLOUD_SCL_CLASSES = (0, 1, 3, 8, 9, 10, 11)  # no data, saturated/defective, shadow, clouds, cirrus, snow/ice


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _parse_date(date_text: str) -> datetime:
    return datetime.strptime(date_text, "%Y-%m-%d")


def _catalog() -> Client:
    return Client.open(STAC_URL)


@lru_cache(maxsize=64)
def resolve_place_bbox(place: str, max_span_deg: float = 0.20) -> tuple[tuple[float, float, float, float], str]:
    """Resolve a place to an approximate WGS84 bbox: west, south, east, north."""
    geocoder = Nominatim(user_agent="dual-mode-deterministic-geoagent-student-project", timeout=20)
    location = geocoder.geocode(place, exactly_one=True)
    if location is None:
        raise ValueError(f"Could not resolve place '{place}'. Try a more specific place name.")
    raw = location.raw.get("boundingbox")
    if not raw or len(raw) != 4:
        raise ValueError("The geocoder did not return a usable bounding box.")
    south, north, west, east = [float(v) for v in raw]
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("The resolved bounding box is invalid.")
    lat_c, lon_c = float(location.latitude), float(location.longitude)
    if (east - west) > max_span_deg:
        west, east = lon_c - max_span_deg / 2.0, lon_c + max_span_deg / 2.0
    if (north - south) > max_span_deg:
        south, north = lat_c - max_span_deg / 2.0, lat_c + max_span_deg / 2.0
    return (round(west, 6), round(south, 6), round(east, 6), round(north, 6)), location.address


def _scene_summary(item) -> dict[str, Any]:
    props = item.properties or {}
    cloud = props.get("eo:cloud_cover")
    assets = list(item.assets.keys())
    preview = None
    for name in ("thumbnail", "preview", "visual", "rendered_preview"):
        if name in item.assets:
            preview = item.assets[name].href
            break
    return {
        "scene_id": item.id,
        "datetime": item.datetime.isoformat() if item.datetime else props.get("datetime", "unknown"),
        "cloud_cover_percent": round(float(cloud), 2) if cloud is not None else None,
        "platform": props.get("platform", "unknown"),
        "collection": item.collection_id or COLLECTION,
        "preview_url": preview,
        "asset_keys": assets,
        "item_url": item.get_self_href(),
    }


def search_sentinel2(
    place: str,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 20.0,
    limit: int = 5,
) -> dict[str, Any]:
    """Search Earth Search for Sentinel-2 L2A scenes over a place and date range."""
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    cloud_max = float(np.clip(float(max_cloud_cover), 0.0, 100.0))
    limit = max(1, min(int(limit), 20))
    bbox, resolved_name = resolve_place_bbox(place)
    catalog = _catalog()
    search = catalog.search(
        collections=[COLLECTION],
        bbox=list(bbox),
        datetime=f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        max_items=200,
    )
    matches = []
    scanned = 0
    for item in search.items():
        scanned += 1
        cloud = (item.properties or {}).get("eo:cloud_cover")
        if cloud is not None and float(cloud) <= cloud_max:
            matches.append(_scene_summary(item))
    matches.sort(key=lambda row: (row["cloud_cover_percent"] is None, row["cloud_cover_percent"] or 100.0))
    return {
        "ok": True,
        "provider": "Element84 Earth Search v1",
        "collection": COLLECTION,
        "place_requested": place,
        "place_resolved_as": resolved_name,
        "bbox_wgs84_wsen": list(bbox),
        "bbox_wsen": list(bbox),
        "date_range": {"start": start_date, "end": end_date},
        "max_scene_cloud_cover_percent": cloud_max,
        "scanned_count": scanned,
        "match_count_returned": min(len(matches), limit),
        "scenes": matches[:limit],
        "caveat": "Cloud cover is a scene-level metadata estimate; it does not guarantee cloud-free pixels inside this AOI. Place-name bounds may be larger than the target field.",
    }


# Earth Search v1 uses readable aliases for its COG assets. The fallbacks permit other
# STAC providers with Sentinel-style band names to be substituted in future.
ASSET_ALIASES = {
    "blue": ("blue", "B02", "blue-jp2"),
    "green": ("green", "B03", "green-jp2"),
    "red": ("red", "B04", "red-jp2"),
    "nir": ("nir", "B08", "nir-jp2"),
    "swir16": ("swir16", "B11", "swir16-jp2"),
    "swir22": ("swir22", "B12", "swir22-jp2"),
    "scl": ("scl", "SCL", "scl-jp2"),
}


def _get_item(scene_id: str):
    results = list(_catalog().search(collections=[COLLECTION], ids=[scene_id], max_items=1).items())
    if not results:
        raise ValueError(f"Scene ID '{scene_id}' was not found in {COLLECTION}.")
    return results[0]


def _asset_key(item, logical_band: str) -> str:
    candidates = ASSET_ALIASES[logical_band]
    lower_to_key = {key.lower(): key for key in item.assets.keys()}
    for alias in candidates:
        key = lower_to_key.get(alias.lower())
        if key and item.assets[key].href:
            # Prefer COG aliases; avoid source JP2K when the COG exists.
            return key
    available = ", ".join(item.assets.keys())
    raise ValueError(f"Band '{logical_band}' was not found for scene {item.id}. Available asset keys: {available}")


def _asset_href(item, logical_band: str) -> str:
    return item.assets[_asset_key(item, logical_band)].href


def _scale_offset(item, logical_band: str) -> tuple[float, float, str]:
    """Return (scale, offset, source) that convert stored digital numbers to surface reflectance.

    Sentinel-2 L2A scenes processed with baseline >= 04.00 (from 2022-01-25) store
    DN = (reflectance + 0.1) * 10000, i.e. a BOA_ADD_OFFSET of -1000 DN. Index math on raw DN is biased
    for those scenes, so the STAC ``raster:bands`` scale/offset is applied first. Where the catalogue
    omits it, fall back to the processing baseline.
    """
    asset = item.assets[_asset_key(item, logical_band)]
    bands = asset.extra_fields.get("raster:bands") or []
    if bands and bands[0].get("scale") is not None:
        return float(bands[0]["scale"]), float(bands[0].get("offset") or 0.0), "stac raster:bands"
    try:
        baseline = float(str((item.properties or {}).get("s2:processing_baseline")))
    except (TypeError, ValueError):
        baseline = None
    if baseline is not None and baseline >= 4.0:
        return 1e-4, -0.1, "processing-baseline fallback (>= 04.00)"
    return 1e-4, 0.0, "default scale, no offset metadata found"


def _window_for_bbox(ds, bbox: list[float]) -> Window:
    """Compute a bounded raster window from a WGS84 bbox."""
    transformed = transform_bounds("EPSG:4326", ds.crs, *bbox, densify_pts=21)
    window = from_bounds(*transformed, transform=ds.transform).round_offsets().round_lengths()
    full = Window(0, 0, ds.width, ds.height)
    try:
        window = window.intersection(full)
    except Exception as exc:
        raise ValueError("The selected area does not overlap this scene.") from exc
    if window.width <= 0 or window.height <= 0:
        raise ValueError("The selected area does not overlap this scene.")
    if window.width > MAX_WINDOW_SIDE or window.height > MAX_WINDOW_SIDE:
        raise ValueError(
            f"AOI is too large for the starter (window {int(window.width)}×{int(window.height)} px). "
            "Use a smaller bounding box or upload a field-level boundary in a future extension."
        )
    return window


def _make_grid(item, bbox: list[float]) -> dict[str, Any]:
    with rasterio.open(_asset_href(item, "nir")) as ref:
        window = _window_for_bbox(ref, bbox)
        return {
            "crs": ref.crs,
            "transform": ref.window_transform(window),
            "width": int(window.width),
            "height": int(window.height),
            "pixel_area_m2": abs(
                ref.window_transform(window).a * ref.window_transform(window).e
                - ref.window_transform(window).b * ref.window_transform(window).d
            ),
        }


def _read_band_to_grid(item, logical_band: str, bbox: list[float], grid: dict[str, Any], is_categorical: bool = False) -> np.ndarray:
    href = _asset_href(item, logical_band)
    with rasterio.open(href) as src:
        src_window = _window_for_bbox(src, bbox)
        source_ma = src.read(1, window=src_window, masked=True)
        source = np.asarray(source_ma.astype("float32").filled(np.nan), dtype=np.float32)
        if not is_categorical:
            # DN 0 is no-data even if the COG lacks a nodata tag. Convert DN -> surface reflectance, and clip
            # small negative values (DN < offset) to 0 so normalized differences stay within [-1, 1].
            source[source == 0] = np.nan
            scale, offset, _ = _scale_offset(item, logical_band)
            source = np.maximum(source * np.float32(scale) + np.float32(offset), np.float32(0.0))
        destination = np.full((grid["height"], grid["width"]), np.nan, dtype=np.float32)
        reproject(
            source=source,
            destination=destination,
            src_transform=src.window_transform(src_window),
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.nearest if is_categorical else Resampling.bilinear,
        )
        return destination


def _calculate_index(item, bbox: list[float], index_name: str, target_grid: dict[str, Any] | None = None):
    index_name = index_name.upper().strip()
    if index_name not in {"NDVI", "NDWI", "NBR"}:
        raise ValueError("index_name must be one of NDVI, NDWI, or NBR")
    grid = target_grid or _make_grid(item, bbox)
    nir = _read_band_to_grid(item, "nir", bbox, grid)
    if index_name == "NDVI":
        red = _read_band_to_grid(item, "red", bbox, grid)
        index = ndvi(nir, red)
    elif index_name == "NDWI":
        green = _read_band_to_grid(item, "green", bbox, grid)
        index = ndwi_green_nir(green, nir)
    else:  # NBR uses NIR and SWIR2 (Sentinel-2 B12, 20 m native GSD).
        swir2 = _read_band_to_grid(item, "swir22", bbox, grid)
        index = nbr(nir, swir2)

    # SCL is categorical and nominally 20 m; resample nearest-neighbour onto the NIR grid.
    scl = _read_band_to_grid(item, "scl", bbox, grid, is_categorical=True)
    mask = np.isin(scl, CLOUD_SCL_CLASSES) | ~np.isfinite(scl)
    index[mask] = np.nan
    return index.astype(np.float32), grid, int(np.count_nonzero(~mask & np.isfinite(index)))


def _slug(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return safe[:80] or "scene"


def _save_index_artifacts(index: np.ndarray, grid: dict[str, Any], scene_id: str, index_name: str) -> tuple[str, str]:
    stem = f"{_slug(scene_id)}_{index_name.lower()}"
    png_path = OUTPUT_DIR / f"{stem}.png"
    tif_path = OUTPUT_DIR / f"{stem}.tif"
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(index, cmap="RdYlGn" if index_name != "NDWI" else "BrBG", vmin=-1, vmax=1)
    ax.set_title(f"{index_name} — {scene_id}")
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, shrink=0.78, label=index_name)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    profile = {
        "driver": "GTiff", "height": index.shape[0], "width": index.shape[1], "count": 1,
        "dtype": "float32", "crs": grid["crs"], "transform": grid["transform"], "nodata": np.nan,
        "compress": "deflate",
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(index.astype(np.float32), 1)
        dst.set_band_description(1, index_name)
    return str(png_path), str(tif_path)


def analyze_spectral_index(
    scene_id: str,
    bbox_wsen: list[float],
    index_name: str = "NDVI",
    threshold: float = 0.3,
) -> dict[str, Any]:
    """Compute a spectral index and resolution-based area summaries from an AOI window."""
    if len(bbox_wsen) != 4:
        raise ValueError("bbox_wsen must be [west, south, east, north] in EPSG:4326")
    bbox = [float(v) for v in bbox_wsen]
    if not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
        raise ValueError("Invalid WGS84 bounding box")
    item = _get_item(scene_id)
    index_name = index_name.upper()
    arr, grid, valid_pixels = _calculate_index(item, bbox, index_name)
    valid = np.isfinite(arr)
    if not np.any(valid):
        raise ValueError("No valid pixels remain after nodata/cloud/shadow masking for this AOI.")
    vals = arr[valid]
    pixel_ha = float(grid["pixel_area_m2"]) / 10000.0
    above = int(np.count_nonzero(vals >= float(threshold)))
    below = int(np.count_nonzero(vals < float(threshold)))
    png_path, tif_path = _save_index_artifacts(arr, grid, scene_id, index_name)
    r_scale, r_offset, r_source = _scale_offset(item, "nir")
    return {
        "ok": True,
        "scene_id": scene_id,
        "index": index_name,
        "reflectance_scale": r_scale,
        "reflectance_offset": r_offset,
        "reflectance_scaling_source": r_source,
        "threshold": float(threshold),
        "mean": round(float(np.mean(vals)), 5),
        "median": round(float(np.median(vals)), 5),
        "std": round(float(np.std(vals)), 5),
        "min": round(float(np.min(vals)), 5),
        "max": round(float(np.max(vals)), 5),
        "valid_pixel_count": int(vals.size),
        "valid_area_ha": round(area_ha(float(grid["pixel_area_m2"]), int(vals.size)), 4),
        "area_above_threshold_ha": round(area_ha(float(grid["pixel_area_m2"]), above), 4),
        "area_below_threshold_ha": round(area_ha(float(grid["pixel_area_m2"]), below), 4),
        "approx_pixel_area_m2": round(float(grid["pixel_area_m2"]), 4),
        "saved_map_path": png_path,
        "saved_geotiff_path": tif_path,
        "method_note": "Stored digital numbers are converted to surface reflectance (scale/offset above) before index math, and negative reflectance is clipped to 0. Areas are raster-resolution estimates after SCL masking; they are not exact ground truth. NBR uses Sentinel-2 SWIR2 (B12), which is natively 20 m and resampled onto the NIR grid for this starter.",
    }


def analyze_change(
    scene_id_t1: str,
    scene_id_t2: str,
    bbox_wsen: list[float],
    severe_loss_threshold: float = -0.2,
    moderate_loss_threshold: float = -0.1,
    gain_threshold: float = 0.1,
) -> dict[str, Any]:
    """Compute ΔNDVI = NDVI(t2)-NDVI(t1) and area bins on a shared pixel grid."""
    bbox = [float(v) for v in bbox_wsen]
    item1, item2 = _get_item(scene_id_t1), _get_item(scene_id_t2)
    ndvi1, grid1, _ = _calculate_index(item1, bbox, "NDVI")
    ndvi2, _, _ = _calculate_index(item2, bbox, "NDVI", target_grid=grid1)
    valid = np.isfinite(ndvi1) & np.isfinite(ndvi2)
    if not np.any(valid):
        raise ValueError("No common valid pixels between the two scenes after masking.")
    delta = np.full(ndvi1.shape, np.nan, dtype=np.float32)
    delta[valid] = ndvi2[valid] - ndvi1[valid]
    vals = delta[valid]
    px_ha = float(grid1["pixel_area_m2"]) / 10000.0
    severe = valid & (delta <= float(severe_loss_threshold))
    moderate = valid & (delta > float(severe_loss_threshold)) & (delta <= float(moderate_loss_threshold))
    stable = valid & (delta > float(moderate_loss_threshold)) & (delta < float(gain_threshold))
    gain = valid & (delta >= float(gain_threshold))
    out = OUTPUT_DIR / f"change_{_slug(scene_id_t1)}_{_slug(scene_id_t2)}.png"
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    img = ax.imshow(delta, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_title("ΔNDVI = later scene − earlier scene")
    ax.set_axis_off()
    fig.colorbar(img, ax=ax, shrink=0.78, label="ΔNDVI")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    tif = out.with_suffix(".tif")
    profile = {
        "driver": "GTiff", "height": delta.shape[0], "width": delta.shape[1], "count": 1,
        "dtype": "float32", "crs": grid1["crs"], "transform": grid1["transform"], "nodata": np.nan,
        "compress": "deflate",
    }
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(delta, 1)
        dst.set_band_description(1, "delta_NDVI")
    return {
        "ok": True,
        "scene_id_t1": scene_id_t1,
        "scene_id_t2": scene_id_t2,
        "mean_delta_ndvi": round(float(np.mean(vals)), 5),
        "median_delta_ndvi": round(float(np.median(vals)), 5),
        "reflectance_offset_t1": _scale_offset(item1, "nir")[1],
        "reflectance_offset_t2": _scale_offset(item2, "nir")[1],
        "common_valid_pixel_count": int(vals.size),
        "approx_common_valid_area_ha": round(area_ha(float(grid1["pixel_area_m2"]), int(vals.size)), 4),
        "severe_loss_area_ha": round(area_ha(float(grid1["pixel_area_m2"]), int(np.count_nonzero(severe))), 4),
        "moderate_loss_area_ha": round(area_ha(float(grid1["pixel_area_m2"]), int(np.count_nonzero(moderate))), 4),
        "stable_area_ha": round(area_ha(float(grid1["pixel_area_m2"]), int(np.count_nonzero(stable))), 4),
        "gain_area_ha": round(area_ha(float(grid1["pixel_area_m2"]), int(np.count_nonzero(gain))), 4),
        "thresholds": {"severe_loss_lte": severe_loss_threshold, "moderate_loss_lte": moderate_loss_threshold, "gain_gte": gain_threshold},
        "saved_map_path": str(out),
        "saved_geotiff_path": str(tif),
        "method_note": "Area estimates use the earlier scene's projected pixel area on a common grid. NDVI change can reflect phenology, atmosphere, soil moisture, and processing differences; interpret with acquisition dates and crop calendar.",
    }


# Agent-facing wrappers: returning compact JSON avoids handing arrays to the model.
from smolagents import tool


@tool
def fetch_sentinel2_stac(place: str, start_date: str, end_date: str, max_cloud_cover: float = 20.0, limit: int = 5) -> dict[str, Any]:
    """Search Sentinel-2 L2A COG scenes through the public Earth Search v1 STAC API.

    Args:
        place: Named city, region, or area (for example 'Karnal, Haryana, India').
        start_date: Inclusive start date in YYYY-MM-DD format.
        end_date: Inclusive end date in YYYY-MM-DD format.
        max_cloud_cover: Maximum scene-level cloud-cover percentage from 0 to 100.
        limit: Maximum number of scenes to return, between 1 and 20.

    Returns:
        A compact dictionary with scene IDs, dates, cloud cover, AOI bbox, preview URLs, and band asset keys.
    """
    try:
        return search_sentinel2(place, start_date, end_date, max_cloud_cover, limit)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@tool
def compute_spectral_index(scene_id: str, bbox_wsen: list[float], index_name: str = "NDVI", threshold: float = 0.3) -> dict[str, Any]:
    """Compute NDVI, NDWI, or NBR from COG windows, cloud-mask with SCL, and estimate area.

    Args:
        scene_id: Exact scene_id returned by fetch_sentinel2_stac.
        bbox_wsen: Four WGS84 coordinates [west, south, east, north] returned by scene search.
        index_name: One of NDVI, NDWI, or NBR. NBR uses NIR and SWIR2/B12.
        threshold: Index threshold used to partition estimated area into above/below categories.

    Returns:
        A compact dictionary containing scalar index statistics, area estimates, and saved PNG/GeoTIFF paths.
    """
    try:
        return analyze_spectral_index(scene_id, bbox_wsen, index_name, threshold)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@tool
def detect_change(scene_id_t1: str, scene_id_t2: str, bbox_wsen: list[float], severe_loss_threshold: float = -0.2, moderate_loss_threshold: float = -0.1, gain_threshold: float = 0.1) -> dict[str, Any]:
    """Compare NDVI between two scene IDs and report threshold-based change area estimates.

    Args:
        scene_id_t1: Earlier Sentinel-2 scene ID returned by fetch_sentinel2_stac.
        scene_id_t2: Later Sentinel-2 scene ID returned by fetch_sentinel2_stac.
        bbox_wsen: AOI in WGS84 [west, south, east, north].
        severe_loss_threshold: ΔNDVI at or below which pixels are severe loss; default -0.2.
        moderate_loss_threshold: Upper ΔNDVI bound for moderate loss; default -0.1.
        gain_threshold: ΔNDVI at or above which pixels are categorized as gain; default 0.1.

    Returns:
        A compact dictionary with common valid pixels, category areas in hectares, and output paths.
    """
    try:
        return analyze_change(scene_id_t1, scene_id_t2, bbox_wsen, severe_loss_threshold, moderate_loss_threshold, gain_threshold)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
