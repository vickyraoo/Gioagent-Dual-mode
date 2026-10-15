"""Small, testable raster math helpers used by the deterministic pipeline."""
from __future__ import annotations
import numpy as np


def normalized_difference(a, b, epsilon: float = 1e-6) -> np.ndarray:
    """Compute (a-b)/(a+b), preserving invalid/near-zero denominators as NaN."""
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    if a_arr.shape != b_arr.shape:
        raise ValueError("Inputs must be aligned to the same pixel grid before index math.")
    denominator = a_arr + b_arr
    out = np.full(a_arr.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(a_arr) & np.isfinite(b_arr) & np.isfinite(denominator) & (np.abs(denominator) > epsilon)
    out[valid] = (a_arr[valid] - b_arr[valid]) / denominator[valid]
    out[~np.isfinite(out)] = np.nan
    return out


def ndvi(nir, red) -> np.ndarray:
    """Normalized Difference Vegetation Index: (NIR - Red)/(NIR + Red)."""
    return normalized_difference(nir, red)


def ndwi_green_nir(green, nir) -> np.ndarray:
    """McFeeters-style green/NIR water index: (Green - NIR)/(Green + NIR)."""
    return normalized_difference(green, nir)


def nbr(nir, swir2) -> np.ndarray:
    """Normalized Burn Ratio: (NIR - SWIR2)/(NIR + SWIR2)."""
    return normalized_difference(nir, swir2)


def area_ha(pixel_area_m2: float, pixel_count: int) -> float:
    """Estimate area in hectares from projected pixel area and a pixel count."""
    if pixel_area_m2 <= 0 or pixel_count < 0:
        raise ValueError("pixel_area_m2 must be positive and pixel_count must be non-negative")
    return float(pixel_area_m2) * int(pixel_count) / 10000.0
