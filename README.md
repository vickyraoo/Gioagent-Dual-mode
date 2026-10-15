# Dual-Mode Deterministic GeoAgent

A student project scaffold for retrieving satellite data and processing natural-language requests with an agentic Hugging Face LLM. It deliberately separates **macro-scale multispectral analysis** from **micro-scale high-resolution vision**.

## Why two modes?

- **Live Multispectral Analyst:** Sentinel-2 L2A data from Element84 Earth Search v1. The app searches public STAC metadata and streams cropped Cloud-Optimized GeoTIFF (COG) windows for NDVI, green-NIR NDWI, NBR, and two-date ΔNDVI.
- **High-Resolution Scene Inspector:** uploaded aerial/benchmark tiles for OWLv2 open-vocabulary object detection and CLIPSeg text-conditioned segmentation. This mode is optional and model-heavy.
- **Agentic planner:** Qwen2.5-Coder-32B-Instruct through Hugging Face Inference Providers + `smolagents.CodeAgent`. Tools return compact JSON, scalar metrics, and saved artifact paths—not NumPy arrays.
- **Qualitative VQA:** optional vision-language model for scene context only. The prompt explicitly forbids using VQA for object counts, area, exact boundaries, or spectral indices.

## Current capabilities

1. Search Sentinel-2 scenes by named location, date range, and maximum scene-level cloud percentage.
2. Display scene IDs, dates, cloud metadata, asset keys, and links.
3. Compute NDVI, green-NIR NDWI, or NBR from raster windows aligned to one projected grid.
4. Apply a basic Sentinel-2 Scene Classification Layer (SCL) mask for cloud, shadow, cirrus, snow/ice and invalid pixels.
5. Export the calculated index and ΔNDVI as PNG and GeoTIFF.
6. Estimate areas above/below an index threshold and bins of NDVI decline/stability/gain.
7. Detect and count objects in high-resolution uploaded images using OWLv2 with overlapping tiles and non-maximum suppression.
8. Create a text-conditioned semantic mask with CLIPSeg and report pixel coverage; estimate square metres only if a known metres-per-pixel value is supplied.
9. Use an agent planner to decide which of the fixed tools to call.

## Setup (Windows PowerShell)

From the extracted project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and replace the placeholder with a Hugging Face token. **Never commit `.env` to GitHub.** Choose `HF_PROVIDER` only if it is available to your account; inference availability, model routing, free credits and rate limits can change. The LLM model itself does not need to run locally.

Launch:

```powershell
streamlit run app.py
```

### Optional high-resolution models

The macro pipeline can be used without installing the heavy vision-model stack. To enable OWLv2 and CLIPSeg locally:

```powershell
pip install -r requirements-vision.txt
```

The first inference downloads model weights. CPU inference may be slow; a CUDA-enabled Google Colab T4 or other compatible GPU is useful for large imagery. GPU availability does not guarantee every model fits or runs at a desired speed.

## Use the app

### Live multispectral analyst

1. Enter a place and dates, then search.
2. Choose a returned scene and an index.
3. Compute the index to display summary metrics and an output map/GeoTIFF.
4. For comparisons, choose two scene IDs and calculate ΔNDVI.
5. Use the natural-language box to have the LLM plan a tool sequence. For reproducible numbers, inspect the underlying tool outputs and saved artifacts.

### High-resolution scene inspector

1. Upload an aerial image/benchmark patch such as an appropriately licensed DOTA/DIOR image.
2. Enter object labels, e.g. `ship, boat`, and run OWLv2 detection.
3. Enter a concept such as `water` and run CLIPSeg.
4. The app reports pixel coverage. It only translates pixels to square metres when a defensible ground sample distance is entered.

## Fixed agent toolbelt

| Tool | Function |
|---|---|
| `fetch_sentinel2_stac` | Search Sentinel-2 L2A scenes and return scene IDs, metadata, AOI bounds, and band asset keys. |
| `compute_spectral_index` | Compute NDVI, NDWI, or NBR with SCL masking and threshold-based area summaries. |
| `detect_change` | Compute ΔNDVI = later − earlier scene and summarize area by configured change bins. |
| `detect_and_count_objects` | Run tiled OWLv2 detection on a user-uploaded high-resolution image; merge overlapping boxes with NMS. |
| `segment_and_measure` | Use CLIPSeg to generate a text-conditioned mask; report pixel coverage and optional GSD-based area estimate. |

The separate `vqa_describe_scene` function is intended only for qualitative visual context and is not registered as a quantitative tool.

## Scientific choices and limitations

- Earth Search v1 exposes Sentinel-2 L2A COG assets, so a small AOI can be window-read instead of downloading the whole product. Actual runtime depends on scene availability, network, catalogue response, AOI size, COG layout, and data provider service limits.
- The Earth Search scene cloud field is an **approximate scene-wide percentage**, not an AOI-specific clear-pixel guarantee.
- Stored Sentinel-2 values are digital numbers, not reflectance. Scenes processed with baseline ≥ 04.00 (from 2022-01-25) carry a −1000 DN offset, so `geo_tools.py` converts every optical band with the scale/offset in the STAC `raster:bands` metadata (falling back to the processing baseline) before computing indices, and reports the values used (`reflectance_scale`, `reflectance_offset`, `reflectance_scaling_source`). Ignoring the offset roughly halves hectare counts near a 0.3 NDVI threshold in the synthetic tests. This was verified on synthetic rasters only, so check `reflectance_scaling_source` on your first real scene.
- Bands B02/B03/B04/B08 are nominally 10 m; B11/B12 and SCL are nominally 20 m. NBR uses SWIR2/B12 and is resampled onto the chosen NIR grid for this scaffold. This alignment does not create extra spatial detail.
- SCL masking is basic. It excludes classes 0, 1, 3, 8, 9, 10, and 11; additional masking or quality checks may be needed for research.
- A projected pixel-area calculation is a **resolution-based estimate**, not exact ground truth. Projection distortion, mixed pixels, resampling, cloud masks, AOI geometry, and sensor/processing differences influence area totals.
- Nominatim's place bounds can be a city/region bounding box and may not match one farm. For defensible field statistics, add a GeoJSON polygon upload and mask the raster to its exact geometry.
- Change thresholds (severe loss ≤ −0.2, moderate loss ≤ −0.1, gain ≥ 0.1) are starter defaults, not universal scientific thresholds. They should be justified or calibrated for the study.
- OWLv2 and CLIPSeg can miss objects or create false detections/masks. Report model, threshold, tile size, NMS settings, class-wise precision/recall, and qualitative failure cases.
- CLIPSeg reports model-derived **pixel coverage**. A ground-area estimate is only valid when the supplied GSD applies to that image and the image has not been resized/cropped without adjusting GSD.
- This project requires internet access for STAC/COGs and Hugging Face inference. Model provider availability, tokens, credits, and rate limits vary.
- `smolagents.CodeAgent` runs model-generated code in the configured executor. Keep the toolbelt allowlisted and do not expose this development app to public users without authentication, a hardened sandbox, resource limits, and appropriate data controls.

## Recommended 12-day build plan

- **Days 1–3:** validate the STAC query; inspect one COG; test NDVI/NDWI/NBR against independently computed arrays; validate projection and mask assumptions.
- **Days 4–6:** test OWLv2 tiling/NMS and CLIPSeg on a small set of high-resolution benchmark images.
- **Days 7–8:** connect the tools to `smolagents`, save tool outputs, and test failure handling and prompt injection boundaries.
- **Days 9–10:** build a labeled evaluation suite with counts, known masks, and reproducible raster index references. Do not invent expected performance numbers; report results actually measured.
- **Days 11–12:** refine the UI, export a CSV/GeoTIFF report, and document limitations and reproducibility.

## Validation

The included `index_math.py` helpers are used by the production raster functions, and `tests/test_index_math.py` checks their algebra and hectare conversion. `tests/test_geo_tools_offline.py` builds synthetic 10 m / 20 m Sentinel-2-style rasters with known answers and runs the real windowing, reprojection, SCL masking, reflectance scaling and hectare code on them without network access. `tests/test_vision_logic.py` covers tiling, non-maximum suppression and non-square tile handling with a stub detector, and `tests/test_evaluate_results.py` covers the benchmark summary. None of this substitutes for validating against real satellite assets and real model outputs. After dependencies are installed, run:

```powershell
python -m unittest discover -s tests -v
```

The folder `evaluation/` contains 30 seed query prompts, a results template, and `evaluate_results.py` for count/coverage/area MAE, relative hectare error and optional mask IoU/Dice, reported separately for each `model_or_tool` value so a VLM baseline and the tool-using agent are never pooled. Populate its expected and measured columns with real labels first; report MAE for counts, IoU/Dice for segmentation, and absolute/relative error against a reference raster for hectare estimates. Avoid claiming specific error reductions until your evaluation supports them.
