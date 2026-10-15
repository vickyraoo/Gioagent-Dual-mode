from __future__ import annotations
import datetime as dt
import json
import os
import hashlib
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from geo_tools import search_sentinel2, analyze_spectral_index, analyze_change
from agent import build_agent
from vision_tools import run_object_detection, run_text_segmentation, vqa_describe_scene
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def render_agent_trace(agent) -> None:
    """Show each step's generated code and tool output so the planner's work can be audited."""
    steps = [s for s in agent.memory.steps if type(s).__name__ == "ActionStep"]
    with st.expander(f"Agent reasoning & tool trace ({len(steps)} step{'s' if len(steps) != 1 else ''})"):
        for number, step in enumerate(steps, start=1):
            st.markdown(f"**Step {number}**")
            if step.code_action:
                st.code(step.code_action, language="python")
            if step.observations:
                st.text(step.observations[:4000])
            if step.error:
                st.error(str(step.error))
            for raw_match in dict.fromkeys(re.findall(r"[^\s'\"<>|]+?\.png", step.observations or "")):
                clean_path = Path(raw_match.replace("\\\\", "\\"))
                candidate = clean_path if clean_path.exists() else (OUTPUT_DIR / clean_path.name)
                if candidate.exists():
                    st.image(str(candidate), width="stretch")


st.set_page_config(page_title="Dual-Mode GeoAgent", page_icon="🛰️", layout="wide")
st.title("🛰️ Dual-Mode Deterministic GeoAgent")
st.caption("Natural-language task planning with measurable geospatial tools — Sentinel-2 analysis for regional patterns and separate high-resolution vision tools for object-scale tasks.")

with st.sidebar:
    st.subheader("Architecture")
    st.markdown("**Planner:** Hugging Face Qwen2.5-Coder + `smolagents`\n\n**Live data:** Earth Search v1 / Sentinel-2 L2A COGs\n\n**Quantitative analysis:** rasterio + NumPy\n\n**High-resolution mode:** OWLv2 + CLIPSeg (optional local model dependencies)")
    st.caption("LLM plans and routes work. Geospatial values are calculated by tools, not estimated by a VLM.")

macro_tab, micro_tab = st.tabs(["1 · Live multispectral analyst", "2 · High-resolution scene inspector"])

with macro_tab:
    st.subheader("Live Satellite Analyst")
    st.write("Search Sentinel-2 scenes, calculate a spectral index from streamed COG windows, compare dates, or ask the agent to plan a multi-step task.")
    c1, c2 = st.columns([1, 1])
    with c1:
        place = st.text_input("Place or region", "Karnal, Haryana, India", key="macro_place")
        # Default to the rabi (wheat) season: low monsoon cloud over north India and a clear NDVI signal.
        # Jun-Aug is monsoon/kharif there, so few scenes pass a 20% cloud filter.
        start_date = st.date_input("Start date", value=dt.date(2026, 2, 1), key="macro_start")
        end_date = st.date_input("End date", value=dt.date(2026, 3, 31), key="macro_end")
    with c2:
        cloud_limit = st.slider("Maximum scene cloud cover (%)", 0, 100, 20, key="macro_cloud")
        scene_limit = st.slider("Maximum scenes to return", 1, 10, 5, key="macro_limit")
        st.caption("The search currently uses a place-name bounding box. For field-level analysis, use a small, precise AOI boundary in a future extension.")

    if st.button("Search Sentinel-2 scenes", type="primary", key="search_s2"):
        if start_date > end_date:
            st.error("Start date must be on or before end date.")
        elif not place.strip():
            st.error("Enter a place name.")
        else:
            with st.spinner("Searching Earth Search STAC metadata…"):
                try:
                    result = search_sentinel2(place.strip(), start_date.isoformat(), end_date.isoformat(), cloud_limit, scene_limit)
                    st.session_state["macro_search_result"] = result
                    st.session_state.pop("macro_index_result", None)
                    st.session_state.pop("macro_change_result", None)
                    st.session_state["macro_search_params"] = {
                        "place": place.strip(), "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
                        "cloud_limit": cloud_limit, "scene_limit": scene_limit,
                    }
                except Exception as exc:
                    st.error(f"Scene search failed: {exc}")

    search_result = st.session_state.get("macro_search_result")
    search_params = st.session_state.get("macro_search_params")
    if search_result and search_params:
        st.markdown("### Search results")
        st.caption(f"Resolved place: {search_result['place_resolved_as']} · bbox W,S,E,N: {search_result['bbox_wgs84_wsen']}")
        scenes = search_result.get("scenes", [])
        if not scenes:
            st.warning("No scenes matched the date and cloud-cover filters. Increase the cloud limit or widen the date range.")
        else:
            st.dataframe([
                {"scene_id": s["scene_id"], "acquired": s["datetime"], "cloud_cover_%": s["cloud_cover_percent"], "platform": s["platform"], "preview": s["preview_url"] or "—"}
                for s in scenes
            ], width="stretch", hide_index=True)
            st.warning(search_result["caveat"])
            scene_ids = [s["scene_id"] for s in scenes]
            def scene_label(scene_id):
                row = next(s for s in scenes if s["scene_id"] == scene_id)
                return f"{row['datetime'][:10]} · cloud {row['cloud_cover_percent']}% · {scene_id}"
            selected_scene = st.selectbox("Scene to analyze", scene_ids, format_func=scene_label, key="macro_selected_scene")
            a1, a2, a3 = st.columns([1, 1, 1])
            with a1:
                index_name = st.selectbox("Spectral index", ["NDVI", "NDWI", "NBR"], key="macro_index")
            with a2:
                threshold = st.number_input("Index threshold", min_value=-1.0, max_value=1.0, value=0.3 if index_name == "NDVI" else 0.0, step=0.05, key="macro_threshold")
            with a3:
                st.write("")
                st.write("")
                compute_clicked = st.button("Compute index", type="primary", key="compute_index")
            if compute_clicked:
                with st.spinner("Streaming only the requested raster windows and computing the index…"):
                    try:
                        result = analyze_spectral_index(selected_scene, search_result["bbox_wgs84_wsen"], index_name, threshold)
                        st.session_state["macro_index_result"] = result
                    except Exception as exc:
                        st.error(f"Index calculation failed: {exc}")
            index_result = st.session_state.get("macro_index_result")
            if index_result and index_result.get("scene_id") == selected_scene and index_result.get("index") == index_name and index_result.get("threshold") == float(threshold):
                st.markdown("### Deterministic result")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mean " + index_name, index_result["mean"])
                m2.metric("Valid area (ha)", index_result["valid_area_ha"])
                m3.metric(f"Area ≥ {threshold}", f"{index_result['area_above_threshold_ha']} ha")
                m4.metric("Valid pixels", f"{index_result['valid_pixel_count']:,}")
                img_col, info_col = st.columns([1.2, 1])
                with img_col:
                    st.image(index_result["saved_map_path"], caption=f"{index_name} raster window", width="stretch")
                with info_col:
                    st.json({k: v for k, v in index_result.items() if k not in {"saved_map_path", "saved_geotiff_path"}})
                    st.download_button("Download index GeoTIFF", data=Path(index_result["saved_geotiff_path"]).read_bytes(), file_name=Path(index_result["saved_geotiff_path"]).name, mime="image/tiff")
                    st.caption(index_result["method_note"])

            with st.expander("Compare two scenes (ΔNDVI)"):
                if len(scene_ids) >= 2:
                    first_id = st.selectbox("Earlier scene", scene_ids, format_func=scene_label, key="change_first")
                    second_id = st.selectbox("Later scene", scene_ids, index=min(1, len(scene_ids)-1), format_func=scene_label, key="change_second")
                    if st.button("Calculate change", key="run_change"):
                        with st.spinner("Aligning two rasters and computing ΔNDVI…"):
                            try:
                                change = analyze_change(first_id, second_id, search_result["bbox_wgs84_wsen"])
                                st.session_state["macro_change_result"] = change
                            except Exception as exc:
                                st.error(f"Change detection failed: {exc}")
                    change_result = st.session_state.get("macro_change_result")
                    if change_result and change_result.get("scene_id_t1") == first_id and change_result.get("scene_id_t2") == second_id:
                        st.image(change_result["saved_map_path"], caption="ΔNDVI change map", width="stretch")
                        q1, q2, q3, q4 = st.columns(4)
                        q1.metric("Severe loss", f"{change_result['severe_loss_area_ha']} ha")
                        q2.metric("Moderate loss", f"{change_result['moderate_loss_area_ha']} ha")
                        q3.metric("Stable", f"{change_result['stable_area_ha']} ha")
                        q4.metric("Gain", f"{change_result['gain_area_ha']} ha")
                        st.caption(change_result["method_note"])
                else:
                    st.info("Search a longer date range or lower cloud threshold to get two scenes.")

    st.markdown("### Natural-language agent")
    request_text = st.text_area(
        "Write the task in natural language",
        value="Find the lowest-cloud Sentinel-2 scene for this area and calculate NDVI. Explain the mean NDVI and estimate how many hectares have NDVI above 0.3.",
        height=100,
        key="macro_nl_prompt",
    )
    if st.button("Ask the agent to plan and run tools", key="run_macro_agent"):
        if not os.getenv("HF_TOKEN", "").strip():
            st.error("HF_TOKEN is missing. Copy .env.example to .env and add your Hugging Face token.")
        elif not place.strip() or start_date > end_date:
            st.error("Enter a valid place and date range before running the agent.")
        else:
            task = (
                f"User request: {request_text}\n"
                f"Place: {place.strip()}\nStart date: {start_date.isoformat()}\nEnd date: {end_date.isoformat()}\n"
                f"Maximum scene cloud cover: {cloud_limit}\nAOI geocoding method: named-place bbox via Nominatim.\n"
                "Call the appropriate fixed tools. If an index is requested, first search, choose a suitable returned scene, "
                "then call compute_spectral_index using the exact scene_id and bbox_wsen from tool output. "
                "If a change comparison is requested, use two returned scene IDs and detect_change. Do not invent missing parameters."
            )
            try:
                with st.spinner("The planner is deciding which tools to call…"):
                    agent = build_agent()
                    response = agent.run(task)
                with st.expander("Agent response", expanded=True):
                    st.markdown(str(response))
                render_agent_trace(agent)
                st.caption("The LLM plans and selects tools; the returned numeric values should come from the deterministic tool outputs.")
            except Exception as exc:
                st.error(f"Agent run failed: {exc}")
                st.info("Check the Hugging Face token, model/provider availability, and the tool error details above.")

with micro_tab:
    st.subheader("High-Resolution Scene Inspector")
    st.write("Use an uploaded aerial/satellite image or benchmark tile for open-vocabulary object detection and semantic mask coverage. This is deliberately separate from Sentinel-2 macro analysis.")
    st.info("Good fit: airplanes, ships, storage tanks, buildings, runways, or water coverage in high-resolution imagery. Do not use 10 m Sentinel-2 imagery for small-object counting.")
    uploaded = st.file_uploader("Upload a high-resolution image", type=["png", "jpg", "jpeg", "webp"], key="micro_upload")
    saved_upload = None
    image_obj = None
    if uploaded is not None:
        safe_name = Path(uploaded.name).stem[:60] or "uploaded_scene"
        file_bytes = uploaded.getvalue()
        digest = hashlib.sha256(file_bytes).hexdigest()[:10]
        saved_upload = UPLOAD_DIR / f"{safe_name}_{digest}{Path(uploaded.name).suffix.lower()}"
        if not saved_upload.exists():
            saved_upload.write_bytes(file_bytes)
        previous_image = st.session_state.get("micro_last_image_path")
        if previous_image and previous_image != str(saved_upload):
            st.session_state.pop("micro_detection_result", None)
            st.session_state.pop("micro_seg_result", None)
        image_obj = Image.open(saved_upload).convert("RGB")
        st.image(image_obj, caption=f"Uploaded scene · {image_obj.width} × {image_obj.height} px", width="stretch")
        st.session_state["micro_last_image_path"] = str(saved_upload)
        st.session_state["micro_last_image_name"] = uploaded.name

    active_image_path = st.session_state.get("micro_last_image_path")
    if active_image_path and Path(active_image_path).exists():
        det_col, seg_col = st.columns(2)
        with det_col:
            st.markdown("#### Object detection and counting")
            object_query_text = st.text_input("Object classes (comma-separated)", "ship, boat", key="micro_object_queries")
            score_threshold = st.slider("Detection threshold", 0.05, 0.9, 0.18, 0.01, key="micro_score")
            tile_size = st.select_slider("Tile size", options=[384, 512, 768, 1024], value=768, key="micro_tiles")
            if st.button("Detect and count objects", type="primary", key="run_detection"):
                queries = [q.strip() for q in object_query_text.split(",") if q.strip()]
                try:
                    with st.spinner("Loading OWLv2 and running tiled detection… first run may download model weights."):
                        output = run_object_detection(active_image_path, queries, score_threshold, int(tile_size), 0.2)
                    st.session_state["micro_detection_result"] = output
                except Exception as exc:
                    st.error(f"Detection failed: {exc}")
                    st.info("Install requirements-vision.txt. CPU execution can be slow; a CUDA-enabled Colab runtime is recommended for larger tiles.")
            det_result = st.session_state.get("micro_detection_result")
            if det_result:
                if Path(det_result["saved_overlay_path"]).exists():
                    st.image(det_result["saved_overlay_path"], caption="Detected boxes after overlap suppression", width="stretch")
                st.metric("Predicted object count", det_result["object_count"])
                st.json({k: v for k, v in det_result.items() if k != "detections"})
                with st.expander("Detection boxes"):
                    st.dataframe(det_result["detections"], width="stretch", hide_index=True)
                st.caption(det_result["warning"])
        with seg_col:
            st.markdown("#### Text-guided segmentation")
            seg_prompt = st.text_input("What should be segmented?", "water", key="micro_seg_prompt")
            seg_threshold = st.slider("Mask threshold", 0.1, 0.9, 0.5, 0.05, key="micro_seg_threshold")
            mpp = st.number_input("Known ground sample distance (metres/pixel; 0 if unknown)", min_value=0.0, value=0.0, step=0.05, key="micro_mpp")
            if st.button("Segment and measure coverage", key="run_segmentation"):
                try:
                    with st.spinner("Running CLIPSeg text-conditioned segmentation… first run downloads model weights."):
                        output = run_text_segmentation(active_image_path, seg_prompt, seg_threshold, mpp if mpp > 0 else None)
                    st.session_state["micro_seg_result"] = output
                except Exception as exc:
                    st.error(f"Segmentation failed: {exc}")
                    st.info("Install requirements-vision.txt. If model loading fails, try a clean virtual environment and a recent Transformers version.")
            seg_result = st.session_state.get("micro_seg_result")
            if seg_result:
                if Path(seg_result["saved_overlay_path"]).exists():
                    st.image(seg_result["saved_overlay_path"], caption=f"Mask overlay: {seg_result['text_prompt']}", width="stretch")
                st.metric("Pixel coverage", f"{seg_result['pixel_coverage_percent']}%")
                if seg_result["estimated_area_m2"] is not None:
                    st.metric("Estimated mask area", f"{seg_result['estimated_area_m2']} m²")
                st.json({k: v for k, v in seg_result.items() if k not in {"saved_overlay_path", "saved_mask_path"}})
                st.caption(seg_result["warning"])

        st.markdown("#### Qualitative visual context (optional)")
        vqa_question = st.text_input("Ask about visible scene context", "Is this scene mainly coastal, agricultural, or urban? Explain uncertainty.", key="micro_vqa_prompt")
        if st.button("Describe visible context", key="run_vqa"):
            if not os.getenv("HF_TOKEN", "").strip():
                st.error("HF_TOKEN is missing in .env.")
            else:
                try:
                    with st.spinner("Requesting qualitative image interpretation…"):
                        answer = vqa_describe_scene(Image.open(active_image_path), vqa_question, os.getenv("HF_TOKEN", "").strip(), os.getenv("HF_VISION_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct"))
                    st.markdown(answer)
                    st.caption("Qualitative description only: this model must not be used for object counts, areas, boundaries, or spectral indices.")
                except Exception as exc:
                    st.error(f"Image Q&A failed: {exc}")
                    st.info("The selected vision model may not be served by your configured Hugging Face inference provider. Update HF_VISION_MODEL_ID in .env.")

    st.markdown("### Optional natural-language high-resolution task")
    micro_request = st.text_area("Instruction", "Count all ships and estimate what percentage of the image is water.", height=75, key="micro_nl_prompt")
    if st.button("Ask agent to use vision tools", key="run_micro_agent"):
        if not active_image_path:
            st.error("Upload an image first.")
        elif not os.getenv("HF_TOKEN", "").strip():
            st.error("HF_TOKEN is missing. Add it to .env.")
        else:
            task = (
                f"User request: {micro_request}\nImage path (user-uploaded file): {active_image_path}\n"
                "Use detect_and_count_objects for object counts and segment_and_measure for pixel coverage. "
                "Do not use qualitative image Q&A to estimate numbers. If ground sample distance is unknown, report pixel coverage only, not square metres."
            )
            try:
                with st.spinner("The planner is routing the task to the available vision tools…"):
                    micro_agent = build_agent()
                    response = micro_agent.run(task)
                st.markdown(str(response))
                render_agent_trace(micro_agent)
            except Exception as exc:
                st.error(f"Agent run failed: {exc}")

with st.expander("Scientific caveats and implementation notes"):
    st.markdown(
        "- Sentinel-2 bands B02/B03/B04/B08 are nominally 10 m; B11/B12 and SCL are nominally 20 m. Different grids are explicitly aligned before index math.\n"
        "- NDVI = (NIR − Red)/(NIR + Red); green-NIR NDWI = (Green − NIR)/(Green + NIR); NBR = (NIR − SWIR2)/(NIR + SWIR2).\n"
        "- SCL classes for cloud, cirrus, shadow, snow/ice, and invalid pixels are excluded in this starter. Cloud masks can miss artifacts.\n"
        "- Area values are projected raster-pixel estimates. Report resolution, CRS, valid-pixel fraction, dates, and thresholds; do not call these values exact ground truth.\n"
        "- Place-name bounding boxes are coarse. Use a surveyed/field boundary GeoJSON for publication-quality AOI statistics.\n"
        "- `smolagents.CodeAgent` runs generated code in its configured execution environment. Keep tools allowlisted, don't expose this development app publicly without authentication and a proper sandbox."
    )
