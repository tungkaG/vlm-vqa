"""Streamlit display widgets for the verification GUI (Phase 17)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import streamlit as st

from constants import CAMERA_NAMES
from schema import CandidateRecord

# Order used for the six-camera grid (front row, then back row).
_GRID_LAYOUT = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def _image(path: str, caption: str) -> None:
    """Display an image, tolerating Streamlit version parameter changes."""
    try:
        st.image(path, caption=caption, use_container_width=True)
    except TypeError:  # older Streamlit
        st.image(path, caption=caption, use_column_width=True)


def _exists(path: Optional[str]) -> bool:
    return bool(path) and Path(path).exists()


def show_multiview_grid(
    preview_paths: Dict[str, Optional[str]], grid_path: Optional[str] = None
) -> None:
    """Show the six-camera surround view (precomputed grid or a 3x2 layout)."""
    st.subheader("Six-camera surround view")
    if _exists(grid_path):
        _image(grid_path, "Multi-view grid")
        return

    rows = [_GRID_LAYOUT[:3], _GRID_LAYOUT[3:]]
    for row in rows:
        columns = st.columns(3)
        for column, camera in zip(columns, row):
            with column:
                path = preview_paths.get(camera)
                if _exists(path):
                    _image(path, camera)
                else:
                    st.caption(f"{camera}: (missing)")


def show_single_camera(preview_paths: Dict[str, Optional[str]]) -> None:
    """Show a selectable single-camera view."""
    st.subheader("Single camera")
    available = [c for c in CAMERA_NAMES if _exists(preview_paths.get(c))]
    if not available:
        st.caption("No camera previews available.")
        return
    selected = st.selectbox("Camera", available, key="single_camera")
    _image(preview_paths[selected], selected)


def show_gemini_outputs(candidate: CandidateRecord) -> None:
    """Show the Gemini suggestions for the current candidate."""
    st.subheader("Gemini suggestions")
    st.markdown(f"**Sample ID:** `{candidate.sample_id}`")
    st.markdown(f"**Scene token:** `{candidate.scene_token}`")
    st.markdown(f"**Source model:** `{candidate.source_model_name}`")
    st.markdown(f"**Priority score:** {candidate.priority_score:g}")

    label = candidate.answerability_label
    st.markdown(f"**Candidate question:** {candidate.question}")
    st.markdown(f"**Answerability:** {label.answerability}")
    st.markdown(f"**Ground-truth answer:** {label.ground_truth_answer}")
    st.markdown(f"**Visible evidence:** {label.visible_evidence or '—'}")
    st.markdown(f"**Missing evidence:** {label.missing_evidence or '—'}")
    st.markdown(f"**Recommended action:** {label.recommended_action}")

    with st.expander("Layered scene description"):
        st.json(candidate.layered_scene_description.model_dump())
    with st.expander("Scenario classification"):
        st.json(candidate.scenario_classification.model_dump())
