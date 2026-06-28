"""Human verification GUI (Phase 17).

Run with:

    streamlit run src/gui/app.py -- --config configs/nuscenes_mini.yaml

Shows the six-camera surround view, a single-camera selector and every
Gemini suggestion for the selected candidate, and lets a human edit the
important labels and accept/reject. A sidebar dropdown (plus Prev/Next
buttons) lets you jump to *any* candidate; accepting or rejecting
auto-advances to the next unverified one. Accepted samples are appended
to ``ground_truth_100.jsonl`` (rejected ones to the rejected file).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the src/ directory importable when launched via `streamlit run`.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import streamlit as st  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from config import load_config  # noqa: E402
from constants import (  # noqa: E402
    ANSWERABILITY_LABELS,
    SAFETY_ACTIONS,
    SCENARIO_CLUSTERS,
    TASK_LAYERS,
    UNCERTAINTY_SOURCES,
)
from gui import state, widgets  # noqa: E402


def _resolve_config_path() -> str:
    """Resolve the config path from `-- --config`, env, or the default."""
    argv = sys.argv[1:]
    if "--config" in argv:
        index = argv.index("--config")
        if index + 1 < len(argv):
            return argv[index + 1]
    return os.environ.get("VLM_VQA_CONFIG", "configs/nuscenes_mini.yaml")


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # older Streamlit
        st.experimental_rerun()


def _index_of(options, value, default=0) -> int:
    return options.index(value) if value in options else default


def _nudge(delta: int, total: int) -> None:
    """Move the navigation cursor by ``delta``, clamped to the list."""
    current = st.session_state.get("nav_index", 0)
    st.session_state.nav_index = max(0, min(total - 1, current + delta))


def _set_index(index: int) -> None:
    st.session_state.nav_index = index


def _status_mark(candidate, verified_keys, rejected_keys) -> str:
    key = state.candidate_key(candidate)
    if key in verified_keys:
        return "✓"
    if key in rejected_keys:
        return "✗"
    return "•"


def _candidate_label(candidate, verified_keys, rejected_keys, i, total) -> str:
    mark = _status_mark(candidate, verified_keys, rejected_keys)
    question = candidate.question.strip()
    if len(question) > 40:
        question = question[:39] + "…"
    return f"{mark} {i + 1}/{total}  {candidate.sample_id[:8]} — {question}"



def _edit_form(candidate) -> None:
    label = candidate.answerability_label
    scenario = candidate.scenario_classification
    current_cluster = scenario.scenario_clusters[0] if scenario.scenario_clusters else SCENARIO_CLUSTERS[0]

    with st.form(key=f"edit_{candidate.sample_id}_{abs(hash(candidate.question))}"):
        st.subheader("Human correction")
        scenario_cluster = st.selectbox(
            "Scenario cluster",
            SCENARIO_CLUSTERS,
            index=_index_of(SCENARIO_CLUSTERS, current_cluster),
        )
        task_layer = st.selectbox(
            "Task layer", TASK_LAYERS, index=_index_of(TASK_LAYERS, scenario.task_layer)
        )
        question = st.text_area("Question", value=candidate.question)
        answerability = st.selectbox(
            "Answerability",
            ANSWERABILITY_LABELS,
            index=_index_of(ANSWERABILITY_LABELS, label.answerability),
        )
        ground_truth_answer = st.text_area(
            "Ground-truth answer", value=label.ground_truth_answer
        )
        abstention_required = st.checkbox(
            "Abstention required (must be true for unanswerable/ambiguous)",
            value=label.abstention_required,
        )
        visible_evidence = st.text_area(
            "Visible evidence", value=label.visible_evidence
        )
        missing_evidence = st.text_area(
            "Missing evidence", value=label.missing_evidence
        )
        uncertainty_source = st.selectbox(
            "Uncertainty source",
            UNCERTAINTY_SOURCES,
            index=_index_of(UNCERTAINTY_SOURCES, label.uncertainty_source),
        )
        recommended_action = st.selectbox(
            "Recommended action",
            SAFETY_ACTIONS,
            index=_index_of(SAFETY_ACTIONS, label.recommended_action),
        )
        human_notes = st.text_area("Human notes", value=candidate.human_notes or "")

        accept_col, reject_col = st.columns(2)
        accepted = accept_col.form_submit_button("Accept", type="primary")
        rejected = reject_col.form_submit_button("Reject")

    edits = {
        "scenario_cluster": scenario_cluster,
        "task_layer": task_layer,
        "question": question,
        "answerability": answerability,
        "ground_truth_answer": ground_truth_answer,
        "abstention_required": abstention_required,
        "visible_evidence": visible_evidence,
        "missing_evidence": missing_evidence,
        "uncertainty_source": uncertainty_source,
        "recommended_action": recommended_action,
        "human_notes": human_notes,
    }

    if accepted:
        try:
            state.accept_candidate(candidate, edits, _config())
        except ValidationError as error:
            st.error(f"Cannot accept — the edited labels are inconsistent:\n\n{error}")
        else:
            st.success("Saved to verified set.")
            st.session_state["advance"] = True
            _rerun()

    if rejected:
        state.reject_candidate(candidate, _config(), notes=human_notes or None)
        st.warning("Candidate rejected.")
        st.session_state["advance"] = True
        _rerun()


@st.cache_resource
def _load_config_cached(config_path: str):
    return load_config(config_path)


def _config():
    return _load_config_cached(st.session_state["config_path"])


def main() -> None:
    st.set_page_config(page_title="nuScenes VQA Verification", layout="wide")
    st.title("nuScenes VQA — human verification")

    default_path = _resolve_config_path()
    config_path = st.sidebar.text_input("Config path", value=default_path)
    st.session_state["config_path"] = config_path

    try:
        config = _config()
    except Exception as error:  # noqa: BLE001
        st.error(f"Failed to load config '{config_path}': {error}")
        return

    candidates = state.load_candidates(config)
    if not candidates:
        st.warning(
            "No candidates found. Run `auto_annotate` first to produce "
            f"{config.paths.auto_candidates_path}."
        )
        return

    total = len(candidates)
    verified_keys, rejected_keys = state.load_status_keys(config)
    processed = verified_keys | rejected_keys

    # Index of the next not-yet-processed candidate (None if all done).
    next_idx = next(
        (i for i, c in enumerate(candidates) if state.candidate_key(c) not in processed),
        None,
    )

    # Initialise / auto-advance the navigation cursor.
    if "nav_index" not in st.session_state:
        st.session_state.nav_index = next_idx if next_idx is not None else 0
    if st.session_state.pop("advance", False) and next_idx is not None:
        st.session_state.nav_index = next_idx
    st.session_state.nav_index = max(0, min(total - 1, st.session_state.nav_index))

    verified_count = state.count_verified(config)
    target = config.pipeline.target_verified_count
    st.sidebar.metric("Verified", f"{verified_count} / {target}")
    st.sidebar.progress(min(1.0, verified_count / target) if target else 0.0)
    if state.is_target_reached(config):
        st.sidebar.success("Target reached — run the `report` stage to summarise.")

    # --- Navigation controls ---------------------------------------------
    st.sidebar.header("Navigate")
    fill = widgets.width_kwargs()
    prev_col, next_col = st.sidebar.columns(2)
    prev_col.button("◀ Prev", on_click=_nudge, args=(-1, total), **fill)
    next_col.button("Next ▶", on_click=_nudge, args=(1, total), **fill)
    if next_idx is not None:
        st.sidebar.button(
            "Go to next unverified",
            on_click=_set_index,
            args=(next_idx,),
            **fill,
        )
    else:
        st.sidebar.caption("All candidates have been processed.")

    st.sidebar.selectbox(
        "Jump to candidate",
        options=list(range(total)),
        key="nav_index",
        format_func=lambda i: _candidate_label(
            candidates[i], verified_keys, rejected_keys, i, total
        ),
    )
    if st.sidebar.button("Refresh"):
        _rerun()

    candidate = candidates[st.session_state.nav_index]

    # Status banner for the selected candidate.
    selected_key = state.candidate_key(candidate)
    if selected_key in verified_keys:
        st.info("✓ This candidate is already in the verified set (accepting again appends another copy).")
    elif selected_key in rejected_keys:
        st.info("✗ This candidate was previously rejected.")

    left, right = st.columns([3, 2])
    with left:
        grid_path = str(
            Path(config.paths.preview_dir) / "grids" / f"{candidate.sample_id}.jpg"
        )
        widgets.show_multiview_grid(candidate.preview_paths, grid_path)
        widgets.show_single_camera(candidate.preview_paths, candidate.sample_id)
    with right:
        widgets.show_gemini_outputs(candidate)
        _edit_form(candidate)



# Streamlit executes this script top-to-bottom on every rerun.
main()
