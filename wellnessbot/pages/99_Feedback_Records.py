from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from wellnessbot.offline.mine_logs import run_mining

# --------------------------------------------
# PAGE TRACKING FOR CHAT RESET
# --------------------------------------------
CURRENT_PAGE = "mining"

if "active_page" not in st.session_state:
    st.session_state.active_page = None

if "force_chat_reset" not in st.session_state:
    st.session_state.force_chat_reset = False

previous_page = st.session_state.active_page
if previous_page is not None and previous_page != CURRENT_PAGE:
    st.session_state.force_chat_reset = True

st.session_state.active_page = CURRENT_PAGE

st.set_page_config(page_title="Feedback Records", layout="wide")
st.title("Feedback Records")
st.caption("Offline mining for interaction logs + feedback logs")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTERACTIONS = PROJECT_ROOT / "logs"
DEFAULT_FEEDBACK = PROJECT_ROOT / "logs"

st.subheader("Input files")

col1, col2 = st.columns(2)

with col1:
    interactions_path_str = st.text_input(
        "Interactions JSONL path",
        value=str(DEFAULT_INTERACTIONS),
    )

with col2:
    feedback_path_str = st.text_input(
        "Feedback JSONL path",
        value=str(DEFAULT_FEEDBACK),
    )

run_clicked = st.button("Run mining", type="primary")

if run_clicked:
    try:
        interactions_path = Path(interactions_path_str)
        feedback_path = Path(feedback_path_str)

        results = run_mining(interactions_path, feedback_path)

        st.success("Mining complete.")

        st.subheader("Basic summary")
        stats = results["basic_stats"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Interactions", stats["interaction_count"])
        c2.metric("Matched feedback", stats["matched_feedback_count"])
        c3.metric("Unmatched feedback", stats["unmatched_feedback_count"])
        c4.metric("Candidate cases", len(results["candidate_cases"]))

        st.markdown("### Action distribution")
        st.dataframe(
            pd.DataFrame(
                [
                    {"action": k, "count": v}
                    for k, v in stats["action_distribution"].items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Feedback distribution")
        st.dataframe(
            pd.DataFrame(
                [
                    {"thumb": k, "count": v}
                    for k, v in stats["feedback_distribution"].items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Rule disagreement")
        rule_df = pd.DataFrame(results["rule_disagreement"])
        if not rule_df.empty:
            st.dataframe(rule_df, use_container_width=True, hide_index=True)
        else:
            st.info("No rule disagreement patterns found.")

        st.subheader("Numeric ambiguity")
        ambiguity_df = pd.DataFrame(results["numeric_ambiguity"])
        if not ambiguity_df.empty:
            st.dataframe(ambiguity_df, use_container_width=True, hide_index=True)
        else:
            st.info("No numeric ambiguity patterns found.")

        st.subheader("Threshold consistency")
        threshold_df = pd.DataFrame(results["threshold_consistency"])
        if not threshold_df.empty:
            st.dataframe(threshold_df, use_container_width=True, hide_index=True)
        else:
            st.info("No threshold consistency data found.")

        st.subheader("Red-flag consistency")
        redflag_df = pd.DataFrame(results["red_flag_consistency"])
        if not redflag_df.empty:
            st.dataframe(redflag_df, use_container_width=True, hide_index=True)
        else:
            st.info("No red-flag cases found.")

        st.subheader("Candidate cases")
        cases_df = pd.DataFrame(results["candidate_cases"])
        if not cases_df.empty:
            st.dataframe(cases_df, use_container_width=True, hide_index=True)
        else:
            st.info("No candidate cases generated.")

        with st.expander("Show unmatched feedback JSON"):
            st.code(json.dumps(results["unmatched_feedback"], indent=2), language="json")

    except Exception as e:
        st.error(f"Mining failed: {e}")