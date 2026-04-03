from __future__ import annotations

import json
import hashlib
import logging
import sys
from datetime import datetime, timezone

import streamlit as st
import streamlit.components.v1 as components

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    stream=sys.stderr,
)

from wellnessbot.pipeline.run import run_pipeline
from wellnessbot.logging.logger import log_feedback
from wellnessbot.storage.profile_store import (
    create_profile,
    load_profile,
    list_profiles,
    save_profile,
    delete_profile,
)
from wellnessbot.storage.exercise_history_store import (
    load_exercise_history,
    append_exercise_history,
    clear_exercise_history,
)
from wellnessbot.kg.kg import phase_from_weeks, get_protocol_for_surgery_type

CURRENT_PAGE = "chat"

TOOL_OPTIONS = [
    "chair",
    "resistance_band",
    "towel",
    "step",
    "socks",
    "sliding_board",
    "strap",
    "stool",
    "wall",
    "table",
    "stationary_bicycle",
    "wobble_board",
    "thick_carpet",
    "foam_block",
    "camping_mattress",
]

SURGERY_TYPE_OPTIONS = [
    "unknown",
    "arthroscopic_knee_surgery",
    "acl_reconstruction",
    "tkr",
    "sprain_non_surgical",
]

SURGERY_TYPE_LABELS = {
    "unknown": "Unknown",
    "arthroscopic_knee_surgery": "Arthroscopic knee surgery",
    "acl_reconstruction": "ACL reconstruction",
    "tkr": "TKR",
    "sprain_non_surgical": "Sprain (non-surgical)",
}


def normalize_surgery_type_value(value: str | None) -> str:
    value = (value or "unknown").strip()
    if value == "post_arthroscopic_knee_surgery":
        return "arthroscopic_knee_surgery"
    return value or "unknown"


def format_surgery_type(value: str) -> str:
    value = normalize_surgery_type_value(value)
    return SURGERY_TYPE_LABELS.get(value, value.replace("_", " ").title())


def make_interaction_id(user_text: str, audit_ts: str) -> str:
    return hashlib.sha256(f"{audit_ts}|{user_text}".encode("utf-8")).hexdigest()[:16]


def _result_should_end_chat(result: dict) -> bool:
    if result.get("mode") != "final":
        return False

    action = (result.get("decision") or {}).get("action")
    conv_state = result.get("conv_state") or {}
    pending_followup_slots = conv_state.get("pending_followup_slots", []) or []

    if action in {"RECOMMEND", "ESCALATE"}:
        return True

    if action == "FORBID":
        return len(pending_followup_slots) == 0

    return False


def build_welcome_message(profile: dict | None = None) -> dict:
    profile = profile or {}

    surgery_type = normalize_surgery_type_value(
        profile.get("surgery_type", profile.get("event_type", "unknown"))
    )
    surgery_date = profile.get("surgery_date", "")

    phase_id = None
    phase_name = None
    weeks_since_event = None

    if surgery_type not in (None, "", "unknown") and surgery_date:
        try:
            dt = datetime.strptime(surgery_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            delta_days = (datetime.now(timezone.utc) - dt).days
            if delta_days >= 0:
                weeks_since_event = round(delta_days / 7, 2)
                phase_id = phase_from_weeks(weeks_since_event, surgery_type)

                protocol = get_protocol_for_surgery_type(surgery_type)
                if protocol and phase_id:
                    for ph in protocol.phases:
                        if ph.phase_id == phase_id:
                            phase_name = ph.name
                            break
        except Exception:
            phase_id = None
            phase_name = None
            weeks_since_event = None

    phase_line = ""
    if phase_id:
        if phase_name:
            phase_line = f"**Current phase:** {phase_id} ({phase_name})\n\n"
        else:
            phase_line = f"**Current phase:** {phase_id}\n\n"

    if surgery_type in (None, "", "unknown"):
        slot_name = "surgery_type"
        question = "What surgery type did you have? (e.g. Arthroscopic knee surgery)"
    elif not surgery_date:
        slot_name = "surgery_date"
        question = (
            "When was your surgery or injury? Please tell me the date (YYYY-MM-DD) "
            "or how many weeks/days ago."
        )
    else:
        slot_name = "symptom_screen"
        question = (
            "Are you having any symptoms today, such as fever or excessive bleeding? "
            "If none, just say 'none'."
        )

    return {
        "role": "assistant",
        "text": (
            "👋 Welcome to the Wellnessbot.\n\n"
            "I can help suggest suitable rehabilitation exercises based on your recovery stage.\n\n"
            f"{phase_line}"
            f"{question}"
        ),
        "result": {
            "mode": "clarify",
            "slot_name": slot_name,
            "nlu_turn": {},
            "audit_trace": {
                "mode": "clarify",
                "asked_slot": slot_name,
                "phase_id": phase_id,
                "phase_name": phase_name,
                "weeks_since_event": weeks_since_event,
                "notes": ["Initial system greeting and first question."],
            },
        },
    }


def _profile_to_conv_state(profile: dict) -> dict:
    surgery_type = normalize_surgery_type_value(
        profile.get("surgery_type", profile.get("event_type", "unknown"))
    )
    return {
        "surgery_type": surgery_type,
        "surgery_date": profile.get("surgery_date", ""),
        "equipment_available": profile.get("equipment_available", []) or [],
    }


def _get_preserved_profile_from_state() -> dict:
    profile_id = (st.session_state.get("profile_id") or "").strip()

    if profile_id:
        try:
            profile = load_profile(profile_id)
            return {
                "surgery_type": normalize_surgery_type_value(
                    profile.get("surgery_type", profile.get("event_type", "unknown"))
                ),
                "surgery_date": profile.get("surgery_date", ""),
                "equipment_available": profile.get("equipment_available", []) or [],
            }
        except Exception:
            pass

    conv = st.session_state.get("conv_state") or {}
    return {
        "surgery_type": normalize_surgery_type_value(
            conv.get("surgery_type", conv.get("event_type", "unknown"))
        ),
        "surgery_date": conv.get("surgery_date", ""),
        "equipment_available": st.session_state.get("equipment_available", []) or [],
    }


def _save_current_profile() -> None:
    profile_id = (st.session_state.get("profile_id") or "").strip()
    if not profile_id:
        return

    conv = st.session_state.conv_state or {}
    surgery_type = normalize_surgery_type_value(
        conv.get("surgery_type", conv.get("event_type", "unknown"))
    )

    profile = {
        "profile_id": profile_id,
        "display_name": st.session_state.get("display_name", ""),
        "surgery_type": surgery_type,
        "surgery_date": conv.get("surgery_date", ""),
        "equipment_available": conv.get("equipment_available", []) or [],
    }
    save_profile(profile)


def _sync_core_profile_widgets_from_state() -> None:
    conv = st.session_state.conv_state or {}

    surgery_type = normalize_surgery_type_value(
        conv.get("surgery_type", conv.get("event_type", "unknown"))
    )
    if surgery_type not in SURGERY_TYPE_OPTIONS:
        surgery_type = "unknown"

    st.session_state.editable_surgery_type = surgery_type
    st.session_state.editable_surgery_date = conv.get("surgery_date", "")


def _sync_equipment_multiselect_from_state() -> None:
    st.session_state.equipment_multiselect = (
        st.session_state.conv_state.get("equipment_available", []) or []
    )


def _bump_chat_input_epoch() -> None:
    st.session_state.chat_input_epoch = st.session_state.get("chat_input_epoch", 0) + 1


def _reset_chat_session(preserved_profile: dict, keep_profile_loaded: bool = True) -> None:
    preserved_profile = dict(preserved_profile or {})

    clean_profile = {
        "surgery_type": normalize_surgery_type_value(
            preserved_profile.get("surgery_type", preserved_profile.get("event_type", "unknown"))
        ),
        "surgery_date": preserved_profile.get("surgery_date", ""),
        "equipment_available": preserved_profile.get("equipment_available", []) or [],
    }

    st.session_state.chat = [build_welcome_message(clean_profile)]
    st.session_state.feedback_state = {}
    st.session_state.conv_state = clean_profile
    st.session_state.sync_core_profile_from_conv = True
    st.session_state.sync_equipment_from_conv = True
    st.session_state.chat_ended = False
    st.session_state.pending_user_text = None
    st.session_state.processing_turn = False
    st.session_state.quick_reply = None
    st.session_state.profile_loaded = keep_profile_loaded
    _bump_chat_input_epoch()


def _get_latest_asked_slot() -> str | None:
    if not st.session_state.chat:
        return None

    for msg in reversed(st.session_state.chat):
        if msg.get("role") != "assistant":
            continue

        result = msg.get("result") or {}
        if result.get("mode") == "clarify":
            return result.get("slot_name")

        if result.get("mode") == "final":
            return None

    return None


def _build_assistant_text(result: dict) -> str:
    mode = result.get("mode", "final")

    if mode == "clarify":
        return result.get("question", "I need a bit more information.")

    action = result["decision"]["action"]

    rules = result["audit_trace"].get("rules_fired", [])
    top_rule = next((r for r in rules if r.get("action") == action), None)
    if top_rule is None and rules:
        top_rule = rules[0]

    primary_rationale = top_rule.get("rationale") if top_rule else "Decision generated."

    extra_lines = []
    for r in rules:
        if r is top_rule:
            continue

        rid = r.get("rule_id", "")
        rationale = (r.get("rationale") or "").strip()
        if not rationale:
            continue

        if rid.startswith("R_SELFCARE_"):
            extra_lines.append(f"**Supportive care:** {rationale}")

    extra_block = ""
    if extra_lines:
        extra_block = "\n\n" + "\n\n".join(extra_lines)

    rule_ids = result["decision"].get("rule_ids", [])
    rule_line = f"\n\nRule IDs: {', '.join(rule_ids)}" if rule_ids else ""

    citations = result["decision"].get("citations", [])
    cite_str = ", ".join(citations[:3]) + (" ..." if len(citations) > 3 else "")
    cite_line = f"\n\nSources: {cite_str}" if citations else ""

    planner = result["audit_trace"].get("planner")
    planner_line = ""
    if planner:
        if planner.get("type") == "alternatives":
            items = planner.get("items", [])
            names = [it.get("name") for it in items if it.get("name")]
            if names:
                planner_line = "\n\n**Available options in this phase:**\n- " + "\n- ".join(names)
        else:
            ex_name = planner.get("exercise_name") or planner.get("exercise_id")
            stop = planner.get("stop_conditions", [])

            planner_line = f"\n\n**Recommended exercise:** {ex_name}"
            if stop:
                planner_line += "\n\nStop if:\n- " + "\n- ".join(stop)

    assistant_text = f"**{action}**\n\n{primary_rationale}{extra_block}{planner_line}{rule_line}{cite_line}"

    if _result_should_end_chat(result):
        assistant_text += (
            "\n\n---\n\n"
            "✅ **Chat ended.**\n\n"
            "Please click **End conversation / Restart** or **Clear chat only** to start a new case."
        )

    return assistant_text


def _handle_pipeline_result(result: dict) -> None:
    if "conv_state" in result:
        conv_state = dict(result["conv_state"] or {})
        conv_state["surgery_type"] = normalize_surgery_type_value(
            conv_state.get("surgery_type", conv_state.get("event_type", "unknown"))
        )

        st.session_state.conv_state = conv_state
        st.session_state.equipment_available = (
            st.session_state.conv_state.get("equipment_available", [])
            or st.session_state.equipment_available
        )
        st.session_state.sync_core_profile_from_conv = True
        st.session_state.sync_equipment_from_conv = True
        _save_current_profile()

    mode = result.get("mode", "final")

    if mode == "final":
        st.session_state.chat_ended = _result_should_end_chat(result)

        planner = (result.get("audit_trace") or {}).get("planner") or {}
        ex_id = planner.get("exercise_id")
        ex_name = planner.get("exercise_name") or ex_id
        profile_id = (st.session_state.get("profile_id") or "").strip()

        if profile_id and ex_id:
            append_exercise_history(
                profile_id=profile_id,
                record={
                    "timestamp_utc": (result.get("audit_trace") or {}).get("timestamp_utc"),
                    "exercise_id": ex_id,
                    "exercise_name": ex_name,
                    "phase_id": planner.get("phase_id"),
                    "priority": planner.get("priority"),
                    "pain_score": st.session_state.conv_state.get("pain_score"),
                    "swelling_score": st.session_state.conv_state.get("swelling_score"),
                    "action": result.get("decision", {}).get("action"),
                },
                keep_last=50,
            )


st.set_page_config(page_title="Wellnessbot - Rehab Decision Support", layout="centered")
st.title("Wellnessbot - Rehab Decision Support")
st.caption("Decision brain = Rules + KG + Planner. LLM (optional) is NLU only. Not medical advice.")

if "conv_state" not in st.session_state:
    st.session_state.conv_state = {}

if "profile_id" not in st.session_state:
    st.session_state.profile_id = ""

if "display_name" not in st.session_state:
    st.session_state.display_name = ""

if "mock_toggle" not in st.session_state:
    st.session_state.mock_toggle = False

if "feedback_state" not in st.session_state:
    st.session_state.feedback_state = {}

if "equipment_available" not in st.session_state:
    st.session_state.equipment_available = (
        st.session_state.conv_state.get("equipment_available", []) or []
    )

if "equipment_multiselect" not in st.session_state:
    st.session_state.equipment_multiselect = (
        st.session_state.conv_state.get("equipment_available", []) or []
    )

if "chat" not in st.session_state:
    st.session_state.chat = []

if "chat_ended" not in st.session_state:
    st.session_state.chat_ended = False

if "profile_loaded" not in st.session_state:
    st.session_state.profile_loaded = bool(st.session_state.get("profile_id"))

if "show_create_profile" not in st.session_state:
    st.session_state.show_create_profile = False

if "editable_surgery_type" not in st.session_state:
    st.session_state.editable_surgery_type = "unknown"

if "editable_surgery_date" not in st.session_state:
    st.session_state.editable_surgery_date = ""

if "sync_core_profile_from_conv" not in st.session_state:
    st.session_state.sync_core_profile_from_conv = False

if "sync_equipment_from_conv" not in st.session_state:
    st.session_state.sync_equipment_from_conv = False

if "pending_user_text" not in st.session_state:
    st.session_state.pending_user_text = None

if "processing_turn" not in st.session_state:
    st.session_state.processing_turn = False

if "chat_input_epoch" not in st.session_state:
    st.session_state.chat_input_epoch = 0

if "quick_reply" not in st.session_state:
    st.session_state.quick_reply = None

if "active_page" not in st.session_state:
    st.session_state.active_page = None

if "force_chat_reset" not in st.session_state:
    st.session_state.force_chat_reset = False

came_from_other_page = st.session_state.active_page not in (None, CURRENT_PAGE)

if st.session_state.get("profile_id") and (
    came_from_other_page or st.session_state.force_chat_reset
):
    preserved_profile = _get_preserved_profile_from_state()
    _reset_chat_session(preserved_profile, keep_profile_loaded=True)
    st.session_state.force_chat_reset = False

st.session_state.active_page = CURRENT_PAGE

if st.session_state.sync_core_profile_from_conv:
    _sync_core_profile_widgets_from_state()
    st.session_state.sync_core_profile_from_conv = False

if st.session_state.sync_equipment_from_conv:
    _sync_equipment_multiselect_from_state()
    st.session_state.sync_equipment_from_conv = False

st.sidebar.header("Profile Memory")

existing_profiles = list_profiles()
profile_options = ["-- Select a profile --"] + existing_profiles

default_idx = 0
active_profile_id = (st.session_state.get("profile_id") or "").strip()
if active_profile_id and active_profile_id in existing_profiles:
    default_idx = 1 + existing_profiles.index(active_profile_id)

with st.sidebar.form("load_profile_form"):
    selected_profile = st.selectbox(
        "Load existing profile",
        options=profile_options,
        index=default_idx,
        key="selected_profile_id",
    )
    load_clicked = st.form_submit_button("Load profile")

if load_clicked:
    try:
        if selected_profile == "-- Select a profile --":
            st.warning("Please choose a profile first.")
        else:
            profile = load_profile(selected_profile)
            st.session_state.profile_id = profile["profile_id"]
            st.session_state.display_name = profile.get("display_name", "")
            st.session_state.equipment_available = profile.get("equipment_available", []) or []
            preserved_profile = _profile_to_conv_state(profile)
            _reset_chat_session(preserved_profile, keep_profile_loaded=True)
            st.session_state.show_create_profile = False
            st.success(f"Loaded profile: {profile['profile_id']}")
            st.rerun()
    except Exception as e:
        st.error(f"Could not load profile: {e}")

col_p1, col_p2 = st.sidebar.columns(2)

with col_p1:
    if st.button("➕ Create", use_container_width=True):
        st.session_state.show_create_profile = not st.session_state.show_create_profile
        st.rerun()

with col_p2:
    if st.button("🗑 Delete", use_container_width=True):
        try:
            active_profile_id = (st.session_state.get("profile_id") or "").strip()

            if not active_profile_id:
                st.warning("No active profile to delete.")
            else:
                delete_profile(active_profile_id)
                clear_exercise_history(active_profile_id)

                st.session_state.profile_id = ""
                st.session_state.display_name = ""
                st.session_state.conv_state = {}
                st.session_state.equipment_available = []
                st.session_state.equipment_multiselect = []
                st.session_state.chat = []
                st.session_state.feedback_state = {}
                st.session_state.chat_ended = False
                st.session_state.profile_loaded = False
                st.session_state.show_create_profile = False
                st.session_state.editable_surgery_type = "unknown"
                st.session_state.editable_surgery_date = ""
                st.session_state.pending_user_text = None
                st.session_state.processing_turn = False
                st.session_state.quick_reply = None
                st.session_state.force_chat_reset = False
                st.session_state.active_page = CURRENT_PAGE
                _bump_chat_input_epoch()

                st.success(f"Deleted profile: {active_profile_id}")
                st.rerun()

        except Exception as e:
            st.error(f"Could not delete profile: {e}")

if st.session_state.show_create_profile:
    with st.sidebar.form("create_profile_form"):
        new_profile_id = st.text_input(
            "New profile ID",
            value="",
            key="new_profile_id_input",
            help="Use a simple unique ID, e.g. kx_demo",
        )

        new_display_name = st.text_input(
            "Display name",
            value="",
            key="new_display_name_input",
        )

        create_clicked = st.form_submit_button("Confirm create profile")

    if create_clicked:
        try:
            profile_id = new_profile_id.strip()
            display_name = new_display_name.strip()

            if not profile_id:
                st.warning("Please enter a profile ID.")
            else:
                profile = create_profile(
                    profile_id=profile_id,
                    display_name=display_name,
                )
                st.session_state.profile_id = profile["profile_id"]
                st.session_state.display_name = profile.get("display_name", "")
                st.session_state.equipment_available = profile.get("equipment_available", []) or []
                preserved_profile = _profile_to_conv_state(profile)
                _reset_chat_session(preserved_profile, keep_profile_loaded=True)
                st.session_state.show_create_profile = False
                st.success(f"Profile created: {profile['profile_id']}")
                st.rerun()
        except Exception as e:
            st.error(f"Could not create profile: {e}")

active_profile_id = (st.session_state.get("profile_id") or "").strip()
active_display_name = (st.session_state.get("display_name") or "").strip()

if active_profile_id:
    st.sidebar.success(
        f"Current profile: {active_profile_id}"
        + (f" ({active_display_name})" if active_display_name else "")
    )
else:
    st.sidebar.info("No profile loaded.")

st.sidebar.divider()
st.sidebar.header("Core Profile")

edited_surgery_type = st.sidebar.selectbox(
    "Surgery type",
    options=SURGERY_TYPE_OPTIONS,
    key="editable_surgery_type",
    disabled=not st.session_state.profile_loaded,
    format_func=format_surgery_type,
)

edited_surgery_date = st.sidebar.text_input(
    "Surgery date (YYYY-MM-DD)",
    key="editable_surgery_date",
    disabled=not st.session_state.profile_loaded,
)

if st.sidebar.button("Save core profile", disabled=not st.session_state.profile_loaded):
    st.session_state.conv_state["surgery_type"] = normalize_surgery_type_value(edited_surgery_type)
    st.session_state.conv_state["surgery_date"] = edited_surgery_date.strip()
    _save_current_profile()

    preserved_profile = _get_preserved_profile_from_state()
    _reset_chat_session(preserved_profile, keep_profile_loaded=True)

    st.success("Core profile updated.")
    st.rerun()

st.sidebar.markdown("**Available tools/equipment**")

selected_tools = st.sidebar.multiselect(
    "Select available tools",
    options=TOOL_OPTIONS,
    key="equipment_multiselect",
    disabled=not st.session_state.profile_loaded,
    format_func=lambda x: x.replace("_", " ").title(),
)

if st.session_state.profile_loaded:
    st.session_state.equipment_available = selected_tools
    st.session_state.conv_state["equipment_available"] = selected_tools
    _save_current_profile()

with st.container():
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.session_state.mock_toggle = st.toggle(
            "MOCK_NLU",
            value=st.session_state.mock_toggle,
            help="ON = deterministic mock. OFF = try OpenAI then fallback to mock on error.",
            disabled=not st.session_state.profile_loaded,
        )

    with col2:
        if st.button("End conversation / Restart", disabled=not st.session_state.profile_loaded):
            preserved_profile = _get_preserved_profile_from_state()
            _reset_chat_session(
                preserved_profile,
                keep_profile_loaded=bool(st.session_state.get("profile_id")),
            )
            _save_current_profile()
            st.rerun()

    with col3:
        if st.button("Clear chat only", disabled=not st.session_state.profile_loaded):
            preserved_profile = _get_preserved_profile_from_state()
            _reset_chat_session(
                preserved_profile,
                keep_profile_loaded=bool(st.session_state.get("profile_id")),
            )
            _save_current_profile()
            st.rerun()

st.divider()

for i, msg in enumerate(st.session_state.chat):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["text"])
        continue

    with st.chat_message("assistant"):
        st.write(msg["text"])

        result = msg.get("result")
        if not result:
            continue

        mode = result.get("mode", "final")

        if mode == "clarify":
            asked_slot = result.get("slot_name", "unknown")
            nlu_turn = result.get("nlu_turn", {})
            nlu_source = nlu_turn.get("nlu_source", "unknown")

            st.caption(
                f"mode: **clarify** · asked_slot: `{asked_slot}` · nlu_source: `{nlu_source}`"
            )

            with st.expander("Show turn NLU JSON"):
                st.code(json.dumps(nlu_turn, indent=2), language="json")

            with st.expander("Show dialog audit"):
                st.code(json.dumps(result.get("audit_trace", {}), indent=2), language="json")

            continue

        action = result["decision"]["action"]
        nlu_source = result["nlu"]["nlu_source"]
        conf = result["decision"]["confidence"]

        st.caption(
            f"mode: **final** · action: **{action}** · confidence: **{conf:.2f}** · nlu_source: `{nlu_source}`"
        )

        interaction_id = result.get("interaction_id")
        if not interaction_id:
            audit_ts = (result.get("audit_trace") or {}).get("timestamp_utc") or ""
            interaction_id = make_interaction_id(result.get("user_text", ""), audit_ts)

        if i not in st.session_state.feedback_state:
            st.session_state.feedback_state[i] = {
                "thumb": None,
                "comment": "",
                "expected_action": "UNKNOWN",
                "submitted": False,
            }

        fb = st.session_state.feedback_state[i]

        c1, c2, _ = st.columns([1, 1, 6])

        with c1:
            if st.button("👍", key=f"thumb_up_{i}", disabled=fb["submitted"]):
                fb["thumb"] = "up"
                fb["submitted"] = True
                log_feedback(interaction_id=interaction_id, thumb="up")
                st.toast("Feedback saved (👍)")
                st.rerun()

        with c2:
            if st.button("👎", key=f"thumb_down_{i}", disabled=fb["submitted"]):
                fb["thumb"] = "down"
                st.rerun()

        if fb["thumb"] == "down" and not fb["submitted"]:
            fb["expected_action"] = st.selectbox(
                "What would be the correct action?",
                options=["UNKNOWN", "RECOMMEND", "FORBID", "CLARIFY", "ESCALATE"],
                index=0,
                key=f"expected_action_{i}",
            )
            fb["comment"] = st.text_input(
                "What went wrong? (optional)",
                value=fb.get("comment", ""),
                key=f"comment_{i}",
            )
            if st.button("Submit feedback", key=f"submit_feedback_{i}"):
                fb["submitted"] = True
                log_feedback(
                    interaction_id=interaction_id,
                    thumb="down",
                    comment=fb["comment"] or None,
                    expected_action=None
                    if fb["expected_action"] == "UNKNOWN"
                    else fb["expected_action"],
                )
                st.toast("Feedback saved (👎)")
                st.rerun()

        with st.expander("Show NLU JSON"):
            st.code(json.dumps(result["nlu"], indent=2), language="json")

        with st.expander("Show Audit Trace (rules, citations, planner)"):
            st.code(json.dumps(result["audit_trace"], indent=2), language="json")

st.divider()

if not st.session_state.profile_loaded:
    st.info("Please load a profile or create a new profile before starting the chat.")
else:
    if st.session_state.processing_turn and st.session_state.pending_user_text:
        pending_text = st.session_state.pending_user_text

        profile_id = (st.session_state.get("profile_id") or "").strip()
        history_for_planner = load_exercise_history(profile_id)

        conv_state_for_run = dict(st.session_state.conv_state or {})
        conv_state_for_run["surgery_type"] = normalize_surgery_type_value(
            conv_state_for_run.get("surgery_type", conv_state_for_run.get("event_type", "unknown"))
        )
        conv_state_for_run["exercise_history"] = history_for_planner

        result = run_pipeline(
            user_text=pending_text,
            conv_state=conv_state_for_run,
            force_mock_nlu=st.session_state.mock_toggle,
        )

        _handle_pipeline_result(result)
        assistant_text = _build_assistant_text(result)

        if (
            st.session_state.chat
            and st.session_state.chat[-1]["role"] == "assistant"
            and st.session_state.chat[-1]["result"] is None
            and st.session_state.chat[-1]["text"] == "Thinking…"
        ):
            st.session_state.chat[-1] = {
                "role": "assistant",
                "text": assistant_text,
                "result": result,
            }
        else:
            st.session_state.chat.append(
                {"role": "assistant", "text": assistant_text, "result": result}
            )

        st.session_state.pending_user_text = None
        st.session_state.processing_turn = False
        st.rerun()

    asked_slot = _get_latest_asked_slot()

    if (
        asked_slot is not None
        and not st.session_state.chat_ended
        and not st.session_state.processing_turn
    ):
        if asked_slot == "pain_score":
            st.markdown("**Select pain level:**")
            c0, c1, c2, c3 = st.columns(4)

            if c0.button("None", key="quick_pain_none", use_container_width=True):
                st.session_state.quick_reply = "pain 0"
                st.rerun()

            if c1.button("Mild (1)", key="quick_pain_1", use_container_width=True):
                st.session_state.quick_reply = "pain 1"
                st.rerun()

            if c2.button("Moderate (2)", key="quick_pain_2", use_container_width=True):
                st.session_state.quick_reply = "pain 2"
                st.rerun()

            if c3.button("Severe (3)", key="quick_pain_3", use_container_width=True):
                st.session_state.quick_reply = "pain 3"
                st.rerun()

        elif asked_slot == "swelling_score":
            st.markdown("**Select swelling level:**")
            c0, c1, c2, c3 = st.columns(4)

            if c0.button("None", key="quick_swell_none", use_container_width=True):
                st.session_state.quick_reply = "swelling 0"
                st.rerun()

            if c1.button("Mild (1)", key="quick_swell_1", use_container_width=True):
                st.session_state.quick_reply = "swelling 1"
                st.rerun()

            if c2.button("Moderate (2)", key="quick_swell_2", use_container_width=True):
                st.session_state.quick_reply = "swelling 2"
                st.rerun()

            if c3.button("Severe (3)", key="quick_swell_3", use_container_width=True):
                st.session_state.quick_reply = "swelling 3"
                st.rerun()

    chat_placeholder = (
        "Type your message here"
        if not st.session_state.chat_ended
        else "Chat ended. Use 'End conversation / Restart' or 'Clear chat only'."
    )

    user_text = st.chat_input(
        chat_placeholder,
        disabled=st.session_state.chat_ended or st.session_state.processing_turn,
        key=f"chat_input_{st.session_state.chat_input_epoch}",
    )

    if st.session_state.quick_reply:
        user_text = st.session_state.quick_reply
        st.session_state.quick_reply = None

    if user_text:
        st.session_state.chat.append(
            {"role": "user", "text": user_text, "result": None}
        )
        st.session_state.chat.append(
            {"role": "assistant", "text": "Thinking…", "result": None}
        )

        st.session_state.pending_user_text = user_text
        st.session_state.processing_turn = True
        st.rerun()