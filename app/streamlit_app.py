from __future__ import annotations

import json
import hashlib
import streamlit as st

from wellnessbot.pipeline.run import run_pipeline
from wellnessbot.logging.logger import log_feedback
from wellnessbot.storage.profile_store import (
    create_profile,
    load_profile,
    list_profiles,
    save_profile,
    delete_profile,
)


def make_interaction_id(user_text: str, audit_ts: str) -> str:
    return hashlib.sha256(f"{audit_ts}|{user_text}".encode("utf-8")).hexdigest()[:16]


def build_welcome_message(profile: dict | None = None) -> dict:
    profile = profile or {}

    event_type = profile.get("event_type", "unknown")
    surgery_date = profile.get("surgery_date", "")

    if event_type in (None, "", "unknown"):
        slot_name = "event_type"
        question = "What surgery or injury did you have? (ACL surgery / TKR / meniscus / sprain)"
    elif not surgery_date:
        slot_name = "surgery_date"
        question = "When was your surgery or injury? Please tell me the date (YYYY-MM-DD) or how many weeks/days ago."
    else:
        slot_name = "symptom_screen"
        question = "Are you having any symptoms today, such as fever, excessive bleeding, unusual swelling, or pain? If none, just say 'none'."

    return {
        "role": "assistant",
        "text": (
            "👋 Welcome to the Knee Rehab Decision Support assistant.\n\n"
            "I can help suggest suitable rehabilitation exercises based on your recovery stage.\n\n"
            f"{question}"
        ),
        "result": {
            "mode": "clarify",
            "slot_name": slot_name,
            "nlu_turn": {},
            "audit_trace": {
                "mode": "clarify",
                "asked_slot": slot_name,
                "notes": ["Initial system greeting and first question."],
            },
        },
    }


def _profile_to_conv_state(profile: dict) -> dict:
    return {
        "event_type": profile.get("event_type", "unknown"),
        "surgery_date": profile.get("surgery_date", ""),
        "equipment_available": profile.get("equipment_available", []) or [],
        "exercise_history": profile.get("exercise_history", []) or [],
    }


def _save_current_profile() -> None:
    profile_id = (st.session_state.get("profile_id") or "").strip()
    if not profile_id:
        return

    conv = st.session_state.conv_state or {}

    profile = {
        "profile_id": profile_id,
        "display_name": st.session_state.get("display_name", ""),
        "event_type": conv.get("event_type", "unknown"),
        "surgery_date": conv.get("surgery_date", ""),
        "equipment_available": conv.get("equipment_available", []) or [],
        "exercise_history": conv.get("exercise_history", []) or [],
    }
    save_profile(profile)


st.set_page_config(page_title="Knee Rehab Decision Support (v2)", layout="centered")
st.title("Knee Rehab Decision Support")
st.caption("Decision brain = Rules + KG + Planner. LLM (optional) is NLU only. Not medical advice.")

# --- Session state bootstrap ---
if "conv_state" not in st.session_state:
    st.session_state.conv_state = {}

if "profile_id" not in st.session_state:
    st.session_state.profile_id = ""

if "display_name" not in st.session_state:
    st.session_state.display_name = ""

if "mock_toggle" not in st.session_state:
    st.session_state.mock_toggle = False

if "pending" not in st.session_state:
    st.session_state.pending = None

if "feedback_state" not in st.session_state:
    st.session_state.feedback_state = {}

if "equipment_available" not in st.session_state:
    st.session_state.equipment_available = (
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

# --- Sidebar: Profile Memory ---
st.sidebar.header("Profile Memory")

existing_profiles = list_profiles()

profile_options = ["-- Select a profile --"] + existing_profiles

default_idx = 0
if (
    st.session_state.get("profile_id")
    and st.session_state.profile_id in existing_profiles
):
    default_idx = 1 + existing_profiles.index(st.session_state.profile_id)

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
            st.session_state.conv_state = _profile_to_conv_state(profile)
            st.session_state.equipment_available = profile.get("equipment_available", []) or []
            st.session_state.chat = [build_welcome_message(st.session_state.conv_state)]
            st.session_state.pending = None
            st.session_state.feedback_state = {}
            st.session_state.chat_ended = False
            st.session_state.profile_loaded = True
            st.session_state.show_create_profile = False
            st.success(f"Loaded profile: {profile['profile_id']}")
            st.rerun()
    except Exception as e:
        st.error(f"Could not load profile: {e}")

col_p1, col_p2 = st.sidebar.columns(2)

with col_p1:
    if st.button("Create new profile"):
        st.session_state.show_create_profile = not st.session_state.show_create_profile
        st.rerun()

with col_p2:
    if st.button("Delete profile"):
        try:
            active_profile_id = (st.session_state.get("profile_id") or "").strip()

            if not active_profile_id:
                st.warning("No active profile to delete.")
            else:
                delete_profile(active_profile_id)

                st.session_state.profile_id = ""
                st.session_state.display_name = ""
                st.session_state.conv_state = {}
                st.session_state.equipment_available = []
                st.session_state.chat = []
                st.session_state.pending = None
                st.session_state.feedback_state = {}
                st.session_state.chat_ended = False
                st.session_state.profile_loaded = False
                st.session_state.show_create_profile = False

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
                st.session_state.conv_state = _profile_to_conv_state(profile)
                st.session_state.equipment_available = profile.get("equipment_available", []) or []
                st.session_state.chat = [build_welcome_message(st.session_state.conv_state)]
                st.session_state.pending = None
                st.session_state.feedback_state = {}
                st.session_state.chat_ended = False
                st.session_state.profile_loaded = True
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

# --- Sidebar: Core Profile ---
st.sidebar.header("Core Profile")

profile = st.session_state.conv_state

event_type_options = ["unknown", "acl_surgery", "tkr", "meniscus", "sprain"]

current_event_type = profile.get("event_type", "unknown")
if current_event_type not in event_type_options:
    current_event_type = "unknown"

edited_event_type = st.sidebar.selectbox(
    "Event type",
    options=event_type_options,
    index=event_type_options.index(current_event_type),
    key="editable_event_type",
    disabled=not st.session_state.profile_loaded,
)

edited_surgery_date = st.sidebar.text_input(
    "Surgery date (YYYY-MM-DD)",
    value=profile.get("surgery_date", ""),
    key="editable_surgery_date",
    disabled=not st.session_state.profile_loaded,
)

if st.sidebar.button("Save core profile", disabled=not st.session_state.profile_loaded):
    st.session_state.conv_state["event_type"] = edited_event_type
    st.session_state.conv_state["surgery_date"] = edited_surgery_date.strip()
    _save_current_profile()
    st.session_state.chat = [build_welcome_message(st.session_state.conv_state)]
    st.session_state.pending = None
    st.session_state.feedback_state = {}
    st.session_state.chat_ended = False
    st.session_state.profile_loaded = True
    st.success("Core profile updated.")
    st.rerun()

tool_options = [
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

selected_tools = []
st.sidebar.markdown("**Available tools/equipment**")
for tool in tool_options:
    checked = st.sidebar.checkbox(
        tool.replace("_", " ").title(),
        value=tool in st.session_state.equipment_available,
        key=f"tool_{tool}",
        disabled=not st.session_state.profile_loaded,
    )
    if checked:
        selected_tools.append(tool)

if st.session_state.profile_loaded:
    st.session_state.equipment_available = selected_tools
    st.session_state.conv_state["equipment_available"] = selected_tools
    _save_current_profile()

# --- Controls (top bar) ---
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
            preserved_profile = {
                "event_type": st.session_state.conv_state.get("event_type", "unknown"),
                "surgery_date": st.session_state.conv_state.get("surgery_date", ""),
                "equipment_available": st.session_state.equipment_available,
                "exercise_history": st.session_state.conv_state.get("exercise_history", []),
            }
            st.session_state.chat = [build_welcome_message(preserved_profile)]
            st.session_state.pending = None
            st.session_state.feedback_state = {}
            st.session_state.conv_state = preserved_profile
            st.session_state.chat_ended = False
            st.session_state.profile_loaded = bool(st.session_state.get("profile_id"))
            _save_current_profile()
            st.rerun()

    with col3:
        if st.button("Clear chat only", disabled=not st.session_state.profile_loaded):
            preserved_profile = {
                "event_type": st.session_state.conv_state.get("event_type", "unknown"),
                "surgery_date": st.session_state.conv_state.get("surgery_date", ""),
                "equipment_available": st.session_state.equipment_available,
                "exercise_history": st.session_state.conv_state.get("exercise_history", []),
            }
            st.session_state.chat = [build_welcome_message(preserved_profile)]
            st.session_state.pending = None
            st.session_state.feedback_state = {}
            st.session_state.conv_state = preserved_profile
            st.session_state.chat_ended = False
            st.session_state.profile_loaded = bool(st.session_state.get("profile_id"))
            _save_current_profile()
            st.rerun()

st.divider()

# --- Render chat history ---
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
                options=["UNKNOWN", "RECOMMEND", "FORBID", "CLARIFY", "ESCALATE", "SUPPORTIVE_CARE"],
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

# --- Pending compute ---
if st.session_state.pending:
    pending_text = st.session_state.pending
    st.session_state.pending = None

    result = run_pipeline(
        user_text=pending_text,
        conv_state=st.session_state.conv_state,
        force_mock_nlu=st.session_state.mock_toggle,
    )

    if "conv_state" in result:
        st.session_state.conv_state = result["conv_state"]
        st.session_state.equipment_available = (
            st.session_state.conv_state.get("equipment_available", [])
            or st.session_state.equipment_available
        )
        _save_current_profile()

    mode = result.get("mode", "final")

    if mode == "final":
        terminal_actions = {"RECOMMEND", "SUPPORTIVE_CARE", "ESCALATE", "FORBID"}
        action = result.get("decision", {}).get("action")
        if action in terminal_actions:
            st.session_state.chat_ended = True

        planner = (result.get("audit_trace") or {}).get("planner") or {}
        ex_name = planner.get("exercise_name") or planner.get("exercise_id")

        if ex_name:
            history = st.session_state.conv_state.get("exercise_history", []) or []
            history.append(
                {
                    "timestamp_utc": (result.get("audit_trace") or {}).get("timestamp_utc"),
                    "exercise_id": ex_name,
                    "phase_id": planner.get("phase_id"),
                    "status": "recommended",
                    "pain_score": st.session_state.conv_state.get("pain_score"),
                    "swelling_level": st.session_state.conv_state.get("swelling_level"),
                    "action": result.get("decision", {}).get("action"),
                }
            )
            st.session_state.conv_state["exercise_history"] = history
            _save_current_profile()

    if mode == "clarify":
        assistant_text = result.get("question", "I need a bit more information.")
    else:
        action = result["decision"]["action"]

        rules = result["audit_trace"].get("rules_fired", [])
        top_rule = next((r for r in rules if r.get("action") == action), None)
        if top_rule is None and rules:
            top_rule = rules[0]
        rationale = top_rule.get("rationale") if top_rule else "Decision generated."

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

        assistant_text = f"**{action}**\n\n{rationale}{planner_line}{rule_line}{cite_line}"

        terminal_actions = {"RECOMMEND", "SUPPORTIVE_CARE", "ESCALATE", "FORBID"}
        if action in terminal_actions:
            assistant_text += (
                "\n\n---\n\n"
                "✅ **Chat ended.**\n\n"
                "Please click **End conversation / Restart** or **Clear chat only** to start a new case."
            )

    if (
        st.session_state.chat
        and st.session_state.chat[-1]["role"] == "assistant"
        and st.session_state.chat[-1]["result"] is None
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

    st.rerun()

# --- Input box ---
if not st.session_state.profile_loaded:
    st.info("Please load a profile or create a new profile before starting the chat.")
elif st.session_state.chat_ended:
    st.info("This chat has ended. Please click **End conversation / Restart** or **Clear chat only** to begin a new case.")
else:
    user_text = st.chat_input("Type your message here")

    if user_text:
        st.session_state.chat.append({"role": "user", "text": user_text, "result": None})
        st.session_state.chat.append(
            {"role": "assistant", "text": "Thinking…", "result": None}
        )
        st.session_state.pending = user_text
        st.rerun()