import json
import hashlib
import streamlit as st

from wellnessbot.pipeline.run import run_pipeline
from wellnessbot.logging.logger import log_feedback


def make_interaction_id(user_text: str, audit_ts: str) -> str:
    return hashlib.sha256(f"{audit_ts}|{user_text}".encode("utf-8")).hexdigest()[:16]


st.set_page_config(page_title="Knee Rehab Decision Support (v2)", layout="centered")
st.title("Knee Rehab Decision Support")
st.caption("Decision brain = Rules + KG + Planner. LLM (optional) is NLU only. Not medical advice.")

# --- Session state ---
if "chat" not in st.session_state:
    st.session_state.chat = []  # {"role": "user"/"assistant", "text": str, "result": dict|None}

if "mock_toggle" not in st.session_state:
    st.session_state.mock_toggle = False  # default OpenAI

# NEW: conversation memory for looping chat
if "conv_state" not in st.session_state:
    st.session_state.conv_state = {}  # dict persisted across turns

# pending message support (Option A)
if "pending" not in st.session_state:
    st.session_state.pending = None

# store feedback UI state per message index
if "feedback_state" not in st.session_state:
    st.session_state.feedback_state = {}  # {msg_index: {"thumb", "comment", "expected_action", "submitted"}}

# --- Controls (top bar) ---
with st.container():
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.session_state.mock_toggle = st.toggle(
            "MOCK_NLU",
            value=st.session_state.mock_toggle,
            help="ON = deterministic mock. OFF = try OpenAI then fallback to mock on error.",
        )

    with col2:
        if st.button("End conversation / Restart"):
            st.session_state.chat = []
            st.session_state.pending = None
            st.session_state.feedback_state = {}
            st.session_state.conv_state = {}  # ✅ clears loop memory
            st.rerun()

    with col3:
        if st.button("Clear chat only"):
            st.session_state.chat = []
            st.session_state.pending = None
            st.session_state.feedback_state = {}
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

        # ========== CLARIFY MODE ==========
        if mode == "clarify":
            asked_slot = result.get("slot_name", "unknown")
            nlu_turn = result.get("nlu_turn", {})
            nlu_source = nlu_turn.get("nlu_source", "unknown")

            st.caption(f"mode: **clarify** · asked_slot: `{asked_slot}` · nlu_source: `{nlu_source}`")

            with st.expander("Show turn NLU JSON"):
                st.code(json.dumps(nlu_turn, indent=2), language="json")

            with st.expander("Show dialog audit"):
                st.code(json.dumps(result.get("audit_trace", {}), indent=2), language="json")

            # No feedback controls during clarify
            continue

        # ========== FINAL MODE ==========
        # Your existing display expects decision + nlu in final mode
        action = result["decision"]["action"]
        nlu_source = result["nlu"]["nlu_source"]
        conf = result["decision"]["confidence"]

        st.caption(f"mode: **final** · action: **{action}** · confidence: **{conf:.2f}** · nlu_source: `{nlu_source}`")

        # --- Rating controls (FINAL mode only) ---
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
                    expected_action=None if fb["expected_action"] == "UNKNOWN" else fb["expected_action"],
                )
                st.toast("Feedback saved (👎)")
                st.rerun()

        with st.expander("Show NLU JSON"):
            st.code(json.dumps(result["nlu"], indent=2), language="json")

        with st.expander("Show Audit Trace (rules, citations, planner)"):
            st.code(json.dumps(result["audit_trace"], indent=2), language="json")

st.divider()

# --- Pending compute (Option A) ---
if st.session_state.pending:
    pending_text = st.session_state.pending
    st.session_state.pending = None

    # ✅ NEW: pass conv_state in, get updated conv_state out
    result = run_pipeline(
        user_text=pending_text,
        conv_state=st.session_state.conv_state,
        force_mock_nlu=st.session_state.mock_toggle,
    )

    # ✅ NEW: always update conversation memory if provided
    if "conv_state" in result:
        st.session_state.conv_state = result["conv_state"]

    mode = result.get("mode", "final")

    # --- craft assistant bubble depending on mode ---
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
                dose = planner.get("dose", {})
                stop = planner.get("stop_conditions", [])
                planner_line = (
                    f"\n\n**Recommended exercise:** {ex_name}"
                    f"\n\nDose: {dose.get('sets')} sets × {dose.get('reps')} reps, {dose.get('frequency_per_day')}×/day"
                )
                if stop:
                    planner_line += "\n\nStop if:\n- " + "\n- ".join(stop)

        assistant_text = f"**{action}**\n\n{rationale}{planner_line}{rule_line}{cite_line}"

    # Replace placeholder “Thinking…”
    if st.session_state.chat and st.session_state.chat[-1]["role"] == "assistant" and st.session_state.chat[-1]["result"] is None:
        st.session_state.chat[-1] = {"role": "assistant", "text": assistant_text, "result": result}
    else:
        st.session_state.chat.append({"role": "assistant", "text": assistant_text, "result": result})

    st.rerun()

# --- Input box ---
user_text = st.chat_input("Type your message (e.g., '3 weeks after ACL surgery, pain 4/10, want squats')")

if user_text:
    st.session_state.chat.append({"role": "user", "text": user_text, "result": None})
    st.session_state.chat.append({"role": "assistant", "text": "Thinking…", "result": None})
    st.session_state.pending = user_text
    st.rerun()