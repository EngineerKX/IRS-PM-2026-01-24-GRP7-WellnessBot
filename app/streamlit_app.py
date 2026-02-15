import json
import streamlit as st

from wellnessbot.pipeline.run import run_pipeline

st.set_page_config(page_title="Knee Rehab Decision Support (v2)", layout="centered")
st.title("Knee Rehab Decision Support")
st.caption("Decision brain = Rules + KG + Planner. LLM (optional) is NLU only. Not medical advice.")

# --- Session state ---
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of dicts: {"role": "user"/"assistant", "text": str, "result": dict|None}

if "mock_toggle" not in st.session_state:
    st.session_state.mock_toggle = False  # default OpenAI

# --- Controls (top bar) ---
with st.container():
    col1, col2 = st.columns([1, 1])
    with col1:
        st.session_state.mock_toggle = st.toggle(
            "MOCK_NLU",
            value=st.session_state.mock_toggle,
            help="ON = deterministic mock. OFF = try OpenAI then fallback to mock on error.",
        )
    with col2:
        if st.button("Clear chat"):
            st.session_state.chat = []
            st.rerun()

st.divider()

# --- Render chat history ---
for i, msg in enumerate(st.session_state.chat):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["text"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["text"])

            result = msg.get("result")
            if result:
                # Compact “status line”
                action = result["decision"]["action"]
                nlu_source = result["nlu"]["nlu_source"]
                conf = result["decision"]["confidence"]

                st.caption(f"action: **{action}** · confidence: **{conf:.2f}** · nlu_source: `{nlu_source}`")

                with st.expander("Show NLU JSON"):
                    st.code(json.dumps(result["nlu"], indent=2), language="json")

                with st.expander("Show Audit Trace (rules, citations, planner)"):
                    st.code(json.dumps(result["audit_trace"], indent=2), language="json")

st.divider()

# --- Input box (chat style) ---
user_text = st.chat_input("Type your message (e.g., '3 weeks after ACL surgery, pain 4/10, want squats')")

if user_text:
    # 1) add user message
    st.session_state.chat.append({"role": "user", "text": user_text, "result": None})

    # 2) run pipeline (thin UI – no decision logic here)
    result = run_pipeline(user_text=user_text, force_mock_nlu=st.session_state.mock_toggle)

    # 3) craft a concise assistant message (no decision changes)
    action = result["decision"]["action"]

    # Select rule matching final action
    rules = result["audit_trace"].get("rules_fired", [])
    top_rule = next((r for r in rules if r.get("action") == action), None)
    if top_rule is None and rules:
        top_rule = rules[0]

    rationale = top_rule.get("rationale") if top_rule else "Decision generated."

    # Rule IDs (traceability)
    rule_ids = result["decision"].get("rule_ids", [])
    rule_line = f"\n\nRule IDs: {', '.join(rule_ids)}" if rule_ids else ""

    # Citations
    citations = result["decision"].get("citations", [])
    cite_str = ", ".join(citations[:3]) + (" ..." if len(citations) > 3 else "")
    cite_line = f"\n\nSources: {cite_str}" if citations else ""

    assistant_text = f"**{action}**\n\n{rationale}{rule_line}{cite_line}"

    st.session_state.chat.append({"role": "assistant", "text": assistant_text, "result": result})

    st.rerun()