import json
import streamlit as st

from wellnessbot.pipeline.run import run_pipeline

st.set_page_config(page_title="Knee Rehab Decision Support (v2)", layout="centered")
st.title("Knee Rehab Decision Support (Decision-first, Auditable)")

default_text = "3 weeks after ACL surgery, pain 4/10, mild swelling, want to do squats. No fever."
user_text = st.text_area("Describe your situation and requested exercise:", value=default_text, height=130)

col1, col2 = st.columns(2)
with col1:
    mock = st.toggle("MOCK_NLU", value=True, help="Mock-first. OpenAI optional later.")
with col2:
    st.caption("Safety: not diagnosis. Outputs: RECOMMEND / FORBID / CLARIFY / ESCALATE.")

if st.button("Run decision", type="primary"):
    result = run_pipeline(user_text=user_text, force_mock_nlu=mock)

    st.subheader("Final Decision")
    st.write(f"**{result['decision']['action']}**")
    st.write("nlu_source:", f"`{result['nlu']['nlu_source']}`")

    st.subheader("Output JSON")
    st.code(json.dumps(result, indent=2), language="json")

    with st.expander("Audit Trace (rules, citations, planner)"):
        st.code(json.dumps(result["audit_trace"], indent=2), language="json")