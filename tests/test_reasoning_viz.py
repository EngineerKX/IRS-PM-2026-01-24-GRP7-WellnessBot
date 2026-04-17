from wellnessbot.nlu.schema import NLUOutput
from wellnessbot.kg.reasoning_viz import (
    collect_full_rule_trace,
    build_full_trace_graph,
    draw_full_trace_graph,
)

nlu = NLUOutput(
    weeks_since_event=1.86,
    surgery_type="arthroscopic_knee_surgery",
    surgery_date="2026-01-01",
    requested_exercise_text="",
    pain_score=0,
    swelling_score=0,
    weight_bearing="unknown",
    symptom_screen_done=True,
    symptom_flags=["none"],
    red_flag_terms=[],
    negated_terms=[],
    missing_fields=[],
    nlu_source="openai",
)

trace, winner = collect_full_rule_trace(nlu)

print("=== TRACE ===")
for t in trace:
    print(t)

print("\n=== WINNER ===")
print(winner)

G = build_full_trace_graph(trace, winner)

print("\n=== GRAPH INFO ===")
print("Nodes:", len(G.nodes()))
print("Edges:", len(G.edges()))

draw_full_trace_graph(G)