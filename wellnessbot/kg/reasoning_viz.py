from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import networkx as nx

from wellnessbot.kg.kg import get_redflag_policies
from wellnessbot.rules.ruleset import (
    DOMINANCE,
    RULES,
    _current_phase,
    _match_redflag_policy,
    _match_selfcare_actions,
)


@dataclass
class ReasoningTrace:
    input_state: Dict[str, Any]
    inferred_phase: Optional[str]
    matched_rules: List[Dict[str, Any]]
    winning_rule: Optional[Dict[str, Any]]
    final_action: Optional[str]
    selfcare_actions: List[str]


def collect_reasoning_trace(nlu) -> ReasoningTrace:
    """
    Builds a reasoning trace focused on red-flag/supportive-care logic.
    This mirrors the current rule engine helpers.
    """
    inferred_phase = _current_phase(nlu)

    input_state = {
        "surgery_type": getattr(nlu, "surgery_type", None),
        "surgery_date": getattr(nlu, "surgery_date", ""),
        "weeks_since_event": getattr(nlu, "weeks_since_event", None),
        "pain_score": getattr(nlu, "pain_score", None),
        "swelling_score": getattr(nlu, "swelling_score", None),
        "symptom_flags": list(getattr(nlu, "symptom_flags", []) or []),
        "red_flag_terms": list(getattr(nlu, "red_flag_terms", []) or []),
    }

    matched_rules: List[Dict[str, Any]] = []

    if inferred_phase:
        policies = get_redflag_policies(nlu.surgery_type, inferred_phase)

        for p in policies:
            matched = False
            reasons = []

            if (
                p.symptom == "pain"
                and nlu.pain_score is not None
                and str(nlu.pain_score) == str(p.severity)
            ):
                matched = True
                reasons.append(f"pain_score={nlu.pain_score}")

            if (
                p.symptom == "swelling"
                and nlu.swelling_score is not None
                and str(nlu.swelling_score) == str(p.severity)
            ):
                matched = True
                reasons.append(f"swelling_score={nlu.swelling_score}")

            if p.symptom == "fever" and "fever" in (nlu.red_flag_terms or []):
                matched = True
                reasons.append("red_flag_terms contains fever")

            if p.symptom == "excessive_bleeding_or_wound_drainage":
                bleeding_terms = {"excessive bleeding", "wound drainage", "pus", "bleeding"}
                if any(t in (nlu.red_flag_terms or []) for t in bleeding_terms):
                    matched = True
                    reasons.append("red_flag_terms contains bleeding/drainage term")

            if matched:
                action_name = (
                    "SUPPORTIVE_CARE"
                    if p.action == "supportive_sequence"
                    else str(p.action).upper()
                )
                matched_rules.append(
                    {
                        "rule_id": p.redflag_id,
                        "phase_ids": list(getattr(p, "phase_ids", []) or []),
                        "symptom": p.symptom,
                        "severity": getattr(p, "severity", None),
                        "policy_action": p.action,
                        "engine_action": action_name,
                        "reasons": reasons,
                        "message": getattr(p, "message", None),
                        "action_steps": list(getattr(p, "action_steps", []) or []),
                    }
                )

    winning_policy = _match_redflag_policy(nlu)

    winning_rule = None
    final_action = None
    selfcare_texts: List[str] = []

    if winning_policy:
        for r in matched_rules:
            if r["rule_id"] == winning_policy.redflag_id:
                winning_rule = r
                break

        if winning_policy.action == "supportive_sequence":
            final_action = "SUPPORTIVE_CARE"
            actions = _match_selfcare_actions(nlu)
            for a in actions:
                selfcare_texts.append(
                    f"{a.care_type} for {a.duration_minutes} min ({a.frequency_condition})"
                )
        elif winning_policy.action == "escalate":
            final_action = "ESCALATE"
        else:
            final_action = str(winning_policy.action).upper()

    return ReasoningTrace(
        input_state=input_state,
        inferred_phase=inferred_phase,
        matched_rules=matched_rules,
        winning_rule=winning_rule,
        final_action=final_action,
        selfcare_actions=selfcare_texts,
    )


def build_reasoning_graph(trace: ReasoningTrace) -> nx.DiGraph:
    G = nx.DiGraph()

    for k, v in trace.input_state.items():
        label = f"{k}={v}"
        node_id = f"input:{k}"
        G.add_node(node_id, label=label, type="Input")

    if trace.inferred_phase:
        phase_node = f"phase:{trace.inferred_phase}"
        G.add_node(phase_node, label=trace.inferred_phase, type="Phase")
        G.add_edge("input:weeks_since_event", phase_node, relation="INFERS")

    for rule in trace.matched_rules:
        rule_node = f"rule:{rule['rule_id']}"
        G.add_node(rule_node, label=rule["rule_id"], type="Rule")

        if trace.inferred_phase:
            G.add_edge(phase_node, rule_node, relation="MATCHES_PHASE")

        if rule["symptom"] == "pain":
            G.add_edge("input:pain_score", rule_node, relation="MATCHES")
        elif rule["symptom"] == "swelling":
            G.add_edge("input:swelling_score", rule_node, relation="MATCHES")
        elif rule["symptom"] in ("fever", "excessive_bleeding_or_wound_drainage"):
            G.add_edge("input:red_flag_terms", rule_node, relation="MATCHES")

        if rule["severity"] is not None:
            sev_node = f"severity:{rule['severity']}"
            G.add_node(sev_node, label=f"severity_{rule['severity']}", type="Severity")
            G.add_edge(sev_node, rule_node, relation="QUALIFIES")

    if trace.winning_rule:
        win_rule_node = f"rule:{trace.winning_rule['rule_id']}"
        action_node = f"action:{trace.final_action}"
        G.add_node(action_node, label=trace.final_action, type="Action")
        G.add_edge(win_rule_node, action_node, relation="WINS")

        for i, text in enumerate(trace.selfcare_actions, start=1):
            care_node = f"selfcare:{i}"
            G.add_node(care_node, label=text, type="SelfCare")
            G.add_edge(action_node, care_node, relation="INCLUDES")

    return G


def draw_reasoning_graph(
    G: nx.DiGraph,
    trace: ReasoningTrace,
    figsize: Tuple[int, int] = (12, 7),
) -> None:
    plt.figure(figsize=figsize)

    pos: Dict[str, Tuple[float, float]] = {}

    input_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Input"]
    phase_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Phase"]
    sev_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Severity"]
    rule_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Rule"]
    action_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Action"]
    care_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "SelfCare"]

    def place(nodes: List[str], x: float, y_start: float = 0.0, y_gap: float = 1.5) -> None:
        if not nodes:
            return
        total_height = (len(nodes) - 1) * y_gap
        start_y = total_height / 2.0
        for i, n in enumerate(nodes):
            pos[n] = (x, start_y - i * y_gap)

    place(input_nodes, 0)
    place(phase_nodes + sev_nodes, 2)
    place(rule_nodes, 4)
    place(action_nodes, 6)
    place(care_nodes, 8)

    def shorten(text: str, max_len: int = 32) -> str:
        text = str(text)
        return text if len(text) <= max_len else text[:max_len] + "..."

    labels = {n: shorten(d.get("label", n)) for n, d in G.nodes(data=True)}

    node_colors = []
    node_sizes = []

    winning_rule_id = None
    if trace.winning_rule:
        winning_rule_id = f"rule:{trace.winning_rule['rule_id']}"

    for n, d in G.nodes(data=True):
        ntype = d.get("type", "Other")
        label = str(d.get("label", ""))

        if n == winning_rule_id:
            node_colors.append("#ff8c42")
            node_sizes.append(2800)
        elif ntype == "Rule":
            node_colors.append("#f5b041")
            node_sizes.append(2200)
        elif ntype == "Action":
            if "ESCALATE" in label:
                node_colors.append("#e74c3c")
            elif "SUPPORTIVE" in label:
                node_colors.append("#8e44ad")
            elif "FORBID" in label:
                node_colors.append("#c0392b")
            elif "CLARIFY" in label:
                node_colors.append("#f1c40f")
            elif "RECOMMEND" in label:
                node_colors.append("#3498db")
            else:
                node_colors.append("#8e44ad")
            node_sizes.append(2400)
        elif ntype == "Input":
            node_colors.append("#d9edf7")
            node_sizes.append(1800)
        elif ntype == "Phase":
            node_colors.append("#aed6f1")
            node_sizes.append(1900)
        elif ntype == "Severity":
            node_colors.append("#d5d8dc")
            node_sizes.append(1600)
        elif ntype == "SelfCare":
            node_colors.append("#58d68d")
            node_sizes.append(2000)
        else:
            node_colors.append("#cccccc")
            node_sizes.append(1800)

    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=9,
        arrows=True,
    )

    edge_labels = {}
    for u, v, d in G.edges(data=True):
        rel = d.get("relation", "")
        if rel in {"INFERS", "MATCHES", "WINS", "INCLUDES"}:
            edge_labels[(u, v)] = rel

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    title = "Reasoning Path Visualization"
    if trace.final_action:
        title += f" — Final Action: {trace.final_action}"

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show(block=True)


def print_reasoning_summary(trace: ReasoningTrace) -> None:
    print("\n=== INPUT STATE ===")
    for k, v in trace.input_state.items():
        print(f"{k}: {v}")

    print(f"\nInferred phase: {trace.inferred_phase}")

    print("\n=== MATCHED RULES ===")
    if not trace.matched_rules:
        print("No red-flag rules matched.")
    else:
        for r in trace.matched_rules:
            print(
                f"- {r['rule_id']} | symptom={r['symptom']} | severity={r['severity']} "
                f"| action={r['engine_action']} | reasons={'; '.join(r['reasons'])}"
            )

    print("\n=== WINNING RULE ===")
    if trace.winning_rule:
        print(trace.winning_rule["rule_id"])
    else:
        print("None")

    print("\n=== FINAL ACTION ===")
    print(trace.final_action)

    if trace.selfcare_actions:
        print("\n=== SELF-CARE ===")
        for x in trace.selfcare_actions:
            print(f"- {x}")


def collect_full_rule_trace(nlu):
    trace = []

    for rule_fn in RULES:
        name = rule_fn.__name__

        try:
            result = rule_fn(nlu)
        except Exception as e:
            trace.append(
                {
                    "rule": name,
                    "status": "ERROR",
                    "action": None,
                    "score": 0,
                    "error": str(e),
                }
            )
            continue

        if result is None:
            trace.append(
                {
                    "rule": name,
                    "status": "NOT_TRIGGERED",
                    "action": None,
                    "score": 0,
                }
            )
        else:
            score = DOMINANCE.get(result.action, 0)
            trace.append(
                {
                    "rule": name,
                    "status": "TRIGGERED",
                    "action": result.action.name,
                    "score": score,
                    "rule_id": result.rule_id,
                    "rationale": result.rationale,
                }
            )

    triggered = [t for t in trace if t["status"] == "TRIGGERED"]

    winner = None
    if triggered:
        winner = sorted(triggered, key=lambda x: x["score"], reverse=True)[0]

    return trace, winner


def build_full_trace_graph(trace, winner):
    G = nx.DiGraph()

    G.add_node("INPUT", label="User Input", type="Input", layer=0)

    for i, r in enumerate(trace):
        rule_node = f"rule_{i}"
        label = r["rule"]

        if r["status"] == "TRIGGERED":
            label += f"\n{r['action']}"
        elif r["status"] == "ERROR":
            label += "\nERROR"
        else:
            label += "\nNOT_TRIGGERED"

        G.add_node(
            rule_node,
            label=label,
            type="Rule",
            layer=1,
            status=r["status"],
        )
        G.add_edge("INPUT", rule_node, relation="EVALUATED")

        if r["status"] == "TRIGGERED":
            action_node = f"action_{i}"
            G.add_node(
                action_node,
                label=r["action"],
                type="Action",
                layer=2,
            )
            G.add_edge(rule_node, action_node, relation="PRODUCES")

            if winner and r["rule"] == winner["rule"]:
                G.add_node(
                    "FINAL",
                    label=f"FINAL: {winner['action']}",
                    type="Final",
                    layer=3,
                )
                G.add_edge(action_node, "FINAL", relation="WINNER")

    return G


def build_combined_kg_reasoning_graph(reasoning_trace, full_trace, winner):
    G = nx.DiGraph()

    if reasoning_trace.inferred_phase:
        phase_node = f"kg:phase:{reasoning_trace.inferred_phase}"
        G.add_node(phase_node, label=reasoning_trace.inferred_phase, type="Phase", layer="KG")

    if reasoning_trace.winning_rule:
        kg_rule = f"kg:rule:{reasoning_trace.winning_rule['rule_id']}"
        G.add_node(kg_rule, label=reasoning_trace.winning_rule["rule_id"], type="Rule", layer="KG")

        action_node = f"kg:action:{reasoning_trace.final_action}"
        G.add_node(action_node, label=reasoning_trace.final_action, type="Action", layer="KG")

        if reasoning_trace.inferred_phase:
            G.add_edge(phase_node, kg_rule, relation="APPLIES_TO")

        symptom = reasoning_trace.winning_rule.get("symptom")
        if symptom:
            symptom_node = f"kg:symptom:{symptom}"
            G.add_node(symptom_node, label=symptom, type="Symptom", layer="KG")
            G.add_edge(symptom_node, kg_rule, relation="TRIGGERS")

        severity = reasoning_trace.winning_rule.get("severity")
        if severity is not None:
            sev_node = f"kg:severity:{severity}"
            G.add_node(sev_node, label=f"severity_{severity}", type="Severity", layer="KG")
            G.add_edge(sev_node, kg_rule, relation="QUALIFIES")

        G.add_edge(kg_rule, action_node, relation="RESULTS_IN")

        for i, text in enumerate(reasoning_trace.selfcare_actions, start=1):
            care_node = f"kg:selfcare:{i}"
            G.add_node(care_node, label=text, type="SelfCare", layer="KG")
            if reasoning_trace.inferred_phase:
                G.add_edge(phase_node, care_node, relation="HAS_SELFCARE")

    G.add_node("trace:input", label="User Input", type="Input", layer="TRACE")

    for i, r in enumerate(full_trace):
        rule_node = f"trace:rule:{i}"
        label = r["rule"]

        if r["status"] == "TRIGGERED":
            if winner and r["rule"] != winner["rule"]:
                label += f"\\n{r['action']}\\nPriority={r['score']}\\nOVERRIDDEN"
            else:
                label += f"\\n{r['action']}\\nPriority={r['score']}"
        elif r["status"] == "ERROR":
            label += f"\\nERROR\\n{r.get('error', 'unknown')}"
        else:
            label += "\\nNOT_TRIGGERED"

        G.add_node(rule_node, label=label, type="TraceRule", layer="TRACE")
        G.add_edge("trace:input", rule_node, relation="EVALUATED")

        if r["status"] == "TRIGGERED":
            action_node = f"trace:action:{i}"
            G.add_node(action_node, label=r["action"], type="TraceAction", layer="TRACE")
            G.add_edge(rule_node, action_node, relation="PRODUCES")

            if winner and r["rule"] == winner["rule"]:
                G.add_node(
                    "trace:final",
                    label=f"FINAL: {winner['action']}",
                    type="Final",
                    layer="TRACE",
                )
                G.add_edge(action_node, "trace:final", relation="WINNER")

    if reasoning_trace.winning_rule and winner:
        G.add_edge(
            f"kg:rule:{reasoning_trace.winning_rule['rule_id']}",
            "trace:input",
            relation="GROUNDS",
        )

    return G


def draw_full_trace_graph(G, figsize=(16, 9)):
    import math
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.patches import Patch

    plt.figure(figsize=figsize)

    # -------------------------
    # Layered positions
    # -------------------------
    pos = {}

    input_nodes = [n for n, d in G.nodes(data=True) if d.get("layer") == 0]
    rule_nodes = [n for n, d in G.nodes(data=True) if d.get("layer") == 1]
    action_nodes = [n for n, d in G.nodes(data=True) if d.get("layer") == 2]
    final_nodes = [n for n, d in G.nodes(data=True) if d.get("layer") == 3]

    def place_vertical(nodes, x, y_gap=1.3):
        if not nodes:
            return
        total_height = (len(nodes) - 1) * y_gap
        start_y = total_height / 2
        for i, n in enumerate(nodes):
            pos[n] = (x, start_y - i * y_gap)

    place_vertical(input_nodes, 0, y_gap=1.3)
    place_vertical(rule_nodes, 3, y_gap=1.15)
    place_vertical(action_nodes, 6, y_gap=1.5)
    place_vertical(final_nodes, 9, y_gap=1.3)

    # -------------------------
    # Labels
    # -------------------------
    def clean_label(text):
        text = str(text).replace("_", " ")
        text = text.replace("rule ", "R: ")
        return text

    labels = {n: clean_label(d.get("label", n)) for n, d in G.nodes(data=True)}

    # -------------------------
    # Node colors and sizes
    # -------------------------
    node_colors = []
    node_sizes = []

    for node, data in G.nodes(data=True):
        label = str(data.get("label", ""))
        ntype = data.get("type", "")

        if ntype == "Final":
            node_colors.append("#FFD700")
            node_sizes.append(4200)

        elif ntype == "Input":
            node_colors.append("#5dade2")
            node_sizes.append(3600)

        elif ntype == "Rule":
            if "ERROR" in label:
                node_colors.append("#ffb3b3")
                node_sizes.append(2800)
            elif "NOT_TRIGGERED" in label:
                node_colors.append("#d3d3d3")
                node_sizes.append(2600)
            elif "ESCALATE" in label:
                node_colors.append("#e74c3c")
                node_sizes.append(3000)
            elif "FORBID" in label:
                node_colors.append("#f39c12")
                node_sizes.append(3000)
            elif "RECOMMEND" in label:
                node_colors.append("#3498db")
                node_sizes.append(3000)
            else:
                node_colors.append("#95a5a6")
                node_sizes.append(2600)

        elif ntype == "Action":
            if "ESCALATE" in label:
                node_colors.append("#e74c3c")
            elif "FORBID" in label:
                node_colors.append("#c0392b")
            elif "RECOMMEND" in label:
                node_colors.append("#3498db")
            else:
                node_colors.append("#8e44ad")
            node_sizes.append(2800)

        else:
            node_colors.append("#cccccc")
            node_sizes.append(2200)

    # -------------------------
    # Draw nodes and labels
    # -------------------------
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=8,
        font_weight="normal",
    )

    # -------------------------
    # Draw edges
    # -------------------------
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        width=1.2,
        min_source_margin=18,
        min_target_margin=18,
    )

    # -------------------------
    # Manual edge labels
    # -------------------------
    for u, v, d in G.edges(data=True):
        rel = d.get("relation")
        if rel not in {"EVALUATED", "PRODUCES", "WINNER"}:
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]

        if rel == "EVALUATED":
            t = 0.45
            offset = 0.18
            fs = 6
        elif rel == "PRODUCES":
            t = 0.45
            offset = 0.16
            fs = 6.5
        else:  # WINNER
            t = 0.50
            offset = 0.16
            fs = 6.5

        xm = x1 + t * (x2 - x1)
        ym = y1 + t * (y2 - y1)

        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)

        if length > 0:
            px = -dy / length
            py = dx / length
        else:
            px, py = 0.0, 0.0

        xm += offset * px
        ym += offset * py

        plt.text(
            xm,
            ym,
            rel,
            fontsize=fs,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.2),
            zorder=10,
        )

    # -------------------------
    # Column headers
    # -------------------------
    plt.text(0, max(y for _, y in pos.values()) + 1.0, "Input", fontsize=12, weight="bold", ha="center")
    plt.text(3, max(y for _, y in pos.values()) + 1.0, "Rules", fontsize=12, weight="bold", ha="center")
    plt.text(6, max(y for _, y in pos.values()) + 1.0, "Triggered Actions", fontsize=12, weight="bold", ha="center")
    plt.text(9, max(y for _, y in pos.values()) + 1.0, "Final Decision", fontsize=12, weight="bold", ha="center")

    # -------------------------
    # Legend
    # -------------------------
    legend_elements = [
        Patch(facecolor="#e74c3c", label="ESCALATE"),
        Patch(facecolor="#f39c12", label="FORBID"),
        Patch(facecolor="#3498db", label="RECOMMEND"),
        Patch(facecolor="#d3d3d3", label="Not Triggered"),
        Patch(facecolor="#ffb3b3", label="Error"),
        Patch(facecolor="#FFD700", label="Final Decision"),
    ]

    plt.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=9,
        frameon=True,
    )

    plt.title("Full Rule Execution Trace", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.show(block=True)

def draw_combined_kg_reasoning_graph(G, figsize=(16, 10)):
    plt.figure(figsize=figsize)
    pos = {}

    kg_nodes = [n for n, d in G.nodes(data=True) if d.get("layer") == "KG"]
    trace_nodes = [n for n, d in G.nodes(data=True) if d.get("layer") == "TRACE"]

    def place(nodes, x_start, y, x_gap=2.2):
        for i, n in enumerate(nodes):
            pos[n] = (x_start + i * x_gap, y)

    place(kg_nodes, 0, 3)
    place(trace_nodes, 0, -3)

    labels = {}
    for n, d in G.nodes(data=True):
        text = str(d.get("label", n))
        labels[n] = text if len(text) <= 28 else text[:28] + "..."

    node_colors = []
    node_sizes = []

    for n, d in G.nodes(data=True):
        t = d.get("type", "")
        label = str(d.get("label", ""))

        if t == "Phase":
            node_colors.append("#aed6f1")
            node_sizes.append(2200)
        elif t in ("Symptom", "Severity"):
            node_colors.append("#f5b7b1")
            node_sizes.append(2000)
        elif t == "Rule":
            node_colors.append("#f5b041")
            node_sizes.append(2200)
        elif t == "Action":
            node_colors.append("#8e44ad")
            node_sizes.append(2200)
        elif t == "SelfCare":
            node_colors.append("#58d68d")
            node_sizes.append(2200)
        elif t == "Input":
            node_colors.append("#5dade2")
            node_sizes.append(2400)
        elif t == "TraceRule":
            if "ERROR" in label:
                node_colors.append("#ffb3b3")
            elif "NOT_TRIGGERED" in label:
                node_colors.append("#d3d3d3")
            elif "OVERRIDDEN" in label:
                node_colors.append("#85c1e9")
            elif "Priority=3" in label or "Priority=4" in label:
                node_colors.append("#f39c12")
            else:
                node_colors.append("#3498db")
            node_sizes.append(2400)
        elif t == "TraceAction":
            node_colors.append("#8e44ad")
            node_sizes.append(2200)
        elif t == "Final":
            node_colors.append("#FFD700")
            node_sizes.append(2600)
        else:
            node_colors.append("#cccccc")
            node_sizes.append(1800)

    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=6,
        font_weight="normal",
        arrows=True,

        # 👇 ADD THESE
        arrowsize=10,              # smaller arrow head
        width=1.0,                # thinner line
        min_source_margin=15,     # shorten from source node
        min_target_margin=15,     # shorten before target node
    )

    edge_labels = {}
    for u, v, d in G.edges(data=True):
        rel = d.get("relation", "")
        if rel in {"RESULTS_IN", "TRIGGERS", "PRODUCES", "WINNER", "GROUNDS"}:
            edge_labels[(u, v)] = rel

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.text(0, 4.5, "Static Knowledge Graph Layer", fontsize=14, weight="bold")
    plt.text(0, -1.5, "Runtime Rule Execution Layer", fontsize=14, weight="bold")

    plt.axis("off")
    plt.tight_layout()
    plt.show(block=True)