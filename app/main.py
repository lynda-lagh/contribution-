"""
Human review queue for the abstained set.

    streamlit run app/main.py

Your pipeline produces two outputs: accepted -> KG, and abstained -> ?
This app is that question mark. It turns abstention from a refusal into a workflow,
which is what Tsaneva's disagreement routing measures: <13% of triples escalated,
and all three metrics improve simultaneously -- the only workflow in the corpus
that improves precision, recall and F1 at once.

⚠️ RUNS OVER CACHED RESULTS, NOT LIVE INFERENCE.
Pre-compute everything into results/; the app only reads. Instant, and it cannot
crash during a defence.

Screens, in priority order:
    1. Review queue      <- the core; everything else is optional
    2. Graph evidence    <- what the decision actually used (see graph_backend.py)
    3. Risk-coverage     <- the number no paper in the corpus reports
    4. Calibration       <- what makes the confidence in screen 1 believable
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="KGC Review Queue", layout="wide",
                   initial_sidebar_state="expanded")

RESULTS = Path("results")
DATA = Path("data")


# ------------------------------------------------------------------ loading
@st.cache_data
def load_json(p: str) -> dict | None:
    f = Path(p)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


@st.cache_resource
def load_graph(dataset: str, backend_kind: str = "memory"):
    from app.graph_backend import make_backend
    from src.data.loaders import load_kg
    kg = load_kg(dataset, str(DATA))
    return make_backend(kg, backend_kind), kg


@st.cache_resource
def load_features(dataset: str):
    from src.data.loaders import load_kg
    from src.routing.features import compute_features
    return compute_features(load_kg(dataset, str(DATA)))


# ------------------------------------------------------------------ graph
def render_graph(sub, height: int = 430) -> None:
    """pyvis -> HTML -> embedded. No Neo4j required for a 1-2 hop neighbourhood."""
    try:
        from pyvis.network import Network
    except ImportError:
        st.info("`pip install pyvis` to enable the graph panel.")
        st.json({"nodes": len(sub.nodes), "edges": len(sub.edges)})
        return

    colours = {"focus": "#1f77b4", "gold": "#2ca02c", "prediction_ok": "#ff7f0e",
               "type_violation": "#d62728", "oov": "#7f2704", "context": "#c7c7c7"}
    net = Network(height=f"{height}px", width="100%", directed=True,
                  bgcolor="#ffffff", font_color="#222222")
    net.barnes_hut(gravity=-4000, spring_length=140)
    for n in sub.nodes:
        g = n["group"]
        net.add_node(n["id"], label=n["label"], title=n.get("title", ""),
                     color=colours.get(g, "#c7c7c7"),
                     size=26 if g in ("focus", "gold") else 16,
                     borderWidth=3 if g in ("type_violation", "oov") else 1)
    for e in sub.edges:
        net.add_edge(e["source"], e["target"], label=e.get("label", ""),
                     color=colours.get(e.get("group", "context"), "#cccccc"),
                     width=3 if e.get("group") != "context" else 1)
    st.components.v1.html(net.generate_html(notebook=False), height=height + 20)


# ------------------------------------------------------------------ screens
def screen_queue(cfg) -> None:
    st.header("Review queue")

    abst = load_json(str(RESULTS / cfg["run"] / "abstention.json"))
    if not abst:
        st.warning(f"No abstention.json under results/{cfg['run']}/. "
                   "Run `python -m chapters.ch4_measurement.run` first.")
        return

    ops = abst.get("operating_points", {})
    if not ops:
        st.error("No usable operating point reached the precision targets.")
        st.json(abst.get("unreachable_targets", {}))
        return

    key = st.selectbox("Operating point", list(ops))
    op = ops[key]

    c = st.columns(5)
    c[0].metric("Coverage", f"{op['coverage']:.1%}")
    c[1].metric("Selective acc.", f"{op['selective_accuracy']:.3f}",
                f"{op['accuracy_gain']:+.3f} vs forced")
    c[2].metric("In queue", op["abstained"])
    c[3].metric("Errors avoided", op["errors_avoided"])
    c[4].metric("Threshold", f"{op['threshold']:.3f}")

    st.caption("Tsaneva: disagreement routing escalates <13% of triples and improves "
               "precision, recall and F1 simultaneously — the only workflow in the "
               "corpus that improves all three.")
    st.divider()

    queue = op.get("queue_sample", [])
    if not queue:
        st.info("No queue sample stored. Pass `explanations=` to `full_report()`.")
        return

    idx = st.number_input("Item", 0, len(queue) - 1, 0)
    item = queue[idx]

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Model output")
        st.code(item.get("reason", ""), language=None)
        st.metric("Confidence", f"{item['confidence']:.3f}")
        st.markdown("**Decision**")
        b = st.columns(4)
        for i, lbl in enumerate(["Accept top", "Accept alt.", "Reject all",
                                 "Not enough info"]):
            if b[i].button(lbl, key=f"{idx}-{lbl}", use_container_width=True):
                st.success(f"Logged: {lbl}")
                st.caption("Every action is logged — this log IS the human-study "
                           "dataset (accept/reject, time-per-item, whether the "
                           "reason was used).")

    with right:
        st.subheader("Audit panel")
        st.markdown("**Abstention reason** — trace 2")
        st.info(item.get("reason", "—"))
        hall = load_json(str(RESULTS / cfg["run"] / "hallucination.json"))
        if hall:
            st.markdown("**Type check** — trace 3")
            m = st.columns(2)
            m[0].metric("type-1 OOV", f"{hall['type1_oov_rate']:.1%}",
                        help="GS-KGC's WN18RR baseline: 38.9–45.3%")
            m[1].metric("★ type-2", f"{hall['type2_rate']:.1%}",
                        help="EGIT defined it; never measured before. MKGL cannot prevent it.")


def screen_graph(cfg) -> None:
    st.header("Graph evidence")
    st.caption("⚠️ This shows the evidence a decision **used** — neighbourhood, type "
               "constraints, candidate positions. It is not a search animation: the "
               "model is generative and does not traverse the graph.")

    try:
        backend, kg = load_graph(cfg["dataset"], cfg["backend"])
    except Exception as e:
        st.error(f"Could not load graph: {e}")
        return

    from app.graph_backend import evidence_subgraph

    ents = list(kg.ent2txt)[:3000]
    head = st.selectbox("Head entity", ents,
                        format_func=lambda e: f"{kg.ent2txt.get(e, e)[:60]}  ({e[:18]})")
    rel = st.selectbox("Relation", list(kg.rel2txt),
                       format_func=lambda r: kg.rel2txt.get(r, r))
    preds = [p.strip() for p in
             st.text_input("Sampled predictions (comma-separated)", "").split(",") if p.strip()]
    hops = st.slider("Hops", 1, 2, 1)

    sub, stats = evidence_subgraph(backend, head, rel, preds, hops=hops)

    m = st.columns(5)
    m[0].metric("Head degree", stats["head_degree"])
    m[1].metric("Range size", stats["range_size"])
    m[2].metric("Type-valid", stats["in_range"])
    m[3].metric("★ Violations", stats["type_violation"])
    m[4].metric("OOV", stats["oov"])

    render_graph(sub)
    st.caption("🔵 focus  🟢 gold  🟠 type-valid prediction  🔴 type violation (type-2)  "
               "🟤 not a KG entity (type-1)  ⚪ context")


def screen_faithfulness(cfg) -> None:
    st.header("Test this reason")
    st.caption("The router says feature X caused the decision. Change X, re-run, and "
               "see whether the decision flips. Possible only because the explanations "
               "are feature attributions rather than prose — 106/188 papers claim "
               "explainability, ~1 evaluates it, and none measures faithfulness.")

    try:
        feats = load_features(cfg["dataset"])
    except Exception as e:
        st.error(f"Could not compute features: {e}")
        return

    from src.routing.faithfulness import test_one
    from src.routing.router import Router

    level = st.selectbox("Ladder level", ["L1", "L2", "L3", "L4"], index=2)
    eid = st.selectbox("Element", list(feats)[:3000])
    f = feats[eid]

    c = st.columns(4)
    c[0].metric("Quality band", f.quality_band)
    c[1].metric("Semantic type", f.semantic_type[:16])
    c[2].metric("Ambiguity", f.ambiguity)
    c[3].metric("Degree", f.degree)

    r = test_one(f, Router(level))
    st.markdown(f"**Action** `{r['action']}`")
    st.info(r["reason"])

    if st.button("★ Test this reason", type="primary"):
        if not r["per_feature"]:
            st.warning("This reason cites no ablatable feature.")
        for feat, res in r["per_feature"].items():
            (st.success if res["flipped"] else st.error)(
                f"change **{feat}** → `{res['action_after']}` "
                f"{'✓ FLIPPED — the reason was operative' if res['flipped'] else '✗ unchanged — the reason was not operative'}")
        st.metric("Verdict", r["verdict"])


def screen_curves(cfg) -> None:
    st.header("Risk–coverage & calibration")
    import pandas as pd

    abst = load_json(str(RESULTS / cfg["run"] / "abstention.json"))
    calib = load_json(str(RESULTS / cfg["run"] / "calibration.json"))

    if abst:
        st.subheader("Risk–coverage")
        st.caption("Verified 0 of 188 papers report this curve.")
        df = pd.DataFrame(abst["curve_points"])
        st.line_chart(df.set_index("coverage")[["precision", "risk"]])
        rc = abst["risk_coverage"]
        c = st.columns(3)
        c[0].metric("AURC", f"{rc['aurc']:.4f}")
        c[1].metric("vs random", f"{rc['aurc_vs_random']:+.4f}")
        c[2].metric("Signal?", "yes" if rc["has_signal"] else "no")
        st.info(abst.get("headline", ""))
        if not rc["has_signal"]:
            st.warning(rc["note"])

    if calib:
        st.subheader("Calibration")
        st.dataframe(pd.DataFrame(calib["ranking"]), use_container_width=True)
        src = st.selectbox("Confidence source", list(calib["sources"]))
        s = calib["sources"][src]
        c = st.columns(4)
        c[0].metric("ECE (raw)", f"{s['uncalibrated']['ece']:.4f}")
        c[1].metric("ECE (best)", f"{s['best_ece']:.4f}", s["best_method"])
        c[2].metric("Brier", f"{s['uncalibrated']['brier']:.4f}")
        c[3].metric("Direction", s["direction"])
        rel = pd.DataFrame(s["uncalibrated"]["reliability"])
        if not rel.empty:
            st.caption("Reliability — in this confidence bin, X% are actually correct.")
            st.bar_chart(rel.set_index("confidence")[["accuracy"]])


# ------------------------------------------------------------------ main
def main() -> None:
    st.sidebar.title("KGC Review Queue")
    cfg = {
        "dataset": st.sidebar.selectbox("Dataset", ["YAGO3-10", "WN18RR", "FB15k-237",
                                                    "WN11", "FB13"]),
        "run": st.sidebar.text_input("Results dir", "ch4_ch2-lora-E123182-T10000-s42"),
        "backend": st.sidebar.radio("Graph backend", ["memory", "neo4j"], index=0,
                                    help="memory reads the TSVs — no server. Use neo4j "
                                         "only if your supervisor requires it."),
    }
    st.sidebar.divider()
    st.sidebar.caption("Runs over cached results. No live inference — instant, and it "
                       "cannot crash during a defence.")

    tabs = st.tabs(["Review queue", "Graph evidence", "Test this reason",
                    "Risk–coverage & calibration"])
    with tabs[0]:
        screen_queue(cfg)
    with tabs[1]:
        screen_graph(cfg)
    with tabs[2]:
        screen_faithfulness(cfg)
    with tabs[3]:
        screen_curves(cfg)


if __name__ == "__main__":
    main()
