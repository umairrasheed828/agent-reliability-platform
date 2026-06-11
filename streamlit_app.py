import streamlit as st

from src.drift.engine import detect_drift
from src.store.metrics_store import all_runs, count_runs, runs_by_cohort

st.set_page_config(page_title="Agent Reliability Platform", layout="wide")
st.title("Agent Reliability Platform")
st.caption("Continuous evaluation and drift monitoring for the multi-agent analyst")

total = count_runs()
if total == 0:
    st.warning(
        "No runs yet. Seed demo data with `uv run python -m scripts.seed_demo`, "
        "or run probes against the live agent."
    )
    st.stop()

baseline = runs_by_cohort("baseline")
recent = runs_by_cohort("live")
if not baseline or not recent:  # fallback if data isn't cohort-tagged
    runs = all_runs()
    half = max(1, len(runs) // 2)
    baseline, recent = runs[:half], runs[half:]

report = detect_drift(baseline, recent)

if report.alert:
    st.error(
        "DRIFT ALERT — recent quality has regressed or its distribution has shifted."
    )
else:
    st.success("Stable — recent quality matches the baseline.")


def _mean(runs: list, axis: str) -> float:
    return sum(getattr(r, axis) for r in runs) / len(runs) if runs else 0.0


col1, col2, col3 = st.columns(3)
col1.metric("Total runs", total)
col2.metric(
    "Faithfulness (recent)",
    f"{_mean(recent, 'faithfulness'):.2f}",
    delta=f"{_mean(recent, 'faithfulness') - _mean(baseline, 'faithfulness'):+.2f}",
)
col3.metric(
    "Relevance (recent)",
    f"{_mean(recent, 'relevance'):.2f}",
    delta=f"{_mean(recent, 'relevance') - _mean(baseline, 'relevance'):+.2f}",
)

st.subheader("Drift verdict (baseline vs recent)")
st.code(report.summary())

st.subheader("Quality over time")
ordered = sorted(all_runs(), key=lambda r: r.timestamp)
st.line_chart(
    {
        "faithfulness": [r.faithfulness for r in ordered],
        "relevance": [r.relevance for r in ordered],
    }
)

st.subheader("Recent runs")
st.dataframe(
    [
        {
            "time": r.timestamp,
            "cohort": r.cohort,
            "question": r.question[:50],
            "faithfulness": r.faithfulness,
            "relevance": r.relevance,
        }
        for r in ordered[-15:]
    ]
)
