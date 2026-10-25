import os, glob
import pandas as pd
import streamlit as st

st.set_page_config(page_title="OS Tutor RAGAS Dashboard", page_icon="chart_with_upwards_trend", layout="wide")

EVAL_DIR = "eval"
MC = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference"]
ML = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "llm_context_precision_with_reference": "Context Precision",
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 1200px; padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("## RAGAS Evaluation Dashboard")
st.markdown("**Qwen2.5-1.5B + LoRA** vs **Groq llama-3.3-70b** — judged by Gemini 2.5 Flash")

# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load(p):
    return pd.read_csv(p)

csvs = sorted(glob.glob(os.path.join(EVAL_DIR, "results_*.csv")), reverse=True)
lat  = os.path.join(EVAL_DIR, "results_latest.csv")
if os.path.exists(lat) and lat not in csvs:
    csvs = [lat] + csvs

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Select Results File")
    if not csvs:
        st.warning("No CSVs found. Run: python eval/run_eval.py")
        st.stop()
    chosen = st.selectbox("CSV file", csvs, format_func=lambda p: os.path.basename(p))

try:
    df = load(chosen)
except Exception as e:
    st.error("Load failed: " + str(e))
    st.stop()

systems_in_file = sorted(df["system"].dropna().unique().tolist()) if "system" in df.columns else []

with st.sidebar:
    st.markdown("---")
    st.markdown("### Systems to Display")
    active = []
    if systems_in_file:
        for sys_name in systems_in_file:
            if st.checkbox(sys_name, value=True, key=f"chk_{sys_name}"):
                active.append(sys_name)
    else:
        active = systems_in_file

    st.markdown("---")
    st.caption("Back to main app: streamlit run app.py")

avail = [c for c in MC if c in df.columns]
if "system" in df.columns:
    df = df[df["system"].isin(active)]
if df.empty:
    st.warning("No data with current filters."); st.stop()

sg_df = df.groupby("system")[avail].mean() if "system" in df.columns else None


def clr(v):
    if pd.isna(v): return "gray"
    return "green" if v >= 0.6 else "orange" if v >= 0.35 else "red"


# ── Summary metrics ───────────────────────────────────────────────────────────
st.subheader("Overall Average Scores")
for nm in active:
    if sg_df is None or nm not in sg_df.index: continue
    r = sg_df.loc[nm]
    st.markdown("**" + nm + "**")
    cs = st.columns(len(avail))
    for c, m in zip(cs, avail):
        v = r.get(m, float("nan"))
        t = "{:.3f}".format(v) if not pd.isna(v) else "N/A"
        c.metric(label=ML.get(m, m), value=t)
    st.markdown("")


# ── Bar chart ─────────────────────────────────────────────────────────────────
st.subheader("Metric Comparison")
if sg_df is not None and avail:
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        pal = {
            "S1 - Groq (No RAG)": "#8b949e",
            "S2 - Groq + RAG": "#58a6ff",
            "S3 - Fine-Tuned (No RAG)": "#d29922",
            "S4 - Fine-Tuned + RAG": "#2ea043",
            "Qwen2.5 RAG": "#58a6ff",
            "Groq RAG": "#3fb950",
        }
        for nm in active:
            if nm not in sg_df.index: continue
            r = sg_df.loc[nm]
            ys = [float(r.get(m, 0)) for m in avail]
            fig.add_trace(go.Bar(
                name=nm,
                x=[ML.get(m, m) for m in avail],
                y=ys,
                marker_color=pal.get(nm, "#a371f7"),
                text=["{:.3f}".format(y) for y in ys],
                textposition="outside"))
        fig.update_layout(
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3", family="Segoe UI"),
            yaxis=dict(range=[0, 1.15]),
            height=380)
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart(sg_df[avail].T)


# ── Per-chapter breakdown ─────────────────────────────────────────────────────
if "chapter" in df.columns and not df["chapter"].isna().all():
    st.subheader("Per-Chapter Breakdown")
    for nm in active:
        ch = df[df["system"] == nm].groupby("chapter")[avail].mean().reset_index()
        if ch.empty: continue
        st.markdown("**" + nm + "**")
        try:
            import plotly.graph_objects as go
            f2 = go.Figure()
            cc = ["#58a6ff", "#3fb950", "#d29922", "#f85149"]
            for i, m in enumerate(avail):
                ys2 = list(ch[m])
                f2.add_trace(go.Bar(
                    name=ML.get(m, m),
                    x=list(ch["chapter"]),
                    y=ys2,
                    marker_color=cc[i % 4],
                    text=["{:.3f}".format(v) for v in ys2],
                    textposition="outside"))
            f2.update_layout(
                barmode="group",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e6edf3", family="Segoe UI"),
                yaxis=dict(range=[0, 1.18]),
                height=320)
            st.plotly_chart(f2, use_container_width=True)
        except ImportError:
            st.dataframe(ch.set_index("chapter"), use_container_width=True)


# ── Per-question table ────────────────────────────────────────────────────────
st.subheader("Per-Question Scores")
dcols = (["user_input"]
         + (["system"]  if "system"  in df.columns else [])
         + (["chapter"] if "chapter" in df.columns else [])
         + avail)
d2 = df[[c for c in dcols if c in df.columns]].copy()
d2.rename(columns={
    "user_input": "Question", "system": "System", "chapter": "Chapter",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "llm_context_precision_with_reference": "Ctx Precision",
}, inplace=True)

sc = [c for c in ["Faithfulness", "Answer Relevancy", "Ctx Precision"] if c in d2.columns]


def hl(v):
    if not isinstance(v, float): return ""
    if v >= 0.6:    return "color:#3fb950;font-weight:600"
    elif v >= 0.35: return "color:#d29922;font-weight:600"
    return "color:#f85149;font-weight:600"


st.dataframe(d2.style.applymap(hl, subset=sc), use_container_width=True, height=480)

st.markdown("---")
st.download_button(
    label="Download filtered CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="ragas_filtered.csv",
    mime="text/csv")
st.caption("File: " + os.path.basename(chosen) + " | " + str(len(df)) + " rows")
