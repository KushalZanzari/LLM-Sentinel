import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title='LLM Eval Dashboard', layout='wide')

st.title('LLM Evaluation Dashboard')

CSV_PATH = 'data/batch_results.csv'

# Load CSV safely
try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    st.error(f'Cannot load CSV at {CSV_PATH}: {e}')
    st.stop()

# Compute quality_score if not present
def compute_quality(row):
    rel = float(row.get('relevance', 0))
    comp = float(row.get('completeness', 0))
    fact = float(row.get('factuality', row.get('factuality_avg', 0)))
    return 0.4 * rel + 0.3 * comp + 0.3 * fact

if "quality_score" not in df.columns:
    df["quality_score"] = df.apply(compute_quality, axis=1)

# ---------------------------
# SUMMARY BOXES
# ---------------------------
st.header('Batch Summary')
col1, col2, col3 = st.columns(3)

col1.metric("Files evaluated", len(df))
col2.metric("Avg quality", f"{df['quality_score'].mean():.3f}")
col3.metric("PASS rate", f"{(df['verdict'] == 'PASS').mean():.2%}")

# ---------------------------
# DATA TABLE
# ---------------------------
st.header("Scores Table")
st.dataframe(df)

# ---------------------------
# DISTRIBUTIONS
# ---------------------------
st.header("Score Distributions")
c1, c2 = st.columns(2)

with c1:
    st.subheader("Histogram: Quality Score")

    # FIX: Convert IntervalIndex → string
    hist = (
        pd.cut(df["quality_score"], bins=10)
        .value_counts()
        .sort_index()
    )
    hist.index = hist.index.astype(str)

    st.bar_chart(hist)

with c2:
    st.subheader("Summary Statistics")
    st.write(df[["relevance", "completeness", "factuality"]].describe())

# ---------------------------
# SCATTER PLOT
# ---------------------------
st.header("Scatter: Quality vs Total Tokens")

chart = (
    alt.Chart(df)
    .mark_circle(size=60)
    .encode(
        x="total_tokens:Q",
        y="quality_score:Q",
        color="verdict:N",
        tooltip=["chat_file", "quality_score", "verdict", "total_tokens"],
    )
    .interactive()
    .properties(width=800, height=400)
)

st.altair_chart(chart, use_container_width=True)

# ---------------------------
# VERDICT COUNTS
# ---------------------------
st.header("Verdict Counts")
st.bar_chart(df["verdict"].value_counts())
