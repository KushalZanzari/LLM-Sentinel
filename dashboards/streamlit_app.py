import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
import os
import sys
import tempfile

# Add parent directory to path so it can find the 'src' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import evaluate

st.set_page_config(page_title='LLM Eval Dashboard', layout='wide')

st.title('LLM Evaluation Dashboard')

tab1, tab2 = st.tabs([" Batch Analytics", " Live Evaluation"])

with tab1:
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

with tab2:
    st.header("Live Interactive Evaluation")
    st.write("Paste your chat and ground truth context below to instantly evaluate a response without needing to create JSON files.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Chat Data")
        user_prompt = st.text_area("User Prompt", value="What is the company's refund policy?", height=100)
        ai_answer = st.text_area("AI Answer", value="You can get a full refund within 30 days of purchase.", height=150)
        
    with col_b:
        st.subheader(" Ground Truth Context")
        context_text = st.text_area("Context Facts", value="Our refund policy allows for full refunds up to 30 days after the initial purchase date. Refunds are processed within 5-7 business days.", height=300)
        
    if st.button(" Evaluate Response", type="primary", use_container_width=True):
        if not user_prompt or not ai_answer or not context_text:
            st.error("Please fill in all text fields before evaluating.")
        else:
            with st.spinner("Evaluating (Generating Embeddings & Scoring)..."):
                # 1. Structure the data exactly as the CLI/Batch scripts expect
                chat_data = {
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": ai_answer}
                    ]
                }
                ctx_data = {
                    "contexts": [
                        {"id": "live-ctx-1", "text": context_text}
                    ]
                }
                
                # 2. Write to temporary files (safest way to interface with existing evaluate function)
                chat_path = ""
                ctx_path = ""
                
                try:
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as c_file:
                        json.dump(chat_data, c_file)
                        chat_path = c_file.name
                        
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as ctx_file:
                        json.dump(ctx_data, ctx_file)
                        ctx_path = ctx_file.name
                    
                    # 3. Call the core pipeline
                    result = evaluate(chat_path, ctx_path)
                    
                    st.divider()
                    st.subheader("Final Verdict")
                    
                    # Display colored verdict
                    verdict = result.get("verdict", "FAIL")
                    if verdict == "PASS":
                        st.success(f" {verdict}: The response meets all quality thresholds.")
                    elif verdict == "WARN":
                        st.warning(f" {verdict}: The response is factually safe but lacks relevance or completeness.")
                    else:
                        st.error(f"{verdict}: The response failed critical checks (likely a hallucination).")
                        
                    # Display metrics in columns
                    mc1, mc2, mc3 = st.columns(3)
                    scores = result.get("scores", {})
                    mc1.metric("Quality Score", f"{scores.get('quality_score', 0):.3f}")
                    mc2.metric("Relevance", f"{scores.get('relevance', 0):.3f}")
                    mc3.metric("Factuality", f"{scores.get('factuality', {}).get('avg_score', 0):.3f}")
                    
                    # Full JSON
                    st.subheader("Detailed JSON Report")
                    st.json(result)
                    
                except Exception as e:
                    st.error(f"Evaluation failed due to an internal error: {str(e)}")
                finally:
                    # 4. Cleanup temporary files
                    if chat_path and os.path.exists(chat_path):
                        os.remove(chat_path)
                    if ctx_path and os.path.exists(ctx_path):
                        os.remove(ctx_path)
