"""
app.py — Streamlit frontend for Redrob AI Candidate Ranker
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from stage2_scorer import score_candidate, _days_since
from rank import generate_reasoning

st.set_page_config(
    page_title="Redrob AI Ranker",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Redrob AI — Intelligent Candidate Ranker")
st.markdown("*Semantic matching · Career analysis · Behavioral signals*")
st.divider()

with st.sidebar:
    st.header("⚙️ Pipeline Info")
    st.success("✅ Precomputed embeddings ready")
    st.info("📊 100,000 candidates in pool")
    st.info("🎯 Returns Top 100 best fits")
    st.divider()
    st.markdown("**Scoring weights:**")
    st.markdown("- 🔍 Semantic match: 40%")
    st.markdown("- 💼 Career quality: 25%")
    st.markdown("- 📊 Behavioral signals: 20%")
    st.markdown("- 🛠️ Skills depth: 15%")
    st.divider()
    st.markdown("**Stage 1 filters:**")
    st.markdown("- ✅ Honeypot detection")
    st.markdown("- ✅ Activity filter")
    st.markdown("- ✅ Experience range")
    st.markdown("- ✅ Consulting-only filter")
    st.markdown("- ✅ Skill relevance")

st.subheader("📋 Job Description")
st.markdown("*The ranker reads the full JD semantically — not just keywords*")

jd_text = st.text_area(
    "Job Description",
    value="""Senior AI Engineer — Founding Team at Redrob AI (Series A).
5-9 years experience. Pune/Noida India, hybrid.

Must have: Production experience with embeddings-based retrieval systems 
(sentence-transformers, OpenAI embeddings, BGE, E5). Production experience with 
vector databases or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, 
Milvus, Elasticsearch, FAISS). Strong Python. Hands-on experience designing 
evaluation frameworks for ranking systems (NDCG, MRR, MAP, A/B testing).

NOT a fit: pure research, consulting-only background (TCS, Infosys, Wipro),
computer vision without NLP/IR, inactive candidates.""",
    height=200,
    label_visibility="collapsed"
)

st.divider()

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_button = st.button(
        "🚀 Rank Candidates",
        type="primary",
        use_container_width=True
    )

if run_button:
    if not Path("output/stage1_scored.csv").exists():
        st.error("❌ Precomputed data not found! Run precompute.py first.")
        st.code("python src/precompute.py --candidates data/candidates.jsonl")
        st.stop()

    with st.spinner("🧠 Scoring candidates..."):
        try:
            # Load precomputed data
            df = pd.read_csv("output/stage1_scored.csv")
            with open("output/raw_candidates.pkl", "rb") as f:
                raw_data = pickle.load(f)
            df["_raw"] = raw_data

            # Score all candidates
            score_rows = []
            for idx in range(len(df)):
                row = df.iloc[idx]
                scores = score_candidate(row.to_dict())
                scores["final_score"] = (
                    float(row.get("semantic_score", 0)) * 0.40 +
                    scores["career_score"] * 0.25 +
                    scores["behavioral_score"] * 0.20 +
                    scores["skills_score"] * 0.15
                )
                score_rows.append(scores)

            scores_df = pd.DataFrame(score_rows)
            df = df.merge(
                scores_df, on="candidate_id", how="left", suffixes=("", "_scored"))

            # Top 100
            top100 = df.nlargest(100, "final_score").copy()
            top100 = top100.sort_values(
                "final_score", ascending=False).reset_index(drop=True)
            top100["rank"] = range(1, 101)

            # Generate reasoning
            reasonings = []
            for idx in range(len(top100)):
                row = top100.iloc[idx]
                c = row.get("_raw", {})
                if not isinstance(c, dict):
                    c = {}
                reasoning = generate_reasoning(
                    c, row.to_dict(), int(row["rank"]))
                reasonings.append(reasoning)

            top100 = top100.copy()
            top100["reasoning"] = reasonings

            st.success(f"✅ Ranked {len(df):,} candidates in seconds!")

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)
            st.stop()

    st.divider()

    # Metrics
    st.subheader("📊 Results Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Pool", "100,000")
    m2.metric("Passed Filter", f"{len(df):,}")
    m3.metric("Top 100 Min Score", f"{top100['final_score'].min():.3f}")
    m4.metric("Top 100 Max Score", f"{top100['final_score'].max():.3f}")

    st.divider()

    # Score Distribution Chart
    st.subheader("📈 Score Distribution — Top 100")
    fig = px.bar(
        top100.head(20),
        x="rank",
        y="final_score",
        color="final_score",
        color_continuous_scale="Viridis",
        labels={"rank": "Rank", "final_score": "Score"},
        title="Top 20 Candidates by Score"
    )
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Score Breakdown Chart
    st.subheader("🔍 Score Breakdown — Top 10")
    top10 = top100.head(10).copy()

    sem_col = "semantic_score" if "semantic_score" in top10.columns else "semantic_score_scored"
    car_col = "career_score" if "career_score" in top10.columns else "career_score_scored"
    beh_col = "behavioral_score" if "behavioral_score" in top10.columns else "behavioral_score_scored"
    ski_col = "skills_score" if "skills_score" in top10.columns else "skills_score_scored"

    breakdown_df = pd.DataFrame({
        "Candidate": [f"#{r} {cid[-7:]}" for r, cid in
                      zip(top10["rank"], top10["candidate_id"])],
        "Semantic": top10[sem_col].values,
        "Career": top10[car_col].values,
        "Behavioral": top10[beh_col].values,
        "Skills": top10[ski_col].values,
    })

    fig2 = px.bar(
        breakdown_df.melt(
            id_vars="Candidate",
            var_name="Component",
            value_name="Score"),
        x="Candidate",
        y="Score",
        color="Component",
        barmode="group",
        title="Score Components for Top 10 Candidates",
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Top 100 Table
    st.subheader("🏆 Top 100 Candidates")
    display_rows = []
    for idx in range(len(top100)):
        row = top100.iloc[idx]
        c = row.get("_raw", {})
        if not isinstance(c, dict):
            c = {}
        profile = c.get("profile", {})
        signals = c.get("redrob_signals", {}) or {}

        display_rows.append({
            "Rank": int(row["rank"]),
            "Candidate ID": row["candidate_id"],
            "Title": profile.get("current_title", "—"),
            "Company": profile.get("current_company", "—"),
            "YoE": profile.get("years_of_experience", "—"),
            "Score": round(row["final_score"], 4),
            "Semantic": round(float(row.get(sem_col, 0)), 3),
            "Career": round(float(row.get(car_col, 0)), 3),
            "Behavioral": round(float(row.get(beh_col, 0)), 3),
            "Open to Work": "✅" if signals.get("open_to_work_flag") else "❌",
            "Reasoning": row.get("reasoning", "—"),
        })

    display_df = pd.DataFrame(display_rows)
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=1),
            "Semantic": st.column_config.ProgressColumn(
                "Semantic", min_value=0, max_value=1),
            "Career": st.column_config.ProgressColumn(
                "Career", min_value=0, max_value=1),
            "Behavioral": st.column_config.ProgressColumn(
                "Behavioral", min_value=0, max_value=1),
        }
    )

    st.divider()

    # Download
    st.subheader("📥 Download Submission")
    submission = top100[["candidate_id", "rank",
                        "final_score", "reasoning"]].copy()
    submission.columns = ["candidate_id", "rank", "score", "reasoning"]
    submission["score"] = submission["score"].round(4)

    csv = submission.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Top 100 CSV",
        data=csv,
        file_name="submission.csv",
        mime="text/csv",
        use_container_width=True
    )

st.divider()
st.markdown(
    "*Built for Redrob Hackathon — Intelligent Candidate Discovery & Ranking*")