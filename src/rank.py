"""
rank.py — Main ranking step (must run under 5 minutes)
-------------------------------------------------------
Loads precomputed embeddings and produces Top 100 CSV.

Usage:
  python src/rank.py --candidates data/candidates.jsonl --out output/submission.csv
"""

import argparse
import time
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stage1_filter import stage1_filter
from stage2_scorer import (
    score_candidate, build_candidate_narrative,
    behavioral_score, skills_depth_score, JD_TEXT, _days_since
)

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


def generate_reasoning(c: dict, scores: dict, rank: int) -> str:
    profile = c.get("profile", {})
    signals = c.get("redrob_signals", {}) or {}
    career = c.get("career_history", [])

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")

    facts = []
    if yoe:
        facts.append(f"{yoe:.1f} years of experience")
    if title and company:
        facts.append(f"currently {title} at {company}")

    for job in career[:2]:
        desc = (job.get("description") or "")
        kws = ["retrieval", "ranking", "embedding", "search",
               "recommendation", "vector"]
        if any(kw in desc.lower() for kw in kws):
            facts.append(
                f"built retrieval/ranking systems at {job.get('company', 'prior company')}")
            break
        elif any(kw in desc.lower() for kw in ["ml", "machine learning", "nlp"]):
            facts.append(
                f"applied ML work at {job.get('company', 'prior company')}")
            break

    days = _days_since(signals.get("last_active_date", ""))
    if days <= 7:
        facts.append("active on platform this week")
    elif days <= 30:
        facts.append("recently active")

    if signals.get("open_to_work_flag"):
        facts.append("actively seeking roles")

    notice = signals.get("notice_period_days")
    if notice is not None and notice <= 30:
        facts.append(f"available within {notice} days")
    elif notice and notice > 90:
        facts.append(f"long notice period ({notice} days)")

    concerns = []
    if scores.get("career_score", 0) < 0.3:
        concerns.append("limited product-company or IR background")
    if scores.get("behavioral_score", 0) < 0.3:
        concerns.append("lower engagement signals")
    if signals.get("github_activity_score", -1) == -1:
        concerns.append("no GitHub linked")

    sentence1 = f"{'; '.join(facts[:3])}." if facts else f"{title} with {yoe} years."
    sentence1 = sentence1[0].upper() + sentence1[1:]

    if concerns and rank > 30:
        sentence2 = f"Concerns: {'; '.join(concerns)}."
    elif scores.get("career_score", 0) > 0.7:
        sentence2 = "Strong product-company background with evidence of shipping real ML/retrieval systems."
    elif scores.get("semantic_score", 0) > 0.7:
        sentence2 = "High semantic alignment with JD requirements across career narrative."
    else:
        sentence2 = "Included based on skill overlap and engagement signals."

    return f"{sentence1} {sentence2}"


def run_pipeline(candidates_path: str, output_path: str):
    t0 = time.time()
    print(f"\n🚀 Starting ranking pipeline")
    print(f"   Input:  {candidates_path}")
    print(f"   Output: {output_path}\n")

    # ── Check for precomputed data ────────────────────────────────────────────
    precomputed = (
        Path("output/stage1_scored.csv").exists() and
        Path("output/raw_candidates.pkl").exists()
    )

    if precomputed:
        print("✅ Found precomputed data — loading from disk...")
        df = pd.read_csv("output/stage1_scored.csv")
        with open("output/raw_candidates.pkl", "rb") as f:
            raw_data = pickle.load(f)
        df["_raw"] = raw_data
        df["semantic_score"] = df.get("semantic_score", 0.0)
        t1 = time.time()
        print(f"   Loaded {len(df)} candidates in {t1-t0:.1f}s\n")
    else:
        print("⚠️  No precomputed data found — running full pipeline...")
        print("   TIP: Run precompute.py first for faster ranking!\n")

        # Stage 1
        print("📋 Stage 1: Fast filtering...")
        df = stage1_filter(candidates_path, verbose=True)
        t1 = time.time()
        print(f"   Done in {t1-t0:.1f}s\n")

        # Embeddings
        if HAS_ST:
            print("🔍 Semantic embedding...")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            narratives = [
                build_candidate_narrative(row["_raw"])
                for _, row in df.iterrows()
            ]
            jd_emb = model.encode([JD_TEXT], normalize_embeddings=True)[0]
            cand_emb = model.encode(
                narratives, batch_size=128,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            df["semantic_score"] = cand_emb @ jd_emb
        else:
            df["semantic_score"] = 0.0

    # ── Deep scoring ──────────────────────────────────────────────────────────
    print("🧠 Deep career + behavioral scoring...")
    score_rows = []
    for _, row in df.iterrows():
        scores = score_candidate(row.to_dict())
        scores["final_score"] = (
            float(row.get("semantic_score", 0)) * 0.40 +
            scores["career_score"] * 0.25 +
            scores["behavioral_score"] * 0.20 +
            scores["skills_score"] * 0.15
        )
        score_rows.append(scores)

    scores_df = pd.DataFrame(score_rows)
    df = df.merge(scores_df, on="candidate_id", how="left")
    t2 = time.time()
    print(f"   Done in {t2-t1:.1f}s\n")

    # ── Top 100 ───────────────────────────────────────────────────────────────
    print("🏆 Selecting Top 100...")
    top100 = df.nlargest(100, "final_score").copy()
    top100 = top100.sort_values(
    ["final_score", "candidate_id"],
    ascending=[False, True]).reset_index(drop=True)
    top100["rank"] = range(1, 101)

    for i in range(1, len(top100)):
        if top100.loc[i, "final_score"] > top100.loc[i-1, "final_score"]:
            top100.loc[i, "final_score"] = top100.loc[i-1, "final_score"]

    # ── Reasoning ─────────────────────────────────────────────────────────────
    print("📝 Generating reasoning...")
    reasonings = []
    for _, row in top100.iterrows():
        c = row.get("_raw", {})
        if not isinstance(c, dict):
            c = {}
        reasoning = generate_reasoning(c, row.to_dict(), int(row["rank"]))
        reasonings.append(reasoning)
    top100["reasoning"] = reasonings

    # ── Save CSV ──────────────────────────────────────────────────────────────
    submission = top100[[
        "candidate_id", "rank", "final_score", "reasoning"]].copy()
    submission.columns = ["candidate_id", "rank", "score", "reasoning"]
    submission["score"] = submission["score"].round(4)

    # Fix tie-breaking
    submission = submission.sort_values(
        ["score", "candidate_id"],
        ascending=[False, True]).reset_index(drop=True)
    submission["rank"] = range(1, 101)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    t_end = time.time()
    total = t_end - t0

    print(f"\n{'='*55}")
    print(f"✅ Done in {total:.1f}s ({total/60:.1f} min)")
    print(f"   Output: {output_path}")
    print(f"\n   Top 5 candidates:")
    for _, row in submission.head(5).iterrows():
        print(f"     Rank {int(row['rank'])}: {row['candidate_id']} "
              f"(score={row['score']:.4f})")
    print(f"{'='*55}\n")

    if total > 300:
        print("⚠️  WARNING: Exceeded 5-minute constraint!")
    else:
        print(f"✅ Within time limit ({300-total:.0f}s to spare)")

    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", default="./output/submission.csv")
    args = parser.parse_args()
    run_pipeline(args.candidates, args.out)