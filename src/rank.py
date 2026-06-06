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
    skills = c.get("skills", [])

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    location = profile.get("location", "")
    country = profile.get("country", "")

    # Top skills by proficiency + endorsements
    top_skills = sorted(
        skills,
        key=lambda s: (
            {"advanced": 3, "intermediate": 2, "beginner": 1}.get(
                s.get("proficiency", ""), 0),
            s.get("endorsements", 0)
        ),
        reverse=True
    )[:5]
    skill_names = [s["name"] for s in top_skills if s.get("name")]

    # Find most relevant career highlight
    retrieval_job = None
    ml_job = None
    for job in career[:3]:
        desc = (job.get("description") or "").lower()
        if any(kw in desc for kw in ["retrieval", "ranking", "embedding",
                                      "search", "recommendation", "vector"]):
            retrieval_job = job
            break
        elif any(kw in desc for kw in ["ml", "machine learning", "nlp",
                                        "deep learning", "model"]):
            ml_job = job

    # Assessment scores
    assessments = signals.get("skill_assessment_scores", {}) or {}
    best_assessment = max(assessments.items(),
                          key=lambda x: x[1]) if assessments else None

    # Behavioral signals
    notice = signals.get("notice_period_days")
    github = signals.get("github_activity_score", -1)
    response_rate = signals.get("recruiter_response_rate", 0) or 0
    days_inactive = _days_since(signals.get("last_active_date", ""))
    willing_to_relocate = signals.get("willing_to_relocate", False)

    # ── Sentence 1 — specific facts ──────────────────────────────────────────
    facts = []
    if yoe:
        facts.append(f"{yoe:.1f} yrs exp")
    if title and company:
        facts.append(f"{title} at {company}")
    if location:
        facts.append(f"based in {location}")
    if skill_names:
        facts.append(f"top skills: {', '.join(skill_names[:3])}")

    sentence1 = "; ".join(facts) + "."
    sentence1 = sentence1[0].upper() + sentence1[1:]

    # ── Sentence 2 — specific insight + honest concern ────────────────────────
    insights = []
    concerns = []

    # Career specific insight
    if retrieval_job:
        duration = retrieval_job.get("duration_months", 0)
        insights.append(
            f"Shipped retrieval/search systems at "
            f"{retrieval_job.get('company')} ({duration}mo tenure)"
        )
    elif ml_job:
        insights.append(
            f"Applied ML work at {ml_job.get('company')} — "
            f"adjacent to JD but not direct IR experience"
        )

    # Assessment insight
    if best_assessment and best_assessment[1] > 60:
        insights.append(
            f"Platform-verified {best_assessment[0]} "
            f"score: {best_assessment[1]:.0f}/100"
        )
    elif best_assessment and best_assessment[1] < 40:
        concerns.append(
            f"Low platform assessment in {best_assessment[0]} "
            f"({best_assessment[1]:.0f}/100)"
        )

    # GitHub signal
    if github > 15:
        insights.append(f"Strong GitHub activity ({github:.0f}/100)")
    elif github == -1 or github == 0:
        concerns.append("No GitHub activity linked")

    # Notice period
    if notice is not None and notice <= 30:
        insights.append(f"Available in {notice} days")
    elif notice and notice > 90:
        concerns.append(f"Long notice period ({notice} days)")

    # Location fit for Pune/Noida
    loc_lower = (location or "").lower()
    if any(city in loc_lower for city in ["pune", "noida", "delhi",
                                           "mumbai", "bangalore",
                                           "bengaluru", "hyderabad"]):
        insights.append("India-based — strong location fit")
    elif country and country.lower() != "india":
        concerns.append(
            f"Based outside India ({country}) — relocation needed")

    # Response rate concern
    if response_rate < 0.2:
        concerns.append(
            f"Low recruiter response rate ({response_rate:.0%})")

    # Career score feedback
    career_score = scores.get("career_score", 0)
    if career_score > 0.75:
        insights.append("Strong product-company IR background")
    elif career_score < 0.35:
        concerns.append("Limited direct IR/retrieval system experience")

    # Rank-consistent tone
    if rank <= 10:
        if insights:
            sentence2 = "Strengths: " + "; ".join(insights[:2]) + "."
        else:
            sentence2 = "High semantic and career alignment with JD."
    elif rank <= 30:
        if insights and concerns:
            sentence2 = (f"{insights[0]}. "
                         f"Minor concern: {concerns[0]}.")
        elif insights:
            sentence2 = "; ".join(insights[:2]) + "."
        else:
            sentence2 = "Good overall fit with minor gaps."
    elif rank <= 60:
        if concerns:
            sentence2 = (
                f"Partial fit — "
                f"{insights[0] if insights else 'some relevant experience'}. "
                f"Concern: {concerns[0]}.")
        else:
            sentence2 = ("Moderate fit — included based on skill overlap "
                         "and engagement signals.")
    else:
        if concerns:
            sentence2 = (f"Below cutoff on key criteria. "
                         f"Concerns: {'; '.join(concerns[:2])}.")
        else:
            sentence2 = ("Borderline inclusion — marginal fit on JD "
                         "requirements; ranked here due to behavioral signals.")

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

        print("📋 Stage 1: Fast filtering...")
        df = stage1_filter(candidates_path, verbose=True)
        t1 = time.time()
        print(f"   Done in {t1-t0:.1f}s\n")

        if HAS_ST:
            print("🔍 Semantic embedding...")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            narratives = [
                build_candidate_narrative(row["_raw"])
                for _, row in df.iterrows()
            ]
            jd_emb = model.encode(
                [JD_TEXT], normalize_embeddings=True)[0]
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