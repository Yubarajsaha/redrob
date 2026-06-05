"""
Stage 1: Fast Filter
--------------------
Cuts 100K candidates to ~3-5K using hard rules.
Must be blazing fast — no embeddings here.
Goal: eliminate obviously wrong candidates, honeypots, and unavailable people.
"""

import gzip
import json
import pandas as pd
from datetime import datetime, timezone
from typing import Generator

# ─── Constants tuned to the JD ────────────────────────────────────────────────
MIN_YOE = 4.0
MAX_YOE = 20.0
MIN_LAST_ACTIVE_DAYS = 180
MIN_PROFILE_COMPLETENESS = 40
MIN_RECRUITER_RESPONSE_RATE = 0.05

RELEVANT_SKILLS = {
    "core": [
        "python", "embeddings", "vector", "faiss", "pinecone", "weaviate",
        "qdrant", "milvus", "elasticsearch", "opensearch", "sentence-transformers",
        "retrieval", "ranking", "nlp", "machine learning", "deep learning",
        "transformers", "bert", "llm", "fine-tuning", "rag",
        "recommendation", "search", "information retrieval",
        "spark", "airflow", "data pipeline", "ml engineering",
    ],
    "bonus": [
        "lora", "qlora", "peft", "xgboost", "learning to rank",
        "ndcg", "mrr", "a/b testing", "kafka", "redis", "pytorch", "tensorflow",
    ]
}

CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "zensar", "l&t infotech", "ltimindtree"
}


def load_candidates(filepath: str) -> Generator[dict, None, None]:
    """Load candidates from .jsonl or .jsonl.gz"""
    opener = gzip.open if filepath.endswith(".gz") else open
    with opener(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def days_since(date_str: str) -> int:
    if not date_str:
        return 9999
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).days
    except Exception:
        return 9999


def is_honeypot(c: dict) -> tuple[bool, str]:
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    skills = c.get("skills", [])
    yoe = profile.get("years_of_experience", 0) or 0

    for job in career:
        duration = job.get("duration_months", 0) or 0
        start = job.get("start_date", "")
        if start:
            try:
                start_dt = datetime.strptime(start[:10], "%Y-%m-%d")
                actual_months = (datetime.now() - start_dt).days / 30
                if duration > actual_months + 3:
                    return True, f"Impossible job duration at {job.get('company')}"
            except Exception:
                pass

    expert_zero_months = sum(
        1 for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and (s.get("duration_months") or 0) == 0
    )
    if expert_zero_months >= 5:
        return True, f"{expert_zero_months} advanced skills with 0 months experience"

    edu = c.get("education", [])
    for e in edu:
        grad_year = e.get("end_year")
        if grad_year:
            max_possible_yoe = datetime.now().year - grad_year + 1
            if yoe > max_possible_yoe + 2:
                return True, f"YoE {yoe} impossible given graduation {grad_year}"

    assessments = list((c.get("redrob_signals", {}) or {}).get("skill_assessment_scores", {}).values())
    if len(assessments) >= 3 and len(set(round(x, 1) for x in assessments)) == 1:
        return True, "All skill assessments identical"

    return False, ""


def skills_relevance_score(skills: list) -> tuple[int, int]:
    skill_names = {(s.get("name") or "").lower() for s in skills}
    full_text = " ".join(skill_names)
    core_hits = sum(1 for kw in RELEVANT_SKILLS["core"] if kw in full_text)
    bonus_hits = sum(1 for kw in RELEVANT_SKILLS["bonus"] if kw in full_text)
    return core_hits, bonus_hits


def is_consulting_only(career: list) -> bool:
    if not career:
        return False
    non_consulting = 0
    for job in career:
        company = (job.get("company") or "").lower()
        if not any(c in company for c in CONSULTING_COMPANIES):
            non_consulting += 1
    return non_consulting == 0


def stage1_filter(filepath: str, verbose: bool = True) -> pd.DataFrame:
    passed = []
    total = 0
    honeypots_caught = 0
    filtered_stats = {
        "honeypot": 0,
        "inactive": 0,
        "low_yoe": 0,
        "consulting_only": 0,
        "no_relevant_skills": 0,
    }

    for c in load_candidates(filepath):
        total += 1
        cid = c.get("candidate_id", "")
        profile = c.get("profile", {})
        signals = c.get("redrob_signals", {}) or {}
        career = c.get("career_history", [])
        skills = c.get("skills", [])

        hp, hp_reason = is_honeypot(c)
        if hp:
            filtered_stats["honeypot"] += 1
            honeypots_caught += 1
            continue

        yoe = profile.get("years_of_experience", 0) or 0
        if yoe < MIN_YOE or yoe > MAX_YOE:
            filtered_stats["low_yoe"] += 1
            continue

        last_active = signals.get("last_active_date", "")
        days_inactive = days_since(last_active)
        if days_inactive > MIN_LAST_ACTIVE_DAYS:
            filtered_stats["inactive"] += 1
            continue

        if is_consulting_only(career):
            filtered_stats["consulting_only"] += 1
            continue

        core_hits, bonus_hits = skills_relevance_score(skills)
        if core_hits < 1:
            filtered_stats["no_relevant_skills"] += 1
            continue

        completeness = signals.get("profile_completeness_score", 0) or 0
        if completeness < MIN_PROFILE_COMPLETENESS:
            continue

        open_to_work = signals.get("open_to_work_flag", False)

        passed.append({
            "candidate_id": cid,
            "yoe": yoe,
            "days_inactive": days_inactive,
            "open_to_work": open_to_work,
            "core_skill_hits": core_hits,
            "bonus_skill_hits": bonus_hits,
            "profile_completeness": completeness,
            "recruiter_response_rate": signals.get("recruiter_response_rate", 0) or 0,
            "interview_completion_rate": signals.get("interview_completion_rate", 0) or 0,
            "offer_acceptance_rate": signals.get("offer_acceptance_rate", 0) or 0,
            "github_activity_score": signals.get("github_activity_score", -1),
            "notice_period_days": signals.get("notice_period_days", 90) or 90,
            "last_active_days": days_inactive,
            "applications_30d": signals.get("applications_submitted_30d", 0) or 0,
            "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d", 0) or 0,
            "willing_to_relocate": signals.get("willing_to_relocate", False),
            "preferred_work_mode": signals.get("preferred_work_mode", ""),
            "_raw": c,
        })

    df = pd.DataFrame(passed)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Stage 1 Filter Results")
        print(f"{'='*50}")
        print(f"Total candidates:     {total:,}")
        print(f"Honeypots caught:     {honeypots_caught:,}")
        for k, v in filtered_stats.items():
            print(f"  {k:<25} {v:,}")
        print(f"Passed Stage 1:       {len(df):,}")
        print(f"{'='*50}\n")

    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/candidates.jsonl.gz"
    df = stage1_filter(path)
    df.drop(columns=["_raw"]).to_csv("output/stage1_passed.csv", index=False)
    print(f"Saved {len(df)} candidates to output/stage1_passed.csv")