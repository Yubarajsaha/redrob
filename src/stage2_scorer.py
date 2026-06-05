"""
Stage 2: Deep Semantic Ranker
------------------------------
Scores candidates on career quality, behavioral signals, and skills depth.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

# ─── JD Text for embedding ────────────────────────────────────────────────────
JD_TEXT = """
Senior AI Engineer — Founding Team at Redrob AI (Series A).
5-9 years experience. Pune/Noida India, hybrid.

Must have: Production experience with embeddings-based retrieval systems 
(sentence-transformers, OpenAI embeddings, BGE, E5). Production experience with 
vector databases or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, 
Milvus, Elasticsearch, FAISS). Strong Python. Hands-on experience designing 
evaluation frameworks for ranking systems (NDCG, MRR, MAP, A/B testing).

Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT). Learning-to-rank models.
HR-tech or marketplace products. Distributed systems. Open-source contributions.

Ideal: 6-8 years total, 4-5 in applied ML/AI at product companies not consulting.
Shipped at least one end-to-end ranking, search, or recommendation system to real 
users at meaningful scale.

NOT a fit: pure research, LangChain tutorials only, consulting-only background,
TCS Infosys Wipro Accenture, computer vision without NLP/IR, inactive candidates.
"""

# ─── Keywords ─────────────────────────────────────────────────────────────────
CONSULTING_SIGNALS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "ltimindtree", "l&t infotech", "deloitte", "kpmg"
]

RETRIEVAL_KEYWORDS = [
    "retrieval", "ranking", "recommendation", "search", "vector",
    "embedding", "semantic", "similarity", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "elasticsearch", "opensearch",
    "bm25", "dense retrieval", "hybrid search", "re-rank", "rerank",
    "ann", "approximate nearest neighbor", "knn", "index"
]

PRODUCTION_EVIDENCE_KEYWORDS = [
    "deployed", "production", "real users", "scaled", "serving",
    "latency", "throughput", "a/b test", "online", "inference",
    "api", "microservice", "pipeline", "shipped", "launched",
    "built", "designed", "architected", "owned", "led"
]

PRE_LLM_ML_KEYWORDS = [
    "xgboost", "lightgbm", "sklearn", "scikit", "tensorflow", "pytorch",
    "deep learning", "neural network", "bert", "transformer",
    "nlp", "natural language", "text classification", "named entity",
    "information retrieval", "collaborative filtering",
    "gradient boosting", "random forest", "feature engineering"
]

STARTUP_FIT_KEYWORDS = [
    "founding", "early stage", "0 to 1", "zero to one", "greenfield",
    "built from scratch", "wore many hats", "cross-functional",
    "ambiguity", "fast-paced", "startup", "series a", "series b", "seed"
]


def build_candidate_narrative(c: dict) -> str:
    """Build rich text for embedding — career history is most important."""
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    skills = c.get("skills", [])

    parts = []

    if profile.get("headline"):
        parts.append(profile["headline"])
    if profile.get("summary"):
        parts.append(profile["summary"])

    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    industry = profile.get("current_industry", "")
    if title:
        parts.append(f"Current role: {title} at {company} ({industry})")

    for job in career[:4]:
        job_parts = []
        if job.get("title"):
            job_parts.append(job["title"])
        if job.get("company"):
            job_parts.append(f"at {job['company']}")
        if job.get("duration_months"):
            job_parts.append(f"({job['duration_months']} months)")
        if job.get("description"):
            job_parts.append(job["description"])
        if job_parts:
            parts.append(" ".join(job_parts))

    top_skills = sorted(
        skills,
        key=lambda s: (
            {"advanced": 3, "intermediate": 2, "beginner": 1}.get(
                s.get("proficiency", ""), 0),
            s.get("endorsements", 0)
        ),
        reverse=True
    )[:15]
    skill_names = [s["name"] for s in top_skills if s.get("name")]
    if skill_names:
        parts.append("Key skills: " + ", ".join(skill_names))

    return " ".join(parts)


def career_quality_score(c: dict) -> tuple[float, dict]:
    """Score career quality on dimensions the JD explicitly cares about."""
    profile = c.get("profile", {})
    career = c.get("career_history", [])

    full_career_text = " ".join([
        (job.get("description") or "") + " " +
        (job.get("company") or "") + " " +
        (job.get("title") or "")
        for job in career
    ]).lower()

    profile_text = (
        (profile.get("summary") or "") + " " +
        (profile.get("headline") or "")
    ).lower()

    all_text = full_career_text + " " + profile_text
    scores = {}

    # 1. Product vs consulting
    consulting_months = sum(
        job.get("duration_months", 0) or 0
        for job in career
        if any(co in (job.get("company") or "").lower()
               for co in CONSULTING_SIGNALS)
    )
    total_months = sum(
        job.get("duration_months", 0) or 0 for job in career) or 1
    consulting_ratio = consulting_months / total_months
    scores["product_vs_consulting"] = max(0.0, 1.0 - consulting_ratio * 1.5)

    # 2. Retrieval/search/ranking experience
    retrieval_hits = sum(1 for kw in RETRIEVAL_KEYWORDS if kw in all_text)
    scores["retrieval_experience"] = min(1.0, retrieval_hits / 5.0)

    # 3. Pre-LLM ML experience
    pre_llm_hits = sum(1 for kw in PRE_LLM_ML_KEYWORDS if kw in all_text)
    scores["pre_llm_ml"] = min(1.0, pre_llm_hits / 4.0)

    # 4. Production deployment evidence
    prod_hits = sum(
        1 for kw in PRODUCTION_EVIDENCE_KEYWORDS if kw in all_text)
    scores["production_evidence"] = min(1.0, prod_hits / 6.0)

    # 5. Career progression
    SENIORITY = {
        "intern": 0, "junior": 1, "associate": 2, "engineer": 3,
        "developer": 3, "analyst": 2, "senior": 4, "lead": 5,
        "staff": 6, "principal": 7, "architect": 6, "manager": 5,
        "director": 7, "head": 7
    }

    def title_level(title: str) -> int:
        t = (title or "").lower()
        return max((v for k, v in SENIORITY.items() if k in t), default=3)

    if len(career) >= 2:
        sorted_career = sorted(
            career, key=lambda j: j.get("start_date") or "")
        levels = [title_level(j.get("title", "")) for j in sorted_career]
        progression = (levels[-1] - levels[0]) / max(len(levels), 1)
        scores["career_progression"] = min(
            1.0, max(0.0, 0.5 + progression * 0.2))
    else:
        scores["career_progression"] = 0.5

    # 6. Startup fit
    startup_hits = sum(
        1 for kw in STARTUP_FIT_KEYWORDS if kw in all_text)
    scores["startup_fit"] = min(1.0, startup_hits / 3.0)

    # 7. Tenure stability
    avg_tenure = total_months / max(len(career), 1)
    scores["tenure_stability"] = min(1.0, avg_tenure / 24.0)

    weights = {
        "product_vs_consulting": 0.25,
        "retrieval_experience":  0.25,
        "pre_llm_ml":            0.15,
        "production_evidence":   0.15,
        "career_progression":    0.10,
        "startup_fit":           0.05,
        "tenure_stability":      0.05,
    }
    final = sum(scores[k] * weights[k] for k in weights)
    return final, scores


def behavioral_score(signals: dict) -> float:
    """Score candidate availability and engagement."""
    if not signals:
        return 0.3

    score = 0.0

    last_active = signals.get("last_active_date", "")
    days_ago = _days_since(last_active)
    if days_ago <= 7:       score += 0.25
    elif days_ago <= 30:    score += 0.20
    elif days_ago <= 90:    score += 0.12
    elif days_ago <= 180:   score += 0.05

    if signals.get("open_to_work_flag"):
        score += 0.15

    rrr = signals.get("recruiter_response_rate", 0) or 0
    score += rrr * 0.15

    icr = signals.get("interview_completion_rate", 0) or 0
    score += icr * 0.10

    github = signals.get("github_activity_score", -1)
    if github > 0:
        score += min(0.10, github / 100 * 0.10)

    notice = signals.get("notice_period_days", 90) or 90
    if notice <= 30:        score += 0.10
    elif notice <= 60:      score += 0.06
    elif notice <= 90:      score += 0.03

    saved = min(signals.get("saved_by_recruiters_30d", 0) or 0, 20)
    score += saved / 20 * 0.05

    if signals.get("willing_to_relocate"):
        score += 0.05

    mode = (signals.get("preferred_work_mode") or "").lower()
    if mode in ("onsite", "hybrid", "flexible"):
        score += 0.05

    return min(1.0, score)


def skills_depth_score(skills: list, signals: dict) -> float:
    """Score skills on depth — proficiency + duration + assessments."""
    if not skills:
        return 0.0

    CORE_SKILLS = {
        "python", "machine learning", "deep learning", "nlp",
        "embeddings", "vector", "retrieval", "ranking", "transformers",
        "bert", "pytorch", "tensorflow", "elasticsearch", "faiss",
        "sentence-transformers", "recommendation", "search"
    }

    PROFICIENCY_WEIGHTS = {
        "advanced": 1.0, "intermediate": 0.6, "beginner": 0.2}

    assessments = (signals or {}).get("skill_assessment_scores", {}) or {}
    total_score = 0.0

    for skill in skills:
        name = (skill.get("name") or "").lower()
        prof = PROFICIENCY_WEIGHTS.get(
            skill.get("proficiency", "beginner"), 0.2)
        duration = min((skill.get("duration_months") or 0), 60) / 60
        endorsements = min((skill.get("endorsements") or 0), 50) / 50

        is_core = any(kw in name for kw in CORE_SKILLS)
        multiplier = 2.0 if is_core else 0.5

        skill_score = (
            prof * 0.4 + duration * 0.35 + endorsements * 0.25
        ) * multiplier
        total_score += skill_score

        for assessed_skill, ascore in assessments.items():
            if assessed_skill.lower() in name or name in assessed_skill.lower():
                total_score += (ascore / 100) * 0.5

    normalized = total_score / max(len(skills), 1)
    return min(1.0, normalized)


def _days_since(date_str: str) -> int:
    if not date_str:
        return 9999
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return (datetime.now() - dt).days
    except Exception:
        return 9999


def score_candidate(row: dict) -> dict:
    """Full scoring for a single candidate."""
    c = row.get("_raw", {})
    signals = c.get("redrob_signals", {}) or {}
    skills = c.get("skills", [])

    career_score, career_breakdown = career_quality_score(c)
    behav_score = behavioral_score(signals)
    skill_score = skills_depth_score(skills, signals)
    semantic_score = row.get("semantic_score", 0.0)

    final = (
        semantic_score * 0.40 +
        career_score   * 0.25 +
        behav_score    * 0.20 +
        skill_score    * 0.15
    )

    return {
        "candidate_id": row.get("candidate_id"),
        "final_score": round(final, 4),
        "semantic_score": round(semantic_score, 4),
        "career_score": round(career_score, 4),
        "behavioral_score": round(behav_score, 4),
        "skills_score": round(skill_score, 4),
        **{f"career_{k}": round(v, 3) for k, v in career_breakdown.items()},
    }