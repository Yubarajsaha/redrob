# Redrob AI — Intelligent Candidate Ranker
### Redrob Hackathon — Talent Acquisition Challenge

## Quick Start — Reproduce Submission

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Pre-computation (run once, ~20 min)
```bash
python src/precompute.py --candidates ./data/candidates.jsonl
```

### Step 3 — Ranking (produces submission CSV, ~6 seconds)
```bash
python src/rank.py --candidates ./data/candidates.jsonl --out ./output/yubaraj459_8412.csv
```

### Step 4 — Validate
```bash
python validate_submission.py output/yubaraj459_8412.csv
```

## Architecture

Three-stage pipeline that finds Top 100 from 100K candidates:

100,000 candidates
↓
Stage 1: Fast Filter (rule-based)     100K → ~19K  | 5.6s
↓
Stage 2a: Semantic Embedding          19K → scored  | pre-computed
↓
Stage 2b: Deep Career Scoring         scored → ranked | 5s
↓
Stage 3: Top 100 + CSV output         Top 100       | <1s

## What makes this different from keyword matching

- **Honeypot detection** — catches impossible profiles before ranking
- **Career trajectory analysis** — reads actual job descriptions
- **Product vs consulting detection** — per JD's explicit disqualifiers
- **Production evidence scoring** — looks for "deployed", "scaled", "shipped"
- **Pre-LLM ML experience** — rewards candidates who knew retrieval before LLMs
- **Behavioral availability signals** — inactive profiles are not real candidates

## Scoring Formula

| Component | Weight | What it measures |
|---|---|---|
| Semantic similarity | 40% | JD embedding vs full candidate narrative |
| Career quality | 25% | Product co, retrieval experience, production evidence |
| Behavioral signals | 20% | Activity, response rate, availability |
| Skills depth | 15% | Proficiency + duration + assessment scores |

## Compute Environment

- Python 3.14.0
- Windows 11, 16GB RAM, CPU only
- No GPU, no network during ranking
- Pre-computation: ~20 min (once)
- Ranking step: ~6 seconds ✅

## Sandbox

Google Colab:
https://colab.research.google.com/drive/18kvPkhJ8xFBbEcmJ0zjZzo1yWZ1lHhsc?usp=sharing

## Files

| File | Description |
|---|---|
| `src/rank.py` | Main ranking pipeline — single command to reproduce |
| `src/precompute.py` | Pre-computes embeddings (run once offline) |
| `src/stage1_filter.py` | Fast filter — honeypot detection + hard rules |
| `src/stage2_scorer.py` | Deep career + behavioral scoring |
| `requirements.txt` | All dependencies |
| `submission_metadata.yaml` | Submission metadata |
| `validate_submission.py` | Format validator |