"""
precompute.py — Run once offline before ranking
------------------------------------------------
Embeds all Stage 1 candidates and saves to disk.
This can take 10-15 min but only needs to run ONCE.

Usage:
  python src/precompute.py --candidates data/candidates.jsonl
"""

import argparse
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from stage1_filter import stage1_filter
from stage2_scorer import build_candidate_narrative, JD_TEXT
from sentence_transformers import SentenceTransformer

def run_precompute(candidates_path: str):
    t0 = time.time()
    print("\n⚙️  Precompute — run once offline")
    print("="*50)

    # Stage 1 filter
    print("\n📋 Stage 1: Fast filtering...")
    df = stage1_filter(candidates_path, verbose=True)
    print(f"   {len(df)} candidates passed\n")

    # Build narratives
    print("📝 Building candidate narratives...")
    narratives = [
        build_candidate_narrative(row["_raw"])
        for _, row in df.iterrows()
    ]
    print(f"   Done — {len(narratives)} narratives built\n")

    # Embed everything
    print("🔍 Embedding candidates (this takes a while)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embed in batches for speed
    embeddings = model.encode(
        narratives,
        batch_size=128,        # bigger batch = faster
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Embed JD
    jd_embedding = model.encode(
        [JD_TEXT],
        normalize_embeddings=True
    )[0]

    # Compute semantic scores
    semantic_scores = embeddings @ jd_embedding
    df["semantic_score"] = semantic_scores

    # Save to disk
    Path("output").mkdir(exist_ok=True)

    # Save candidate IDs + semantic scores + raw data
    df.drop(columns=["_raw"]).to_csv(
        "output/stage1_scored.csv", index=False)

    # Save embeddings
    np.save("output/candidate_embeddings.npy", embeddings)
    np.save("output/jd_embedding.npy", jd_embedding)

    # Save candidate IDs in order
    pd.Series(df["candidate_id"].values).to_csv(
        "output/candidate_ids.csv", index=False, header=False)

    # Save raw candidate data for reasoning generation
    import pickle
    raw_data = [row["_raw"] for _, row in df.iterrows()]
    with open("output/raw_candidates.pkl", "wb") as f:
        pickle.dump(raw_data, f)

    t_end = time.time()
    print(f"\n✅ Precompute done in {t_end-t0:.1f}s ({(t_end-t0)/60:.1f} min)")
    print(f"   Saved to output/")
    print(f"   Now run: python src/rank.py --candidates data/candidates.jsonl --out output/submission.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    args = parser.parse_args()
    run_precompute(args.candidates)