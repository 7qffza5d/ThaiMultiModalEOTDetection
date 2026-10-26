# analyze_unk_counts.py
"""Standalone: measure how <unk>/<td>/<sil> tags and real-word counts are
distributed across timestamped/, to decide the section 4.1 filtering rule
from data. Requires the per-tag columns (n_unk_tags, n_td_tags,
n_sil_tags, n_words) added to attach_timestamps_from_alignment, so
re-run alignTextGridsToParquet.py first."""
import glob
import os

import pandas as pd

REQUIRED = ["n_unk_tags", "n_td_tags", "n_sil_tags", "n_words", "tag_density"]


def load_all(timestamped_dir):
    paths = sorted(glob.glob(os.path.join(timestamped_dir, "*.parquet")))
    if not paths:
        raise SystemExit(f"No parquet files found in {timestamped_dir}/")
    dfs = []
    for p in paths:
        df = pd.read_parquet(p)
        df["session"] = os.path.basename(p).removesuffix(".parquet")
        dfs.append(df)
    all_rows = pd.concat(dfs, ignore_index=True)
    missing = [c for c in REQUIRED if c not in all_rows.columns]
    if missing:
        raise SystemExit(f"Missing columns {missing}; regenerate timestamped/ "
                         f"with the updated attach_timestamps_from_alignment.")
    return all_rows, len(paths)


def main(timestamped_dir="timestamped"):
    rows, n_sessions = load_all(timestamped_dir)
    total = len(rows)
    print(f"Total rows: {total} across {n_sessions} sessions\n")

    # --- How common is each tag type? ---
    print("Rows containing each tag type:")
    for col, label in [("n_unk_tags", "<unk>"), ("n_td_tags", "<td>"), ("n_sil_tags", "<sil>")]:
        n = (rows[col] > 0).sum()
        print(f"  {label}: {n} rows ({n / total:.2%})")
    print()

    # --- Overall length picture ---
    print("n_words percentiles (all rows):")
    for p in [1, 5, 25, 50, 75, 95]:
        print(f"  p{p}: {rows['n_words'].quantile(p / 100):.0f}")
    print(f"  rows with exactly 1 word: {(rows['n_words'] == 1).sum()} "
          f"({(rows['n_words'] == 1).mean():.2%})")
    print(f"  rows with 0 words:        {(rows['n_words'] == 0).sum()}")
    print()

    # --- The <unk> question: does a 'few real words' rule drop anything meaningful? ---
    unk = rows[rows["n_unk_tags"] > 0]
    print(f"Rows with <unk>: {len(unk)}")
    if len(unk):
        buckets = pd.cut(unk["n_words"], bins=[-1, 0, 1, 2, 5, 10, 10_000],
                         labels=["0", "1", "2", "3-5", "6-10", "11+"])
        print("  n_words distribution within <unk> rows:")
        print(buckets.value_counts().reindex(["0", "1", "2", "3-5", "6-10", "11+"]).to_string())
        print()
        print("  Candidate rule: <unk> AND n_words <= N  (rows that would be flagged)")
        for n in [0, 1, 2, 3, 5]:
            k = (unk["n_words"] <= n).sum()
            print(f"    n_words <= {n}: {k} rows ({k / total:.2%} of all rows)")
        print()
        print("  Sessions with the most <unk> rows:")
        print(unk.groupby("session").size().sort_values(ascending=False).head(5).to_string())
        print()
        print("  Sample of <unk> rows with <= 2 real words (eyeball these):")
        sample = unk[unk["n_words"] <= 2].head(10)
        for _, r in sample.iterrows():
            print(f"    {r['session']} @ {r['start_time']:.1f}s | "
                  f"n_words={r['n_words']} | {r['sentence']}")
    print()

    # --- Re-check the extreme density rows now that <sil> is counted ---
        # --- Where does the <unk> sit? (needs unk_leading / unk_trailing) ---
    if {"unk_leading", "unk_trailing"} <= set(rows.columns):
        lead = rows["unk_leading"].astype(bool)
        trail = rows["unk_trailing"].astype(bool)
        has_unk = rows["n_unk_tags"] > 0
        print("<unk> position:")
        print(f"  leading:  {lead.sum()} rows ({lead.sum() / total:.2%} of all rows)")
        print(f"  trailing: {trail.sum()} rows ({trail.sum() / total:.2%} of all rows)")
        print(f"  both:     {(lead & trail).sum()}")
        print(f"  <unk> rows where it is mid-utterance only: {(has_unk & ~lead & ~trail).sum()}")
        print(f"  sanity (flag set but n_unk_tags == 0, should be 0): {((lead | trail) & ~has_unk).sum()}")
        print("  Sample of trailing-<unk> rows (last characters of the sentence):")
        for _, r in rows[trail].head(10).iterrows():
            print(f"    {r['session']} @ {r['start_time']:.1f}s | ...{str(r['sentence'])[-25:]}")
        print()
    high = rows[rows["tag_density"] >= 0.7]
    print(f"Rows with tag_density >= 0.7: {len(high)}")
    for _, r in high.iterrows():
        print(f"  {r['session']} @ {r['start_time']:.1f}s | "
              f"density={r['tag_density']:.2f} | n_words={r['n_words']} | {r['sentence']}")


if __name__ == "__main__":
    main()