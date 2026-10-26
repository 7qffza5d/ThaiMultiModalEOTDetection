# analyze_tag_density.py
"""Standalone: summarize tag_density across all session files in
timestamped/, to pick the section 4.1 filtering threshold from actual
data rather than a guess. Run after re-generating timestamped/ with the
tag_density field (see attach_timestamps_from_alignment)."""
import glob
import pandas as pd

def main(timestamped_dir="timestamped"):
    paths = sorted(glob.glob(f"{timestamped_dir}/*.parquet"))
    if not paths:
        print(f"No parquet files found in {timestamped_dir}/")
        return

    dfs = []
    for p in paths:
        df = pd.read_parquet(p)
        df["session"] = p.split("/")[-1].removesuffix(".parquet")
        dfs.append(df)
    all_rows = pd.concat(dfs, ignore_index=True)

    density = all_rows["tag_density"].dropna()
    print(f"Total rows: {len(all_rows)} across {len(paths)} sessions")
    print(f"Rows with tag_density computed: {len(density)}")
    print()
    print("Percentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  p{p}: {density.quantile(p / 100):.3f}")
    print()
    print("Rows at exactly 1.0 (100% tag tokens, zero real words):",
          (density == 1.0).sum())
    print()
    print("Counts at candidate thresholds (rows that WOULD be dropped/flagged):")
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        n = (density >= t).sum()
        print(f"  >= {t}: {n} rows ({n / len(density):.1%})")

    print()
    print("Top 5 sessions by mean tag_density (worth a manual look):")
    by_session = all_rows.groupby("session")["tag_density"].mean().sort_values(ascending=False)
    print(by_session.head(5).to_string())


if __name__ == "__main__":
    main()