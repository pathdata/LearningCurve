"""
JUMP-CP Batch QC Script  —  v2  (aligned with jump_cp_pipeline_PN_v4.py)
=========================================================================
Fixes applied from v4 pipeline

  FIX L1  — top-5 worst NaN offender features logged (mirrors Stage 3 v4).
  Thresholds mirror v4 config:
      MAX_NAN_RATE = 0.05   (5 %)
      MIN_VARIANCE = 1e-6
      MAX_CORR     = 0.95
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
import warnings
import datetime

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Uncomment the batch you want to run:
# root = Path(r"C:\Users\priya\jump_cp_profiles\2021_08_23_Batch12")
root = Path(r"C:\Users\priya\jump_cp_profiles\2021_08_09_Batch11")

# Feature-selection thresholds — kept identical to v4 pipeline config
MAX_NAN_RATE = 0.05    # drop features with > 5 % NaN across dataset
MIN_VARIANCE = 1e-6    # drop near-zero variance features
MAX_CORR     = 0.95    # drop one of each highly-correlated pair

# Output directory — excluded from glob so re-runs don't double-count (FIX L1)
OUT_DIR = root / "_batch_qc_outputs"

# ── LOGGING ───────────────────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}]  {msg}")


# ── INVENTORY  (FIX B1 + FIX L1) ─────────────────────────────────────────────
def inventory(root: Path) -> list[dict]:
    """
    Walk root recursively for parquet files.

    FIX L1: skips any file inside OUT_DIR so previously written QC parquets
            are not re-ingested as raw data (would double well counts and
            inflate NaN rates from column-count mismatch).

    FIX B1: batch name comes from the first subdirectory of root, not from
            the file stem.  Falls back to 'unknown' and warns if all parquets
            are flat in root with no subdirectories.
    """
    out_resolved = OUT_DIR.resolve()
    records = []
    for pq in sorted(root.rglob("*.parquet")):
        # FIX L1 — skip files inside the output folder
        try:
            pq.relative_to(out_resolved)
            continue
        except ValueError:
            pass

        parts = pq.relative_to(root).parts
        # FIX B1 — need at least batch/plate.parquet
        batch = parts[0] if len(parts) >= 2 else "unknown"
        records.append({"batch": batch, "plate": pq.stem, "path": str(pq)})

    # FIX B1 — warn if no subdirectory structure found
    if records and all(r["batch"] == "unknown" for r in records):
        log("WARNING: no batch subdirectories found — all files assigned "
            "batch='unknown'.  Check that parquets live under "
            "root/<batch>/<plate>.parquet")

    log(f"Found {len(records)} parquet files, "
        f"{len({r['batch'] for r in records})} batches")
    return records


# ── LOAD ONE  (FIX B2) ────────────────────────────────────────────────────────
def load_one(path: str, plate: str) -> pd.DataFrame:
    """
    FIX B2: df.get() silently leaves NaN values in an existing but partially
    null Metadata_Plate column.  Explicit fillna() ensures every row has a
    valid plate label so groupby('Metadata_Plate') never creates a NaN group.
    """
    df = pd.read_parquet(path)
    if "Metadata_Plate" not in df.columns:
        df["Metadata_Plate"] = plate
    else:
        df["Metadata_Plate"] = df["Metadata_Plate"].fillna(plate)
    return df


# ── CONCAT WITH SCHEMA CHECK  (FIX B3) ───────────────────────────────────────
def load_all(records: list[dict]) -> pd.DataFrame:
    """
    FIX B3: checks for column-set mismatches across plates before concat.
    Mismatched schemas cause silent NaN inflation in every feature column that
    is absent from at least one plate — this shows up as an inflated NaN rate
    and can cause the NaN filter to drop the majority of features.
    """
    frames = []
    for rec in records:
        try:
            frames.append(load_one(rec["path"], rec["plate"]))
        except Exception as exc:
            log(f"  WARNING: skipped {rec['plate']}: {exc}")

    if not frames:
        raise RuntimeError("No parquet files could be loaded.")

    # FIX B3 — schema mismatch check
    col_sets = [set(f.columns) for f in frames]
    if len(set(map(frozenset, col_sets))) > 1:
        common = set.intersection(*col_sets)
        max_cols = max(len(s) for s in col_sets)
        log(f"  WARNING: schema mismatch across plates — "
            f"{len(common)} shared columns out of up to {max_cols}.  "
            f"Concat will fill missing columns with NaN.")

    df = pd.concat(frames, ignore_index=True, sort=False)
    log(f"Combined shape: {df.shape}")
    return df


# ── PER-PLATE QC ──────────────────────────────────────────────────────────────
def per_plate_qc(df: pd.DataFrame) -> pd.DataFrame:
    """Compute QC statistics per plate, matching Stage 2 of the v4 pipeline."""
    rows = []
    for plate, grp in df.groupby("Metadata_Plate"):
        feat_cols = [c for c in grp.columns if not c.startswith("Metadata")]
        feats = grp[feat_cols].select_dtypes(include=np.number)
        variances = feats.var(skipna=True)
        rows.append({
            "plate":             plate,
            "n_wells":           len(grp),
            "nan_rate":          feats.isnull().mean().mean(),
            "median_variance":   variances.median(),
            "min_variance":      variances.min(),
            "max_variance":      variances.max(),
            # threshold matches MIN_VARIANCE in v4 config
            "zero_var_features": int((variances < MIN_VARIANCE).sum()),
            # flag plates that exceed the NaN threshold
            "nan_flag":          feats.isnull().mean().mean() > MAX_NAN_RATE,
        })
    return pd.DataFrame(rows)


# ── DATASET-LEVEL FEATURE SELECTION  (FIX B4, B5, L2) ───────────────────────
def feature_selection(df: pd.DataFrame) -> dict:
    """
    Apply the same three-stage feature selection as Stage 3 of the v4 pipeline.

    3-a  NaN filter      : drop columns with > MAX_NAN_RATE missing values.
         FIX L2          : log top-5 worst NaN offenders.
    3-b  Variance filter : drop near-zero-variance columns (< MIN_VARIANCE).
    3-c  Correlation filter: drop one of each pair with |r| >= MAX_CORR.
         FIX B4          : boolean mask used to extract variance-passing
                           columns from the imputed array — avoids O(n²)
                           list scans and duplicate-name bugs.
         FIX B5          : track column NAMES to drop, not integer positions,
                           so filtering is unambiguous regardless of ordering.
    """
    feat_cols = [c for c in df.columns if not c.startswith("Metadata")]
    feats = df[feat_cols].select_dtypes(include=np.number)
    n0 = feats.shape[1]
    log(f"Feature selection starting with {n0:,} features")

    # 3-a  NaN filter
    nan_rates = feats.isnull().mean()
    keep_nan  = nan_rates[nan_rates <= MAX_NAN_RATE].index.tolist()
    dropped_nan = n0 - len(keep_nan)
    log(f"  After NaN filter  ({MAX_NAN_RATE*100:.0f}%): "
        f"{len(keep_nan):,} / {n0:,}  (dropped {dropped_nan:,})")
    # FIX L2 — log top offenders so schema mismatches are immediately visible
    if dropped_nan > 0:
        top_nan = (nan_rates[nan_rates > MAX_NAN_RATE]
                   .sort_values(ascending=False).head(5))
        log("  Top NaN features dropped:")
        for feat, rate in top_nan.items():
            log(f"    {feat}: {rate*100:.1f}% NaN")
        if dropped_nan > 5:
            log(f"    ... and {dropped_nan - 5:,} more")

    # 3-b  Variance filter (impute first so var() is not skewed by NaN)
    imp  = SimpleImputer(strategy="median")
    arr  = imp.fit_transform(feats[keep_nan])
    vars_ = arr.var(axis=0)
    keep_var = [c for c, v in zip(keep_nan, vars_) if v > MIN_VARIANCE]
    log(f"  After variance filter ({MIN_VARIANCE}): {len(keep_var):,}")

    # 3-c  Correlation filter
    # FIX B4 — boolean mask to select variance-passing columns from arr
    keep_nan_arr = np.array(keep_nan)
    var_mask     = np.isin(keep_nan_arr, keep_var)
    arr_v        = arr[:, var_mask]
    sample = min(2000, arr_v.shape[0])
    idx    = np.random.default_rng(42).choice(arr_v.shape[0], sample, replace=False)
    corr_m = np.corrcoef(arr_v[idx].T)

    # FIX B5 — track names to drop, not positions
    drop_names = set()
    for i in range(len(keep_var)):
        if keep_var[i] in drop_names:
            continue
        for j in range(i + 1, len(keep_var)):
            if abs(corr_m[i, j]) >= MAX_CORR:
                drop_names.add(keep_var[j])
    keep_final = [c for c in keep_var if c not in drop_names]
    log(f"  After correlation filter ({MAX_CORR}): {len(keep_final):,}")

    return {
        "n_raw":        n0,
        "n_after_nan":  len(keep_nan),
        "n_after_var":  len(keep_var),
        "n_final":      len(keep_final),
        "dropped_nan":  dropped_nan,
        "dropped_var":  len(keep_nan) - len(keep_var),
        "dropped_corr": len(keep_var) - len(keep_final),
        "keep_final":   keep_final,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log(f"Root: {root}")
    log(f"Out:  {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Inventory
    records = inventory(root)
    if not records:
        log("No parquet files found. Check root path.")
        raise SystemExit(1)

    # 2. Load
    df = load_all(records)

    # 3. Per-plate QC (mirrors Stage 2)
    log("Running per-plate QC …")
    qc = per_plate_qc(df)
    flagged = qc["nan_flag"].sum()
    log(f"  {flagged} / {len(qc)} plates exceed NaN threshold ({MAX_NAN_RATE*100:.0f}%)")
    print("\n── Per-plate QC ──────────────────────────────────────────────────────")
    print(qc.to_string(index=False))

    # 4. Dataset-level feature selection (mirrors Stage 3)
    log("\nRunning dataset-level feature selection …")
    fs = feature_selection(df)
    print("\n── Feature selection summary ─────────────────────────────────────────")
    print(f"  Raw features          : {fs['n_raw']:,}")
    print(f"  After NaN filter      : {fs['n_after_nan']:,}  (dropped {fs['dropped_nan']:,})")
    print(f"  After variance filter : {fs['n_after_var']:,}  (dropped {fs['dropped_var']:,})")
    print(f"  After corr filter     : {fs['n_final']:,}  (dropped {fs['dropped_corr']:,})")

    # 5. Save outputs
    qc_path = OUT_DIR / "plate_qc.csv"
    qc.to_csv(qc_path, index=False)
    log(f"\nSaved plate QC → {qc_path}")

    fs_summary = pd.DataFrame([{k: v for k, v in fs.items() if k != "keep_final"}])
    fs_path = OUT_DIR / "feature_selection_summary.csv"
    fs_summary.to_csv(fs_path, index=False)
    log(f"Saved feature selection summary → {fs_path}")

    kept_path = OUT_DIR / "kept_features.txt"
    kept_path.write_text("\n".join(fs["keep_final"]))
    log(f"Saved {fs['n_final']:,} kept feature names → {kept_path}")

    log("Done.")
