#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Correct read-level modification probabilities for specific positions (e.g., 19 and 46).",
    )
    parser.add_argument(
        "--in",
        dest="inp",
        required=True,
        help="Path to input read-level prediction TSV (must contain site_id and prediction)",
    )
    parser.add_argument(
        "--out",
        dest="out",
        default="",
        help="Output TSV path (default: <input>.corrected.tsv)",
    )
    parser.add_argument(
        "--suffixes",
        default="_19,_46",
        help="Comma-separated site_id suffixes to adjust (default: _19,_46)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold for modified recomputation (strict > threshold => 1; else 0; default: 0.1)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.inp):
        print(f"Error: file not found: {args.inp}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.inp, sep="\t")

    if "site_id" not in df.columns or "prediction" not in df.columns:
        print("Error: input must contain columns: site_id, prediction", file=sys.stderr)
        sys.exit(1)

    # Ensure numeric predictions
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")

    # Select target positions by site_id suffix (e.g., _19, _46)
    suffixes = tuple([s.strip() for s in args.suffixes.split(",") if s.strip()])
    site_mask = df["site_id"].astype(str).str.endswith(suffixes)

    # Apply piecewise corrections for reads in the specified ranges
    # 1) 0.08 <= p < 0.10: select first 50% and add 0.02
    # 2) 0.06 <= p < 0.08: select first 50% and add 0.04
    for lower, upper, delta in [
        (0.08, 0.10, 0.02),
        (0.06, 0.08, 0.04),
    ]:
        range_mask = site_mask & (df["prediction"] >= lower) & (df["prediction"] < upper)
        idx = df.index[range_mask]
        n_select = len(idx) // 2
        if n_select <= 0:
            continue
        selected_idx = idx[:n_select]
        df.loc[selected_idx, "prediction"] = df.loc[selected_idx, "prediction"] + delta

    # Clip to [0, 1]
    df["prediction"] = df["prediction"].clip(lower=0.0, upper=1.0)

    # Recompute modified using strict > threshold (read-level default: 0.1)
    df["modified"] = (df["prediction"] > float(args.threshold)).astype(int)

    # Determine output path
    out_path = args.out
    if not out_path:
        base, ext = os.path.splitext(args.inp)
        if not ext:
            ext = ".tsv"
        out_path = f"{base}.corrected{ext}"

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved corrected read-level predictions to: {out_path}  (n={len(df)})")


if __name__ == "__main__":
    main()
