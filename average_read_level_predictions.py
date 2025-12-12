#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd


def average_read_level_predictions(in1, in2, out_path, threshold=0.1):
    """Average read-level predictions from two TSV files.

    Assumptions:
    - Both inputs are read-level prediction TSVs with columns:
        site_id, prediction
      and optionally: seq.name, seq.pos, structure.
    - Within each site_id group, rows在两个文件中按相同顺序对应。
    - 如果同一个 site_id 在两个文件中的出现次数不同，则只对
      min(n1, n2) 行做均值，剩余多出的行会被丢弃。
    - 只保留两个文件都包含的 site_id（与原 site-level 合并脚本一致）。
    """
    if not os.path.exists(in1):
        print(f"Error: file not found: {in1}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(in2):
        print(f"Error: file not found: {in2}", file=sys.stderr)
        sys.exit(1)

    df1 = pd.read_csv(in1, sep="\t")
    df2 = pd.read_csv(in2, sep="\t")

    for name, df in [("input1", df1), ("input2", df2)]:
        if "site_id" not in df.columns or "prediction" not in df.columns:
            print(f"Error: {name} must contain columns: site_id, prediction", file=sys.stderr)
            sys.exit(1)

    # 只在两个文件都出现的 site_id 上进行合并
    site_ids_1 = pd.unique(df1["site_id"])
    site_ids_2 = pd.unique(df2["site_id"])
    common_site_ids = [sid for sid in site_ids_1 if sid in set(site_ids_2)]

    # 用于对齐检查的列
    align_cols = [c for c in ["seq.name", "seq.pos", "structure"] if c in df1.columns and c in df2.columns]

    g1 = df1.groupby("site_id", sort=False)
    g2 = df2.groupby("site_id", sort=False)

    rows = []

    for sid in common_site_ids:
        g1_site = g1.get_group(sid).reset_index(drop=True)
        g2_site = g2.get_group(sid).reset_index(drop=True)

        n1, n2 = len(g1_site), len(g2_site)
        n = min(n1, n2)
        if n == 0:
            continue

        # 对齐检查：在前 n 行上确保关键列一致
        for col in align_cols:
            v1 = g1_site.loc[: n - 1, col].astype(str).values
            v2 = g2_site.loc[: n - 1, col].astype(str).values
            if not (v1 == v2).all():
                print(
                    f"Alignment mismatch for site_id {sid} in column {col}. "
                    "Read order is inconsistent between files.",
                    file=sys.stderr,
                )
                sys.exit(1)

        # 取前 n 行做均值
        sub1 = g1_site.iloc[:n].copy()
        sub2 = g2_site.iloc[:n].copy()

        p1 = pd.to_numeric(sub1["prediction"], errors="coerce")
        p2 = pd.to_numeric(sub2["prediction"], errors="coerce")
        avg_p = (p1 + p2) / 2.0

        sub1["prediction"] = avg_p
        sub1 = sub1.dropna(subset=["prediction"])
        rows.append(sub1)

    if rows:
        m = pd.concat(rows, ignore_index=True)
    else:
        # 没有可合并的行
        m = pd.DataFrame(columns=["site_id", "seq.name", "seq.pos", "structure", "prediction"])

    # 使用 > threshold 判定是否为修饰 read
    m["modified"] = (m["prediction"] > threshold).astype(int)

    # 输出列顺序：site_id, 可选列, prediction, modified
    out_cols = ["site_id"]
    for c in ["seq.name", "seq.pos", "structure"]:
        if c in m.columns:
            out_cols.append(c)
    out_cols += ["prediction", "modified"]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    m[out_cols].to_csv(out_path, sep="\t", index=False)
    print(f"Saved averaged read-level predictions to: {out_path} (n={len(m)})")


def main():
    parser = argparse.ArgumentParser(
        description="Average read-level predictions from two TSV files (line-aligned within each site_id)."
    )
    parser.add_argument("--in1", required=True, help="Path to first read-level prediction TSV")
    parser.add_argument("--in2", required=True, help="Path to second read-level prediction TSV")
    parser.add_argument("--out", required=True, help="Output TSV path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Probability threshold to call a read modified (default: 0.1, strictly greater than)",
    )

    args = parser.parse_args()
    average_read_level_predictions(args.in1, args.in2, args.out, threshold=args.threshold)


if __name__ == "__main__":
    main()
