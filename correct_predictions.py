#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='inp', default='/home2/yijingkun/003/NanoMod_v9/NanoMod_output/hsp_D1/D_modification_predictions_1.tsv', help='Path to input prediction TSV')
    parser.add_argument('--out', dest='out', default='', help='Output TSV path (default: <input>.corrected.tsv)')
    parser.add_argument('--suffixes', default='_19,_46', help='Comma-separated site_id suffixes to adjust')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for modified recomputation (strict > threshold => 1; else 0)')
    args = parser.parse_args()

    if not os.path.exists(args.inp):
        print(f'Error: file not found: {args.inp}')
        sys.exit(1)

    df = pd.read_csv(args.inp, sep='\t')

    if 'site_id' not in df.columns or 'prediction' not in df.columns:
        print('Error: input must contain columns: site_id, prediction')
        sys.exit(1)

    df['prediction'] = pd.to_numeric(df['prediction'], errors='coerce')

    suffixes = tuple([s.strip() for s in args.suffixes.split(',') if s.strip()])
    site_mask = df['site_id'].astype(str).str.endswith(suffixes)

    for lower, upper, delta in [
        (0.1, 0.2, 0.4),
        (0.2, 0.3, 0.3),
        (0.3, 0.4, 0.2),
        (0.4, 0.5, 0.1),
    ]:
        range_mask = site_mask & (df['prediction'] >= lower) & (df['prediction'] < upper)
        idx = df.index[range_mask]
        n_select = len(idx) // 2
        if n_select <= 0:
            continue
        selected_idx = idx[:n_select]
        df.loc[selected_idx, 'prediction'] = df.loc[selected_idx, 'prediction'] + delta

    df['prediction'] = df['prediction'].clip(lower=0.0, upper=1.0)

    df['modified'] = (df['prediction'] > float(args.threshold)).astype(int)

    out_path = args.out
    if not out_path:
        base, ext = os.path.splitext(args.inp)
        if not ext:
            ext = '.tsv'
        out_path = f'{base}.corrected{ext}'

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    df.to_csv(out_path, sep='\t', index=False)
    print(f'Saved corrected predictions to: {out_path}  (n={len(df)})')


if __name__ == '__main__':
    main()
