"""Slide-level KFold splitting; each input row must be one unique slide.

The historical filename is retained. The outer KFold is NOT stratified;
only the train/validation split is stratified by STAS. Patients are not grouped.
"""

import argparse
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import os

def kfold_with_val(seed, total_label, output_folder, k=5, val_ratio=0.2):
    os.makedirs(output_folder, exist_ok=True)

    # 读取数据
    df = pd.read_csv(total_label) #encoding='gbk'
    if df[['path', 'STAS']].isna().any().any():
        raise ValueError('Slide IDs and labels must not be missing')
    if df['path'].duplicated().any():
        raise ValueError('Expected one row per unique slide ID')
    X = df['path']
    y = df['STAS']
    # X = list(df['path'])
    # y = [str(i) for i in list(df['STAS'])]
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    for fold, (trainval_idx, test_idx) in enumerate(kf.split(X, y)):
        # 拆出 Test
        test_df = df.iloc[test_idx]
        trainval_df = df.iloc[trainval_idx]

        # 从 TrainVal 中再划分 Val
        train_df, val_df = train_test_split(
            trainval_df,
            test_size=val_ratio,
            stratify=trainval_df['STAS'],
            random_state=seed
        )

        # 保存每一折的数据
        train_df.to_csv(os.path.join(output_folder, f'fold{fold}_train.csv'), index=False)
        val_df.to_csv(os.path.join(output_folder, f'fold{fold}_val.csv'), index=False)
        test_df.to_csv(os.path.join(output_folder, f'fold{fold}_test.csv'), index=False)

    print(f"{k}-折交叉验证划分完成（每折含 train / val / test）")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', required=True, help='CSV containing path and STAS')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Fraction of the train+validation pool used for validation')
    args = parser.parse_args()
    kfold_with_val(args.seed, args.input_csv, args.output_dir, args.folds, args.val_ratio)

if __name__ == '__main__':
    main()
