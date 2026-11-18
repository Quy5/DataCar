"""
imbalance_experiment.py - Thử nghiệm xử lý mất cân bằng lớp cho ID3 (Nhóm 06).

So sánh 3 cách trên cùng cách chia dữ liệu:
  1. Giữ nguyên phân bố gốc (mô hình chính của nhóm)
  2. Random Oversampling: nhân bản ngẫu nhiên mẫu lớp ít trên TẬP TRAIN cho bằng lớp nhiều nhất
  3. Random Undersampling: cắt ngẫu nhiên mọi lớp trên TẬP TRAIN về bằng lớp ít nhất
Tập kiểm tra luôn giữ nguyên phân bố gốc (không lấy mẫu lại).

Đánh giá: Hold-out 80/20 (random_state=42) và 10-fold phân tầng.
Chạy:  python imbalance_experiment.py   -> in bảng và lưu output/imbalance_results.json
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from data_utils import FEATURE_COLUMNS, TARGET_COLUMN, load_data, split_data
from evaluation import evaluate_predictions
from id3 import ID3DecisionTree

SEED = 42


def keep(train: pd.DataFrame) -> pd.DataFrame:
    return train


def oversample(train: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.RandomState(SEED)
    biggest = train[TARGET_COLUMN].value_counts().max()
    parts = []
    for _, g in train.groupby(TARGET_COLUMN):
        parts.append(g)
        if len(g) < biggest:
            parts.append(g.sample(biggest - len(g), replace=True, random_state=rng))
    return pd.concat(parts, ignore_index=True)


def undersample(train: pd.DataFrame) -> pd.DataFrame:
    smallest = train[TARGET_COLUMN].value_counts().min()
    return pd.concat([g.sample(smallest, random_state=SEED) for _, g in train.groupby(TARGET_COLUMN)],
                     ignore_index=True)


METHODS = {"Giữ nguyên": keep, "Random Oversampling": oversample, "Random Undersampling": undersample}


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    model = ID3DecisionTree().fit(train, attributes=FEATURE_COLUMNS, target=TARGET_COLUMN)
    return evaluate_predictions(test[TARGET_COLUMN].tolist(), model.predict(test))


def main():
    df = load_data("data/car.data")
    train, test = split_data(df, test_size=0.2, random_state=SEED, stratify=True)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    out = {}
    for name, fn in METHODS.items():
        hold = evaluate(fn(train), test)
        folds = [evaluate(fn(df.iloc[a]), df.iloc[b]) for a, b in skf.split(df, df[TARGET_COLUMN])]
        mean = lambda key: float(np.mean([f[key] for f in folds]))
        std = lambda key: float(np.std([f[key] for f in folds], ddof=1))
        rec = lambda c: float(np.mean([f["per_class"][c]["recall"] for f in folds]))
        out[name] = {
            "train_size": len(fn(train)),
            "holdout": {"accuracy": hold["accuracy"], "macro_f1": hold["macro_f1"],
                        "confusion_matrix": hold["confusion_matrix"]},
            "cv10": {"accuracy_mean": round(mean("accuracy"), 4), "accuracy_std": round(std("accuracy"), 4),
                     "macro_f1_mean": round(mean("macro_f1"), 4), "macro_f1_std": round(std("macro_f1"), 4),
                     "recall_good": round(rec("good"), 4), "recall_vgood": round(rec("vgood"), 4)},
        }
        cvr = out[name]["cv10"]
        print(f"{name:22s} train={out[name]['train_size']:5d} | Hold-out acc={hold['accuracy']:.4f} "
              f"mF1={hold['macro_f1']:.4f} | 10-fold acc={cvr['accuracy_mean']:.4f}±{cvr['accuracy_std']:.4f} "
              f"mF1={cvr['macro_f1_mean']:.4f} recall good={cvr['recall_good']:.3f} vgood={cvr['recall_vgood']:.3f}")
    os.makedirs("output", exist_ok=True)
    with open("output/imbalance_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
