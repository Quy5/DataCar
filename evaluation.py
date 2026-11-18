"""
evaluation.py - Đánh giá hiệu năng mô hình ID3 và Baseline đối chứng.
Đồ án môn Khai phá dữ liệu - Nhóm 06 - QT01.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

TARGET_CLASSES = ["unacc", "acc", "good", "vgood"]

class BaselineMajorityClassifier:
    """
    Mô hình baseline đối chứng: luôn dự đoán lớp chiếm đa số trong tập huấn luyện.
    """
    def __init__(self):
        self.majority_class: str = ""
        self.train_distribution: Dict[str, int] = {}
        
    def fit(self, y_train: List[str]) -> "BaselineMajorityClassifier":
        counts = pd.Series(y_train).value_counts()
        self.majority_class = str(counts.index[0])
        self.train_distribution = counts.to_dict()
        return self
        
    def predict(self, n_samples: int) -> List[str]:
        return [self.majority_class] * n_samples

def evaluate_predictions(y_true: List[str], y_pred: List[str], labels: List[str] = TARGET_CLASSES) -> Dict[str, Any]:
    """
    Tính toán toàn bộ các chỉ số đánh giá:
    - Accuracy
    - Precision, Recall, F1 theo từng nhãn
    - Macro Precision, Macro Recall, Macro F1
    - Weighted Precision, Weighted Recall, Weighted F1
    - Confusion Matrix (mảng 2D và dict)
    """
    acc = float(accuracy_score(y_true, y_pred))
    
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )
    
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )
    
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    per_class_metrics = {}
    for i, cls in enumerate(labels):
        per_class_metrics[cls] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1_score": round(float(f1[i]), 4),
            "support": int(sup[i])
        }
        
    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_prec), 4),
        "weighted_recall": round(float(weighted_rec), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "labels": labels
    }

def compare_models(
    y_test: List[str],
    id3_preds: List[str],
    baseline_preds: List[str],
    labels: List[str] = TARGET_CLASSES
) -> Dict[str, Any]:
    """
    So sánh chi tiết hiệu năng giữa mô hình ID3 và Baseline.
    """
    id3_metrics = evaluate_predictions(y_test, id3_preds, labels=labels)
    baseline_metrics = evaluate_predictions(y_test, baseline_preds, labels=labels)
    
    comparison_table = []
    comparison_table.append({
        "Metric": "Độ chính xác (Accuracy)",
        "Baseline": f"{baseline_metrics['accuracy']:.4f} ({baseline_metrics['accuracy']*100:.2f}%)",
        "ID3": f"{id3_metrics['accuracy']:.4f} ({id3_metrics['accuracy']*100:.2f}%)",
        "Chênh lệch": f"{(id3_metrics['accuracy'] - baseline_metrics['accuracy'])*100:+.2f}%"
    })
    comparison_table.append({
        "Metric": "Macro-F1",
        "Baseline": f"{baseline_metrics['macro_f1']:.4f}",
        "ID3": f"{id3_metrics['macro_f1']:.4f}",
        "Chênh lệch": f"{id3_metrics['macro_f1'] - baseline_metrics['macro_f1']:+.4f}"
    })
    comparison_table.append({
        "Metric": "Macro-Recall",
        "Baseline": f"{baseline_metrics['macro_recall']:.4f}",
        "ID3": f"{id3_metrics['macro_recall']:.4f}",
        "Chênh lệch": f"{id3_metrics['macro_recall'] - baseline_metrics['macro_recall']:+.4f}"
    })
    
    for cls in labels:
        b_f1 = baseline_metrics["per_class"][cls]["f1_score"]
        i_f1 = id3_metrics["per_class"][cls]["f1_score"]
        comparison_table.append({
            "Metric": f"F1-score lớp [{cls}]",
            "Baseline": f"{b_f1:.4f}",
            "ID3": f"{i_f1:.4f}",
            "Chênh lệch": f"{i_f1 - b_f1:+.4f}"
        })
        
    return {
        "id3": id3_metrics,
        "baseline": baseline_metrics,
        "comparison_table": comparison_table
    }

def analyze_errors(
    test_df: pd.DataFrame,
    y_true: List[str],
    y_pred: List[str],
    target_col: str = "class"
) -> pd.DataFrame:
    """
    Phân tích chi tiết các mẫu dự đoán sai.
    """
    df = test_df.copy()
    df["Thực tế"] = y_true
    df["Dự đoán"] = y_pred
    df["Đúng/Sai"] = [t == p for t, p in zip(y_true, y_pred)]
    
    errors_df = df[df["Đúng/Sai"] == False].copy()
    return errors_df



def cross_validate_id3(
    df: pd.DataFrame,
    features: List[str],
    target: str = "class",
    k: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Đánh giá chéo k-fold phân tầng (Stratified k-fold) cho ID3 trên toàn bộ dữ liệu.
    Mỗi fold: huấn luyện trên k-1 phần, kiểm tra trên phần còn lại.
    Bổ sung cho phương pháp Hold-out (80/20) theo bài giảng.
    """
    from id3 import ID3DecisionTree  # import tại chỗ để tránh vòng lặp import

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    folds = []
    for i, (tr_idx, te_idx) in enumerate(skf.split(df, df[target]), 1):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        model = ID3DecisionTree().fit(tr, attributes=features, target=target)
        pred = model.predict(te)
        m = evaluate_predictions(te[target].tolist(), pred)
        folds.append({"fold": i, "test_samples": len(te), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})

    accs = np.array([f["accuracy"] for f in folds])
    f1s = np.array([f["macro_f1"] for f in folds])
    return {
        "k": k,
        "folds": folds,
        "accuracy_mean": round(float(accs.mean()), 4),
        "accuracy_std": round(float(accs.std(ddof=1)), 4),
        "macro_f1_mean": round(float(f1s.mean()), 4),
        "macro_f1_std": round(float(f1s.std(ddof=1)), 4),
    }

def analyze_empty_leaf_predictions(model: Any, test_df: pd.DataFrame, target: str = "class") -> Dict[str, Any]:
    """
    Thống kê các mẫu test rơi vào lá của nhánh rỗng (lá không có mẫu huấn luyện,
    nhãn lấy theo đa số của nút cha) và độ chính xác ở nhóm này so với phần còn lại.
    """
    empty_total = empty_correct = other_total = other_correct = 0
    for rec in test_df.to_dict(orient="records"):
        res = model.predict_with_path(rec)
        ok = res["prediction"] == rec[target]
        if res.get("is_empty_leaf"):
            empty_total += 1
            empty_correct += ok
        else:
            other_total += 1
            other_correct += ok
    total_errors = (empty_total - empty_correct) + (other_total - other_correct)
    return {
        "empty_leaf_samples": empty_total,
        "empty_leaf_correct": empty_correct,
        "empty_leaf_accuracy": round(empty_correct / empty_total, 4) if empty_total else None,
        "other_samples": other_total,
        "other_correct": other_correct,
        "other_accuracy": round(other_correct / other_total, 4) if other_total else None,
        "total_errors": total_errors,
        "errors_from_empty_leaves": empty_total - empty_correct,
    }
