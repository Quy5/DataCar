"""
test_prediction.py - Kiểm thử cơ chế dự đoán, fallback khi gặp giá trị chưa thấy, đường đi quyết định và metrics.
"""

import pytest
import pandas as pd
from id3 import ID3DecisionTree
from evaluation import evaluate_predictions, BaselineMajorityClassifier

def test_prediction_batch_and_single():
    """10. Dự đoán một và nhiều bản ghi."""
    df = pd.DataFrame({
        "safety": ["low", "high", "low", "high"],
        "persons": ["2", "4", "2", "4"],
        "class": ["unacc", "acc", "unacc", "acc"]
    })
    tree = ID3DecisionTree()
    tree.fit(df, attributes=["safety", "persons"], target="class")
    
    sample1 = {"safety": "low", "persons": "2"}
    assert tree.predict_one(sample1) == "unacc"
    
    sample2 = {"safety": "high", "persons": "4"}
    assert tree.predict_one(sample2) == "acc"
    
    batch_preds = tree.predict(df)
    assert batch_preds == ["unacc", "acc", "unacc", "acc"]

def test_unseen_value_fallback_mechanism():
    """8, 9. Xử lý giá trị chưa xuất hiện trong nhánh hoặc nhánh rỗng: kích hoạt fallback."""
    df = pd.DataFrame({
        "safety": ["low", "high"],
        "class": ["unacc", "acc"]
    })
    tree = ID3DecisionTree()
    tree.fit(df, attributes=["safety"], target="class")
    
    # "med" chưa từng xuất hiện trong tập huấn luyện của thuộc tính "safety"
    unseen_sample = {"safety": "med"}
    res = tree.predict_with_path(unseen_sample)
    
    assert res["is_fallback"] is True
    assert "không có nhánh tương ứng" in res["fallback_reason"]
    # Fallback phải trả về nhãn đa số tại nút hiện tại
    assert res["prediction"] in ["unacc", "acc"]

def test_decision_path_explanation():
    """Đường đi quyết định ghi nhận đầy đủ các nút và thuộc tính đã duyệt."""
    df = pd.DataFrame({
        "safety": ["low", "high", "high"],
        "persons": ["2", "2", "4"],
        "class": ["unacc", "unacc", "vgood"]
    })
    tree = ID3DecisionTree()
    tree.fit(df, attributes=["safety", "persons"], target="class")
    
    sample = {"safety": "high", "persons": "4"}
    res = tree.predict_with_path(sample)
    
    assert res["prediction"] == "vgood"
    assert len(res["path"]) >= 2
    # Nút đầu tiên phải có thuộc tính được kiểm tra
    assert "attribute" in res["path"][0]
    # Nút cuối cùng là nút lá
    assert res["path"][-1].get("is_leaf") is True

def test_metrics_evaluation_matches_ground_truth():
    """14. Các chỉ số đánh giá tính toán chính xác theo công thức."""
    y_true = ["unacc", "unacc", "acc", "good"]
    y_pred = ["unacc", "unacc", "acc", "vgood"]  # 3 đúng, 1 sai
    
    metrics = evaluate_predictions(y_true, y_pred, labels=["unacc", "acc", "good", "vgood"])
    assert metrics["accuracy"] == 0.75
    # unacc: 2 true pos, 0 false pos -> prec=1.0, rec=1.0, f1=1.0
    assert metrics["per_class"]["unacc"]["precision"] == 1.0
    assert metrics["per_class"]["unacc"]["recall"] == 1.0
    assert metrics["per_class"]["unacc"]["f1_score"] == 1.0
    
    # good: 1 true pos nhưng pred sai thành vgood -> recall = 0
    assert metrics["per_class"]["good"]["recall"] == 0.0

def test_baseline_classifier():
    """Baseline đối chứng luôn dự đoán lớp phổ biến nhất."""
    y_train = ["unacc", "unacc", "unacc", "acc"]
    base = BaselineMajorityClassifier()
    base.fit(y_train)
    assert base.majority_class == "unacc"
    assert base.predict(3) == ["unacc", "unacc", "unacc"]

