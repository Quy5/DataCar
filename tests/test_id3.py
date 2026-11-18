"""
test_id3.py - Kiểm thử thuật toán ID3: Entropy, Information Gain, Dừng đệ quy, Cây đa nhánh.
"""

import math
import pytest
import pandas as pd
from id3 import entropy, information_gain, get_majority_class, ID3DecisionTree

def test_entropy_pure_set():
    """1. Entropy của tập thuần nhất phải bằng 0.0."""
    pure_unacc = ["unacc"] * 100
    assert entropy(pure_unacc) == 0.0
    
    pure_vgood = ["vgood"] * 50
    assert entropy(pure_vgood) == 0.0

def test_entropy_empty_set():
    """Entropy của tập rỗng trả về 0.0 và không lỗi."""
    assert entropy([]) == 0.0

def test_entropy_multiclass():
    """2. Entropy trên dữ liệu nhiều lớp tính chính xác theo lý thuyết."""
    # 2 lớp cân bằng 50/50: - (0.5 * log2(0.5) + 0.5 * log2(0.5)) = 1.0
    binary_balanced = ["acc"] * 50 + ["unacc"] * 50
    assert math.isclose(entropy(binary_balanced), 1.0, rel_tol=1e-6)
    
    # 4 lớp cân bằng 25% mỗi lớp: log2(4) = 2.0
    four_classes = ["unacc"] * 25 + ["acc"] * 25 + ["good"] * 25 + ["vgood"] * 25
    assert math.isclose(entropy(four_classes), 2.0, rel_tol=1e-6)

def test_majority_class_and_tie_breaking():
    """Kiểm tra nhãn đa số và quy tắc giải quyết hòa (tie-breaking) nhất quán."""
    labels = ["unacc", "unacc", "acc", "good"]
    assert get_majority_class(labels) == "unacc"
    
    # Hòa 1-1 giữa unacc và vgood -> ưu tiên theo DEFAULT_CLASS_ORDER: unacc trước
    tied_labels = ["vgood", "unacc"]
    assert get_majority_class(tied_labels, class_order=["unacc", "acc", "good", "vgood"]) == "unacc"

def test_information_gain_calculation():
    """3. Information Gain tính đúng theo công thức."""
    # Tạo toy dataset có thể kiểm chứng bằng tay
    data = pd.DataFrame({
        "color": ["red", "red", "blue", "blue"],
        "class": ["yes", "yes", "no", "no"]
    })
    # Ở đây 'color' phân chia hoàn hảo, Gain = Entropy(S) - 0 = 1.0
    gain, base_ent, sub_ents = information_gain(data, "color", "class")
    assert math.isclose(base_ent, 1.0, rel_tol=1e-6)
    assert math.isclose(gain, 1.0, rel_tol=1e-6)
    assert sub_ents["red"] == 0.0
    assert sub_ents["blue"] == 0.0

def test_best_attribute_selection():
    """4. Thuật toán chọn thuộc tính có Gain cao nhất làm nút gốc."""
    data = pd.DataFrame({
        "weather": ["sunny", "sunny", "rain", "rain"],
        "temp": ["hot", "cold", "hot", "cold"],
        "class": ["yes", "yes", "no", "no"]
    })
    # 'weather' quyết định nhãn 100%, 'temp' không phân tách được
    tree = ID3DecisionTree()
    tree.fit(data, attributes=["weather", "temp"], target="class")
    assert tree.root.attribute == "weather"
    assert tree.root.is_leaf is False

def test_stopping_condition_all_pure():
    """6. Điều kiện dừng: Khi tất cả mẫu cùng nhãn, tạo ngay nút lá."""
    data = pd.DataFrame({
        "feature_a": ["A", "B", "C"],
        "class": ["good", "good", "good"]
    })
    tree = ID3DecisionTree()
    tree.fit(data, attributes=["feature_a"], target="class")
    assert tree.root.is_leaf is True
    assert tree.root.prediction == "good"

def test_stopping_condition_no_attributes_left_multiclass():
    """7. Hết thuộc tính nhưng còn nhiều lớp -> Gán nhãn đa số."""
    data = pd.DataFrame({
        "feature_a": ["A", "A", "A"],
        "class": ["acc", "acc", "unacc"]
    })
    # feature_a có 1 giá trị duy nhất, sau khi chọn xong không còn thuộc tính nào
    tree = ID3DecisionTree()
    tree.fit(data, attributes=["feature_a"], target="class")
    # Thuộc tính có Gain = 0 nên dừng ngay và lấy majority là 'acc'
    assert tree.root.is_leaf is True
    assert tree.root.prediction == "acc"

def test_multi_branch_tree_structure():
    """5. Tạo cây đa nhánh (multi-way split)."""
    data = pd.DataFrame({
        "size": ["S", "M", "L", "XL"],
        "class": ["unacc", "acc", "good", "vgood"]
    })
    tree = ID3DecisionTree()
    tree.fit(data, attributes=["size"], target="class")
    assert tree.root.attribute == "size"
    assert len(tree.root.children) == 4
    for val in ["S", "M", "L", "XL"]:
        assert val in tree.root.children
        assert tree.root.children[val].is_leaf is True

