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

def test_stopping_condition_zero_gain():
    """7a. Gain lớn nhất = 0 (thuộc tính không mang thông tin) -> dừng, gán nhãn đa số."""
    data = pd.DataFrame({
        "feature_a": ["A", "A", "A"],
        "class": ["acc", "acc", "unacc"]
    })
    tree = ID3DecisionTree()
    tree.fit(data, attributes=["feature_a"], target="class")
    assert tree.root.is_leaf is True
    assert tree.root.prediction == "acc"

def test_stopping_condition_no_attributes_left_multiclass():
    """7b. Đã dùng hết thuộc tính trên đường đi nhưng nút vẫn còn nhiều lớp -> lá mang nhãn đa số."""
    # Dữ liệu mâu thuẫn: nhánh x = "A" có 2 acc + 1 unacc nhưng không còn thuộc tính nào để chia tiếp
    data = pd.DataFrame({
        "x": ["A", "A", "A", "B", "B"],
        "class": ["acc", "acc", "unacc", "unacc", "unacc"]
    })
    tree = ID3DecisionTree()
    tree.fit(data, attributes=["x"], target="class")
    assert tree.root.attribute == "x"
    leaf = tree.root.children["A"]
    assert leaf.is_leaf is True
    assert leaf.entropy_val > 0          # lá không thuần nhất
    assert leaf.prediction == "acc"      # nhãn đa số 2/3

# Ví dụ AllElectronic trong bài giảng "Phân lớp dữ liệu" (Han & Kamber, Bảng 8.1)
ALL_ELECTRONICS = pd.DataFrame([
    ("youth", "high", "no", "fair", "no"), ("youth", "high", "no", "excellent", "no"),
    ("middle_aged", "high", "no", "fair", "yes"), ("senior", "medium", "no", "fair", "yes"),
    ("senior", "low", "yes", "fair", "yes"), ("senior", "low", "yes", "excellent", "no"),
    ("middle_aged", "low", "yes", "excellent", "yes"), ("youth", "medium", "no", "fair", "no"),
    ("youth", "low", "yes", "fair", "yes"), ("senior", "medium", "yes", "fair", "yes"),
    ("youth", "medium", "yes", "excellent", "yes"), ("middle_aged", "medium", "no", "excellent", "yes"),
    ("middle_aged", "high", "yes", "fair", "yes"), ("senior", "medium", "no", "excellent", "no"),
], columns=["age", "income", "student", "credit_rating", "buys_computer"])

def test_lecture_example_allelectronic_gain():
    """Khớp số liệu tính tay trong bài giảng: Info(D)=0.940, Gain(age)=0.246, ... và gốc là age."""
    expected = {"age": 0.246, "income": 0.029, "student": 0.151, "credit_rating": 0.048}
    for attr, g in expected.items():
        gain, base, _ = information_gain(ALL_ELECTRONICS, attr, "buys_computer")
        assert abs(base - 0.940) < 1e-3
        assert abs(gain - g) < 1e-3, f"Gain({attr}) = {gain:.4f}, kỳ vọng {g}"
    tree = ID3DecisionTree(attribute_order=["age", "income", "student", "credit_rating"],
                           class_order=["no", "yes"])
    tree.fit(ALL_ELECTRONICS, attributes=["age", "income", "student", "credit_rating"], target="buys_computer")
    assert tree.root.attribute == "age"
    assert tree.root.children["middle_aged"].is_leaf and tree.root.children["middle_aged"].prediction == "yes"
    assert tree.root.children["youth"].attribute == "student"
    assert tree.root.children["senior"].attribute == "credit_rating"

def test_empty_branch_leaf_and_rules():
    """Nhánh rỗng -> lá 0 mẫu mang nhãn đa số nút cha; get_rules() mặc định không tính lá này là luật."""
    train = pd.DataFrame({
        "x": ["a", "a", "a", "b", "b", "b"],
        "y": ["p", "p", "q", "p", "p", "p"],
        "class": ["acc", "acc", "unacc", "unacc", "unacc", "unacc"]
    })
    # Thêm mẫu (b, r): giá trị y = "r" có trong miền giá trị nhưng KHÔNG xuất hiện ở nhánh x = "a"
    # -> khi nhánh x = "a" chia theo y sẽ sinh ra nhánh rỗng y = "r"
    tree2 = ID3DecisionTree(attribute_order=["x", "y"])
    train2 = pd.concat([train, pd.DataFrame({"x": ["b"], "y": ["r"], "class": ["unacc"]})], ignore_index=True)
    tree2.fit(train2, attributes=["x", "y"], target="class")
    node_a = tree2.root.children["a"]
    assert node_a.attribute == "y"
    empty = node_a.children["r"]
    assert empty.is_leaf and empty.samples_count == 0
    assert empty.prediction == node_a.majority_class == "acc"
    stats = tree2.get_tree_stats()
    assert stats["empty_leaves"] == 1
    assert stats["rules_with_support"] == stats["leaf_nodes"] - 1
    assert len(tree2.get_rules()) == stats["rules_with_support"]
    assert len(tree2.get_rules(include_empty=True)) == stats["leaf_nodes"]
    res = tree2.predict_with_path({"x": "a", "y": "r"})
    assert res["is_empty_leaf"] is True and res["prediction"] == "acc"

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

