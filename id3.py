"""
id3.py - Thuật toán cây quyết định ID3 (Iterative Dichotomiser 3) tự cài đặt.
Đồ án môn Khai phá dữ liệu - Nhóm 06 - QT01.
Tác giả: Nhóm 06
"""

import math
import json
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

# Thứ tự cố định để giải quyết tie-break (đảm bảo tính tái lập)
DEFAULT_ATTRIBUTE_ORDER = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
DEFAULT_CLASS_ORDER = ["unacc", "acc", "good", "vgood"]

def entropy(labels: List[Any]) -> float:
    """
    Tính Entropy của tập nhãn S theo công thức:
    Entropy(S) = - sum(p_i * log2(p_i))
    
    Xử lý:
    - Tập rỗng -> trả về 0.0
    - Tập thuần nhất -> trả về 0.0
    - Không xảy ra lỗi log2(0)
    """
    n = len(labels)
    if n == 0:
        return 0.0
    
    counts = Counter(labels)
    ent = 0.0
    for count in counts.values():
        p = count / n
        if p > 0.0:
            ent -= p * math.log2(p)
            
    # Xử lý sai số dấu phẩy động làm entropy âm siêu nhỏ (-0.0)
    return max(0.0, float(ent))

def get_majority_class(labels: List[Any], class_order: Optional[List[str]] = None) -> str:
    """
    Tìm nhãn đa số. Nếu số lượng bằng nhau, giải quyết bằng thứ tự ưu tiên class_order cố định.
    """
    if not labels:
        raise ValueError("Danh sách nhãn không được rỗng khi lấy nhãn đa số.")
        
    counts = Counter(labels)
    max_count = max(counts.values())
    candidates = [c for c, cnt in counts.items() if cnt == max_count]
    
    if len(candidates) == 1:
        return candidates[0]
        
    # Giải quyết tie-break theo thứ tự ưu tiên
    order = class_order or DEFAULT_CLASS_ORDER
    for c in order:
        if c in candidates:
            return c
    return candidates[0]

def information_gain(
    df: pd.DataFrame,
    attribute: str,
    target: str = "class"
) -> Tuple[float, float, Dict[str, float]]:
    """
    Tính Information Gain của thuộc tính A đối với tập dữ liệu S:
    Gain(S, A) = Entropy(S) - sum( (|S_v| / |S|) * Entropy(S_v) )
    
    Trả về:
    - gain: float
    - base_entropy: float (Entropy của S)
    - subset_entropies: dict {giá trị v: Entropy(S_v)}
    """
    total_samples = len(df)
    if total_samples == 0:
        return 0.0, 0.0, {}
        
    base_ent = entropy(df[target].tolist())
    subsets = df.groupby(attribute)
    
    weighted_child_entropy = 0.0
    subset_entropies = {}
    
    for val, group in subsets:
        n_v = len(group)
        ent_v = entropy(group[target].tolist())
        subset_entropies[str(val)] = ent_v
        weighted_child_entropy += (n_v / total_samples) * ent_v
        
    gain = base_ent - weighted_child_entropy
    return max(0.0, float(gain)), base_ent, subset_entropies

class ID3Node:
    """Đại diện cho một nút trong cây quyết định ID3."""
    def __init__(
        self,
        node_id: int,
        is_leaf: bool = False,
        prediction: Optional[str] = None,
        attribute: Optional[str] = None,
        majority_class: Optional[str] = None,
        entropy_val: float = 0.0,
        samples_count: int = 0,
        class_distribution: Optional[Dict[str, int]] = None,
        gain_table: Optional[Dict[str, float]] = None,
        depth: int = 0
    ):
        self.node_id = node_id
        self.is_leaf = is_leaf
        self.prediction = prediction           # Chỉ có ở nút lá
        self.attribute = attribute             # Thuộc tính phân chia nếu là nút quyết định
        self.majority_class = majority_class   # Nhãn đa số tại nút này (dùng làm fallback)
        self.entropy_val = entropy_val
        self.samples_count = samples_count
        self.class_distribution = class_distribution or {}
        self.gain_table = gain_table or {}     # Gain của các thuộc tính tại nút này
        self.depth = depth
        self.children: Dict[str, ID3Node] = {} # Ánh xạ giá trị thuộc tính -> nút con

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển node thành dictionary để serialize JSON."""
        return {
            "node_id": self.node_id,
            "is_leaf": self.is_leaf,
            "prediction": self.prediction,
            "attribute": self.attribute,
            "majority_class": self.majority_class,
            "entropy": round(self.entropy_val, 5),
            "samples_count": self.samples_count,
            "class_distribution": self.class_distribution,
            "gain_table": {k: round(v, 5) for k, v in self.gain_table.items()},
            "depth": self.depth,
            "children": {str(val): child.to_dict() for val, child in self.children.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ID3Node":
        """Khôi phục node từ dictionary."""
        node = cls(
            node_id=data["node_id"],
            is_leaf=data["is_leaf"],
            prediction=data.get("prediction"),
            attribute=data.get("attribute"),
            majority_class=data.get("majority_class"),
            entropy_val=data.get("entropy", 0.0),
            samples_count=data.get("samples_count", 0),
            class_distribution=data.get("class_distribution", {}),
            gain_table=data.get("gain_table", {}),
            depth=data.get("depth", 0)
        )
        for val, child_dict in data.get("children", {}).items():
            node.children[val] = cls.from_dict(child_dict)
        return node

class ID3DecisionTree:
    """
    Thuật toán cây quyết định ID3 chuẩn (Iterative Dichotomiser 3).
    Tự cài đặt cho bài toán phân loại đa lớp với thuộc tính danh mục.
    """
    def __init__(
        self,
        attribute_order: Optional[List[str]] = None,
        class_order: Optional[List[str]] = None,
        min_gain_threshold: float = 1e-12,
        max_depth: Optional[int] = None
    ):
        self.attribute_order = attribute_order or DEFAULT_ATTRIBUTE_ORDER
        self.class_order = class_order or DEFAULT_CLASS_ORDER
        self.min_gain_threshold = min_gain_threshold
        self.max_depth = max_depth
        self.root: Optional[ID3Node] = None
        self._next_node_id = 0
        self.features: List[str] = []
        self.target: str = "class"
        self.known_attribute_values: Dict[str, List[str]] = {}
        self.train_samples_count: int = 0
        self.train_class_distribution: Dict[str, int] = {}

    def _get_new_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def fit(self, df: pd.DataFrame, attributes: Optional[List[str]] = None, target: str = "class") -> "ID3DecisionTree":
        """
        Huấn luyện cây quyết định ID3 trên tập dữ liệu danh mục df.
        """
        self.target = target
        if attributes is None:
            self.features = [c for c in df.columns if c != target]
        else:
            self.features = list(attributes)
            
        self.train_samples_count = len(df)
        self.train_class_distribution = df[target].value_counts().to_dict()
        
        # Lưu lại tất cả giá trị danh mục đã quan sát của mỗi thuộc tính trong tập train
        self.known_attribute_values = {
            col: sorted(df[col].dropna().unique().tolist())
            for col in self.features
        }
        
        self._next_node_id = 0
        self.root = self._build_tree(
            df=df,
            remaining_attributes=list(self.features),
            depth=0,
            parent_majority=get_majority_class(df[target].tolist(), self.class_order)
        )
        return self

    def _build_tree(
        self,
        df: pd.DataFrame,
        remaining_attributes: List[str],
        depth: int,
        parent_majority: str
    ) -> ID3Node:
        """
        Thuật toán đệ quy xây dựng cây ID3.
        """
        node_id = self._get_new_id()
        
        # Trường hợp tập dữ liệu rỗng (nhánh con không có mẫu từ dữ liệu cha)
        if len(df) == 0:
            return ID3Node(
                node_id=node_id,
                is_leaf=True,
                prediction=parent_majority,
                majority_class=parent_majority,
                entropy_val=0.0,
                samples_count=0,
                depth=depth
            )
            
        labels = df[self.target].tolist()
        node_entropy = entropy(labels)
        current_majority = get_majority_class(labels, self.class_order)
        current_distribution = Counter(labels)
        
        # Điều kiện dừng 1: Tất cả mẫu cùng nhãn (Node thuần nhất)
        if len(current_distribution) == 1:
            only_label = list(current_distribution.keys())[0]
            return ID3Node(
                node_id=node_id,
                is_leaf=True,
                prediction=only_label,
                majority_class=only_label,
                entropy_val=0.0,
                samples_count=len(df),
                class_distribution=dict(current_distribution),
                depth=depth
            )
            
        # Điều kiện dừng 2: Hết thuộc tính để phân chia
        if not remaining_attributes:
            return ID3Node(
                node_id=node_id,
                is_leaf=True,
                prediction=current_majority,
                majority_class=current_majority,
                entropy_val=node_entropy,
                samples_count=len(df),
                class_distribution=dict(current_distribution),
                depth=depth
            )
            
        # Điều kiện dừng bổ sung: Đạt độ sâu tối đa (nếu có cấu hình)
        if self.max_depth is not None and depth >= self.max_depth:
            return ID3Node(
                node_id=node_id,
                is_leaf=True,
                prediction=current_majority,
                majority_class=current_majority,
                entropy_val=node_entropy,
                samples_count=len(df),
                class_distribution=dict(current_distribution),
                depth=depth
            )
            
        # Tính Information Gain cho tất cả thuộc tính còn lại
        gain_table = {}
        for attr in remaining_attributes:
            gain, _, _ = information_gain(df, attr, self.target)
            gain_table[attr] = gain
            
        # Lựa chọn thuộc tính có Gain lớn nhất
        # Tie-break bằng thứ tự ưu tiên trong self.attribute_order
        best_attr = None
        max_gain = -1.0
        
        # Sắp xếp remaining_attributes theo thứ tự ưu tiên để đảm bảo deterministic
        sorted_candidates = sorted(
            remaining_attributes,
            key=lambda a: self.attribute_order.index(a) if a in self.attribute_order else 999
        )
        
        for attr in sorted_candidates:
            g = gain_table[attr]
            if g > max_gain:
                max_gain = g
                best_attr = attr
                
        # Điều kiện dừng 3: Max Gain <= 1e-12 (không phân chia thêm được thông tin)
        if max_gain <= self.min_gain_threshold or best_attr is None:
            return ID3Node(
                node_id=node_id,
                is_leaf=True,
                prediction=current_majority,
                majority_class=current_majority,
                entropy_val=node_entropy,
                samples_count=len(df),
                class_distribution=dict(current_distribution),
                gain_table=gain_table,
                depth=depth
            )
            
        # Tạo nút quyết định
        decision_node = ID3Node(
            node_id=node_id,
            is_leaf=False,
            attribute=best_attr,
            majority_class=current_majority,
            entropy_val=node_entropy,
            samples_count=len(df),
            class_distribution=dict(current_distribution),
            gain_table=gain_table,
            depth=depth
        )
        
        # Tạo nhánh cho từng giá trị danh mục của thuộc tính được chọn
        # Lưu ý: Mỗi nhánh phải xét trên các giá trị đã quan sát (known_attribute_values)
        attr_values = self.known_attribute_values.get(best_attr, sorted(df[best_attr].unique().tolist()))
        next_remaining_attrs = [a for a in remaining_attributes if a != best_attr]
        
        for val in attr_values:
            subset_df = df[df[best_attr] == val]
            child_node = self._build_tree(
                df=subset_df,
                remaining_attributes=next_remaining_attrs,
                depth=depth + 1,
                parent_majority=current_majority
            )
            decision_node.children[str(val)] = child_node
            
        return decision_node

    def predict_one(self, sample: Dict[str, Any]) -> str:
        """Dự đoán nhãn cho 1 mẫu dictionary."""
        result = self.predict_with_path(sample)
        return result["prediction"]

    def predict(self, data: Any) -> List[str]:
        """Dự đoán cho danh sách mẫu hoặc pandas DataFrame."""
        if isinstance(data, pd.DataFrame):
            records = data.to_dict(orient="records")
        elif isinstance(data, list):
            records = data
        else:
            raise TypeError("data phải là pandas DataFrame hoặc danh sách dict.")
            
        return [self.predict_one(rec) for rec in records]

    def predict_with_path(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dự đoán và trả về chi tiết đường đi quyết định (decision path):
        - prediction: nhãn dự đoán
        - is_fallback: bool (có sử dụng cơ chế dự phòng không)
        - fallback_reason: lý do nếu fallback
        - path: danh sách các bước qua từng nút
        - rules_applied: chuỗi mô tả các điều kiện
        """
        if self.root is None:
            raise ValueError("Mô hình ID3 chưa được huấn luyện. Vui lòng gọi fit() trước.")
            
        curr = self.root
        path = []
        is_fallback = False
        fallback_reason = ""
        
        while not curr.is_leaf:
            attr = curr.attribute
            val = str(sample.get(attr, ""))
            
            step = {
                "node_id": curr.node_id,
                "attribute": attr,
                "checked_value": val,
                "entropy": curr.entropy_val,
                "samples_count": curr.samples_count,
                "majority_at_node": curr.majority_class,
                "available_branches": list(curr.children.keys())
            }
            path.append(step)
            
            if val in curr.children:
                curr = curr.children[val]
            else:
                # Giá trị chưa từng xuất hiện hoặc thiếu -> Dùng nhãn đa số tại nút hiện tại (fallback)
                is_fallback = True
                fallback_reason = f"Giá trị '{val}' của thuộc tính '{attr}' không có nhánh tương ứng. Sử dụng nhãn đa số ({curr.majority_class}) tại nút ID {curr.node_id}."
                break
                
        # Nút kết thúc
        if is_fallback:
            pred = curr.majority_class
        else:
            pred = curr.prediction
            path.append({
                "node_id": curr.node_id,
                "is_leaf": True,
                "prediction": pred,
                "entropy": curr.entropy_val,
                "samples_count": curr.samples_count,
                "class_distribution": curr.class_distribution
            })
            
        return {
            "prediction": pred,
            "is_fallback": is_fallback,
            "fallback_reason": fallback_reason,
            "path": path,
            "leaf_node_id": curr.node_id
        }

    def get_tree_stats(self) -> Dict[str, Any]:
        """Thống kê tổng số nút, số lá, độ sâu lớn nhất, thuộc tính gốc."""
        if self.root is None:
            return {"total_nodes": 0, "leaf_nodes": 0, "max_depth": 0, "root_attribute": None}
            
        total_nodes = 0
        leaf_nodes = 0
        max_depth = 0
        
        def traverse(node: ID3Node):
            nonlocal total_nodes, leaf_nodes, max_depth
            total_nodes += 1
            if node.depth > max_depth:
                max_depth = node.depth
            if node.is_leaf:
                leaf_nodes += 1
            else:
                for child in node.children.values():
                    traverse(child)
                    
        traverse(self.root)
        return {
            "total_nodes": total_nodes,
            "leaf_nodes": leaf_nodes,
            "decision_nodes": total_nodes - leaf_nodes,
            "max_depth": max_depth,
            "root_attribute": self.root.attribute if not self.root.is_leaf else None,
            "root_gain_table": self.root.gain_table if self.root else {}
        }

    def get_rules(self) -> List[Dict[str, Any]]:
        """Xuất danh sách các quy tắc IF-THEN từ cây."""
        rules = []
        if self.root is None:
            return rules
            
        def extract(node: ID3Node, current_conditions: List[Tuple[str, str]]):
            if node.is_leaf:
                rules.append({
                    "conditions": list(current_conditions),
                    "prediction": node.prediction,
                    "samples": node.samples_count,
                    "entropy": node.entropy_val
                })
            else:
                for val, child in node.children.items():
                    extract(child, current_conditions + [(node.attribute, val)])
                    
        extract(self.root, [])
        return rules

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển toàn bộ mô hình thành dictionary."""
        return {
            "features": self.features,
            "target": self.target,
            "attribute_order": self.attribute_order,
            "class_order": self.class_order,
            "min_gain_threshold": self.min_gain_threshold,
            "max_depth": self.max_depth,
            "train_samples_count": self.train_samples_count,
            "train_class_distribution": self.train_class_distribution,
            "known_attribute_values": self.known_attribute_values,
            "tree": self.root.to_dict() if self.root else None,
            "stats": self.get_tree_stats()
        }

    def save_model(self, filepath: str) -> None:
        """Lưu mô hình sang định dạng JSON."""
        data = self.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_model(cls, filepath: str) -> "ID3DecisionTree":
        """Nạp mô hình từ file JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        model = cls(
            attribute_order=data.get("attribute_order", DEFAULT_ATTRIBUTE_ORDER),
            class_order=data.get("class_order", DEFAULT_CLASS_ORDER),
            min_gain_threshold=data.get("min_gain_threshold", 1e-12),
            max_depth=data.get("max_depth")
        )
        model.features = data.get("features", [])
        model.target = data.get("target", "class")
        model.train_samples_count = data.get("train_samples_count", 0)
        model.train_class_distribution = data.get("train_class_distribution", {})
        model.known_attribute_values = data.get("known_attribute_values", {})
        
        if data.get("tree"):
            model.root = ID3Node.from_dict(data["tree"])
        return model

