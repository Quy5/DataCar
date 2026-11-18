"""
visualization.py - Trực quan hóa dữ liệu, cây quyết định ID3 và kết quả thực nghiệm.
Đồ án môn Khai phá dữ liệu - Nhóm 06 - QT01.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from id3 import ID3DecisionTree, ID3Node
from data_utils import ATTRIBUTE_NAMES_VI, VALUE_MAPPING_VI, CLASS_MAPPING_VI

# Cấu hình phong cách biểu đồ thống nhất
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI", "Tahoma"]
plt.rcParams["axes.unicode_minus"] = False

# Bảng màu chuẩn dự án: Navy, Blue, Teal/Green, Amber, Crimson
PALETTE = {
    "navy": "#1B365D",
    "blue": "#2E86AB",
    "teal": "#2A9D8F",
    "amber": "#E76F51",
    "crimson": "#D90429",
    "unacc": "#E63946",
    "acc": "#F4A261",
    "good": "#2A9D8F",
    "vgood": "#1D3557"
}

def plot_class_distribution(df: pd.DataFrame, target_col: str = "class", output_path: Optional[str] = None):
    """Vẽ biểu đồ phân bố 4 lớp của Car Evaluation."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[target_col].value_counts()
    classes = ["unacc", "acc", "good", "vgood"]
    counts = counts.reindex(classes, fill_value=0)
    
    colors = [PALETTE[c] for c in classes]
    bars = ax.bar(classes, counts.values, color=colors, edgecolor="black", linewidth=1.2, width=0.55)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:,}\n({height/len(df)*100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )
        
    ax.set_title("Phân bố nhãn lớp - Car Evaluation Dataset", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Mức độ phù hợp (Class)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Số lượng mẫu (Count)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(counts.values) * 1.15)
    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig

def plot_confusion_matrix(cm_matrix: List[List[int]], labels: List[str], output_path: Optional[str] = None):
    """Vẽ ma trận nhầm lẫn (Confusion Matrix)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_arr = np.array(cm_matrix)
    
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
        linewidths=1,
        linecolor="white",
        annot_kws={"size": 12, "weight": "bold"}
    )
    
    ax.set_title("Ma trận nhầm lẫn (Confusion Matrix) - Thuật toán ID3", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Nhãn dự đoán (Predicted Label)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Nhãn thực tế (True Label)", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig

def plot_model_comparison(comparison_data: Dict[str, Any], output_path: Optional[str] = None):
    """Vẽ biểu đồ so sánh giữa Baseline và ID3."""
    id3 = comparison_data["id3"]
    baseline = comparison_data["baseline"]
    
    metrics = ["Accuracy", "Macro-F1", "Macro-Recall", "Macro-Precision"]
    id3_vals = [id3["accuracy"], id3["macro_f1"], id3["macro_recall"], id3["macro_precision"]]
    base_vals = [baseline["accuracy"], baseline["macro_f1"], baseline["macro_recall"], baseline["macro_precision"]]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, base_vals, width, label="Baseline (Majority Class)", color="#A0AAB2", edgecolor="black")
    rects2 = ax.bar(x + width/2, id3_vals, width, label="Cây quyết định ID3 (Nhóm 06)", color=PALETTE["blue"], edgecolor="black")
    
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                    
    ax.set_ylabel("Giá trị Score (0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("So sánh hiệu năng: Cây quyết định ID3 vs Baseline", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", frameon=True)
    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig

def plot_root_information_gain(gain_table: Dict[str, float], output_path: Optional[str] = None):
    """Vẽ biểu đồ Information Gain tại nút gốc."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sorted_gains = sorted(gain_table.items(), key=lambda x: x[1], reverse=True)
    attrs = [k for k, _ in sorted_gains]
    gains = [v for _, v in sorted_gains]
    
    colors = [PALETTE["blue"] if i == 0 else PALETTE["navy"] for i in range(len(attrs))]
    bars = ax.barh(attrs[::-1], gains[::-1], color=colors[::-1], edgecolor="black", height=0.6)
    
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.4f}", xy=(w, bar.get_y() + bar.get_height()/2), xytext=(5, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=10, fontweight="bold")
                    
    ax.set_xlabel("Information Gain (bits)", fontsize=11, fontweight="bold")
    ax.set_title("Information Gain của các thuộc tính tại Nút Gốc (Root Node)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(0, max(gains) * 1.25)
    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig

def export_tree_to_dot(
    tree: ID3DecisionTree,
    output_path: Optional[str] = None,
    max_depth: Optional[int] = None
) -> str:
    """
    Xuất cây quyết định sang định dạng Graphviz DOT.
    Hỗ trợ giới hạn max_depth để tạo bản xem trước (preview).
    """
    lines = [
        "digraph ID3DecisionTree {",
        "  rankdir=TB;",
        "  node [fontname=\"Segoe UI, Arial\", shape=box, style=\"rounded,filled\", color=\"#1B365D\", penwidth=1.5];",
        "  edge [fontname=\"Segoe UI, Arial\", fontsize=10, color=\"#4A5568\", penwidth=1.2];"
    ]
    
    class_colors = {
        "unacc": "#FFD6D6",
        "acc": "#FFE8CC",
        "good": "#D4EDDA",
        "vgood": "#D0E1FD"
    }
    
    def visit(node: ID3Node):
        if max_depth is not None and node.depth > max_depth:
            return
            
        nid = f"n{node.node_id}"
        if node.is_leaf:
            color = class_colors.get(node.prediction, "#FFFFFF")
            if node.samples_count == 0:
                # Lá của nhánh rỗng: không có mẫu huấn luyện, nhãn lấy theo đa số nút cha
                label = (
                    f"LÁ ID: {node.node_id}\\n"
                    f"KẾT QUẢ: {node.prediction.upper()}\\n"
                    f"Nhánh rỗng (0 mẫu)\\n"
                    f"Nhãn đa số của nút cha"
                )
                lines.append(f'  {nid} [label="{label}", fillcolor="{color}", shape=ellipse, style="dashed,filled"];')
            else:
                label = (
                    f"LÁ ID: {node.node_id}\\n"
                    f"KẾT QUẢ: {node.prediction.upper()}\\n"
                    f"Số mẫu: {node.samples_count}\\n"
                    f"Entropy: {node.entropy_val:.3f}"
                )
                lines.append(f'  {nid} [label="{label}", fillcolor="{color}", shape=ellipse];')
        else:
            # Nếu đạt max_depth thì hiển thị nút quyết định thu gọn
            if max_depth is not None and node.depth == max_depth:
                color = "#E2E8F0"
                label = (
                    f"Nút {node.node_id}: {node.attribute}\\n"
                    f"(Thu gọn... N={node.samples_count})\\n"
                    f"Đa số: {node.majority_class}"
                )
                lines.append(f'  {nid} [label="{label}", fillcolor="{color}"];')
                return
                
            color = "#EBF4FF"
            label = (
                f"Nút ID: {node.node_id}\\n"
                f"Thuộc tính: {node.attribute}\\n"
                f"Entropy: {node.entropy_val:.3f}\\n"
                f"Mẫu: {node.samples_count}"
            )
            lines.append(f'  {nid} [label="{label}", fillcolor="{color}"];')
            
            for val, child in node.children.items():
                if max_depth is not None and child.depth > max_depth:
                    continue
                cid = f"n{child.node_id}"
                lines.append(f'  {nid} -> {cid} [label=" {val} "];')
                visit(child)
                
    if tree.root:
        visit(tree.root)
        
    lines.append("}")
    dot_str = "\n".join(lines)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(dot_str)
            
    return dot_str

def generate_text_tree(tree: ID3DecisionTree, max_depth: int = 4) -> str:
    """
    Tạo cấu trúc cây dạng văn bản phân cấp (ASCII text tree).
    Phương án thay thế tin cậy 100% khi môi trường không có binary Graphviz.
    """
    if tree.root is None:
        return "Cây quyết định trống."
        
    lines = []
    lines.append("=" * 65)
    lines.append("CẤU TRÚC CÂY QUYẾT ĐỊNH ID3 (VĂN BẢN PHÂN CẤP)")
    lines.append("=" * 65)
    
    def print_node(node: ID3Node, prefix: str = "", branch_label: str = ""):
        if branch_label:
            branch_str = f"└── [{branch_label}] --> "
        else:
            branch_str = "ROOT: "
            
        if node.is_leaf:
            if node.samples_count == 0:
                lines.append(f"{prefix}{branch_str}[LÁ - NHÁNH RỖNG] DỰ ĐOÁN = {node.prediction.upper()} (0 mẫu, lấy nhãn đa số của nút cha)")
            else:
                lines.append(f"{prefix}{branch_str}[LÁ] DỰ ĐOÁN = {node.prediction.upper()} (Mẫu: {node.samples_count}, Entropy: {node.entropy_val:.3f})")
            return
            
        if node.depth >= max_depth:
            lines.append(f"{prefix}{branch_str}[NÚT {node.node_id}] {node.attribute} ... (Đã giới hạn độ sâu {max_depth}, Đa số: {node.majority_class})")
            return
            
        lines.append(f"{prefix}{branch_str}[NÚT {node.node_id}] Kiểm tra: '{node.attribute}' (Entropy: {node.entropy_val:.3f}, Mẫu: {node.samples_count})")
        
        children_keys = list(node.children.keys())
        for i, val in enumerate(children_keys):
            child = node.children[val]
            child_prefix = prefix + ("    " if branch_label else "  ")
            print_node(child, child_prefix, val)
            
    print_node(tree.root)
    lines.append("=" * 65)
    return "\n".join(lines)

