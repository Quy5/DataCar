"""
app.py - Giao diện Web Streamlit cho Ứng dụng Phân loại Ô tô bằng Thuật toán ID3.
Đồ án môn Khai phá dữ liệu - Nhóm 06 - QT01.
Phong cách thiết kế: Modern AI Dashboard - Clean - Minimal - Academic - Professional.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Phân loại ô tô - ID3 Decision Tree | Nhóm 06",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho phong cách Modern AI Dashboard (Navy, Blue, Teal, Indigo, Slate)
CUSTOM_CSS = """
<style>
    /* Nhập font hiện đại Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Màu nền chính & sidebar */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Tùy biến Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        border-right: 1px solid #334155;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #F8FAFC !important;
    }
    
    /* Brand Card trong Sidebar */
    .brand-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        color: #34D399 !important;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
    }
    
    /* Header chính */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F2D59 100%);
        padding: 28px 36px;
        border-radius: 18px;
        color: white;
        margin-bottom: 26px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15), 0 8px 10px -6px rgba(15, 23, 42, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .hero-banner::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(14, 165, 233, 0.15) 0%, rgba(0,0,0,0) 70%);
        pointer-events: none;
    }
    
    .hero-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #FFFFFF !important;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .hero-subtitle {
        font-size: 14px;
        color: #94A3B8 !important;
        margin-top: 8px;
        line-height: 1.5;
    }
    
    .tag-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
    }
    
    .hero-tag {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #E2E8F0 !important;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 500;
    }

    /* Modern KPI Metric Card */
    .kpi-card {
        background: #FFFFFF;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: all 0.2s ease-in-out;
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.08);
        border-color: #CBD5E1;
    }
    
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .kpi-title {
        font-size: 12px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    
    .kpi-icon {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }
    
    .kpi-val {
        font-size: 32px;
        font-weight: 800;
        color: #0F172A;
        margin-top: 10px;
        letter-spacing: -0.5px;
    }
    
    .kpi-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 12px;
        font-weight: 600;
        margin-top: 6px;
        padding: 2px 8px;
        border-radius: 6px;
    }
    
    .badge-green {
        background: #ECFDF5;
        color: #059669;
    }
    
    .badge-blue {
        background: #EFF6FF;
        color: #2563EB;
    }
    
    .badge-slate {
        background: #F1F5F9;
        color: #475569;
    }

    /* Container Card cho Form và Nội dung */
    .content-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    }
    
    .content-box-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 18px;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 12px;
    }
    
    .content-box-title {
        font-size: 16px;
        font-weight: 700;
        color: #1E293B;
        margin: 0;
    }

    /* Result Card hiện đại */
    .result-banner {
        padding: 28px;
        border-radius: 18px;
        color: white;
        margin: 20px 0;
        box-shadow: 0 12px 28px -5px rgba(0, 0, 0, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .res-unacc {
        background: linear-gradient(135deg, #E11D48 0%, #9F1239 100%);
        border: 1px solid #F43F5E;
    }
    .res-acc {
        background: linear-gradient(135deg, #D97706 0%, #92400E 100%);
        border: 1px solid #F59E0B;
    }
    .res-good {
        background: linear-gradient(135deg, #059669 0%, #064E3B 100%);
        border: 1px solid #10B981;
    }
    .res-vgood {
        background: linear-gradient(135deg, #2563EB 0%, #1E3A8A 100%);
        border: 1px solid #3B82F6;
    }

    /* Vertical Timeline Stepper cho Decision Path */
    .timeline-container {
        position: relative;
        padding-left: 28px;
        margin: 20px 0;
    }
    
    .timeline-container::before {
        content: '';
        position: absolute;
        top: 15px;
        bottom: 15px;
        left: 11px;
        width: 2px;
        background: #CBD5E1;
    }
    
    .timeline-item {
        position: relative;
        margin-bottom: 18px;
    }
    
    .timeline-item:last-child {
        margin-bottom: 0;
    }
    
    .timeline-dot {
        position: absolute;
        left: -28px;
        top: 6px;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: #FFFFFF;
        border: 2px solid #0EA5E9;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 700;
        color: #0284C7;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .timeline-dot.leaf {
        border-color: #10B981;
        background: #10B981;
        color: white;
    }
    
    .timeline-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.02);
    }

    /* Workflow Cards */
    .wf-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 20px 0;
    }
    
    .wf-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        border-top: 4px solid #2563EB;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
    
    .wf-step {
        font-size: 11px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    
    .wf-name {
        font-size: 14px;
        font-weight: 700;
        color: #0F172A;
    }
    
    .wf-desc {
        font-size: 12px;
        color: #64748B;
        margin-top: 6px;
        line-height: 1.4;
    }
    
    /* Nút bấm Streamlit */
    div.stButton > button:first-child {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        border: none;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
    }
    
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(2, 132, 199, 0.4);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Import các module cốt lõi của dự án
from data_utils import (
    load_data,
    validate_dataframe,
    split_data,
    get_dataset_summary,
    COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ATTRIBUTE_NAMES_VI,
    VALUE_MAPPING_VI,
    CLASS_MAPPING_VI,
    VALID_ATTRIBUTE_VALUES,
    convert_vi_input_to_uci
)
from id3 import ID3DecisionTree
from visualization import export_tree_to_dot, generate_text_tree

# Đường dẫn file
DATA_FILE = "data/car.data"
MODEL_FILE = "output/model.json"
RESULTS_FILE = "output/results.json"
PREDICTIONS_FILE = "output/predictions.csv"

@st.cache_data
def get_or_load_data():
    """Tải và lưu trữ tạm dữ liệu car.data."""
    if not os.path.exists(DATA_FILE):
        from download_data import download_data
        download_data()
    return load_data(DATA_FILE)

@st.cache_resource
def get_or_train_model():
    """Nạp mô hình ID3 đã huấn luyện hoặc kích hoạt pipeline huấn luyện."""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(RESULTS_FILE):
        from train import run_training_pipeline
        run_training_pipeline()
        
    model = ID3DecisionTree.load_model(MODEL_FILE)
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
    return model, results

# Tải dữ liệu và mô hình
try:
    df_data = get_or_load_data()
    id3_model, exp_results = get_or_train_model()
    is_ready = True
except Exception as e:
    is_ready = False
    init_error = str(e)

# ==============================================================================
# SIDEBAR ĐIỀU HƯỚNG HIỆN ĐẠI
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div class="brand-box">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: linear-gradient(135deg, #0284C7, #2563EB); width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px;">
                🚗
            </div>
            <div>
                <div style="font-size: 15px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.3px;">CAR EVALUATION</div>
                <div style="font-size: 12px; color: #94A3B8; font-weight: 500;">Thuật toán ID3 thuần túy</div>
            </div>
        </div>
        <div style="margin-top: 14px; display: flex; justify-content: space-between; align-items: center;">
            <span class="status-badge"><span class="status-dot"></span> ĐÃ HUẤN LUYỆN</span>
            <span style="font-size: 11px; color: #64748B; font-weight: 600;">Nhóm 06 – QT01</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 11px; text-transform: uppercase; color: #64748B; font-weight: 700; letter-spacing: 0.8px; margin-bottom: 8px;'>ĐIỀU HƯỚNG HỆ THỐNG</p>", unsafe_allow_html=True)
    
    menu = st.radio(
        "Menu điều hướng:",
        [
            "🏠 Tổng quan (Dashboard)",
            "🚗 Dự đoán mức độ phù hợp",
            "📊 Khám phá dữ liệu (EDA)",
            "🌳 Cây quyết định ID3",
            "📈 Đánh giá mô hình",
            "ℹ️ Giới thiệu đề tài"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Widget thông số nhanh trong Sidebar
    if is_ready:
        id3_eval = exp_results["id3_evaluation"]
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 14px; margin-top: 10px;">
            <div style="font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px;">HIỆU NĂNG TEST SET</div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 8px;">
                <span style="font-size: 13px; color: #CBD5E1;">Độ chính xác (Acc):</span>
                <span style="font-size: 16px; font-weight: 800; color: #34D399;">{id3_eval['accuracy']*100:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                <span style="font-size: 13px; color: #CBD5E1;">Macro-F1:</span>
                <span style="font-size: 15px; font-weight: 700; color: #38BDF8;">{id3_eval['macro_f1']:.4f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                <span style="font-size: 13px; color: #CBD5E1;">Mẫu kiểm thử:</span>
                <span style="font-size: 13px; font-weight: 600; color: #F8FAFC;">346 mẫu (20%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("© 2026 Đồ án Khai phá dữ liệu | Nhóm 06")

if not is_ready:
    st.error(f"⚠️ Có lỗi xảy ra trong quá trình khởi tạo ứng dụng: {init_error}")
    st.info("Vui lòng chạy lệnh: `python train.py` trong terminal để tạo dữ liệu và mô hình chuẩn.")
    st.stop()

# ==============================================================================
# 1. TRANG TỔNG QUAN (DASHBOARD)
# ==============================================================================
if menu == "🏠 Tổng quan (Dashboard)":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">
            <span>🚗</span>
            <span>HỆ THỐNG PHÂN LOẠI Ô TÔ BẰNG THUẬT TOÁN ID3</span>
        </div>
        <div class="hero-subtitle">
            Ứng dụng khai phá dữ liệu hỗ trợ đánh giá mức độ phù hợp của phương tiện giao thông dựa trên cây quyết định ID3 (Iterative Dichotomiser 3). Xây dựng trên bộ dữ liệu chuẩn Car Evaluation từ UCI Machine Learning Repository.
        </div>
        <div class="tag-container">
            <span class="hero-tag">🔬 Khai phá dữ liệu</span>
            <span class="hero-tag">🧠 Thuật toán ID3 tự lập trình</span>
            <span class="hero-tag">📁 UCI Car Evaluation</span>
            <span class="hero-tag">⚡ Độ chính xác: 94.51%</span>
            <span class="hero-tag">👥 Nhóm 06 – QT01</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 KPI Hero Cards
    stats = exp_results["tree_stats"]
    id3_eval = exp_results["id3_evaluation"]
    base_eval = exp_results["baseline_evaluation"]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Tổng số mẫu xe</span>
                <div class="kpi-icon" style="background: #EFF6FF; color: #2563EB;">📁</div>
            </div>
            <div class="kpi-val">{len(df_data):,}</div>
            <div class="kpi-badge badge-blue">100% dữ liệu sạch (0 missing)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Đặc trưng kỹ thuật</span>
                <div class="kpi-icon" style="background: #F5F3FF; color: #7C3AED;">🧩</div>
            </div>
            <div class="kpi-val">{len(FEATURE_COLUMNS)}</div>
            <div class="kpi-badge badge-slate">6 thuộc tính danh mục rời rạc</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        acc_gain = (id3_eval['accuracy'] - base_eval['accuracy']) * 100
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Độ chính xác (Accuracy)</span>
                <div class="kpi-icon" style="background: #ECFDF5; color: #059669;">🎯</div>
            </div>
            <div class="kpi-val" style="color: #059669;">{id3_eval['accuracy']*100:.2f}%</div>
            <div class="kpi-badge badge-green">↑ +{acc_gain:.2f}% so với Baseline</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Macro-F1 Score</span>
                <div class="kpi-icon" style="background: #F0F9FF; color: #0284C7;">⚖️</div>
            </div>
            <div class="kpi-val" style="color: #0284C7;">{id3_eval['macro_f1']:.4f}</div>
            <div class="kpi-badge badge-blue">Đánh giá cân bằng 4 lớp</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quy trình khai phá dữ liệu 4 bước
    st.markdown("#### 🔄 Quy trình Khai phá Dữ liệu & Xây dựng Cây Quyết định ID3")
    st.markdown("""
    <div class="wf-grid">
        <div class="wf-card" style="border-top-color: #0284C7;">
            <div class="wf-step">BƯỚC 1: DỮ LIỆU GỐC</div>
            <div class="wf-name">Thu thập UCI Dataset</div>
            <div class="wf-desc">1.728 bản ghi chuẩn từ UCI Repository với 6 thuộc tính và 4 mức độ phù hợp.</div>
        </div>
        <div class="wf-card" style="border-top-color: #2563EB;">
            <div class="wf-step">BƯỚC 2: TIỀN XỬ LÝ</div>
            <div class="wf-name">Phân tầng 80% / 20%</div>
            <div class="wf-desc">Chia 1.382 mẫu Train và 346 mẫu Test, đảm bảo tỷ lệ 4 lớp được bảo toàn.</div>
        </div>
        <div class="wf-card" style="border-top-color: #7C3AED;">
            <div class="wf-step">BƯỚC 3: HUẤN LUYỆN ID3</div>
            <div class="wf-name">Entropy & Information Gain</div>
            <div class="wf-desc">Tự cài đặt thuật toán Ross Quinlan: chia đa nhánh theo Gain lớn nhất.</div>
        </div>
        <div class="wf-card" style="border-top-color: #10B981;">
            <div class="wf-step">BƯỚC 4: SUY DIỄN</div>
            <div class="wf-name">Dự đoán & Giải thích</div>
            <div class="wf-desc">Truy vết đường đi từng bước từ Nút gốc đến Nút lá kết luận (Explainable AI).</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hàng 2: Bảng thông số mô hình và Biểu đồ lớp
    left_c, right_c = st.columns([1.1, 0.9])
    
    with left_c:
        st.markdown("""
        <div class="content-box">
            <div class="content-box-header">
                <span style="font-size: 20px;">⚙️</span>
                <span class="content-box-title">Thông số Cấu trúc Cây Quyết định Đã Huấn luyện</span>
            </div>
        """, unsafe_allow_html=True)
        
        tree_meta_df = pd.DataFrame([
            {"Thuộc tính kiến trúc": "Thuộc tính nút gốc (Root)", "Giá trị": f"⭐ {stats['root_attribute']} ({ATTRIBUTE_NAMES_VI.get(stats['root_attribute'])})"},
            {"Thuộc tính kiến trúc": "Information Gain tại nút gốc", "Giá trị": f"{stats['root_gain_table'].get(stats['root_attribute'], 0):.5f} bits"},
            {"Thuộc tính kiến trúc": "Tổng số nút (Nodes)", "Giá trị": f"{stats['total_nodes']} nút"},
            {"Thuộc tính kiến trúc": "Số nút lá quyết định (Leaves)", "Giá trị": f"{stats['leaf_nodes']} lá (279 quy tắc)"},
            {"Thuộc tính kiến trúc": "Số nút rẽ nhánh (Decision Nodes)", "Giá trị": f"{stats['decision_nodes']} nút"},
            {"Thuộc tính kiến trúc": "Độ sâu lớn nhất của cây (Max Depth)", "Giá trị": f"{stats['max_depth']} tầng"},
            {"Thuộc tính kiến trúc": "Số mẫu huấn luyện (Train set)", "Giá trị": "1.382 mẫu (80%)"}
        ])
        st.dataframe(tree_meta_df, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_c:
        st.markdown("""
        <div class="content-box">
            <div class="content-box-header">
                <span style="font-size: 20px;">📊</span>
                <span class="content-box-title">Phân bố Mẫu trên 4 Lớp Phân loại</span>
            </div>
        """, unsafe_allow_html=True)
        if os.path.exists("output/class_distribution.png"):
            st.image("output/class_distribution.png", use_container_width=True)
        else:
            st.info("Chưa có biểu đồ phân bố lớp.")
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 2. TRANG DỰ ĐOÁN MỨC ĐỘ PHÙ HỢP CỦA Ô TÔ
# ==============================================================================
elif menu == "🚗 Dự đoán mức độ phù hợp":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">
            <span>🚗</span>
            <span>DỰ ĐOÁN MỨC ĐỘ PHÙ HỢP CỦA Ô TÔ</span>
        </div>
        <div class="hero-subtitle">
            Cấu hình 6 thông số kỹ thuật để Cây quyết định ID3 phân loại mức độ chấp nhận của xe và giải thích chi tiết chuỗi suy luận từng bước.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quản lý session state cho form
    if "form_inputs" not in st.session_state:
        st.session_state.form_inputs = {
            "buying": "Trung bình",
            "maint": "Trung bình",
            "doors": "4 cửa",
            "persons": "4 người",
            "lug_boot": "Trung bình",
            "safety": "Cao"
        }
        
    # Preset chips chọn nhanh
    st.markdown("<p style='font-size: 13px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.5px;'>⚡ THỬ NGHIỆM NHANH VỚI CÁC MẪU XE TIÊU BIỂU:</p>", unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        if st.button("🔴 Mẫu 1: Xe không phù hợp (unacc)", use_container_width=True):
            st.session_state.form_inputs = {"buying": "Rất cao", "maint": "Rất cao", "doors": "2 cửa", "persons": "2 người", "lug_boot": "Nhỏ", "safety": "Thấp"}
            st.rerun()
    with p2:
        if st.button("🟠 Mẫu 2: Xe chấp nhận được (acc)", use_container_width=True):
            st.session_state.form_inputs = {"buying": "Trung bình", "maint": "Trung bình", "doors": "4 cửa", "persons": "4 người", "lug_boot": "Trung bình", "safety": "Trung bình"}
            st.rerun()
    with p3:
        if st.button("🟢 Mẫu 3: Xe chất lượng Tốt (good)", use_container_width=True):
            st.session_state.form_inputs = {"buying": "Thấp", "maint": "Thấp", "doors": "4 cửa", "persons": "4 người", "lug_boot": "Lớn", "safety": "Trung bình"}
            st.rerun()
    with p4:
        if st.button("🔵 Mẫu 4: Xe Rất tốt (vgood)", use_container_width=True):
            st.session_state.form_inputs = {"buying": "Thấp", "maint": "Thấp", "doors": "4 cửa", "persons": "4 người", "lug_boot": "Lớn", "safety": "Cao"}
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Form nhập thông số 6 đặc trưng
    with st.form("car_prediction_form"):
        col_group1, col_group2 = st.columns(2)
        
        with col_group1:
            st.markdown("""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                <div style="font-size: 14px; font-weight: 700; color: #1E293B; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                    <span>💰</span> 1. NHÓM THÔNG SỐ CHI PHÍ & THÂN VỎ
                </div>
            """, unsafe_allow_html=True)
            
            val_buying = st.selectbox(
                "Giá mua xe ban đầu (buying):",
                options=["Rất cao", "Cao", "Trung bình", "Thấp"],
                index=["Rất cao", "Cao", "Trung bình", "Thấp"].index(st.session_state.form_inputs["buying"]),
                help="Mức giá mua xe: vhigh, high, med, low"
            )
            val_maint = st.selectbox(
                "Chi phí bảo dưỡng định kỳ (maint):",
                options=["Rất cao", "Cao", "Trung bình", "Thấp"],
                index=["Rất cao", "Cao", "Trung bình", "Thấp"].index(st.session_state.form_inputs["maint"]),
                help="Chi phí duy trì bảo dưỡng: vhigh, high, med, low"
            )
            val_doors = st.selectbox(
                "Số cửa xe (doors):",
                options=["2 cửa", "3 cửa", "4 cửa", "Từ 5 cửa trở lên"],
                index=["2 cửa", "3 cửa", "4 cửa", "Từ 5 cửa trở lên"].index(st.session_state.form_inputs["doors"]),
                help="Số cửa ra vào: 2, 3, 4, 5more"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_group2:
            st.markdown("""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; margin-bottom: 12px;">
                <div style="font-size: 14px; font-weight: 700; color: #1E293B; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                    <span>🛡️</span> 2. NHÓM TIỆN ÍCH & ĐỘ AN TOÀN
                </div>
            """, unsafe_allow_html=True)
            
            val_persons = st.selectbox(
                "Sức chứa hành khách (persons):",
                options=["2 người", "4 người", "Trên 4 người"],
                index=["2 người", "4 người", "Trên 4 người"].index(st.session_state.form_inputs["persons"]),
                help="Sức chứa tối đa: 2, 4, more"
            )
            val_lug_boot = st.selectbox(
                "Khoang chứa hành lý (lug_boot):",
                options=["Nhỏ", "Trung bình", "Lớn"],
                index=["Nhỏ", "Trung bình", "Lớn"].index(st.session_state.form_inputs["lug_boot"]),
                help="Kích thước khoang cốp xe: small, med, big"
            )
            val_safety = st.selectbox(
                "Mức độ an toàn ước tính (safety):",
                options=["Thấp", "Trung bình", "Cao"],
                index=["Thấp", "Trung bình", "Cao"].index(st.session_state.form_inputs["safety"]),
                help="Đánh giá mức độ an toàn: low, med, high"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
        submitted = st.form_submit_button("🚀 PHÂN LOẠI Ô TÔ NGAY VỚI MÔ HÌNH ID3", use_container_width=True, type="primary")

    if submitted:
        input_vi = {
            "buying": val_buying,
            "maint": val_maint,
            "doors": val_doors,
            "persons": val_persons,
            "lug_boot": val_lug_boot,
            "safety": val_safety
        }
        
        try:
            # Chuyển đổi mã UCI và đưa vào mô hình
            uci_sample = convert_vi_input_to_uci(input_vi)
            pred_result = id3_model.predict_with_path(uci_sample)
            predicted_class = pred_result["prediction"]
            cls_info = CLASS_MAPPING_VI.get(predicted_class, {
                "vi": predicted_class,
                "name": predicted_class,
                "description": "Không có mô tả",
                "color": "#4A5568"
            })
            
            # Kết quả Banner nổi bật
            css_class = f"res-{predicted_class}"
            st.markdown(f"""
            <div class="result-banner {css_class}">
                <div style="font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">
                    ✨ KẾT QUẢ DỰ ĐOÁN TỪ CÂY QUYẾT ĐỊNH ID3
                </div>
                <div style="font-size: 36px; font-weight: 800; margin-top: 6px; letter-spacing: -0.5px;">
                    {cls_info['name'].upper()} ({predicted_class})
                </div>
                <div style="font-size: 15px; margin-top: 8px; opacity: 0.95; line-height: 1.5; max-width: 800px;">
                    {cls_info['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Giải thích đường đi quyết định (Decision Path Stepper)
            st.subheader("🔍 Quá trình suy diễn logic trên Cây quyết định ID3")
            st.markdown("Cây quyết định duyệt qua các nút theo thuộc tính được chọn để đi đến nút lá kết luận:")
            
            path_steps = pred_result["path"]
            
            st.markdown('<div class="timeline-container">', unsafe_allow_html=True)
            for idx, step in enumerate(path_steps):
                if not step.get("is_leaf", False):
                    attr = step["attribute"]
                    val = step["checked_value"]
                    attr_vi = ATTRIBUTE_NAMES_VI.get(attr, attr)
                    val_vi = VALUE_MAPPING_VI.get(attr, {}).get(val, val)
                    st.markdown(f"""
                    <div class="timeline-item">
                        <div class="timeline-dot">{idx + 1}</div>
                        <div class="timeline-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 14px; font-weight: 700; color: #1E293B;">
                                    Kiểm tra thuộc tính: <code style="color: #0284C7; font-size: 14px;">{attr}</code> ({attr_vi})
                                </span>
                                <span style="font-size: 12px; background: #F1F5F9; color: #475569; padding: 2px 8px; border-radius: 4px; font-weight: 600;">
                                    Nút ID: {step['node_id']}
                                </span>
                            </div>
                            <div style="font-size: 14px; color: #334155; margin-top: 6px;">
                                👉 Xe có giá trị: <strong>{val} ({val_vi})</strong> &nbsp;•&nbsp; 
                                Entropy tại nút: <code>{step['entropy']:.3f}</code> &nbsp;•&nbsp; 
                                Số mẫu học tập: <code>{step['samples_count']}</code>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="timeline-item">
                        <div class="timeline-dot leaf">🏁</div>
                        <div class="timeline-card" style="border-left: 4px solid #10B981; background: #F0FDF4;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 15px; font-weight: 800; color: #065F46;">
                                    KẾT LUẬN CUỐI CÙNG: {predicted_class.upper()} ({cls_info['name']})
                                </span>
                                <span style="font-size: 12px; background: #DCFCE7; color: #166534; padding: 2px 8px; border-radius: 4px; font-weight: 700;">
                                    Nút lá ID: {step['node_id']}
                                </span>
                            </div>
                            <div style="font-size: 13px; color: #14532D; margin-top: 6px;">
                                Thỏa mãn điều kiện dừng thuần nhất. Độ hỗn loạn Entropy = <code>{step['entropy']:.3f}</code> với <code>{step['samples_count']}</code> mẫu ủng hộ trong tập huấn luyện.
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if pred_result["is_fallback"]:
                st.warning(f"⚠️ Cơ chế dự phòng đã được kích hoạt: {pred_result['fallback_reason']}")
                
        except Exception as e:
            st.error(f"Lỗi khi xử lý dự đoán: {e}")

# ==============================================================================
# 3. TRANG KHÁM PHÁ DỮ LIỆU (EDA)
# ==============================================================================
elif menu == "📊 Khám phá dữ liệu (EDA)":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">
            <span>📊</span>
            <span>KHÁM PHÁ DỮ LIỆU (EXPLORATORY DATA ANALYSIS)</span>
        </div>
        <div class="hero-subtitle">
            Khảo sát cấu trúc, kiểm định tính toàn vẹn và phân tích phân bố đặc trưng của bộ dữ liệu Car Evaluation từ UCI.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    summary = get_dataset_summary(df_data)
    
    ec1, ec2, ec3, ec4 = st.columns(4)
    with ec1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Tổng số dòng</div>
            <div class="kpi-val">{summary['total_rows']:,}</div>
            <div class="kpi-badge badge-blue">1.728 bản ghi</div>
        </div>
        """, unsafe_allow_html=True)
    with ec2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Tổng số cột</div>
            <div class="kpi-val">{summary['total_cols']}</div>
            <div class="kpi-badge badge-slate">6 đặc trưng + 1 nhãn</div>
        </div>
        """, unsafe_allow_html=True)
    with ec3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Giá trị thiếu (Missing)</div>
            <div class="kpi-val" style="color: #059669;">{summary['missing_values']}</div>
            <div class="kpi-badge badge-green">Dữ liệu hoàn chỉnh 100%</div>
        </div>
        """, unsafe_allow_html=True)
    with ec4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Bản ghi trùng lặp</div>
            <div class="kpi-val">{summary['duplicates_count']}</div>
            <div class="kpi-badge badge-slate">Tập dữ liệu chuẩn xác</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📋 Bảng dữ liệu tương tác", "📊 Thống kê từng thuộc tính", "📖 Từ điển thuộc tính"])
    
    with tab1:
        st.subheader("Bộ dữ liệu Car Evaluation (UCI)")
        fc1, fc2 = st.columns(2)
        with fc1:
            filter_class = st.multiselect("Lọc theo mức độ phù hợp (class):", options=["unacc", "acc", "good", "vgood"], default=[])
        with fc2:
            filter_safety = st.multiselect("Lọc theo mức độ an toàn (safety):", options=["low", "med", "high"], default=[])
            
        filtered_df = df_data.copy()
        if filter_class:
            filtered_df = filtered_df[filtered_df["class"].isin(filter_class)]
        if filter_safety:
            filtered_df = filtered_df[filtered_df["safety"].isin(filter_safety)]
            
        st.caption(f"Đang hiển thị {len(filtered_df):,} / {len(df_data):,} bản ghi:")
        st.dataframe(filtered_df, use_container_width=True, height=380)
        
    with tab2:
        st.subheader("Phân bố tần suất các giá trị danh mục")
        selected_attr = st.selectbox(
            "Chọn thuộc tính để xem phân bố chi tiết:",
            options=FEATURE_COLUMNS,
            format_func=lambda x: f"{x} - {ATTRIBUTE_NAMES_VI.get(x)}"
        )
        
        attr_counts = df_data[selected_attr].value_counts().reset_index()
        attr_counts.columns = ["Giá trị (UCI)", "Số lượng mẫu"]
        attr_counts["Ý nghĩa tiếng Việt"] = [VALUE_MAPPING_VI.get(selected_attr, {}).get(v, v) for v in attr_counts["Giá trị (UCI)"]]
        attr_counts["Tỷ lệ (%)"] = (attr_counts["Số lượng mẫu"] / len(df_data) * 100).round(2)
        
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.dataframe(attr_counts, hide_index=True, use_container_width=True)
        with col_c2:
            st.bar_chart(attr_counts.set_index("Giá trị (UCI)")["Số lượng mẫu"])
            
    with tab3:
        st.subheader("Từ điển 6 thuộc tính kỹ thuật và nhãn")
        meta_rows = [
            {"Thuộc tính": "buying", "Tên tiếng Việt": "Giá mua xe", "Miền giá trị hợp lệ": "vhigh, high, med, low", "Kiểu": "Danh mục (Categorical)"},
            {"Thuộc tính": "maint", "Tên tiếng Việt": "Chi phí bảo dưỡng", "Miền giá trị hợp lệ": "vhigh, high, med, low", "Kiểu": "Danh mục (Categorical)"},
            {"Thuộc tính": "doors", "Tên tiếng Việt": "Số cửa xe", "Miền giá trị hợp lệ": "2, 3, 4, 5more", "Kiểu": "Danh mục (Categorical)"},
            {"Thuộc tính": "persons", "Tên tiếng Việt": "Sức chứa (số người)", "Miền giá trị hợp lệ": "2, 4, more", "Kiểu": "Danh mục (Categorical)"},
            {"Thuộc tính": "lug_boot", "Tên tiếng Việt": "Khoang chứa hành lý", "Miền giá trị hợp lệ": "small, med, big", "Kiểu": "Danh mục (Categorical)"},
            {"Thuộc tính": "safety", "Tên tiếng Việt": "Mức độ an toàn", "Miền giá trị hợp lệ": "low, med, high", "Kiểu": "Danh mục (Categorical)"},
            {"Thuộc tính": "class", "Tên tiếng Việt": "Mức độ phù hợp", "Miền giá trị hợp lệ": "unacc, acc, good, vgood", "Kiểu": "Nhãn mục tiêu (Target)"}
        ]
        st.dataframe(pd.DataFrame(meta_rows), hide_index=True, use_container_width=True)

# ==============================================================================
# 4. TRANG CÂY QUYẾT ĐỊNH ID3
# ==============================================================================
elif menu == "🌳 Cây quyết định ID3":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">
            <span>🌳</span>
            <span>TRỰC QUAN HÓA CÂY QUYẾT ĐỊNH ID3</span>
        </div>
        <div class="hero-subtitle">
            Cơ chế phân tách Information Gain tại nút gốc, cấu trúc cây quyết định đa nhánh và trích xuất tập luật IF-THEN.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    stats = exp_results["tree_stats"]
    
    # 1. Bảng Information Gain tại nút gốc
    st.subheader("1. Bảng xếp hạng Information Gain tại Nút Gốc (Root Node)")
    st.markdown("Tại nút gốc (1.382 mẫu Train), ID3 tính Entropy và Information Gain của cả 6 thuộc tính để chọn thuộc tính phân chia tối ưu:")
    
    gain_table = stats.get("root_gain_table", {})
    sorted_gain = sorted(gain_table.items(), key=lambda x: x[1], reverse=True)
    gain_rows = []
    for rank, (attr, g) in enumerate(sorted_gain, 1):
        gain_rows.append({
            "Xếp hạng": f"#{rank}",
            "Thuộc tính": attr,
            "Tên tiếng Việt": ATTRIBUTE_NAMES_VI.get(attr, attr),
            "Information Gain (bits)": f"{g:.5f}",
            "Chọn làm Nút Gốc": "⭐ CÓ (Gain lớn nhất)" if rank == 1 else "Không"
        })
    st.dataframe(pd.DataFrame(gain_rows), hide_index=True, use_container_width=True)
    
    if os.path.exists("output/root_information_gain.png"):
        st.image("output/root_information_gain.png", caption="Biểu đồ trực quan hóa Information Gain tại Nút Gốc", use_container_width=True)
        
    st.markdown("---")
    
    # 2. Sơ đồ cây quyết định
    st.subheader("2. Sơ đồ Cây Quyết định (Decision Tree Graphviz)")
    preview_depth = st.slider("Giới hạn độ sâu hiển thị sơ đồ (Depth Preview):", min_value=1, max_value=4, value=2, step=1)
    
    dot_code = export_tree_to_dot(id3_model, max_depth=preview_depth)
    try:
        st.graphviz_chart(dot_code, use_container_width=True)
    except Exception as e:
        st.warning(f"Lưu ý hiển thị Graphviz: {e}")
        st.info("Hệ thống tự động hiển thị sơ đồ cây dạng văn bản phân cấp bên dưới.")
        
    # Văn bản phân cấp ASCII Text Tree
    with st.expander("📄 Xem sơ đồ cây dạng văn bản phân cấp (ASCII Text Tree)", expanded=False):
        text_tree = generate_text_tree(id3_model, max_depth=preview_depth + 1)
        st.code(text_tree, language="text")
        
    st.markdown("---")
    
    # 3. Trích xuất tập luật IF - THEN
    st.subheader("3. Trích xuất một số quy tắc quyết định (IF - THEN Rules)")
    rules = id3_model.get_rules()
    st.markdown(f"Toàn bộ cây quyết định gồm **{len(rules)} quy tắc độc lập**. Dưới đây là 10 quy tắc tiêu biểu:")
    
    sample_rules = rules[:10]
    for i, r in enumerate(sample_rules, 1):
        conds = " VÀ ".join([f"[{ATTRIBUTE_NAMES_VI.get(attr, attr)} = '{val}']" for attr, val in r["conditions"]])
        pred_vi = CLASS_MAPPING_VI.get(r["prediction"], {}).get("name", r["prediction"])
        st.markdown(f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #0284C7; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; font-size: 13px;">
            <strong>Quy tắc #{i}:</strong> NẾU {conds} THÌ Phân loại = <strong style="color: #0284C7;">{pred_vi.upper()} ({r['prediction']})</strong> 
            &nbsp;•&nbsp; Số mẫu ủng hộ: <code>{r['samples']}</code> &nbsp;•&nbsp; Entropy: <code>{r['entropy']:.2f}</code>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        if os.path.exists("output/id3_tree.dot"):
            with open("output/id3_tree.dot", "r", encoding="utf-8") as f:
                st.download_button("💾 Tải file Graphviz DOT (id3_tree.dot)", f.read(), file_name="id3_tree.dot", mime="text/plain", use_container_width=True)
    with bcol2:
        if os.path.exists("output/id3_tree_hierarchy.txt"):
            with open("output/id3_tree_hierarchy.txt", "r", encoding="utf-8") as f:
                st.download_button("💾 Tải file cây phân cấp (id3_tree_hierarchy.txt)", f.read(), file_name="id3_tree_hierarchy.txt", mime="text/plain", use_container_width=True)

# ==============================================================================
# 5. TRANG ĐÁNH GIÁ MÔ HÌNH
# ==============================================================================
elif menu == "📈 Đánh giá mô hình":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">
            <span>📈</span>
            <span>KẾT QUẢ ĐÁNH GIÁ & ĐỐI CHỨNG MÔ HÌNH</span>
        </div>
        <div class="hero-subtitle">
            Bảng chỉ số hiệu năng trên tập test độc lập (346 mẫu), đối chứng toàn diện với Baseline và ma trận nhầm lẫn Confusion Matrix.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    id3_eval = exp_results["id3_evaluation"]
    base_eval = exp_results["baseline_evaluation"]
    comp_table = exp_results["comparison_table"]
    
    st.subheader("1. So sánh chi tiết: Cây quyết định ID3 vs Baseline đối chứng")
    st.markdown("""
    - **Baseline Model:** Luôn dự đoán lớp đa số trong tập huấn luyện (`unacc`).
    - Do lớp `unacc` chiếm ~70% dữ liệu, chỉ số **Accuracy** có thể gây ngộ nhận. **Macro-F1** là thước đo quan trọng nhất phản ánh năng lực phân loại công bằng trên cả 4 lớp.
    """)
    st.table(pd.DataFrame(comp_table))
    
    st.markdown("---")
    
    col_e1, col_e2 = st.columns([1, 1])
    with col_e1:
        st.subheader("2. Ma trận nhầm lẫn (Confusion Matrix)")
        if os.path.exists("output/confusion_matrix.png"):
            st.image("output/confusion_matrix.png", use_container_width=True)
        else:
            st.info("Chưa có biểu đồ Confusion Matrix.")
    with col_e2:
        st.subheader("3. Biểu đồ so sánh trực quan các chỉ số")
        if os.path.exists("output/model_comparison.png"):
            st.image("output/model_comparison.png", use_container_width=True)
        else:
            st.info("Chưa có biểu đồ so sánh mô hình.")
            
    st.markdown("---")
    
    st.subheader("4. Hiệu năng chi tiết theo từng lớp của ID3")
    per_class_rows = []
    for cls, m in id3_eval["per_class"].items():
        per_class_rows.append({
            "Lớp (Class)": cls,
            "Tên tiếng Việt": CLASS_MAPPING_VI.get(cls, {}).get("name", cls),
            "Độ chuẩn xác (Precision)": f"{m['precision']*100:.2f}% ({m['precision']:.4f})",
            "Độ bao phủ (Recall)": f"{m['recall']*100:.2f}% ({m['recall']:.4f})",
            "F1-Score": f"{m['f1_score']*100:.2f}% ({m['f1_score']:.4f})",
            "Số mẫu test (Support)": m["support"]
        })
    st.dataframe(pd.DataFrame(per_class_rows), hide_index=True, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                st.download_button("💾 Tải Báo cáo kết quả đầy đủ (results.json)", f.read(), file_name="results.json", mime="application/json", use_container_width=True)
    with d2:
        if os.path.exists(PREDICTIONS_FILE):
            with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
                st.download_button("💾 Tải Dữ liệu dự đoán Test Set (predictions.csv)", f.read(), file_name="predictions.csv", mime="text/csv", use_container_width=True)

# ==============================================================================
# 6. TRANG GIỚI THIỆU ĐỀ TÀI
# ==============================================================================
elif menu == "ℹ️ Giới thiệu đề tài":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">
            <span>ℹ️</span>
            <span>THÔNG TIN ĐỒ ÁN & NHÓM THỰC HIỆN</span>
        </div>
        <div class="hero-subtitle">
            Học phần Khai phá dữ liệu (Data Mining) – Nhóm 06 – Lớp QT01.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 1. Thông tin chung về đề tài
    - **Tên đề tài:** Phân loại mức độ phù hợp của ô tô bằng thuật toán ID3.
    - **Tên tiếng Anh:** *Car Acceptability Classification Using the ID3 Decision Tree Algorithm*.
    - **Nhóm thực hiện:** Nhóm 06 – Lớp QT01.
    - **Thuật toán chính:** Cây quyết định ID3 (Iterative Dichotomiser 3) tự cài đặt 100% bằng Python, không sử dụng thư viện đóng gói sẵn của Scikit-learn.
    - **Khung giao diện:** Streamlit Dashboard.

    ### 👥 2. Nhóm thực hiện
    - **Nhóm 06 – Lớp QT01**, học phần Khai phá dữ liệu theo phân công học thuật.

    ### 📚 3. Nguồn dữ liệu & Tài liệu tham khảo
    1. **UCI Machine Learning Repository:** *Car Evaluation Dataset*  
       Website: [https://archive.ics.uci.edu/dataset/19/car+evaluation](https://archive.ics.uci.edu/dataset/19/car+evaluation)  
       Tác giả bộ dữ liệu: Marko Bohanec, Vladislav Rajkovič (1988, 1990).
    2. Quinlan, J. R. (1986). *Induction of decision trees*. Machine Learning, 1(1), 81–106.
    3. Shannon, C. E. (1948). *A mathematical theory of communication*. Bell System Technical Journal, 27(3), 379–423.
    4. Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques*. Morgan Kaufmann.

    ### ⚡ 4. Lệnh khởi chạy hệ thống
    Khởi động ứng dụng Streamlit:
    ```bash
    python -m streamlit run app.py
    ```
    Chạy kiểm thử đơn vị:
    ```bash
    python -m pytest tests/ -v
    ```
    Huấn luyện lại mô hình:
    ```bash
    python train.py
    ```
    """)
