"""
app.py - Ứng dụng web phân lớp mức độ chấp nhận ô tô bằng cây quyết định ID3.
Đồ án môn Khai phá dữ liệu - Nhóm 06 - QT01.

Bố cục: 4 tab, mở vào là dùng được ngay.
  1. Dự đoán        - chọn 6 thông số, xem nhãn và đường đi trên cây (tab mặc định)
  2. Cây ID3        - Gain tại nút gốc, sơ đồ cây, tập luật IF-THEN
  3. Đánh giá       - chỉ số trên tập test, ma trận nhầm lẫn, 10-fold, phân tích lỗi
  4. Dữ liệu        - phân bố nhãn, bảng dữ liệu, ý nghĩa thuộc tính
"""

import json
import os

import altair as alt
import pandas as pd
import streamlit as st

from data_utils import (
    ATTRIBUTE_NAMES_VI,
    CLASS_MAPPING_VI,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    VALID_ATTRIBUTE_VALUES,
    VALUE_MAPPING_VI,
    load_data,
    split_data,
)
from id3 import ID3DecisionTree, entropy, information_gain

st.set_page_config(
    page_title="Phân lớp ô tô bằng ID3 – Nhóm 06",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_FILE = "data/car.data"
MODEL_FILE = "output/model.json"
RESULTS_FILE = "output/results.json"

CLASS_ORDER = ["unacc", "acc", "good", "vgood"]
# Màu theo thang đánh giá: đỏ -> cam -> xanh lá -> xanh lam (chữ trắng đều đạt tương phản AA)
CLASS_COLOR = {"unacc": "#B42318", "acc": "#B54708", "good": "#2E7D32", "vgood": "#1D4ED8"}
CLASS_NAME = {c: CLASS_MAPPING_VI[c]["name"] for c in CLASS_ORDER}


def fmt(n: int) -> str:
    """Định dạng số kiểu Việt Nam: 1382 -> 1.382."""
    return f"{n:,}".replace(",", ".")

# Chỉ một khối CSS nhỏ cho thang nhãn kiểu "nhãn đánh giá" và danh sách đường đi
st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1180px;}
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {display: none;}
.grade {display: flex; flex-direction: column; gap: 6px; margin: 4px 0 12px;}
.grade-row {display: flex; align-items: center; gap: 10px;}
.grade-bar {height: 34px; display: flex; align-items: center; padding: 0 12px;
            color: #fff; font-weight: 600; border-radius: 4px 0 0 4px; opacity: .28;
            clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 50%, calc(100% - 14px) 100%, 0 100%);}
.grade-bar.on {opacity: 1;}
.grade-tag {font-weight: 700; font-size: 1.05rem;}
.path {margin: 0; padding-left: 1.4rem;}
.path li {margin: 4px 0;}
.path .val {font-weight: 600;}
.muted {opacity: .7; font-size: .9rem;}
@media (prefers-reduced-motion: no-preference) {.grade-bar.on {transition: opacity .2s;}}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ dữ liệu & mô hình
@st.cache_data
def get_data() -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        from download_data import download_data
        download_data()
    return load_data(DATA_FILE)


@st.cache_resource
def get_model_and_results():
    if not (os.path.exists(MODEL_FILE) and os.path.exists(RESULTS_FILE)):
        from train import run_training_pipeline
        run_training_pipeline()
    model = ID3DecisionTree.load_model(MODEL_FILE)
    with open(RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)
    return model, results


LEAF_FILL = {"unacc": "#FDE2E1", "acc": "#FDEBD3", "good": "#DDF0DE", "vgood": "#DCE6FB"}


@st.cache_data
def get_dot(depth: int) -> str:
    """Sơ đồ cây với nhãn tiếng Việt, dễ đọc cho người xem (khác bản kỹ thuật trong output/id3_tree.dot)."""
    model, _ = get_model_and_results()
    lines = ['digraph G {', '  rankdir=TB; nodesep=0.25; ranksep=0.45;',
             '  node [fontname="Helvetica", fontsize=11, shape=box, style="rounded,filled", color="#94A3B8", fillcolor="#F8FAFC"];',
             '  edge [fontname="Helvetica", fontsize=10, color="#94A3B8", fontcolor="#334155"];']

    def visit(node):
        nid = f"n{node.node_id}"
        if node.is_leaf:
            if node.samples_count == 0:
                lines.append(f'  {nid} [label="{CLASS_NAME[node.prediction]}\\n(không có mẫu học)", '
                             f'fillcolor="{LEAF_FILL[node.prediction]}", style="rounded,filled,dashed"];')
            else:
                lines.append(f'  {nid} [label="{CLASS_NAME[node.prediction]}\\n{fmt(node.samples_count)} mẫu", '
                             f'fillcolor="{LEAF_FILL[node.prediction]}", color="{CLASS_COLOR[node.prediction]}"];')
            return
        name = ATTRIBUTE_NAMES_VI[node.attribute]
        if node.depth >= depth:
            lines.append(f'  {nid} [label="{name} …\\n{fmt(node.samples_count)} mẫu, đa số {node.majority_class}", '
                         f'style="rounded,dashed", fillcolor="white"];')
            return
        lines.append(f'  {nid} [label="{name}?\\n{fmt(node.samples_count)} mẫu", penwidth=1.6, color="#1D4ED8"];')
        order = VALID_ATTRIBUTE_VALUES.get(node.attribute, [])
        for v, child in sorted(node.children.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99):
            lines.append(f'  {nid} -> n{child.node_id} [label=" {VALUE_MAPPING_VI[node.attribute].get(v, v)} "];')
            visit(child)

    visit(model.root)
    lines.append("}")
    return "\n".join(lines)


@st.cache_resource
def get_node_index() -> dict:
    """node_id -> nút, để hiển thị Gain tại từng bước của đường đi."""
    model, _ = get_model_and_results()
    idx, stack = {}, [model.root]
    while stack:
        n = stack.pop()
        idx[n.node_id] = n
        stack.extend(n.children.values())
    return idx


@st.cache_data
def get_train() -> pd.DataFrame:
    """Đúng tập train mà train.py dùng (80%, phân tầng, random_state=42)."""
    train_df, _ = split_data(get_data(), test_size=0.2, random_state=42, stratify=True)
    return train_df


@st.cache_data
def get_rules_df() -> pd.DataFrame:
    model, _ = get_model_and_results()
    rows = []
    for r in model.get_rules():
        cond = " và ".join(
            f"{ATTRIBUTE_NAMES_VI[a]} = {VALUE_MAPPING_VI[a].get(v, v)}" for a, v in r["conditions"]
        )
        rows.append({"Nếu": cond, "Thì": CLASS_NAME[r["prediction"]], "Nhãn": r["prediction"],
                     "Số mẫu": r["samples"], "Số điều kiện": len(r["conditions"])})
    return pd.DataFrame(rows).sort_values("Số mẫu", ascending=False, ignore_index=True)


try:
    df = get_data()
    model, results = get_model_and_results()
except Exception as e:  # thiếu file hoặc lỗi dữ liệu: nói rõ cách khắc phục
    st.error(f"Không khởi tạo được ứng dụng: {e}")
    st.info("Chạy `python train.py` trong thư mục dự án để tạo lại dữ liệu và mô hình, rồi tải lại trang.")
    st.stop()

stats = model.get_tree_stats()
ev = results["id3_evaluation"]
base = results["baseline_evaluation"]
cv = results.get("cross_validation")
el = results.get("empty_leaf_analysis")
n_train = results["dataset"]["train_samples"]
n_test = results["dataset"]["test_samples"]

# ------------------------------------------------------------------ tiêu đề
st.title("Phân lớp mức độ chấp nhận ô tô bằng cây quyết định ID3")
st.caption(
    f"Nhóm 06, lớp QT01, môn Khai phá dữ liệu. Thuật toán ID3 tự cài đặt, học từ {fmt(n_train)} mẫu của "
    f"bộ UCI Car Evaluation; độ chính xác trên {n_test} mẫu kiểm tra: {ev['accuracy']*100:.2f}%."
)

tab_pred, tab_tree, tab_eval, tab_data = st.tabs(["Dự đoán", "Cây ID3", "Đánh giá", "Dữ liệu"])

# ================================================================== 1. DỰ ĐOÁN
DEFAULT_INPUT = {"buying": "med", "maint": "med", "doors": "4", "persons": "4", "lug_boot": "med", "safety": "high"}
PRESETS = {
    "Xe không phù hợp": {"buying": "vhigh", "maint": "vhigh", "doors": "2", "persons": "2", "lug_boot": "small", "safety": "low"},
    "Xe chấp nhận được": {"buying": "med", "maint": "med", "doors": "4", "persons": "4", "lug_boot": "med", "safety": "med"},
    "Xe tốt": {"buying": "low", "maint": "low", "doors": "4", "persons": "4", "lug_boot": "big", "safety": "med"},
    "Xe rất tốt": {"buying": "low", "maint": "low", "doors": "4", "persons": "4", "lug_boot": "big", "safety": "high"},
}
for a, v in DEFAULT_INPUT.items():
    st.session_state.setdefault(f"in_{a}", v)


def apply_preset(values: dict) -> None:
    for a, v in values.items():
        st.session_state[f"in_{a}"] = v


GROUPS = [("Chi phí", ["buying", "maint"]), ("Kích thước", ["doors", "persons", "lug_boot"]), ("An toàn", ["safety"])]

with tab_pred:
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.subheader("Thông số xe")
        st.write("Ví dụ có sẵn:")
        pc = st.columns(len(PRESETS))
        for col, (label, values) in zip(pc, PRESETS.items()):
            col.button(label, on_click=apply_preset, args=(values,), width="stretch")

        for group, attrs in GROUPS:
            with st.container(border=True):
                st.markdown(f"**{group}**")
                for a in attrs:
                    st.segmented_control(
                        ATTRIBUTE_NAMES_VI[a],
                        options=VALID_ATTRIBUTE_VALUES[a],
                        format_func=lambda v, a=a: VALUE_MAPPING_VI[a][v],
                        key=f"in_{a}",
                        required=True,  # luôn có đúng một giá trị được chọn
                    )

    sample = {a: st.session_state.get(f"in_{a}") for a in FEATURE_COLUMNS}
    missing = [ATTRIBUTE_NAMES_VI[a].lower() for a, v in sample.items() if v is None]

    with right:
        st.subheader("Kết quả")
        if missing:
            st.info("Chọn thêm: " + ", ".join(missing) + ".")
        else:
            res = model.predict_with_path(sample)
            pred = res["prediction"]

            bars = []
            for i, c in enumerate(CLASS_ORDER):
                on = " on" if c == pred else ""
                width = 40 + i * 12  # thanh dài dần như nhãn đánh giá sản phẩm
                tag = f'<span class="grade-tag">◀ {CLASS_NAME[c]}</span>' if c == pred else ""
                bars.append(
                    f'<div class="grade-row"><div class="grade-bar{on}" '
                    f'style="width:{width}%;background:{CLASS_COLOR[c]}">{c}</div>{tag}</div>'
                )
            st.markdown('<div class="grade">' + "".join(bars) + "</div>", unsafe_allow_html=True)
            st.write(CLASS_MAPPING_VI[pred]["description"])

            st.markdown("**Cây đã đi qua các bước:**")
            items = []
            for step in res["path"]:
                if step.get("is_leaf"):
                    continue
                a, v = step["attribute"], step["checked_value"]
                g = get_node_index()[step["node_id"]].gain_table.get(a, 0.0)
                items.append(
                    f'<li>{ATTRIBUTE_NAMES_VI[a]}: <span class="val">{VALUE_MAPPING_VI[a].get(v, v)}</span> '
                    f'<span class="muted">(Gain cao nhất = {g:.3f}; nút có {fmt(step["samples_count"])} mẫu học)</span></li>'
                )
            leaf = res["path"][-1] if res["path"] and res["path"][-1].get("is_leaf") else None
            items.append(f'<li>Kết luận: <span class="val">{CLASS_NAME[pred]} ({pred})</span></li>')
            st.markdown('<ol class="path">' + "".join(items) + "</ol>", unsafe_allow_html=True)

            if res["is_fallback"]:
                st.warning(res["fallback_reason"])
            elif res.get("is_empty_leaf"):
                st.warning(
                    "Tập huấn luyện không có xe nào giống cấu hình này, nên cây lấy nhãn chiếm đa số ở nút "
                    "phía trên. Kết quả này kém tin cậy: trên tập kiểm tra, nhóm trường hợp này chỉ đúng "
                    f"{el['empty_leaf_correct']}/{el['empty_leaf_samples']}." if el else
                    "Tập huấn luyện không có xe nào giống cấu hình này; kết quả kém tin cậy."
                )
            elif leaf:
                dist = ", ".join(f"{k}: {v}" for k, v in leaf["class_distribution"].items())
                st.caption(f"Nút lá này có {leaf['samples_count']} mẫu học ({dist}).")

            row = df[(df[FEATURE_COLUMNS] == pd.Series(sample)).all(axis=1)]
            if len(row):
                truth = row.iloc[0][TARGET_COLUMN]
                verdict = "trùng với dự đoán" if truth == pred else "khác với dự đoán"
                st.caption(f"Nhãn trong bộ dữ liệu gốc của cấu hình này: {CLASS_NAME[truth]} ({truth}), {verdict}.")

# ================================================================== 2. CÂY ID3
with tab_tree:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nút gốc", stats["root_attribute"])
    c2.metric("Số nút", stats["total_nodes"])
    c3.metric("Số luật có mẫu", stats["rules_with_support"],
              help=f"{stats['leaf_nodes']} lá, trong đó {stats['empty_leaves']} lá không có mẫu học.")
    c4.metric("Độ sâu", stats["max_depth"])

    g1, g2 = st.columns([1, 1.6], gap="large")
    with g1:
        st.subheader("Chọn nút gốc")
        st.write(f"ID3 tính Information Gain của 6 thuộc tính trên {fmt(n_train)} mẫu học và chọn thuộc tính lớn nhất.")
        gain = pd.DataFrame(
            [{"Thuộc tính": ATTRIBUTE_NAMES_VI[a], "Gain": g,
              "Loại": "Nút gốc" if a == stats["root_attribute"] else "Khác"}
             for a, g in stats["root_gain_table"].items()]
        )
        chart = (
            alt.Chart(gain)
            .mark_bar()
            .encode(
                x=alt.X("Gain:Q", title="Information Gain (bit)"),
                y=alt.Y("Thuộc tính:N", sort="-x", title=None),
                color=alt.Color("Loại:N", scale=alt.Scale(domain=["Nút gốc", "Khác"], range=["#1D4ED8", "#94A3B8"]),
                                legend=None),
                tooltip=["Thuộc tính", alt.Tooltip("Gain:Q", format=".4f")],
            )
            .properties(height=230)
        )
        text = chart.mark_text(align="left", dx=4).encode(text=alt.Text("Gain:Q", format=".3f"), color=alt.value("#475569"))
        st.altair_chart(chart + text)
    with g2:
        st.subheader("Sơ đồ cây")
        depth = st.slider("Số tầng hiển thị", 1, 4, 2, help="Chỉ thay đổi hình vẽ, không thay đổi mô hình.")
        st.graphviz_chart(get_dot(depth), width="stretch")

    with st.expander("Cách tính tại nút gốc (công thức theo bài giảng)", expanded=False):
        train = get_train()
        root = stats["root_attribute"]
        y = train[TARGET_COLUMN]
        cnt = y.value_counts().reindex(CLASS_ORDER, fill_value=0)
        st.markdown("**Bước 1.** Lượng thông tin cần để phân loại một phần tử trong D (Entropy của D), "
                    "với pᵢ là tỷ lệ phần tử thuộc lớp Cᵢ:")
        st.latex(r"Info(D) = -\sum_{i=1}^{m} p_i \log_2(p_i)")
        terms = " ".join(f"-\\tfrac{{{cnt[c]}}}{{{len(y)}}}\\log_2\\tfrac{{{cnt[c]}}}{{{len(y)}}}" for c in CLASS_ORDER)
        st.latex(f"Info(D) = {terms} = {entropy(y.tolist()):.4f}")
        st.markdown("**Bước 2.** Thuộc tính A chia D thành v phân hoạch D₁…Dᵥ; tính lượng thông tin sau khi chia "
                    "và độ lợi thông tin. ID3 chọn thuộc tính có Gain lớn nhất:")
        st.latex(r"Info_A(D) = \sum_{j=1}^{v} \frac{|D_j|}{|D|}\, Info(D_j) \qquad Gain(A) = Info(D) - Info_A(D)")
        rows = []
        for a in FEATURE_COLUMNS:
            gain_a, base_ent, _ = information_gain(train, a, TARGET_COLUMN)
            rows.append({"Thuộc tính": f"{ATTRIBUTE_NAMES_VI[a]} ({a})", "Info_A(D)": base_ent - gain_a, "Gain(A)": gain_a})
        st.dataframe(pd.DataFrame(rows).sort_values("Gain(A)", ascending=False), hide_index=True,
                     column_config={k: st.column_config.NumberColumn(format="%.4f") for k in ["Info_A(D)", "Gain(A)"]})
        st.markdown(f"**Bước 3.** Gain({root}) lớn nhất nên {ATTRIBUTE_NAMES_VI[root].lower()} làm nút gốc. "
                    "Các phân hoạch của nút gốc:")
        part = []
        for v in VALID_ATTRIBUTE_VALUES[root]:
            dj = train[train[root] == v][TARGET_COLUMN]
            c = dj.value_counts().reindex(CLASS_ORDER, fill_value=0)
            part.append({"Giá trị": VALUE_MAPPING_VI[root][v], "|Dⱼ|": len(dj), **{k: int(c[k]) for k in CLASS_ORDER},
                         "Info(Dⱼ)": entropy(dj.tolist())})
        st.dataframe(pd.DataFrame(part), hide_index=True,
                     column_config={"Info(Dⱼ)": st.column_config.NumberColumn(format="%.4f")})
        st.markdown("**Bước 4.** Lặp lại trên từng phân hoạch với các thuộc tính còn lại. Theo thuật toán trong bài giảng, "
                    "một nhánh dừng khi: mọi mẫu cùng lớp → lá mang lớp đó; hết thuộc tính → lá mang lớp đa số; "
                    "phân hoạch rỗng → lá mang lớp đa số của nút cha. Trên dữ liệu này cây có "
                    f"{stats['rules_with_support']} lá thuần nhất và {stats['empty_leaves']} lá từ phân hoạch rỗng.")

    st.subheader("Luật IF–THEN")
    rules = get_rules_df()
    pick = st.segmented_control("Lọc theo kết luận", ["Tất cả"] + [CLASS_NAME[c] for c in CLASS_ORDER], default="Tất cả", required=True)
    shown = rules if pick in (None, "Tất cả") else rules[rules["Thì"] == pick]
    st.caption(f"{len(shown)} luật, sắp theo số mẫu học ủng hộ.")
    st.dataframe(shown.drop(columns=["Nhãn"]), hide_index=True, height=320)

# ================================================================== 3. ĐÁNH GIÁ
with tab_eval:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy (tập kiểm tra)", f"{ev['accuracy']*100:.2f}%",
              f"{(ev['accuracy'] - base['accuracy'])*100:+.2f} điểm so với đoán toàn unacc")
    m2.metric("Macro-F1", f"{ev['macro_f1']:.4f}", f"{ev['macro_f1'] - base['macro_f1']:+.4f}")
    if cv:
        m3.metric(f"Accuracy {cv['k']}-fold", f"{cv['accuracy_mean']*100:.2f}%", f"± {cv['accuracy_std']*100:.2f}",
                  delta_color="off")
        m4.metric(f"Macro-F1 {cv['k']}-fold", f"{cv['macro_f1_mean']:.4f}", f"± {cv['macro_f1_std']:.4f}",
                  delta_color="off")
    st.caption(f"Tập kiểm tra gồm {n_test} mẫu tách riêng (20%, giữ tỷ lệ nhãn), không dùng khi xây cây. "
               "Mô hình đối chứng luôn đoán nhãn phổ biến nhất (unacc).")

    e1, e2 = st.columns(2, gap="large")
    with e1:
        st.subheader("Ma trận nhầm lẫn")
        labels = ev["labels"]
        cm = pd.DataFrame(ev["confusion_matrix"], index=[f"Thật: {l}" for l in labels],
                          columns=[f"Đoán: {l}" for l in labels])
        st.dataframe(cm.style.background_gradient(cmap="Blues", axis=None))
        st.caption(f"Đường chéo là số mẫu đoán đúng: {sum(ev['confusion_matrix'][i][i] for i in range(4))}/{n_test}.")
    with e2:
        st.subheader("Theo từng nhãn")
        cmx = ev["confusion_matrix"]
        rows = []
        for i, c in enumerate(labels):
            tp = cmx[i][i]
            fn = sum(cmx[i]) - tp
            fp = sum(cmx[r][i] for r in range(len(labels))) - tp
            tn = n_test - tp - fn - fp
            m = ev["per_class"][c]
            rows.append({"Lớp": c, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "Accuracy(cᵢ)": (tp + tn) / n_test,
                         "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1_score"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     column_config={k: st.column_config.NumberColumn(format="%.4f")
                                    for k in ["Accuracy(cᵢ)", "Precision", "Recall", "F1"]})
        st.caption("TP, FP, FN, TN đọc từ ma trận nhầm lẫn bên trái theo từng lớp cᵢ.")

    with st.expander("Công thức đánh giá (theo bài giảng)"):
        st.markdown("Độ chính xác trên tập kiểm thử, với o(x) là lớp hệ thống dự đoán và c(x) là lớp thực sự của ví dụ x:")
        st.latex(r"Accuracy = \frac{1}{|D\_test|}\sum_{x \in D\_test} Identical(o(x), c(x))")
        st.markdown("Với từng lớp cᵢ (TP, FP, TN, FN lấy từ ma trận nhầm lẫn):")
        st.latex(r"Accuracy(c_i)=\frac{TP_i+TN_i}{TP_i+FP_i+TN_i+FN_i}\qquad Precision(c_i)=\frac{TP_i}{TP_i+FP_i}")
        st.latex(r"Recall(c_i)=\frac{TP_i}{TP_i+FN_i}\qquad F1=2\times\frac{Precision\times Recall}{Precision+Recall}")
        st.markdown("Macro-F1 là trung bình F1 của 4 lớp (nhóm bổ sung vì dữ liệu lệch lớp). "
                    "Hold-out: D_train và D_test không giao nhau; bài giảng gợi ý tỷ lệ 2/3–1/3, nhóm dùng 80/20 "
                    "để tập học đủ các tổ hợp hơn. Cross-validation: chia D thành k = 10 fold phân tầng, "
                    "mỗi lần 1 fold làm tập kiểm thử, lấy trung bình k kết quả.")

    if el:
        st.subheader("Mô hình sai ở đâu")
        st.write(
            f"{el['errors_from_empty_leaves']}/{el['total_errors']} lỗi rơi vào trường hợp tập huấn luyện không có cấu hình "
            f"tương tự (lá không có mẫu học). Ở nhóm này mô hình chỉ đúng {el['empty_leaf_correct']}/{el['empty_leaf_samples']} "
            f"mẫu, trong khi các mẫu còn lại đúng {el['other_correct']}/{el['other_samples']} "
            f"({el['other_accuracy']*100:.1f}%)."
        )
    if cv:
        with st.expander(f"Chi tiết {cv['k']} lần đánh giá chéo"):
            st.write(f"Tỷ lệ lỗi trung bình của {cv['k']} fold: {(1 - cv['accuracy_mean'])*100:.2f}% "
                     f"(= 1 − Accuracy trung bình).")
            st.dataframe(pd.DataFrame(cv["folds"]).rename(columns={
                "fold": "Lần", "test_samples": "Số mẫu kiểm tra", "accuracy": "Accuracy", "macro_f1": "Macro-F1"}),
                hide_index=True)

# ================================================================== 4. DỮ LIỆU
with tab_data:
    d1, d2 = st.columns([1, 1.4], gap="large")
    with d1:
        st.subheader("Phân bố nhãn")
        dist = df[TARGET_COLUMN].value_counts().reindex(CLASS_ORDER).reset_index()
        dist.columns = ["Nhãn", "Số mẫu"]
        dist["Tỷ lệ"] = dist["Số mẫu"] / len(df)
        st.altair_chart(
            alt.Chart(dist).mark_bar().encode(
                x=alt.X("Số mẫu:Q"),
                y=alt.Y("Nhãn:N", sort=CLASS_ORDER, title=None),
                color=alt.Color("Nhãn:N", scale=alt.Scale(domain=CLASS_ORDER, range=[CLASS_COLOR[c] for c in CLASS_ORDER]),
                                legend=None),
                tooltip=["Nhãn", "Số mẫu", alt.Tooltip("Tỷ lệ:Q", format=".1%")],
            ).properties(height=180),
                    )
        st.caption(f"{fmt(len(df))} cấu hình xe, đủ mọi tổ hợp của 6 thuộc tính, không thiếu giá trị. "
                   "Nhãn unacc chiếm 70%, nên cần xem Macro-F1 thay vì chỉ Accuracy.")
        st.subheader("Ý nghĩa thuộc tính")
        st.dataframe(pd.DataFrame([
            {"Thuộc tính": a, "Ý nghĩa": ATTRIBUTE_NAMES_VI[a],
             "Giá trị": ", ".join(VALUE_MAPPING_VI[a][v] for v in VALID_ATTRIBUTE_VALUES[a])}
            for a in FEATURE_COLUMNS]), hide_index=True)
    with d2:
        st.subheader("Bảng dữ liệu")
        f1, f2 = st.columns(2)
        fc = f1.multiselect("Nhãn", CLASS_ORDER)
        fs = f2.multiselect("Mức an toàn", VALID_ATTRIBUTE_VALUES["safety"],
                            format_func=lambda v: VALUE_MAPPING_VI["safety"][v])
        view = df
        if fc:
            view = view[view[TARGET_COLUMN].isin(fc)]
        if fs:
            view = view[view["safety"].isin(fs)]
        st.caption(f"{len(view)} / {len(df)} dòng")
        st.dataframe(view, hide_index=True, height=420)
    st.caption("Nguồn: Bohanec, M. (1988). Car Evaluation. UCI Machine Learning Repository, "
               "https://archive.ics.uci.edu/dataset/19/car+evaluation")
