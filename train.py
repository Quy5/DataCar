"""
train.py - Script huấn luyện cây quyết định ID3, đánh giá trên tập test và xuất kết quả.
Đồ án môn Khai phá dữ liệu - Nhóm 06 - QT01.
"""

import os
import json
import pandas as pd
from data_utils import load_data, validate_dataframe, split_data, FEATURE_COLUMNS, TARGET_COLUMN
from id3 import ID3DecisionTree
from evaluation import BaselineMajorityClassifier, evaluate_predictions, compare_models, analyze_errors
from visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_root_information_gain,
    export_tree_to_dot,
    generate_text_tree
)

DATA_PATH = "data/car.data"
OUTPUT_DIR = "output"

def run_training_pipeline():
    """Thực thi toàn bộ quy trình: Load -> Split -> Train -> Eval -> Export."""
    print("=" * 65)
    print("QUY TRÌNH HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH ID3 - NHÓM 06 - QT01")
    print("=" * 65)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Đọc dữ liệu
    print("\n[BƯỚC 1] Đọc và kiểm định bộ dữ liệu Car Evaluation...")
    if not os.path.exists(DATA_PATH):
        from download_data import download_data
        print("Chưa có file car.data. Đang tiến hành tải tự động...")
        download_data()
        
    df = load_data(DATA_PATH)
    is_valid, errors = validate_dataframe(df)
    if not is_valid:
        raise ValueError(f"Dữ liệu không hợp lệ: {errors}")
    print(f"-> Đã đọc thành công {len(df)} dòng dữ liệu từ {DATA_PATH}.")
    print(f"-> Phân bố nhãn toàn tập: {df[TARGET_COLUMN].value_counts().to_dict()}")
    
    # 2. Chia tập Train (80%) và Test (20%) có phân tầng (stratified)
    print("\n[BƯỚC 2] Chia dữ liệu 80% Train, 20% Test (random_state=42, stratify=class)...")
    train_df, test_df = split_data(df, test_size=0.2, random_state=42, stratify=True)
    print(f"-> Số mẫu Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"-> Phân bố Train: {train_df[TARGET_COLUMN].value_counts().to_dict()}")
    print(f"-> Số mẫu Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"-> Phân bố Test:  {test_df[TARGET_COLUMN].value_counts().to_dict()}")
    
    # 3. Huấn luyện mô hình ID3
    print("\n[BƯỚC 3] Huấn luyện cây quyết định ID3 trên tập Train...")
    id3_model = ID3DecisionTree()
    id3_model.fit(train_df, attributes=FEATURE_COLUMNS, target=TARGET_COLUMN)
    stats = id3_model.get_tree_stats()
    print("-> Huấn luyện thành công!")
    print(f"   - Tổng số nút (Nodes): {stats['total_nodes']}")
    print(f"   - Số nút lá (Leaves): {stats['leaf_nodes']}")
    print(f"   - Độ sâu cây (Depth): {stats['max_depth']}")
    print(f"   - Thuộc tính nút gốc (Root): '{stats['root_attribute']}'")
    print(f"   - Information Gain nút gốc: {stats['root_gain_table']}")
    
    # 4. Dự đoán trên tập Test
    print("\n[BƯỚC 4] Dự đoán trên tập Test độc lập...")
    y_test = test_df[TARGET_COLUMN].tolist()
    id3_preds = id3_model.predict(test_df)
    
    # 5. Huấn luyện Baseline đối chứng (Majority class)
    print("\n[BƯỚC 5] Huấn luyện Baseline đối chứng (Majority Classifier)...")
    baseline = BaselineMajorityClassifier()
    baseline.fit(train_df[TARGET_COLUMN].tolist())
    baseline_preds = baseline.predict(len(test_df))
    print(f"-> Baseline luôn dự đoán: '{baseline.majority_class}'")
    
    # 6. Đánh giá và so sánh
    print("\n[BƯỚC 6] Đánh giá các chỉ số (Accuracy, Macro-F1, Precision, Recall, Confusion Matrix)...")
    comparison = compare_models(y_test, id3_preds, baseline_preds)
    id3_eval = comparison["id3"]
    base_eval = comparison["baseline"]
    
    print("\n" + "=" * 50)
    print(f"{'Chỉ số (Metric)':<25} | {'Baseline':<10} | {'ID3 (Nhóm 06)':<12}")
    print("-" * 50)
    print(f"{'Accuracy':<25} | {base_eval['accuracy']:<10.4f} | {id3_eval['accuracy']:<12.4f}")
    print(f"{'Macro-F1':<25} | {base_eval['macro_f1']:<10.4f} | {id3_eval['macro_f1']:<12.4f}")
    print(f"{'Macro-Recall':<25} | {base_eval['macro_recall']:<10.4f} | {id3_eval['macro_recall']:<12.4f}")
    print(f"{'Macro-Precision':<25} | {base_eval['macro_precision']:<10.4f} | {id3_eval['macro_precision']:<12.4f}")
    print("=" * 50)
    
    print("\nChi tiết từng lớp của ID3:")
    for cls, m in id3_eval["per_class"].items():
        print(f"  - Lớp [{cls}]: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, F1={m['f1_score']:.4f}, Số mẫu test={m['support']}")
        
    # 7. Xuất kết quả ra file
    print("\n[BƯỚC 7] Xuất báo cáo, biểu đồ và file mô hình ra thư mục output/...")
    
    # 7.1. Lưu mô hình model.json
    model_path = os.path.join(OUTPUT_DIR, "model.json")
    id3_model.save_model(model_path)
    print(f"-> Đã lưu mô hình ID3 tại: {model_path}")
    
    # 7.2. Lưu kết quả results.json
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    results_data = {
        "dataset": {
            "name": "Car Evaluation (UCI)",
            "total_samples": len(df),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "class_distribution_all": df[TARGET_COLUMN].value_counts().to_dict(),
            "class_distribution_train": train_df[TARGET_COLUMN].value_counts().to_dict(),
            "class_distribution_test": test_df[TARGET_COLUMN].value_counts().to_dict()
        },
        "tree_stats": stats,
        "id3_evaluation": id3_eval,
        "baseline_evaluation": base_eval,
        "comparison_table": comparison["comparison_table"]
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    print(f"-> Đã lưu kết quả đánh giá tại: {results_path}")
    
    # 7.3. Lưu predictions.csv
    preds_df = test_df.copy()
    preds_df["Dự đoán ID3"] = id3_preds
    preds_df["Dự đoán Baseline"] = baseline_preds
    preds_df["Kết quả ID3"] = ["Đúng" if t == p else "Sai" for t, p in zip(y_test, id3_preds)]
    preds_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    preds_df.to_csv(preds_path, index=False, encoding="utf-8-sig")
    print(f"-> Đã lưu dự đoán chi tiết tại: {preds_path}")
    
    # 7.4. Biểu đồ phân bố lớp
    plot_class_distribution(df, target_col=TARGET_COLUMN, output_path=os.path.join(OUTPUT_DIR, "class_distribution.png"))
    
    # 7.5. Biểu đồ Confusion Matrix
    plot_confusion_matrix(id3_eval["confusion_matrix"], labels=id3_eval["labels"], output_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    
    # 7.6. Biểu đồ so sánh Baseline vs ID3
    plot_model_comparison(comparison, output_path=os.path.join(OUTPUT_DIR, "model_comparison.png"))
    
    # 7.7. Biểu đồ Root Information Gain
    plot_root_information_gain(stats["root_gain_table"], output_path=os.path.join(OUTPUT_DIR, "root_information_gain.png"))
    
    # 7.8. Xuất file DOT
    export_tree_to_dot(id3_model, output_path=os.path.join(OUTPUT_DIR, "id3_tree.dot"))
    export_tree_to_dot(id3_model, output_path=os.path.join(OUTPUT_DIR, "id3_tree_preview.dot"), max_depth=2)
    
    # 7.9. Xuất file text tree
    text_tree_path = os.path.join(OUTPUT_DIR, "id3_tree_hierarchy.txt")
    with open(text_tree_path, "w", encoding="utf-8") as f:
        f.write(generate_text_tree(id3_model, max_depth=3))
    print(f"-> Đã xuất toàn bộ biểu đồ và sơ đồ cây vào thư mục: {OUTPUT_DIR}")
    
    print("\n" + "=" * 65)
    print("[HOÀN TẤT] Pipeline huấn luyện và xuất kết quả đã kết thúc thành công!")
    print("=" * 65)
    return results_data

if __name__ == "__main__":
    run_training_pipeline()

