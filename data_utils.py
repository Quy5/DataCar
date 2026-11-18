"""
data_utils.py - Tiền xử lý, kiểm định và chuyển đổi dữ liệu Car Evaluation.
Đồ án môn Khai phá dữ liệu - Nhóm 06 - QT01.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Danh sách cột chuẩn
COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
FEATURE_COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
TARGET_COLUMN = "class"

# Miền giá trị hợp lệ cho từng thuộc tính
VALID_ATTRIBUTE_VALUES = {
    "buying": ["vhigh", "high", "med", "low"],
    "maint": ["vhigh", "high", "med", "low"],
    "doors": ["2", "3", "4", "5more"],
    "persons": ["2", "4", "more"],
    "lug_boot": ["small", "med", "big"],
    "safety": ["low", "med", "high"]
}

VALID_TARGET_VALUES = ["unacc", "acc", "good", "vgood"]

# Thứ tự chuẩn của các thuộc tính để đảm bảo tính tái lập (deterministic)
ATTRIBUTE_ORDER = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

# Ánh xạ tên thuộc tính sang tiếng Việt
ATTRIBUTE_NAMES_VI = {
    "buying": "Giá mua xe",
    "maint": "Chi phí bảo dưỡng",
    "doors": "Số cửa xe",
    "persons": "Sức chứa (số người)",
    "lug_boot": "Khoang hành lý",
    "safety": "Mức độ an toàn",
    "class": "Mức độ phù hợp"
}

# Ánh xạ giá trị thuộc tính sang tiếng Việt
VALUE_MAPPING_VI = {
    "buying": {
        "vhigh": "Rất cao",
        "high": "Cao",
        "med": "Trung bình",
        "low": "Thấp"
    },
    "maint": {
        "vhigh": "Rất cao",
        "high": "Cao",
        "med": "Trung bình",
        "low": "Thấp"
    },
    "doors": {
        "2": "2 cửa",
        "3": "3 cửa",
        "4": "4 cửa",
        "5more": "Từ 5 cửa trở lên"
    },
    "persons": {
        "2": "2 người",
        "4": "4 người",
        "more": "Trên 4 người"
    },
    "lug_boot": {
        "small": "Nhỏ",
        "med": "Trung bình",
        "big": "Lớn"
    },
    "safety": {
        "low": "Thấp",
        "med": "Trung bình",
        "high": "Cao"
    }
}

# Ánh xạ nhãn sang tiếng Việt và màu sắc hiển thị
CLASS_MAPPING_VI = {
    "unacc": {
        "vi": "Không phù hợp (unacc)",
        "name": "Không phù hợp",
        "description": "Xe không đáp ứng các tiêu chuẩn cơ bản hoặc tính an toàn/chi phí không phù hợp.",
        "color": "#e74c3c",   # Đỏ
        "badge": "badge-danger"
    },
    "acc": {
        "vi": "Chấp nhận được (acc)",
        "name": "Chấp nhận được",
        "description": "Xe đáp ứng các yêu cầu tối thiểu về an toàn và tiện ích ở mức cơ bản.",
        "color": "#f39c12",   # Vàng cam
        "badge": "badge-warning"
    },
    "good": {
        "vi": "Tốt (good)",
        "name": "Tốt",
        "description": "Xe có các thông số tốt về chi phí, an toàn và độ tiện dụng.",
        "color": "#27ae60",   # Xanh lục
        "badge": "badge-success"
    },
    "vgood": {
        "vi": "Rất tốt (vgood)",
        "name": "Rất tốt",
        "description": "Xe có độ an toàn cao, chi phí hợp lý và đáp ứng tối đa mọi tiêu chuẩn.",
        "color": "#2980b9",   # Xanh dương
        "badge": "badge-primary"
    }
}

# Ánh xạ ngược từ tiếng Việt sang mã gốc UCI
REVERSE_VALUE_MAPPING = {
    attr: {v: k for k, v in vals.items()}
    for attr, vals in VALUE_MAPPING_VI.items()
}

def load_data(filepath="data/car.data"):
    """
    Đọc dữ liệu từ file car.data (không header).
    Trả về DataFrame với 7 cột chuẩn.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {filepath}")
    
    df = pd.read_csv(filepath, header=None, names=COLUMNS, dtype=str)
    # Loại bỏ khoảng trắng thừa nếu có
    for col in df.columns:
        df[col] = df[col].str.strip()
    return df

def validate_dataframe(df):
    """
    Kiểm tra tính hợp lệ của DataFrame:
    - Đúng 7 cột
    - Không chứa giá trị thiếu (missing)
    - Miền giá trị từng cột chuẩn
    """
    errors = []
    if list(df.columns) != COLUMNS:
        errors.append(f"Tên cột không đúng chuẩn: {list(df.columns)}")
        # Nếu thiếu cột thì không thể kiểm tra tiếp các giá trị từng cột
        return False, errors
    
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        errors.append(f"Dữ liệu có {missing_count} ô bị thiếu (missing).")
        
    for col, valid_vals in VALID_ATTRIBUTE_VALUES.items():
        if col in df.columns:
            invalid = set(df[col].unique()) - set(valid_vals)
            if invalid:
                errors.append(f"Cột {col} chứa giá trị không hợp lệ: {invalid}")
        else:
            errors.append(f"Thiếu cột {col}.")
            
    if TARGET_COLUMN in df.columns:
        invalid_target = set(df[TARGET_COLUMN].unique()) - set(VALID_TARGET_VALUES)
        if invalid_target:
            errors.append(f"Cột {TARGET_COLUMN} chứa nhãn không hợp lệ: {invalid_target}")
    else:
        errors.append(f"Thiếu cột nhãn {TARGET_COLUMN}.")
        
    return len(errors) == 0, errors

def split_data(df, test_size=0.2, random_state=42, stratify=True):
    """
    Chia dữ liệu thành train (80%) và test (20%) theo phân tầng (stratified).
    Trả về train_df, test_df.
    """
    stratify_col = df[TARGET_COLUMN] if stratify else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def get_dataset_summary(df):
    """
    Trả về tóm tắt thống kê của DataFrame:
    - Số dòng, số cột
    - Phân bố nhãn
    - Số lượng trùng lặp
    - Miền giá trị từng cột
    """
    class_dist = df[TARGET_COLUMN].value_counts().to_dict()
    duplicates_count = int(df.duplicated().sum())
    
    attr_distributions = {}
    for col in FEATURE_COLUMNS:
        attr_distributions[col] = df[col].value_counts().to_dict()
        
    return {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "class_distribution": class_dist,
        "duplicates_count": duplicates_count,
        "missing_values": int(df.isnull().sum().sum()),
        "attribute_distributions": attr_distributions
    }

def convert_vi_input_to_uci(input_vi_dict):
    """
    Chuyển dictionary đầu vào bằng tiếng Việt sang mã UCI.
    Ví dụ: {"buying": "Rất cao"} -> {"buying": "vhigh"}
    """
    raw_dict = {}
    for col in FEATURE_COLUMNS:
        vi_val = input_vi_dict.get(col)
        if vi_val is None:
            raise ValueError(f"Thiếu giá trị cho thuộc tính '{col}'.")
            
        # Nếu đã là mã UCI thì giữ nguyên
        if vi_val in VALID_ATTRIBUTE_VALUES[col]:
            raw_dict[col] = vi_val
        elif vi_val in REVERSE_VALUE_MAPPING[col]:
            raw_dict[col] = REVERSE_VALUE_MAPPING[col][vi_val]
        else:
            raise ValueError(
                f"Giá trị '{vi_val}' không hợp lệ cho thuộc tính '{col}'. "
                f"Các giá trị hợp lệ: {list(REVERSE_VALUE_MAPPING[col].keys())} hoặc {VALID_ATTRIBUTE_VALUES[col]}"
            )
    return raw_dict

