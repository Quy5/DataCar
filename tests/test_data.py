"""
test_data.py - Kiểm thử tiền xử lý, ánh xạ tiếng Việt, schema và phân tầng dữ liệu.
"""

import pytest
import pandas as pd
from data_utils import (
    COLUMNS,
    FEATURE_COLUMNS,
    VALID_ATTRIBUTE_VALUES,
    VALID_TARGET_VALUES,
    validate_dataframe,
    split_data,
    convert_vi_input_to_uci
)

def test_vietnamese_to_uci_mapping_valid():
    """11. Mapping tiếng Việt sang mã UCI chuẩn xác."""
    sample_vi = {
        "buying": "Rất cao",
        "maint": "Cao",
        "doors": "4 cửa",
        "persons": "4 người",
        "lug_boot": "Lớn",
        "safety": "Cao"
    }
    converted = convert_vi_input_to_uci(sample_vi)
    assert converted["buying"] == "vhigh"
    assert converted["maint"] == "high"
    assert converted["doors"] == "4"
    assert converted["persons"] == "4"
    assert converted["lug_boot"] == "big"
    assert converted["safety"] == "high"

def test_vietnamese_to_uci_invalid_value():
    """Xác thực và bắt lỗi khi nhập giá trị không tồn tại trong từ điển."""
    bad_sample = {
        "buying": "Siêu rẻ",  # Không hợp lệ
        "maint": "Cao",
        "doors": "4 cửa",
        "persons": "4 người",
        "lug_boot": "Lớn",
        "safety": "Cao"
    }
    with pytest.raises(ValueError):
        convert_vi_input_to_uci(bad_sample)

def test_validate_dataframe_invalid_schema():
    """12. Dữ liệu không đúng schema hoặc thiếu cột."""
    bad_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    valid, errors = validate_dataframe(bad_df)
    assert valid is False
    assert len(errors) > 0

def test_train_test_split_no_leakage_and_stratified():
    """13. Tập train/test không bị trùng lặp chỉ số và giữ nguyên tổng số dòng."""
    toy_df = pd.DataFrame({
        "buying": ["low"] * 10 + ["high"] * 10,
        "maint": ["low"] * 10 + ["high"] * 10,
        "doors": ["2"] * 10 + ["4"] * 10,
        "persons": ["2"] * 10 + ["4"] * 10,
        "lug_boot": ["small"] * 10 + ["big"] * 10,
        "safety": ["low"] * 10 + ["high"] * 10,
        "class": ["unacc"] * 14 + ["acc"] * 6
    })
    
    train_df, test_df = split_data(toy_df, test_size=0.2, random_state=42, stratify=True)
    assert len(train_df) + len(test_df) == len(toy_df)
    assert len(train_df) == 16
    assert len(test_df) == 4
    
    # Kiểm tra không trùng lặp dòng nếu index được giữ
    # Phân tầng: tỷ lệ unacc trong train và test tương đương nhau
    train_unacc = (train_df["class"] == "unacc").sum() / len(train_df)
    test_unacc = (test_df["class"] == "unacc").sum() / len(test_df)
    assert abs(train_unacc - (14/20)) < 0.1

