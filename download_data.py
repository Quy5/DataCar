"""
Script tải và xác thực bộ dữ liệu Car Evaluation từ UCI Machine Learning Repository.
Lưu file vào: data/car.data
"""

import os
import sys
import urllib.request
import zipfile
import io

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE = os.path.join(DATA_DIR, "car.data")

# Các nguồn tải dữ liệu dự phòng từ UCI
URLS = [
    # Direct raw data from classic UCI URL
    "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
    # UCI static zip package
    "https://archive.ics.uci.edu/static/public/19/car+evaluation.zip",
    # Reliable mirror on github datasets
    "https://raw.githubusercontent.com/selva86/datasets/master/car.data",
]

EXPECTED_ROWS = 1728
EXPECTED_CLASS_COUNTS = {
    "unacc": 1210,
    "acc": 384,
    "good": 69,
    "vgood": 65
}

VALID_VALUES = {
    "buying": {"vhigh", "high", "med", "low"},
    "maint": {"vhigh", "high", "med", "low"},
    "doors": {"2", "3", "4", "5more"},
    "persons": {"2", "4", "more"},
    "lug_boot": {"small", "med", "big"},
    "safety": {"low", "med", "high"},
    "class": {"unacc", "acc", "good", "vgood"}
}

def validate_dataset(filepath):
    """Kiểm tra tính toàn vẹn của dataset."""
    if not os.path.exists(filepath):
        return False, f"File {filepath} không tồn tại."
    
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if len(lines) != EXPECTED_ROWS:
        return False, f"Số dòng ({len(lines)}) không khớp kỳ vọng ({EXPECTED_ROWS})."
    
    class_counts = {"unacc": 0, "acc": 0, "good": 0, "vgood": 0}
    cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            return False, f"Dòng {i+1} có {len(parts)} cột, kỳ vọng 7 cột."
        
        for col_name, val in zip(cols, parts):
            if val not in VALID_VALUES[col_name]:
                return False, f"Dòng {i+1}, cột {col_name} có giá trị không hợp lệ: '{val}'"
        
        target = parts[6]
        class_counts[target] += 1
        
    for cls, count in EXPECTED_CLASS_COUNTS.items():
        if class_counts.get(cls) != count:
            return False, f"Lớp '{cls}' có {class_counts.get(cls)} mẫu, kỳ vọng {count} mẫu."
            
    return True, class_counts

def download_data():
    """Tải dữ liệu từ UCI và lưu vào data/car.data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(DATA_FILE):
        valid, result = validate_dataset(DATA_FILE)
        if valid:
            print(f"[OK] File car.data đã tồn tại và hợp lệ: {result}")
            return True
        else:
            print(f"[WARN] File car.data hiện tại không hợp lệ ({result}), tiến hành tải lại...")
            
    print("Đang tải dữ liệu Car Evaluation từ UCI...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    downloaded = False
    for url in URLS:
        try:
            print(f"Thử tải từ: {url}")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read()
                
            # Nếu là file zip
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    for name in z.namelist():
                        if name.endswith("car.data"):
                            with open(DATA_FILE, "wb") as f_out:
                                f_out.write(z.read(name))
                            downloaded = True
                            break
            else:
                # File text car.data trực tiếp
                text = content.decode("utf-8", errors="ignore").strip()
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                if len(lines) == EXPECTED_ROWS:
                    with open(DATA_FILE, "w", encoding="utf-8") as f_out:
                        f_out.write("\n".join(lines) + "\n")
                    downloaded = True
            
            if downloaded:
                valid, result = validate_dataset(DATA_FILE)
                if valid:
                    print(f"[THÀNH CÔNG] Đã tải và kiểm định thành công: {result}")
                    return True
                else:
                    print(f"[LỖI XÁC THỰC] Dữ liệu tải về không đạt yêu cầu: {result}")
                    downloaded = False
        except Exception as e:
            print(f"[THẤT BẠI] Lỗi khi tải từ {url}: {e}")
            
    if not downloaded:
        print("[LỖI] Không thể tải bộ dữ liệu từ các nguồn tự động.")
        print("Vui lòng tải thủ công file 'car.data' từ https://archive.ics.uci.edu/dataset/19/car+evaluation")
        print(f"và lưu vào đường dẫn: {DATA_FILE}")
        return False

if __name__ == "__main__":
    success = download_data()
    sys.exit(0 if success else 1)

