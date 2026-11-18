# ĐỒ ÁN KHAI PHÁ DỮ LIỆU: PHÂN LOẠI MỨC ĐỘ PHÙ HỢP CỦA Ô TÔ BẰNG THUẬT TOÁN ID3

> **Nhóm thực hiện:** Nhóm 06 – Lớp QT01  
> **Tên đề tài:** Phân loại mức độ phù hợp của ô tô bằng thuật toán ID3  
> **Tên tiếng Anh:** *Car Acceptability Classification Using the ID3 Decision Tree Algorithm*  
> **Thuật toán chính:** ID3 (Iterative Dichotomiser 3) tự cài đặt 100% bằng Python  
> **Giao diện:** Ứng dụng web Streamlit gồm 4 tab: Dự đoán, Cây ID3, Đánh giá, Dữ liệu  
> **Bộ dữ liệu:** UCI Machine Learning Repository - Car Evaluation Dataset (1.728 mẫu)  

---

## 1. GIỚI THIỆU ĐỀ TÀI & MỤC TIÊU

### 1.1. Giới thiệu đề tài
Trong quá trình lựa chọn và đánh giá chất lượng ô tô, người tiêu dùng và chuyên gia đánh giá phương tiện phải đối mặt với nhiều tiêu chí định tính và định lượng phức tạp: từ chi phí ban đầu, chi phí bảo dưỡng định kỳ, kết cấu thân vỏ, sức chứa đến mức độ an toàn.

Đề tài ứng dụng thuật toán **cây quyết định ID3 (Iterative Dichotomiser 3)** nhằm xây dựng một mô hình học máy có khả năng tự động phân loại mức độ phù hợp của ô tô dựa trên 6 đặc trưng kỹ thuật, đồng thời cung cấp khả năng **giải thích tường minh (explainable AI)** từng bước suy diễn trên cây quyết định.

### 1.2. Mục tiêu nghiên cứu
1. **Tự cài đặt trọn vẹn thuật toán ID3** bằng ngôn ngữ Python thuần, tuân thủ nguyên lý toán học của Ross Quinlan (Entropy, Information Gain, Multi-way split), **không sử dụng** `sklearn.tree.DecisionTreeClassifier`.
2. **Khai phá và tiền xử lý** bộ dữ liệu chuẩn **Car Evaluation** từ UCI Machine Learning Repository (1.728 bản ghi, 6 thuộc tính danh mục, 4 nhãn phân loại).
3. **Huấn luyện và đánh giá mô hình khoa học** trên tập kiểm thử độc lập (Train 80% / Test 20%, Stratified), bổ sung đánh giá chéo 10-fold phân tầng, so sánh với mô hình Baseline đối chứng.
4. **Xây dựng ứng dụng web (Streamlit)** cho phép chọn thông số xe và xem ngay nhãn dự đoán cùng đường đi trên cây; trực quan hóa cây, tập luật, kết quả đánh giá và dữ liệu.

---

## 2. BỘ DỮ LIỆU CAR EVALUATION (UCI)

- **Nguồn chính thức:** [UCI Machine Learning Repository - Car Evaluation](https://archive.ics.uci.edu/dataset/19/car+evaluation)
- **Tổng số mẫu:** 1.728 bản ghi
- **Số thuộc tính đầu vào:** 6 thuộc tính danh mục
- **Giá trị khuyết thiếu (Missing):** 0 (Bộ dữ liệu chuẩn không có giá trị khuyết)
- **Tập tin lưu trữ:** `data/car.data`

### 2.1. 6 thuộc tính đầu vào

| STT | Thuộc tính | Tên tiếng Việt | Các giá trị hợp lệ (UCI) | Giá trị hiển thị tiếng Việt |
|---|---|---|---|---|
| 1 | `buying` | Giá mua xe | `vhigh`, `high`, `med`, `low` | Rất cao, Cao, Trung bình, Thấp |
| 2 | `maint` | Chi phí bảo dưỡng | `vhigh`, `high`, `med`, `low` | Rất cao, Cao, Trung bình, Thấp |
| 3 | `doors` | Số cửa xe | `2`, `3`, `4`, `5more` | 2 cửa, 3 cửa, 4 cửa, Từ 5 cửa trở lên |
| 4 | `persons` | Sức chứa (người) | `2`, `4`, `more` | 2 người, 4 người, Trên 4 người |
| 5 | `lug_boot` | Khoang hành lý | `small`, `med`, `big` | Nhỏ, Trung bình, Lớn |
| 6 | `safety` | Mức độ an toàn | `low`, `med`, `high` | Thấp, Trung bình, Cao |

### 2.2. Nhãn phân loại (Target Class)

| Nhãn (UCI) | Ý nghĩa tiếng Việt | Số lượng mẫu | Tỷ lệ (%) |
|---|---|---|---|
| `unacc` | Không phù hợp (Unacceptable) | 1.210 | 70.02% |
| `acc` | Chấp nhận được (Acceptable) | 384 | 22.22% |
| `good` | Tốt (Good) | 69 | 3.99% |
| `vgood` | Rất tốt (Very Good) | 65 | 3.76% |

---

## 3. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
KPDL_Nhom06_ID3_Car/
│
├── app.py                  # Ứng dụng web Streamlit (4 tab: Dự đoán, Cây ID3, Đánh giá, Dữ liệu)
├── id3.py                  # Thuật toán ID3 tự cài đặt (Entropy, Gain, Tree, Predict)
├── data_utils.py           # Tiền xử lý, kiểm định dữ liệu, ánh xạ tiếng Việt
├── train.py                # Pipeline huấn luyện, đối chứng baseline và xuất kết quả
├── evaluation.py           # Tính toán các chỉ số đánh giá (Accuracy, Macro-F1, CM)
├── visualization.py        # Vẽ biểu đồ, xuất sơ đồ Graphviz DOT và Text Hierarchy
├── download_data.py        # Tự động tải và kiểm định tính toàn vẹn của dataset
│
├── data/
│   └── car.data            # File dữ liệu chuẩn 1.728 dòng từ UCI
│
├── output/                 # Thư mục chứa kết quả thực nghiệm và biểu đồ
│   ├── model.json          # Mô hình cây ID3 được serialize
│   ├── results.json        # Chỉ số đánh giá và bảng đối chứng
│   ├── predictions.csv     # Kết quả dự đoán chi tiết trên tập test
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   ├── root_information_gain.png
│   ├── id3_tree.dot        # Toàn bộ cây quyết định ID3
│   ├── id3_tree_preview.dot# Cây quyết định rút gọn (preview)
│   └── id3_tree_hierarchy.txt # Sơ đồ cây dạng văn bản phân cấp
│
├── tests/                  # Bộ bài kiểm thử đơn vị tự động (Unit Tests)
│   ├── test_id3.py         # Kiểm thử Entropy, Gain, dừng đệ quy, đa nhánh
│   ├── test_data.py        # Kiểm thử tiền xử lý, ánh xạ và chia tập stratified
│   └── test_prediction.py  # Kiểm thử suy diễn, fallback và các metrics
│
├── docs/
│   └── theory.md           # Tài liệu lý thuyết chuyên sâu và ví dụ tính tay
│
├── requirements.txt        # Danh sách thư viện phụ thuộc
├── README.md               # Tài liệu hướng dẫn sử dụng và báo cáo đồ án
└── .gitignore              # Cấu hình bỏ qua file tạm và môi trường ảo
```

---

## 4. NỀN TẢNG THUẬT TOÁN ID3 TỰ CÀI ĐẶT

### 4.1. Công thức Entropy
$$\text{Entropy}(S) = - \sum_{i=1}^{K} p_i \log_2(p_i)$$
- Đo lường mức độ hỗn loạn của các nhãn trong tập mẫu $S$.
- Nếu tập thuần nhất: $\text{Entropy}(S) = 0$.
- Xử lý mượt mà trường hợp $p_i = 0$ (quy ước $0 \log_2(0) = 0$).

### 4.2. Công thức Information Gain
$$\text{Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \times \text{Entropy}(S_v)$$
- Đo lường lượng thông tin thu được khi phân chia tập $S$ theo thuộc tính $A$.
- Tại mỗi bước đệ quy, thuật toán chọn thuộc tính có $\text{Gain}$ lớn nhất làm nút quyết định.

### 4.3. Các cơ chế xử lý đặc biệt
- **Tie-Breaking Deterministic:** Sử dụng thứ tự thuộc tính cố định (`buying` $\rightarrow$ `maint` $\rightarrow$ `doors` $\rightarrow$ `persons` $\rightarrow$ `lug_boot` $\rightarrow$ `safety`) và thứ tự nhãn chuẩn để đảm bảo kết quả có tính tái lập 100%.
- **Điều kiện dừng chặt chẽ:** Tập thuần nhất, hết thuộc tính, $\text{Gain} \le 10^{-12}$, hoặc nhánh con rỗng.
- **Cơ chế Fallback thông minh:** Khi gặp nhánh rỗng hoặc giá trị chưa từng quan sát, mô hình sử dụng nhãn đa số được lưu tại nút cha và đánh dấu cờ cảnh báo rõ ràng.

---

## 5. HƯỚNG DẪN CÀI ĐẶT VÀ SỬ DỤNG

### 5.1. Yêu cầu môi trường
- Python 3.10 hoặc 3.11.
- Môi trường Windows, Linux hoặc macOS.

### 5.2. Các bước cài đặt
1. **Mở terminal tại thư mục dự án:**
   ```bash
   cd d:\KPDL\APP
   ```

2. **Kích hoạt môi trường ảo (nếu có) hoặc cài đặt thư viện:**
   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Tải và kiểm định dữ liệu từ UCI:**
   ```bash
   python download_data.py
   ```

4. **Chạy kiểm thử đơn vị (Unit Tests):**
   ```bash
   python -m pytest tests/ -v
   ```

5. **Huấn luyện mô hình và xuất biểu đồ:**
   ```bash
   python train.py
   ```

6. **Khởi chạy ứng dụng web Streamlit:**
   ```bash
   python -m streamlit run app.py
   ```
   Trình duyệt sẽ tự động mở tại địa chỉ: `http://localhost:8501`

---

## 6. KỊCH BẢN DEMO BÁO CÁO TRƯỚC GIẢNG VIÊN (5 - 7 PHÚT)

Ứng dụng có 4 tab, mở vào là tab **Dự đoán**:

- **Bước 1 (1,5 phút) – Tab Dự đoán:** bấm ví dụ "Xe không phù hợp": cây dừng ngay ở bước "Mức độ an toàn: Thấp" (465 mẫu học). Bấm "Xe rất tốt": cây đi qua 5 thuộc tính rồi kết luận vgood. Đổi thử một thông số để thấy kết quả cập nhật ngay; dòng cuối cho biết nhãn gốc trong bộ dữ liệu để so sánh.
- **Bước 2 (1 phút) – Ca kém tin cậy:** chọn Giá mua Rất cao, Bảo dưỡng Thấp, 3 cửa, Trên 4 người, Cốp Trung bình, An toàn Trung bình. Ứng dụng cảnh báo vì tập huấn luyện không có xe nào như vậy (lá không có mẫu học).
- **Bước 3 (1,5 phút) – Tab Cây ID3:** biểu đồ Information Gain cho thấy `safety` lớn nhất nên làm nút gốc; kéo thanh "Số tầng hiển thị" lên 2–3 để xem cây; lọc luật IF–THEN theo kết luận.
- **Bước 4 (1,5 phút) – Tab Đánh giá:** Accuracy 94,51% và Macro-F1 0,8975 so với mô hình luôn đoán unacc; 10-fold cho 92,94% ± 1,53%; ma trận nhầm lẫn; 12/19 lỗi rơi vào lá không có mẫu học.
- **Bước 5 (30 giây) – Tab Dữ liệu:** 1.728 cấu hình, unacc chiếm 70% nên phải xem Macro-F1.

---

## 7. KẾT QUẢ THỰC NGHIỆM THỰC TẾ

*(Các số liệu được tính toán tự động và đồng bộ từ pipeline thực thi `train.py` trên tập test độc lập 20% - 346 mẫu)*

| Tiêu chí đánh giá | Baseline (Majority Class) | Cây quyết định ID3 (Nhóm 06) | Ý nghĩa đánh giá |
|---|---|---|---|
| **Độ chính xác (Accuracy)** | 69.94% | **94.51%** | ID3 nâng cao độ chính xác thêm **+24.57%** |
| **Macro-F1 Score** | 0.2058 | **0.8975** | Đánh giá toàn diện trên cả 4 lớp mất cân bằng (**+0.6917**) |
| **Macro-Recall** | 0.2500 | **0.8890** | Khả năng phát hiện mẫu đúng trên từng lớp (**+0.6390**) |
| **Macro-Precision** | 0.1749 | **0.9094** | Độ chuẩn xác trung bình trên 4 lớp (**+0.7345**) |
| **Thuộc tính Nút Gốc** | Không áp dụng | `safety` (Gain: 0.26216) | Mức độ an toàn là yếu tố quyết định cao nhất |
| **Tổng số nút (Nodes)** | 1 | 384 nút | Cây đa nhánh phân tách hoàn chỉnh |
| **Số nút lá (Leaves)** | 1 | 279 lá | Gồm 252 lá có mẫu (= **252 luật IF-THEN**) và 27 lá của nhánh rỗng (0 mẫu huấn luyện) |
| **Độ sâu lớn nhất (Depth)**| 0 | 6 tầng | Tối đa 6 thuộc tính trên một nhánh suy diễn |

### Chi tiết hiệu năng từng lớp của ID3 trên tập Test (346 mẫu):
- **Lớp `unacc` (242 mẫu test):** Precision = 96.72%, Recall = 97.52%, F1-Score = **97.12%**
- **Lớp `acc` (77 mẫu test):** Precision = 89.47%, Recall = 88.31%, F1-Score = **88.89%**
- **Lớp `good` (14 mẫu test):** Precision = 86.67%, Recall = 92.86%, F1-Score = **89.66%**
- **Lớp `vgood` (13 mẫu test):** Precision = 90.91%, Recall = 76.92%, F1-Score = **83.33%**

### Đánh giá chéo 10-fold phân tầng (toàn bộ 1.728 mẫu)
- Accuracy = **0.9294 ± 0.0153**, Macro-F1 = **0.8455 ± 0.0550** (trung bình ± độ lệch chuẩn của 10 fold).
- Kết quả Hold-out (0.9451) nằm trong khoảng dao động của 10-fold, cho thấy con số trên tập test không phải do may mắn khi chia dữ liệu.

### Phân tích lỗi: lá của nhánh rỗng
- Cây có 27 lá không có mẫu huấn luyện (tổ hợp thuộc tính không xuất hiện trong tập train); nhãn của các lá này lấy theo **nhãn đa số của nút cha** (khi hòa thì ưu tiên theo thứ tự `unacc > acc > good > vgood`).
- Trên tập test có **27 mẫu** rơi vào các lá này, chỉ đúng **15/27 (55,6%)**, trong khi các mẫu còn lại đúng **312/319 (97,8%)**.
- Như vậy **12/19 lỗi** của mô hình đến từ nhánh rỗng – hạn chế điển hình của ID3 khi dữ liệu học không phủ hết các tổ hợp. Giao diện web cảnh báo khi một dự đoán rơi vào lá loại này.

---

## 8. HẠN CHẾ CỦA MÔ HÌNH & HƯỚNG PHÁT TRIỂN

### Hạn chế
1. **Thiên vị thuộc tính nhiều giá trị:** Thuật toán ID3 có xu hướng ưu tiên các thuộc tính có nhiều giá trị danh mục hơn do Entropy giảm nhanh.
2. **Hiện tượng quá khớp (Overfitting):** Vì ID3 nguyên bản không tích hợp cơ chế cắt tỉa (pruning), cây phát triển sâu tối đa để khớp toàn bộ tập huấn luyện, có thể làm giảm khả năng khái quát hóa trên dữ liệu ngoại lai.
3. **Chỉ hỗ trợ dữ liệu danh mục:** Không hỗ trợ trực tiếp các thuộc tính số liên tục nếu chưa qua bước rời rạc hóa (discretization).

### Hướng phát triển
1. Nâng cấp lên thuật toán **C4.5** sử dụng **Gain Ratio** và tích hợp cắt tỉa cây sau huấn luyện (Post-pruning).
2. Kết hợp kỹ thuật Ensemble Learning (**Random Forest / Gradient Boosting**) để nâng cao độ bền vững.

---

## 9. TÀI LIỆU THAM KHẢO

1. Quinlan, J. R. (1986). *Induction of decision trees*. Machine Learning, 1(1), 81–106.
2. Shannon, C. E. (1948). *A mathematical theory of communication*. Bell System Technical Journal, 27(3), 379–423.
3. Bohanec, M., & Rajkovič, V. (1990). *Knowledge acquisition and explanation for multi-attribute decision making*. In 8th Intl Workshop on Expert Systems and their Applications (pp. 59–78). Avignon, France.
4. UCI Machine Learning Repository. *Car Evaluation Dataset*. https://archive.ics.uci.edu/dataset/19/car+evaluation

