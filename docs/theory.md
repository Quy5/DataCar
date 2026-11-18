# LÝ THUYẾT THUẬT TOÁN CÂY QUYẾT ĐỊNH ID3 (ITERATIVE DICHOTOMISER 3)

**Đồ án môn học:** Khai phá dữ liệu (Data Mining)  
**Nhóm thực hiện:** Nhóm 06 – Lớp QT01  
**Đề tài:** Phân loại mức độ phù hợp của ô tô bằng thuật toán ID3 (*Car Acceptability Classification Using the ID3 Decision Tree Algorithm*)  

---

## 1. TỔNG QUAN VỀ THUẬT TOÁN ID3

Thuật toán **ID3** (*Iterative Dichotomiser 3*) được phát minh bởi nhà khoa học máy tính **Ross Quinlan** vào năm 1986. Đây là một trong những thuật toán cơ bản và kinh điển nhất trong học máy (Machine Learning) và khai phá dữ liệu (Data Mining) dùng để xây dựng cây quyết định (Decision Tree) cho bài toán phân loại với các thuộc tính danh mục (categorical attributes).

ID3 tuân theo chiến lược học **top-down** kết hợp tìm kiếm **tham lam (greedy search)**:
- Bắt đầu từ nút gốc (root node) chứa toàn bộ tập dữ liệu huấn luyện.
- Tại mỗi bước, thuật toán đánh giá tất cả các thuộc tính chưa được sử dụng dựa trên thước đo **Information Gain (Độ lợi thông tin)**.
- Thuộc tính có Information Gain lớn nhất sẽ được chọn làm nút quyết định.
- Dữ liệu được chia thành các nhánh con tương ứng với từng giá trị danh mục của thuộc tính được chọn (multi-way split).
- Quá trình được lặp lại đệ quy cho từng nhánh con cho đến khi thỏa mãn các điều kiện dừng.

---

## 2. NỀN TẢNG TOÁN HỌC VÀ CÁC CÔNG THỨC CỐT LÕI

### 2.1. Entropy (Độ hỗn loạn thông tin)

Khái niệm **Entropy** xuất phát từ lý thuyết thông tin của Claude Shannon (1948), đo lường mức độ không chắc chắn hoặc độ hỗn loạn của một tập dữ liệu.

Cho tập dữ liệu mẫu $S$ gồm $|S|$ phần tử thuộc về $K$ lớp phân loại khác nhau ($c_1, c_2, \dots, c_K$). Gọi $p_i$ là tỷ lệ các mẫu trong $S$ thuộc về lớp $c_i$:

$$p_i = \frac{|S_i|}{|S|}$$

Công thức tính Entropy của tập $S$ là:

$$\text{Entropy}(S) = - \sum_{i=1}^{K} p_i \log_2(p_i)$$

**Tính chất quan trọng của Entropy:**
1. **Tập thuần nhất (Pure set):** Nếu tất cả các mẫu trong $S$ đều thuộc cùng một lớp duy nhất ($p_1 = 1$, $p_j = 0$ với mọi $j \neq 1$), thì $\text{Entropy}(S) = 0$. Mức độ hỗn loạn bằng 0, thông tin hoàn toàn chắc chắn.
2. **Tập phân bố đều tối đa (Maximal uncertainty):** Nếu các mẫu phân bố đồng đều giữa $K$ lớp ($p_i = \frac{1}{K}$ với mọi $i$), thì $\text{Entropy}(S) = \log_2(K)$. Ví dụ:
   - Với 2 lớp: $\text{Entropy}_{\max} = \log_2(2) = 1.0 \text{ bit}$.
   - Với 4 lớp: $\text{Entropy}_{\max} = \log_2(4) = 2.0 \text{ bits}$.
3. **Quy ước toán học:** Khi $p_i = 0$, ta quy ước $0 \times \log_2(0) = 0$ (xử lý không gây lỗi chia hoặc log cho 0).

---

### 2.2. Information Gain (Độ lợi thông tin)

**Information Gain** (độ lợi thông tin) của thuộc tính $A$ đo lường lượng thông tin kỳ vọng thu được (tức mức độ giảm Entropy) sau khi phân chia tập dữ liệu $S$ theo thuộc tính $A$.

Giả sử thuộc tính $A$ có $V$ giá trị danh mục rời rạc $\{v_1, v_2, \dots, v_V\}$. Khi phân chia $S$ theo $A$, ta thu được $V$ tập con $\{S_{v_1}, S_{v_2}, \dots, S_{v_V}\}$, trong đó $S_v$ là tập các mẫu trong $S$ có giá trị thuộc tính $A = v$.

Công thức tính Information Gain:

$$\text{Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \times \text{Entropy}(S_v)$$

Trong đó:
- $\text{Entropy}(S)$: Entropy ban đầu của tập dữ liệu tại nút hiện tại.
- $\sum_{v} \frac{|S_v|}{|S|} \text{Entropy}(S_v)$: Entropy kỳ vọng có trọng số sau khi phân chia dữ liệu theo các giá trị của thuộc tính $A$.
- $\text{Gain}(S, A)$ càng lớn thì việc phân chia dữ liệu theo thuộc tính $A$ càng giúp các tập con trở nên thuần nhất hơn.

### 2.3. Đối chiếu ký hiệu với bài giảng

Bài giảng "Phân lớp dữ liệu" dùng ký hiệu theo Han & Kamber; hai cách viết là tương đương:

| Tài liệu này / code | Bài giảng | Ý nghĩa |
|---|---|---|
| $\text{Entropy}(S)$ | $Info(D) = -\sum_{i=1}^{m} p_i \log_2(p_i)$ | Entropy của tập dữ liệu |
| $\sum_v \frac{\lvert S_v\rvert}{\lvert S\rvert}\text{Entropy}(S_v)$ | $Info_A(D) = \sum_{j=1}^{v} \frac{\lvert D_j\rvert}{\lvert D\rvert} Info(D_j)$ | Entropy sau khi chia theo A |
| $\text{Gain}(S, A)$ | $Gain(A) = Info(D) - Info_A(D)$ | Độ lợi thông tin |

Kiểm chứng với ví dụ AllElectronic trong bài giảng: $Info(D) = 0.940$, $Gain(age) = 0.246$, $Gain(income) = 0.029$, $Gain(student) = 0.151$, $Gain(credit\_rating) = 0.048$ → chọn `age` làm nút gốc. Test `test_lecture_example_allelectronic_gain` trong `tests/test_id3.py` kiểm tra đúng các số này.

---

### 2.4. Điều kiện dừng thuật toán (Stopping Criteria)

Một nút sẽ trở thành **nút lá (Leaf Node)** và dừng phân chia nếu rơi vào một trong các trường hợp sau:
1. **Tất cả các mẫu cùng một nhãn:** $\text{Entropy}(S) = 0$. Gán nhãn lá bằng nhãn thuần nhất đó.
2. **Không còn thuộc tính nào để phân chia:** Danh sách thuộc tính ứng viên rỗng. Gán nhãn lá bằng **nhãn đa số** (majority class) trong tập $S$.
3. **Information Gain không đáng kể:** $\max_{A} \text{Gain}(S, A) \le \epsilon$ (với $\epsilon = 10^{-12}$). Phân chia không làm giảm độ hỗn loạn, dừng và gán nhãn đa số.
4. **Nhánh con rỗng ($|S_v| = 0$):** Một giá trị danh mục $v$ của thuộc tính cha không có mẫu nào trong dữ liệu con. Gán nhãn lá bằng nhãn đa số của nút cha (khi hòa, ưu tiên theo thứ tự `unacc > acc > good > vgood`). Trên Car Evaluation, cây có 27 lá loại này; đây là nơi tập trung 12/19 lỗi trên tập test.

---

## 3. VÍ DỤ TÍNH TOÁN MINH HỌA TỪNG BƯỚC

Để hiểu rõ cơ chế toán học và đối chiếu trực tiếp với code trong `id3.py`, xét tập dữ liệu mẫu mini gồm 10 chiếc xe với 2 thuộc tính: `safety` (độ an toàn) và `persons` (sức chứa), dự đoán nhãn `class`:

| Mẫu ID | safety | persons | class |
|---|---|---|---|
| 1 | low | 2 | unacc |
| 2 | low | 4 | unacc |
| 3 | low | more | unacc |
| 4 | low | 2 | unacc |
| 5 | med | 2 | unacc |
| 6 | med | 4 | acc |
| 7 | med | more | acc |
| 8 | high | 2 | unacc |
| 9 | high | 4 | vgood |
| 10 | high | more | vgood |

Tổng số mẫu: $|S| = 10$.  
Phân bố nhãn trong $S$:
- `unacc`: 6 mẫu ($p = 6/10 = 0.6$)
- `acc`: 2 mẫu ($p = 2/10 = 0.2$)
- `vgood`: 2 mẫu ($p = 2/10 = 0.2$)

---

### Bước 1: Tính Entropy ban đầu của tập $S$

$$\text{Entropy}(S) = - [0.6 \log_2(0.6) + 0.2 \log_2(0.2) + 0.2 \log_2(0.2)]$$
- $0.6 \times \log_2(0.6) \approx 0.6 \times (-0.73697) = -0.44218$
- $0.2 \times \log_2(0.2) \approx 0.2 \times (-2.32193) = -0.46439$
- $0.2 \times \log_2(0.2) \approx 0.2 \times (-2.32193) = -0.46439$

$$\text{Entropy}(S) = - (-0.44218 - 0.46439 - 0.46439) = 1.37095 \text{ bits}$$

---

### Bước 2: Tính Information Gain cho thuộc tính `safety`

Thuộc tính `safety` có 3 giá trị: `low`, `med`, `high`.

1. **Nhánh `safety = low`:** 4 mẫu (Mẫu 1, 2, 3, 4).
   - Nhãn: 4 `unacc`.
   - $\text{Entropy}(S_{\text{low}}) = 0.0$ (thuần nhất tuyệt đối).

2. **Nhánh `safety = med`:** 3 mẫu (Mẫu 5, 6, 7).
   - Nhãn: 1 `unacc` ($p = 1/3$), 2 `acc` ($p = 2/3$).
   - $\text{Entropy}(S_{\text{med}}) = - [\frac{1}{3} \log_2(\frac{1}{3}) + \frac{2}{3} \log_2(\frac{2}{3})] \approx 0.91830 \text{ bits}$.

3. **Nhánh `safety = high`:** 3 mẫu (Mẫu 8, 9, 10).
   - Nhãn: 1 `unacc` ($p = 1/3$), 2 `vgood` ($p = 2/3$).
   - $\text{Entropy}(S_{\text{high}}) = - [\frac{1}{3} \log_2(\frac{1}{3}) + \frac{2}{3} \log_2(\frac{2}{3})] \approx 0.91830 \text{ bits}$.

**Entropy có trọng số sau khi chia theo `safety`:**

$$\text{Entropy}(S, \text{safety}) = \frac{4}{10} \times 0.0 + \frac{3}{10} \times 0.91830 + \frac{3}{10} \times 0.91830 = 0.55098 \text{ bits}$$

**Information Gain của `safety`:**

$$\text{Gain}(S, \text{safety}) = 1.37095 - 0.55098 = 0.81997 \text{ bits}$$

---

### Bước 3: Tính Information Gain cho thuộc tính `persons`

Thuộc tính `persons` có 3 giá trị: `2`, `4`, `more`.

1. **Nhánh `persons = 2`:** 4 mẫu (Mẫu 1, 4, 5, 8).
   - Nhãn: 4 `unacc`.
   - $\text{Entropy}(S_{2}) = 0.0$.

2. **Nhánh `persons = 4`:** 3 mẫu (Mẫu 2, 6, 9).
   - Nhãn: 1 `unacc`, 1 `acc`, 1 `vgood` ($p_i = 1/3$).
   - $\text{Entropy}(S_{4}) = \log_2(3) \approx 1.58496 \text{ bits}$.

3. **Nhánh `persons = more`:** 3 mẫu (Mẫu 3, 7, 10).
   - Nhãn: 1 `unacc`, 1 `acc`, 1 `vgood` ($p_i = 1/3$).
   - $\text{Entropy}(S_{\text{more}}) = \log_2(3) \approx 1.58496 \text{ bits}$.

**Entropy có trọng số sau khi chia theo `persons`:**

$$\text{Entropy}(S, \text{persons}) = \frac{4}{10} \times 0.0 + \frac{3}{10} \times 1.58496 + \frac{3}{10} \times 1.58496 = 0.95098 \text{ bits}$$

**Information Gain của `persons`:**

$$\text{Gain}(S, \text{persons}) = 1.37095 - 0.95098 = 0.41997 \text{ bits}$$

---

### Bước 4: Lựa chọn nút gốc và cấu trúc cây con

So sánh Information Gain:
- $\text{Gain}(S, \text{safety}) = 0.81997$
- $\text{Gain}(S, \text{persons}) = 0.41997$

Do $\text{Gain}(S, \text{safety}) > \text{Gain}(S, \text{persons})$, thuật toán ID3 **chọn thuộc tính `safety` làm nút gốc**.

**Cây được tạo ra:**
- Nhánh `safety = low` $\rightarrow$ Nút lá: `unacc`.
- Nhánh `safety = med` $\rightarrow$ Tiếp tục chia theo `persons`:
  - `persons = 2` $\rightarrow$ Nút lá: `unacc`.
  - `persons = 4` $\rightarrow$ Nút lá: `acc`.
  - `persons = more` $\rightarrow$ Nút lá: `acc`.
- Nhánh `safety = high` $\rightarrow$ Tiếp tục chia theo `persons`:
  - `persons = 2` $\rightarrow$ Nút lá: `unacc`.
  - `persons = 4` $\rightarrow$ Nút lá: `vgood`.
  - `persons = more` $\rightarrow$ Nút lá: `vgood`.

---

### Bước 5: Ví dụ dự đoán bản ghi mới và đường đi quyết định

*Ghi chú:* `id3.py` đánh số nút theo thứ tự duyệt sâu (DFS), các nhánh con được xét theo thứ tự chữ cái của giá trị (`high`, `low`, `med`; `2`, `4`, `more`). Vì vậy nhánh `safety = high` là Nút ID 1 và lá `persons = 4` bên dưới là Nút ID 3.

Giả sử ta cần phân loại mẫu xe mới:  
$\mathbf{x} = \{\text{safety: "high"}, \text{persons: "4"}\}$

1. **Tại Nút Gốc (ID 0):**
   - Thuộc tính kiểm tra: `safety`
   - Giá trị của xe: `"high"`
   - Thuộc tính tiếp theo: đi theo nhánh `"high"` tới Nút ID 1.
2. **Tại Nút ID 1:**
   - Thuộc tính kiểm tra: `persons`
   - Giá trị của xe: `"4"`
   - Thuộc tính tiếp theo: đi theo nhánh `"4"` tới Nút ID 3.
3. **Tại Nút ID 3 (Nút lá):**
   - Nhãn dự đoán: `vgood` (Rất tốt).
   - Giải thích: "Vì độ an toàn cao (safety = high) và sức chứa 4 người (persons = 4), mẫu xe đạt phân loại Rất tốt (vgood)."

---

## 4. ĐỐI CHIẾU VỚI CÁC THUẬT TOÁN KHÁC

| Tiêu chí | ID3 | C4.5 | CART |
|---|---|---|---|
| **Tác giả** | Ross Quinlan (1986) | Ross Quinlan (1993) | Breiman et al. (1984) |
| **Tiêu chuẩn phân tách** | Information Gain | Gain Ratio | Gini Impurity |
| **Kiểu phân chia** | Đa nhánh (Multi-way) | Đa nhánh (Multi-way) | Nhị phân (Binary split) |
| **Dữ liệu số liên tục** | Không hỗ trợ trực tiếp | Có hỗ trợ chia ngưỡng | Có hỗ trợ chia ngưỡng |
| **Xử lý giá trị thiếu** | Không hỗ trợ nguyên bản | Có xử lý xác suất | Có xử lý nhánh thay thế |
| **Cắt tỉa cây (Pruning)** | Không cắt tỉa (dễ quá khớp) | Cắt tỉa sau (Post-pruning) | Cost-complexity pruning |

---

## 5. TÀI LIỆU THAM KHẢO

1. Quinlan, J. R. (1986). *Induction of decision trees*. Machine Learning, 1(1), 81–106.
2. Shannon, C. E. (1948). *A mathematical theory of communication*. Bell System Technical Journal, 27(3), 379–423.
3. Bohanec, M., & Rajkovič, V. (1990). *Knowledge acquisition and explanation for multi-attribute decision making*. In 8th Intl Workshop on Expert Systems and their Applications (pp. 59–78). Avignon, France.
4. UCI Machine Learning Repository. *Car Evaluation Dataset*. https://archive.ics.uci.edu/dataset/19/car+evaluation

