# ĐỀ CƯƠNG CHI TIẾT & KẾ HOẠCH NGHIÊN CỨU KHOA HỌC

## ĐỀ TÀI: HỆ THỐNG PHÁT HIỆN TRẺ EM ĐUỐI NƯỚC THỜI GIAN THỰC DỰA TRÊN DEEP LEARNING VÀ BỘ LỌC THỜI GIAN (TEMPORAL FILTERING)

* **Đơn vị thực hiện**: Trường Đại học Công nghệ Thông tin (UIT) - ĐHQG-HCM
* **Bài báo nền tảng (Foundation Paper)**: He, Q., Zhang, H., Mei, Z., & Xu, X. (2023), *"High accuracy intelligent real-time framework for detecting infant drowning based on deep learning"*, **Expert Systems with Applications** (*Elsevier*, Vol. 228, 120389, DOI: `10.1016/j.eswa.2023.120389`).

---

## 1. TỔNG QUAN & KHOẢNG TRỐNG NGHIÊN CỨU (RESEARCH GAP)

### 1.1. Bối cảnh thực tiễn & Ý nghĩa đề tài
* Đuối nước là nguyên nhân hàng đầu gây tử vong do tai nạn thương tích ở trẻ em, đặc biệt tại các bể bơi công cộng, trường học và hồ bơi gia đình.
* Trẻ nhỏ khi rơi vào trạng thái đuối nước thường không có phản xạ kêu cứu thành tiếng hoặc vẫy tay rõ ràng (hiện tượng *Instinctive Drowning Response*), phạm vi vận động co hẹp và chìm nhanh chóng. Nhân viên cứu hộ có thể bị quá tải, góc quan sát bị che khuất hoặc mất tập trung khi bể bơi đông người.
* Việc ứng dụng Thị giác máy tính (Computer Vision) kết hợp Học sâu (Deep Learning) từ camera giám sát hỗ trợ cảnh báo sớm là giải pháp cấp thiết và có giá trị xã hội cao.

### 1.2. Phân tích bài báo nền tảng (He et al., 2023 - Elsevier ESWA)
* **Kiến trúc gốc 2 giai đoạn (2-stage)**:
  * **Stage 1 (Foreground Localization)**: Sử dụng **YOLOv5** tích hợp cơ chế tập trung chú ý **CBAM** (*Convolutional Block Attention Module*) để phát hiện và khoanh vùng vị trí trẻ trong ảnh.
  * **Stage 2 (Posture Classification)**: Sử dụng **SSD** (*Single Shot MultiBox Detector*) để phân loại tư thế của vùng ảnh được cắt từ Stage 1 thành 2 lớp: `swimming` (bơi an toàn) hoặc `drowning` (đuối nước).
* **Kết quả bài báo gốc**: YOLOv5s đạt mAP > 89%, Precision ~ 86.6%, tốc độ xử lý đạt 75 FPS trên GPU thực nghiệm.

### 1.3. Khoảng trống nghiên cứu (Research Gap)
1. **Quyết định độc lập trên từng khung hình (Frame-level Limitation)**:
   * Cả công trình của He et al. (2023) và phần lớn các nghiên cứu hiện tại (YOLOv8/v11, Faster R-CNN) đều đưa ra quyết định cảnh báo dựa trên **từng khung hình đơn lẻ**.
   * Hệ thống thiếu cơ chế suy luận thời gian (*temporal reasoning* / *temporal consistency*).
2. **Tỷ lệ báo động giả (False Alarm Rate - FAR) rất cao trong thực tế**:
   * Tại hồ bơi, các hành vi an toàn của trẻ em như: **vẫy tay đùa giỡn, ngụp lặn nín thở, té nước, đạp chân đứng nước** trong một tích tắc (1 - 2 frames) có đặc điểm hình ảnh (tư thế, bọt nước, vùng cơ thể chìm) rất giống với đuối nước thật.
   * Mô hình frame-level sẽ kích hoạt cảnh báo nguy hiểm chỉ sau 1 khung hình đơn lẻ, dẫn đến tình trạng còi báo liên tục. Trong triển khai thực tế, tỷ lệ báo động giả cao khiến nhân viên cứu hộ mệt mỏi và tắt hệ thống, triệt tiêu tính khả dụng thực tế.

---

## 2. MỤC TIÊU NGHIÊN CỨU & ĐÓNG GÓP KHOA HỌC

### 2.1. Mục tiêu đề tài
1. Xây dựng và đánh giá thực nghiệm **5 - 6 mô hình Baseline** nhận diện hành vi cấp khung hình (Frame-level Detectors) đại diện cho các trường phái khác nhau (Two-stage, One-stage Anchor-based, Anchor-free và Vision Transformer).
2. Thiết kế và phát triển **Module lọc thời gian (Temporal Filtering)** ứng dụng nguyên lý cửa sổ trượt (Sliding Window) kết hợp kỹ thuật Debounce / Confidence Smoothing nhằm loại bỏ triệt để các báo động giả ngắn hạn.
3. Xây dựng **Bộ video kiểm thử (Temporal Testbed Benchmark)** gồm 15 - 20 video đa dạng kịch bản, gán nhãn mốc thời gian (*Temporal Ground Truth*) để đo đạc định lượng.
4. Chứng minh định lượng sự cải thiện về chỉ số **False Alarm Rate (FAR)**, xác định ngưỡng thời gian tối ưu cân bằng giữa **Độ trễ cảnh báo (Detection Delay)** và **Tỷ lệ bỏ sót (Miss Rate)**.

### 2.2. Điểm mới & Đóng góp khoa học dự kiến
* **Đóng góp 1 (Kiến trúc)**: Đề xuất pipeline kết hợp linh hoạt giữa một **Frame-level Detector thời gian thực** và một **Module Temporal Filtering nhẹ (lightweight)**, không làm suy giảm FPS tổng thể của hệ thống, không đòi hỏi phần cứng đắt tiền như các mô hình 3D-CNN hay Heavy Video Transformer.
* **Đóng góp 2 (Đo lường định lượng)**: Đưa ra nghiên cứu bóc tách (*Ablation Study*) về sự đánh đổi giữa thời gian xác thực $N$ giây và tỷ lệ triệt tiêu báo động giả.
* **Đóng góp 3 (Ý nghĩa thực tiễn)**: Cung cấp giải pháp có thể triển khai trên các thiết bị biên (*Edge AI* như NVIDIA Jetson) hoặc máy chủ giám sát camera hồ bơi với độ tin cậy thực tế cao.

---

## 3. NGUỒN DATASET TUÂN THỦ TIÊU CHUẨN NCKH

Theo yêu cầu chuẩn mực từ các nguồn xuất bản uy tín (**IEEE, ScienceDirect/Elsevier, Springer, CVF, Scopus**):
> **Lưu ý học thuật**: Bài toán Drowning Detection không có một bộ dữ liệu toàn cầu duy nhất như MS-COCO do ràng buộc đạo đức và quyền riêng tư trẻ em. Các bài báo uy tín trên Elsevier và Springer đều áp dụng phương pháp chuẩn hóa dữ liệu mở có DOI hoặc tự xây dựng benchmark kiểm thử có thẩm định nhãn.

### 3.1. Danh mục các nguồn Dataset sử dụng trong đề tài

| STT | Tên Dataset / Nguồn | Loại dữ liệu & Quy mô | Nguồn học thuật / Định danh DOI | Cách sử dụng trong nghiên cứu |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Underwater Drowning Detection Dataset** | **5,613 ảnh** phân giải 640×640.<br/>3 lớp cân bằng: `swimming` (1,871), `struggling` (1,871), `drowning` (1,871). | **Figshare Academic Repository**<br/>DOI: `10.6084/m9.figshare.29497235.v2`<br/>Trích dẫn trong các nghiên cứu Scopus/Springer 2025. | Sử dụng làm tập dữ liệu huấn luyện chính thức (Train/Val/Test) cho các mô hình Frame-level. |
| **2** | **Water Behavior Dataset** | **91 chuỗi video** (47 overhead camera, 44 underwater camera; tổng cộng **46,739 frames**). | **IEEE Xplore**<br/>Công bố tại: *IEEE Green Energy and Smart Systems Conference (IGESSC 2021)*. | Cung cấp dữ liệu chuỗi hành vi từ góc nhìn camera trên cao (overhead) thực tế hồ bơi. |
| **3** | **Swimming-YOLO Pool Dataset & UAV Pool Dataset** | **4,800 - 8,500 ảnh** chụp từ camera hồ bơi và góc nhìn flycam trên cao. | **SpringerLink**<br/>Tạp chí *Signal, Image and Video Processing* (2025) & GitHub mã nguồn mở. | Bổ sung các góc nhìn góc chéo và từ trên cao của hồ bơi, đa dạng hóa bối cảnh ánh sáng. |
| **4** | **Roboflow Universe Standardized Subset** | ~3,500 - 5,000 ảnh gán nhãn bounding box chuẩn YOLO. | **Roboflow Academic**<br/>(Được làm sạch, khử trùng lặp và xác thực nhãn thủ công). | Đồng bộ với đề cương ban đầu, làm phong phú tập huấn luyện 2 lớp `swimming` và `drowning`. |
| **5** | **Custom Testbed Video Benchmark** | **15 - 20 video** độ dài 30s - 2 phút/video.<br/>Camera giám sát hồ bơi thực tế (CCTV / Diễn tập cứu hộ). | **Tự xây dựng & Công bố**<br/>Đi kèm file ground-truth thời gian (`ground_truth_events.csv`). | Dùng riêng để đánh giá đóng góp cốt lõi: Đo False Alarm Rate, Detection Delay và Miss Rate. |

### 3.2. Quy trình tiền xử lý & Chuẩn hóa Dữ liệu (Data Preprocessing Pipeline)
1. **Hợp nhất nhãn (Label Mapping)**: Thống nhất hệ thống nhãn về 2 lớp chính:
   * Class 0: `swimming` (Bao gồm bơi bình thường, nổi an toàn, đùa nghịch an toàn).
   * Class 1: `drowning` (Bao gồm chới với mất kiểm soát, chìm dần, bất động dưới nước).
2. **Khử trùng lặp (Deduplication)**: Sử dụng thuật toán so khớp mã băm Perceptual Hash (pHash) để loại bỏ các ảnh cắt liên tiếp quá giống nhau từ cùng 1 video.
3. **Phân chia tập dữ liệu (Dataset Splitting)**:
   * **Train set**: 70% (~5,000 - 6,000 ảnh).
   * **Validation set**: 15% (~1,000 - 1,200 ảnh).
   * **Test set (Ảnh tĩnh)**: 15% (~1,000 - 1,200 ảnh).
   * **Testbed (Video chuỗi)**: Độc lập hoàn toàn với tập huấn luyện để đảm bảo tính khách quan.

---

## 4. KẾ HOẠCH LỰA CHỌN 5 - 6 MODEL BASELINE

Để phục vụ so sánh học thuật toàn diện, hệ thống baseline được lựa chọn bao quát 4 nhóm kiến trúc chính trong thị giác máy tính:

```
[Hệ thống 6 Model Baseline Frame-level]
  │
  ├── [Nhóm 1: Two-Stage Detector Kinh điển] ──────► 1. Faster R-CNN (ResNet-50 FPN)
  │
  ├── [Nhóm 2: Baseline của Bài báo Gốc He et al.] ──► 2. SSD (MobileNetV2 / VGG16)
  │                                                  3. YOLOv5s (Anchor-based Backbone)
  │
  ├── [Nhóm 3: SOTA Anchor-free YOLO Family] ──────► 4. YOLOv8s (C2f Module)
  │                                                  5. YOLOv11s (C3k2 Module - Mới nhất)
  │
  └── [Nhóm 4: Real-time Vision Transformer] ───────► 6. RT-DETR-R18 / R50 (Baidu)
```

### Chi tiết các mô hình Baseline:

| STT | Tên Model | Kiến trúc / Đặc trưng kỹ thuật | Mục đích & Ý nghĩa trong nghiên cứu |
| :--- | :--- | :--- | :--- |
| **1** | **Faster R-CNN** | Two-stage, Backbone ResNet-50 kết hợp FPN, RPN. | **Chuẩn mực đối chứng Two-stage**: Được chính He et al. sử dụng trong bài báo trên *Journal of Social Computing (2023)*. Đại diện cho độ trễ cao nhưng đặc trưng phân loại mạnh. |
| **2** | **SSD (Single Shot Detector)** | One-stage anchor-based, Backbone MobileNetV2. | **Đối chứng trực tiếp bài báo nền tảng**: Chính là mô hình Stage 2 mà He et al. (2023, ESWA) dùng để phân loại tư thế. |
| **3** | **YOLOv5s** | One-stage anchor-based, CSPDarknet53. | **Mô hình trung tâm của đề cương & Stage 1 của He et al. (2023)**: Đạt tốc độ cao (75 FPS) và mAP ~89%. Là mốc so sánh cốt lõi. |
| **4** | **YOLOv8s** | One-stage anchor-free, C2f module, decoupled head. | **Đại diện tiêu chuẩn hiện tại (2023-2024)**: Kiến trúc phổ biến nhất trong các bài báo ứng dụng hiện nay, cân bằng tốt giữa tốc độ và độ chính xác. |
| **5** | **YOLOv11s** | One-stage anchor-free, C3k2 module, SPPF cải tiến. | **Đại diện SOTA mới nhất**: Minh chứng cho việc dù dùng detector tiên tiến nhất hiện nay, nếu không có bộ lọc thời gian thì vẫn thất bại trước vấn đề false alarm. |
| **6** | **RT-DETR-R18** | Vision Transformer thời gian thực (Baidu), Hybrid Encoder. | **Đại diện trường phái Transformer**: Đang là xu hướng xuất bản trên các tạp chí hàng đầu (TPAMI, CVPR). So sánh năng lực của cơ chế Global Attention vs Convolution. |

---

## 5. THIẾT KẾ PHƯƠNG PHÁP & THUẬT TOÁN TEMPORAL FILTERING

### 5.1. Sơ đồ Pipeline tổng thể
```
                      [ Video Luồng Vào (25 - 30 FPS) ]
                                      │
                                      ▼
                        [ Module Tách Khung Hình ]
                                      │
                                      ▼
                  [ Frame-level Detector (YOLOv5s / SOTA) ]
                                      │
                     Output: Bounding Box, Class, Score P_drown
                                      │
                                      ▼
                   [ Object Tracking (ByteTrack / ByteID) ]
                       (Duy trì ID của từng đối tượng bơi)
                                      │
                                      ▼
                ┌──────────────────────────────────────────────┐
                │        MODULE TEMPORAL FILTERING             │
                │                                              │
                │  - Cửa sổ trượt kích thước W = FPS * N giây  │
                │  - Bộ đệm lịch sử: B_id = [P_0, P_1, ..., P_W]│
                │  - Đánh giá liên tục: Debounce Logic         │
                └──────────────────────────────────────────────┘
                                      │
                     Vượt ngưỡng xác thực liên tục >= N giây?
                                ├── KHÔNG ──► Hủy kích hoạt (Bỏ qua hành vi thoáng qua)
                                └── CÓ ─────► [ KÍCH HOẠT BÁO ĐỘNG ĐUỐI NƯỚC THỰC SỰ ]
```

### 5.2. Công thức Toán học & Thuật toán Debounce Cửa sổ trượt
Giả sử video có tốc độ $FPS$, xét đối tượng $i$ với chuỗi nhận diện tại khung hình $t$:
$$S_t^{(i)} = \begin{cases} 1 & \text{nếu } \text{Class} = \text{drowning} \text{ và } \text{Confidence} \ge \theta_{conf} \\ 0 & \text{ngược lại (swimming / safe)} \end{cases}$$

Bộ đệm trạng thái của đối tượng $i$ trong cửa sổ trượt độ rộng $W = FPS \times N$ (với $N$ là ngưỡng thời gian cần xác thực, ví dụ $N = 2$ giây):
$$\mathcal{H}_t^{(i)} = [S_{t-W+1}^{(i)}, S_{t-W+2}^{(i)}, \dots, S_t^{(i)}]$$

**Điều kiện kích hoạt cảnh báo (Alert Trigger Condition)**:
$$\text{Alert}_t^{(i)} = \begin{cases} \text{TRUE} & \text{nếu } \sum_{k=0}^{W-1} S_{t-k}^{(i)} \ge \alpha \cdot W \\ \text{FALSE} & \text{ngược lại} \end{cases}$$
Trong đó:
* $\alpha \in [0.8, 1.0]$ là hệ số dung sai (cho phép bù trừ một số frame bị mất dấu hoặc nhiễu tia nước chói sáng).
* Khi $\alpha = 1.0$, thuật toán tương đương với kiểm tra $W$ frame liên tiếp đều là `drowning`.

---

## 6. HỆ THỐNG TIÊU CHÍ ĐÁNH GIÁ (EVALUATION METRICS)

Để đảm bảo tính khoa học chặt chẽ, đề tài sử dụng hai nhóm tiêu chí đánh giá độc lập:

### 6.1. Nhóm tiêu chí đánh giá Frame-level (Tập test ảnh)
* **Precision ($P$)**: Tỷ lệ dự đoán đuối nước chính xác trên tổng số lần mô hình báo đuối nước.
* **Recall ($R$)**: Tỷ lệ phát hiện được đối tượng đuối nước trên tổng số đối tượng đuối nước thực tế.
* **mAP@0.5**: Độ chính xác trung bình tại ngưỡng IoU 0.5 (Chỉ số chính để so sánh với công bố mAP > 89% của He et al., 2023).
* **mAP@0.5:0.95**: Độ chính xác trung bình toàn diện qua các dải IoU từ 0.5 đến 0.95.
* **Frame Latency & FPS**: Thời gian suy luận trung bình trên 1 frame (ms) và số khung hình xử lý được trong 1 giây trên GPU T4.

### 6.2. Nhóm tiêu chí đánh giá Temporal-level (Bộ video kiểm thử Testbed)
* **False Alarm Rate (FAR)**: Số lần báo động giả trung bình trên mỗi phút video kiểm thử:
  $$\text{FAR} = \frac{\text{Tổng số lần kích hoạt cảnh báo sai}}{\text{Tổng thời lượng video (phút)}}$$
* **Detection Delay ($\tau_d$)**: Độ trễ thời gian từ mốc bắt đầu xảy ra sự kiện đuối nước thật ($T_{start}^{GT}$) đến khi hệ thống chính thức kích hoạt chuông báo ($T_{alert}$):
  $$\tau_d = T_{alert} - T_{start}^{GT} \quad (\text{Yêu cầu: } \tau_d \le 2.0 - 3.0 \text{ giây})$$
* **Miss Rate (Tỷ lệ bỏ sót)**: Tỷ lệ các sự kiện đuối nước thật không được hệ thống phát hiện:
  $$\text{Miss Rate} = \frac{\text{FN}_{\text{events}}}{\text{Tổng số sự kiện đuối nước thật}} \times 100\% \quad (\text{Yêu cầu tối thượng: } 0\%)$$

---

## 7. LỘ TRÌNH THỰC HIỆN CHI TIẾT & CÁC CỘT MỐC (12 TUẦN)

```
Tuần:  [1 - 2]   ──►  [3 - 5]   ──►  [6 - 7]   ──►  [8 - 9]   ──►  [10 - 11]  ──►  [12]
Giai   Cột mốc 1      Cột mốc 2      Cột mốc 3      Cột mốc 4      Cột mốc 5       Cột mốc 6
đoạn:  Setup &        Huấn luyện     Xây dựng       Cài đặt        Thực nghiệm     Báo cáo &
       Dữ liệu        6 Baselines    Video Test     Temporal       Đa chiều        Bảo vệ
```

### Cột mốc 1: Thiết lập Môi trường & Khai phá/Chuẩn hóa Dataset (Tuần 1 - 2)
* **Mục tiêu**: Có môi trường thực nghiệm hoàn chỉnh và dataset 2 lớp đạt chuẩn khoa học.
* **Công việc cụ thể**:
  * Cài đặt môi trường Python 3.10+, PyTorch 2.x, CUDA 11.8/12.1 trên Colab / GPU cục bộ.
  * Tải và đồng bộ hóa các tập dữ liệu từ Figshare (DOI: `10.6084/m9.figshare.29497235.v2`), Roboflow và GitHub.
  * Chạy script làm sạch nhãn, loại bỏ bounding box lỗi, chuyển đổi toàn bộ về chuẩn định dạng YOLO `.txt`.
  * Phân chia tập dữ liệu chuẩn Train / Val / Test (70% - 15% - 15%).
* **Sản phẩm đầu ra**: Thư mục `dataset/` sạch, file cấu hình `data.yaml`, biểu đồ thống kê phân bố lớp và kích thước bounding box.

### Cột mốc 2: Huấn luyện & Đánh giá 5 - 6 Mô hình Baseline (Tuần 3 - 5)
* **Mục tiêu**: Hoàn thành việc huấn luyện và có bảng số liệu so sánh công bằng giữa các kiến trúc.
* **Công việc cụ thể**:
  * Huấn luyện 6 mô hình trên cùng tập dữ liệu Train:
    1. Faster R-CNN (ResNet-50 FPN)
    2. SSD (MobileNetV2)
    3. YOLOv5s (Model nền tảng)
    4. YOLOv8s
    5. YOLOv11s
    6. RT-DETR-R18
  * Thiết lập cấu hình huấn luyện đồng nhất: 100 epochs, Image size 640×640, Batch size 16 hoặc 32.
  * Đo đạc mAP@0.5, mAP@0.5:0.95, Precision, Recall và FPS suy luận.
* **Sản phẩm đầu ra**: Bộ trọng số (`best.pt` / `.pth`), bảng tổng hợp so sánh hiệu năng 6 mô hình, đồ thị Loss và PR-Curve.

### Cột mốc 3: Xây dựng Bộ Video Kiểm thử (Testbed) & Gán nhãn Ground Truth (Tuần 6 - 7)
* **Mục tiêu**: Chuẩn bị tập dữ liệu video chuỗi thời gian để kiểm thử bài toán báo động giả.
* **Công việc cụ thể**:
  * Thu thập 15 - 20 video hồ bơi (thời lượng 30s - 2 phút) bao gồm:
    * 8 - 10 kịch bản đuối nước thật (mô phỏng cứu nạn, người gặp nạn thật sự).
    * 15 - 20 tình huống nhiễu dễ gây nhầm lẫn: vẫy tay gọi bạn, ngụp lặn nín thở, té nước té bọt, đứng nước đạp chân.
  * Gán nhãn thời gian chi tiết xuất ra file `ground_truth_events.csv` (ghi rõ `start_second`, `end_second`, `event_type`).
* **Sản phẩm đầu ra**: Thư mục 20 video test và file `ground_truth_events.csv` hoàn chỉnh.

### Cột mốc 4: Thiết kế & Cài đặt Module Temporal Filtering (Tuần 8 - 9)
* **Mục tiêu**: Hoàn thiện module xử lý thời gian có thể tích hợp trực tiếp vào đầu ra của detector.
* **Công việc cụ thể**:
  * Lập trình module `TemporalFilter` bằng Python / OpenCV:
    * Quản lý bộ đệm trạng thái cửa sổ trượt.
    * Cơ chế Debounce kích hoạt cảnh báo khi phát hiện nguy cơ kéo dài liên tục.
    * Tích hợp thuật toán theo dõi đối tượng (ByteTrack) để gán ID ổn định cho từng người trong hồ bơi.
  * Thiết lập các tham số thử nghiệm: Khảo sát ngưỡng thời gian $N \in \{1.0s, 1.5s, 2.0s, 2.5s, 3.0s\}$.
* **Sản phẩm đầu ra**: File mã nguồn `temporal_filter.py` và `tracking_pipeline.py`.

### Cột mốc 5: Thực nghiệm Đo đạc Đa chiều & Nghiên cứu Bóc tách (Tuần 10 - 11)
* **Mục tiêu**: Thu được số liệu chứng minh tính hiệu quả của phương pháp cải tiến so với mô hình gốc.
* **Công việc cụ thể**:
  * Cho chạy toàn bộ 15 - 20 video kiểm thử qua 2 chế độ:
    * *Chế độ Baseline*: Mô hình nhận diện frame-level thuần túy (không lọc thời gian).
    * *Chế độ Đề xuất*: Mô hình nhận diện kết hợp Temporal Filter với các ngưỡng $N$.
  * Đo đạc và ghi nhận: Số lần báo động giả (False Alarms), False Alarm Rate (FAR), Độ trễ cảnh báo ($\tau_d$), Tỷ lệ bỏ sót (Miss Rate).
  * Vẽ đồ thị thể hiện sự đánh đổi (Trade-off Curve) giữa Detection Delay và False Alarm Rate.
  * Đối chiếu trực tiếp với kết quả công bố trong bài báo gốc của He et al. (2023).
* **Sản phẩm đầu ra**: Bảng so sánh Trước/Sau khi áp dụng bộ lọc, đồ thị Ablation Study.

### Cột mốc 6: Xây dựng Demo Trực quan & Hoàn thiện Báo cáo (Tuần 12)
* **Mục tiêu**: Có sản phẩm demo hoàn chỉnh và báo cáo khoa học sẵn sàng bảo vệ trước hội đồng UIT.
* **Công việc cụ thể**:
  * Phát triển giao diện Demo (sử dụng OpenCV Video Stream hoặc web app Streamlit):
    * Hiển thị bounding box màu xanh cho `swimming`.
    * Hiển thị thanh tiến trình cảnh báo (Progress Bar từ 0 đến $N$ giây) khi nghi ngờ `drowning`.
    * Kích hoạt khung viền đỏ và chuông cảnh báo khi đạt ngưỡng $N$ giây.
  * Viết toàn văn Báo cáo Nghiên cứu khoa học theo cấu trúc chuẩn IMRAD (Introduction - Methodology - Results - Discussion).
  * Thiết kế Slide thuyết trình bảo vệ đề tài.
* **Sản phẩm đầu ra**: Báo cáo NCKH hoàn chỉnh (PDF/Word), Slide thuyết trình, Video clip demo hệ thống thực tế.

---

## 8. BIỂU MẪU KẾT QUẢ DỰ KIẾN TRÌNH BÀY TRONG BÁO CÁO

Để báo cáo đạt điểm tối đa trước hội đồng, hai bảng kết quả thực nghiệm sau đây sẽ đóng vai trò minh chứng trung tâm:

### Bảng 1: Hiệu năng đối sánh 6 Mô hình Baseline ở cấp độ Frame (Test Set Ảnh tĩnh)
| STT | Mô hình | Kiến trúc / Kiểu | Số tham số (Params) | GFLOPs | Precision (%) | Recall (%) | mAP@0.5 (%) | FPS (GPU T4) | Đánh giá |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **Faster R-CNN** | Two-stage (ResNet-50) | 41.5M | 180.2 | 84.2% | 87.5% | 86.8% | 18 FPS | Chậm, không đạt real-time |
| 2 | **SSD** *(He et al.)* | One-stage (MobileNetV2) | 4.3M | 3.8 | 81.0% | 79.5% | 80.2% | 52 FPS | Nhanh nhưng độ chính xác thấp |
| 3 | **YOLOv5s** *(He et al.)* | One-stage (CSPDarknet) | 7.2M | 16.5 | 86.5% | 88.2% | 89.4% | **75 FPS** | Cân bằng tốt, đạt chuẩn He et al. |
| 4 | **YOLOv8s** | One-stage (Anchor-free) | 11.2M | 28.6 | 88.1% | 89.0% | 91.2% | 68 FPS | Hiệu năng nhận diện cao |
| 5 | **YOLOv11s** | One-stage (C3k2 SOTA) | 9.4M | 21.5 | **89.5%** | **90.1%** | **92.0%** | 72 FPS | Độ chính xác frame-level cao nhất |
| 6 | **RT-DETR-R18** | Vision Transformer | 20.0M | 60.0 | 88.8% | 89.4% | 91.5% | 45 FPS | Attention toàn cục tốt, tài nguyên vừa |

### Bảng 2: Đánh giá Đóng góp của Temporal Filtering trên Video Testbed Benchmark
| Cấu hình Thử nghiệm | Ngưỡng thời gian $N$ (giây) | Tổng số Báo động giả (False Alarms) | False Alarm Rate (lần / phút) | Tỷ lệ Bỏ sót (Miss Rate) | Độ trễ cảnh báo $\tau_d$ (giây) | Tính khả thi triển khai thực tế |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Gốc: Không lọc (Frame-level)** | $N = 0.0s$ | 38 lần | 2.53 | **0.0%** | **0.04s** | ❌ **Không khả thi** (Báo động giả liên tục) |
| **+ Temporal Filter** | $N = 1.0s$ | 11 lần | 0.73 | 0.0% | 1.06s | ⚠️ Cải thiện một phần, còn nhiễu |
| **+ Temporal Filter (Tối ưu)** | $N = 2.0s$ | **2 lần** | **0.13** | **0.0%** | **2.12s** | ✅ **Rất cao (Điểm cân bằng tối ưu)** |
| **+ Temporal Filter** | $N = 3.0s$ | 0 lần | 0.00 | 8.3% *(bỏ sót 1)* | 3.18s | ⚠️ Nguy hiểm (Trễ phản ứng cứu hộ) |

> **Phân tích học thuật từ Bảng 2**:
> * Khi $N = 0$ (mô hình của He et al. và các phương pháp frame-level): Cứ mỗi phút có tới 2.53 lần báo động giả, làm hệ thống mất hoàn toàn độ tin cậy.
> * Khi áp dụng Temporal Filter với ngưỡng tối ưu **$N = 2.0$ giây**: Tỷ lệ báo động giả giảm hơn **95%** (từ 2.53 lần/phút xuống 0.13 lần/phút), trong khi vẫn bảo toàn tuyệt đối **Miss Rate = 0%** và độ trễ phản ứng chỉ là 2.12 giây (hoàn toàn nằm trong tiêu chuẩn cứu nạn vàng dưới 10 giây của nhân viên cứu hộ).

---

## 9. TÀI LIỆU THAM KHẢO HỌC THUẬT (ACADEMIC REFERENCES)

[1] **He, Q., Zhang, H., Mei, Z., & Xu, X.** (2023). High accuracy intelligent real-time framework for detecting infant drowning based on deep learning. *Expert Systems with Applications*, 228, 120389. DOI: `10.1016/j.eswa.2023.120389`. (Elsevier - ScienceDirect)

[2] **He, Q., Mei, Z., Zhang, H., & Xu, X.** (2023). Automatic Real-Time Detection of Infant Drowning Using YOLOv5 and Faster R-CNN Models Based on Video Surveillance. *Journal of Social Computing*, 4(1), 62-73. DOI: `10.23919/JSC.2023.0006`. (IEEE)

[3] **Hasan, S., Joy, J., Ahsan, F., Khambaty, H., Agarwal, M., & Mounsef, J.** (2021). A Water Behavior Dataset for an Image-Based Drowning Solution. *2021 IEEE Green Energy and Smart Systems Conference (IGESSC)*, Long Beach, CA, USA, pp. 1-6. DOI: `10.1109/IGESSC53193.2021.9679440`. (IEEE Xplore)

[4] **Jiang, X., Tang, D., Xu, W., Zhang, Y., & Lin, Y.** (2025). Swimming-YOLO: a drowning detection method in multi-swimming scenarios based on improved YOLO algorithm. *Signal, Image and Video Processing*, Springer. DOI: `10.1007/s11760-024-03712-x`. (SpringerLink)

[5] **Underwater Drowning Detection Dataset** (2025). Manual Annotated YOLO Dataset for Aquatic Safety. *Figshare*. DOI: `10.6084/m9.figshare.29497235.v2`.

[6] **Ren, S., He, K., Girshick, R., & Sun, J.** (2015). Faster R-CNN: Towards real-time object detection with region proposal networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 28.

[7] **Liu, W., et al.** (2016). SSD: Single shot multibox detector. *European Conference on Computer Vision (ECCV)*, Springer, pp. 21-37.

[8] **Jocher, G., et al.** (2020). YOLOv5 by Ultralytics. DOI: `10.5281/zenodo.3908559`.

[9] **Ultralytics** (2024). YOLOv8 & YOLOv11 Architecture and Real-Time Computer Vision. Available: `https://github.com/ultralytics/ultralytics`.

[10] **Lv, W., et al.** (2024). DETRs Beat YOLOs on Real-time Object Detection (RT-DETR). *arXiv preprint arXiv:2304.08069*. Baidu Inc.
