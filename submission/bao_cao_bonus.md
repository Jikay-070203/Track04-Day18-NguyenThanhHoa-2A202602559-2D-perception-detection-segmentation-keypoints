# Báo cáo bonus — Lab Ngày 18

Môi trường: cuda, ultralytics 8.4.171, torch 2.11.0+cu128.

## 1. Bonus 4C — tập val lật gương

Val gốc là val tiger-pose của lab (53 ảnh, mọi con hổ quay phải). Val lật gương là chính 53 ảnh đó, lật ngang, nhãn chuyển theo quy ước giải phẫu (x → 1 − x rồi hoán đổi cặp trái/phải theo `FLIP_IDX`). Hai model cùng cấu hình (40 epoch, imgsz 640, seed 0), chỉ khác `flip_idx` lúc train.

| Model | Box mAP50-95 — val gốc | Pose mAP50 — val gốc | Pose mAP50-95 — val gốc | Box mAP50-95 — val lật gương | Pose mAP50 — val lật gương | Pose mAP50-95 — val lật gương |
|---|---|---|---|---|---|---|
| flip_idx giải phẫu | 0.914 | 0.995 | 0.436 | 0.907 | 0.995 | 0.425 |
| flip_idx đồng nhất | 0.908 | 0.995 | 0.415 | 0.882 | 0.877 | 0.275 |

Kết quả: trên val gốc hai model gần như không phân biệt được. Pose mAP50 đều 0.995, Pose mAP50-95 là 0.436 (giải phẫu) so với 0.415 (đồng nhất), Box mAP50-95 là 0.914 so với 0.908. Trên val lật gương, model giải phẫu gần như không đổi (Pose mAP50 0.995, Pose mAP50-95 0.425), còn model đồng nhất tụt: Pose mAP50 từ 0.995 xuống 0.877 (−0.118) và Pose mAP50-95 từ 0.415 xuống 0.275 (−0.140, mất khoảng 34%). Khoảng cách giữa hai model là 0.021 điểm Pose mAP50-95 trên val gốc nhưng 0.150 điểm trên val lật gương.

Metric nào đã che lỗi: (1) Mọi metric trên val gốc đều che lỗi, vì val chỉ có hổ quay phải (53/0). Ở đó quy ước "chân phía camera là right" mà model đồng nhất học được tự nhất quán với nhãn, nên lỗi `flip_idx` không bao giờ bị kích hoạt; model sai chỉ lộ ra khi hổ quay trái. (2) Box mAP50-95 gần như mù với lỗi này (0.908 → 0.882 trên val lật gương, chỉ −0.026), vì lỗi nằm ở nhãn keypoint chứ không ở vị trí box. (3) Pose mAP50 che nhiều nhất: nó bão hoà ở 0.995 trên cả hai model ở val gốc, và chỉ lộ −0.118 trên val lật gương vì ngưỡng OKS 0.5 với σ đều 1/12 khá rộng, nên chân trái và chân phải nằm gần nhau vẫn được tính là khớp. Metric nhạy nhất là Pose mAP50-95 (−0.140).

Độ tin cậy: mỗi model chỉ train một lần (seed 0) và val gồm 53 khung hình liên tiếp của một video nên tương quan cao. Chênh lệch nhỏ như 0.021 trên val gốc có thể nằm trong nhiễu, còn chênh lệch 0.150 trên val lật gương đủ lớn để đáng tin hơn nhưng chưa được kiểm chứng qua nhiều seed. Val lật gương cũng là dữ liệu tổng hợp (lật chính ảnh val), không phải hổ quay trái thật.

Thiết kế val tốt hơn: (a) bảo đảm val có cả hai hướng quay (chia tầng theo hướng) và báo cáo mAP riêng cho từng hướng; (b) giữ val lật gương như một bài kiểm tra cho augmentation; (c) thêm metric nhạy với đảo trái/phải, ví dụ tỉ lệ ảnh có OKS tăng sau khi hoán đổi cặp trái/phải (như phân tích lỗi ở 4B) và mAP theo từng keypoint; (d) tách train/val theo đoạn video để các khung gần nhau không rơi vào cả hai tập; (e) thu thập một ít ảnh hổ quay trái thật.

## 2. Bài tập về nhà 3 — export ONNX và đo latency trên CPU

| Cấu hình | preprocess (ms) | inference (ms) | postprocess (ms) | số box | tổng (ms) |
|---|---|---|---|---|---|
| one-to-many + NMS, conf 0.25 | 3.990 | 76.540 | 1.590 | 5 | 82.120 |
| one-to-many + NMS, conf 0.001 | 3.880 | 75.320 | 2.370 | 186 | 81.570 |
| one-to-one NMS-free, conf 0.25 | 3.970 | 51.040 | 0.470 | 5 | 55.480 |
| one-to-one NMS-free, conf 0.001 | 3.720 | 47.640 | 0.460 | 177 | 51.820 |

Postprocess là cột so sánh được. One-to-many + NMS tốn 1.59 ms ở conf 0.25 (5 box) và 2.37 ms ở conf 0.001 (186 box), còn one-to-one NMS-free chỉ 0.47 ms và 0.46 ms (5 và 177 box). Chi phí NMS tăng 0.78 ms (khoảng +49%) khi số box tăng từ 5 lên 186, trong khi postprocess của NMS-free gần như phẳng quanh 0.46 ms. Bỏ NMS giảm postprocess từ 70% đến 81%, đúng hướng dự đoán là lợi ích rõ hơn ở conf thấp. Tuy nhiên về tuyệt đối chỉ là 1.1 ms (conf 0.25) và 1.9 ms (conf 0.001), tức khoảng 2–4% tổng thời gian, vì bus.jpg chỉ có vài trăm box.

Cảnh báo về cột inference: nó khác nhau bất thường (≈ 75–77 ms ở one-to-many so với ≈ 48–51 ms ở one-to-one) dù hai đồ thị cùng kiến trúc, cùng kích thước file (9.4 MB) và NMS nằm ở bước postprocess chứ không thuộc inference. Chênh lệch này vì vậy không thể quy cho việc bỏ NMS. Nhiều khả năng đó là nhiễu đo: CPU Kaggle dùng chung, và one-to-many được đo trước, ngay sau khi ultralytics tự cài onnxruntime. Do đó tổng 82 → 52 ms (−37%) không nên đọc là lợi ích của NMS-free. Để kết luận chắc về inference cần đo xen kẽ hai model nhiều vòng và lấy median; trong lần đo này chỉ kết luận được về cột postprocess.

Kết luận: NMS-free làm postprocess nhanh hơn và ổn định theo số box (≈ 0.46 ms bất kể 5 hay 177 box), nên có giá trị nhất khi conf thấp, cảnh đông, thiết bị yếu hoặc runtime không hỗ trợ NMS. Trên CPU với một ảnh nhỏ, lợi ích thực đo được chỉ khoảng 1–2 ms.
