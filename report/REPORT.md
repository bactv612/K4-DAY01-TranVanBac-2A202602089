# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy: 110/09/2026**

**Runtime Colab:** T4 GPU
**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không 

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
- Record này mô tả toàn ảnh như thế nào?

Trả lời: - `class_id` và `class_name` là kết quả phân loại mà mô hình phát hiện - mã số lớp và tên lớp
         - `rank` là đánh giá của mô hình về độ tin cậy của mô hình khi phân loại, thì theo code hiện chỉ lấy top 5 kết quả tốt nhất để đánh giá
         - `score` là điểm độ tin cậy của đánh giá chạy từ 0.0 tới 1
         - `taxonomy_name` là tên bộ từ điển nhãn, trong code hiện tại đang được fix cứng là *ImageNet-1K*

- Ai định nghĩa class list mà checkpoint có thể dự đoán?

Trả lời: các mô hình YOLO trước đó đã được huấn luyện trên nhiều tập dữ liệu được các nhà khoa học thiết kế các bộ phân loại riêng. Sau khi huấn luyện thành công các dữ liệu về class này sẽ được lưu vào trong checkpoint hoặc model weight.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?

Trả lời: - Cần class_id (Số thứ tự của lớp) để phục vụ Máy tính dễ tra cứu
         - Cần class_name (Tên lớp bằng chữ) để phục vụ Con người
         - Cần taxonomy_name (Tên bộ tiêu chuẩn) để xác nhận Nguồn gốc đối chiếu

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?

Trả lời: 
Tùy theo yêu cầu của chủ thể ta có thể tạo ra các quy định:
- Quy định về "Diện tích chiếm ưu thế" (Dominant Object): Chủ thể nào chiếm tỷ lệ diện tích khung hình lớn nhất (hoặc lớn hơn x%) thì lấy đó làm nhãn gốc.
- Quy định về "Vị trí trung tâm" (Center Focus): Đôi khi diện tích không quan trọng bằng bối cảnh hội tụ ảnh. Guideline sẽ ghi: "Chọn vật thể nằm rõ và đầy đủ nhất ở vùng lưới trung tâm 3x3 của ảnh."
-  Quy định theo "Thứ tự độ ưu tiên nghiệp vụ" (Business Priority): Ví dụ nếu dự án làm về y tế: "Nếu ảnh có con người, giường bệnh và xe lăn. Bất chấp tất cả, ưu tiên số 1 phải gán nhãn là Người."

- Vì sao model score không phải ground truth?
Trả lời: score chỉ phản ánh độ tự tin toán học của mô hình dựa trên dữ liệu đã huấn luyện trước đó. Mô hình có thể rất tự tin (score = 0.99) nhưng vẫn dự đoán sai hoàn toàn. Ground Truth phải là kết quả do con người gán và kiểm định theo đúng Guideline.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
- Diễn giải vị trí box bằng lời:
Trả lời: Vị trí box có 4 điểm bbox_xyxy = [x_min, y_min, x_max, y_max]. Với gốc tọa độ luôn nằm góc trên cùng bên trái, x_min nói bounding_box cách mép trái ảnh là x_min pixel, y_min nói bounding_box cách mép trên cùng bức ảnh là y_min pixel, x_max là chiều rộng của bounding_box, y_max là chiều cao của bounding_box.

- So sánh số prediction ở hai threshold:
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?

Trả lời: Với mức threshold thấp, mô hình đề xuất ra nhiều vật thể hơn. Reviewer sẽ phải làm nhiều việc hơn, thay vì chỉ việc vẽ thêm hộp nếu AI bỏ sót, giờ đây Reviewer phải căng mắt rà soát toàn bộ ảnh để click chuột xóa thủ công (Delete / Reject) hàng loạt các chiếc hộp rác bị khoanh nhầm.

- Đề xuất một quy tắc box chặt:

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?

Trả lời: Guideline hoặc escalation cần quyết định :
- Nên vẽ bounding_box theo tưởng tượng hay theo trực quan. Ví dụ 1 người bị che cánh tay và chân bởi tủ thì nên vẽ bounding_box cho những phần đang hiển thị hay vẽ cho cả phần bị che.
- Nên vẽ 1 hộp trùm hay cắt thành nhiều hộp nhỏ: Vẫn ví dụ trên nhưng bây giờ phía người đó đã bị che gần hết người bởi cái tủ, chỉ còn hiển thị đầu, 2 tay, chân thì nên vẽ bounding_box trùm lên cả hãy chỉ vẽ bounding_box cho từng bộ phận.
- Cần quyết định ngưỡng loại bỏ: Với ví dụ như trên thì tỷ lệ hiển thị bao nhiêu là mình có thể bỏ, không quan tâm.



## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
- Polygon bổ sung chi tiết gì so với box?

Trả lời: Box vơ vét cả những điểm ảnh nhiễu (Background Noise), dù Bounding Box có đóng chặt đến mấy, bên trong cái hộp đó vẫn sẽ lọt vào rất nhiều điểm ảnh (pixels) của nền đường, cây cỏ, hoặc không khí... chứ không chỉ chứa mỗi cái xe đạp. Polygon đem lại "Đường viền biên", bổ sung các chi tiết về đường viền thực tế, hình dáng và ranh giới sắc nét từng điểm ảnh của sự vật, giúp bóc tách hoàn toàn sự vật đó ra khỏi bức hình (như hành động tách nền Photoshop) mà Bounding Box không làm được.

- `instance_id` dùng để làm gì và không phải loại ID nào?

Trả lời: `instance_id`dùng để tách biệt các sự vật nằm trong cùng một danh mục, giả sử trong một tấm ảnh chụp một đĩa trái cây có 3 quả táo, thì nhờ `instance_id`, dù hai quả táo có dính sát, cọ xát vào nhau, hoặc quả này đè lên quả kia, máy tính vẫn hiểu và tách bóc được chúng là 2 vật thể vật lý hoàn toàn riêng biệt. Và nó không phải `class_id`, 1 `class_id` có thể có nhiều `instance_id`.
 
- Đề xuất một quy tắc biên mask:

Trả lời: Quy tắc Bám sát viền và Đục lỗ rỗng : Đa giác (Polygon) phải được bấm dọc theo đúng mép màu ranh giới phân tách giữa vật thể và background. Độ lệch (sai số) cho phép dao động lẹm ra ngoài nền hoặc lấn vào trong thịt vật thể phải nhỏ nhất có thể. Nếu ở giữa vật thể có không gian trống nhìn xuyên thấu qua nền phía sau (ví dụ: khoảng trống ở giữa chiếc vòng cổ, hoặc lỗ rỗng trong lõi cuộn băng dính), Reviewer phải khoanh thêm một đường biên âm ở giữa để "đục lỗ" (subtract/make hole) loại bỏ phần nền đó ra khỏi Mask.


- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?

Trả lời: - Đối với `Vùng mờ` guideline quyết định: Hướng dẫn vẽ theo Khối cứng  hay vẽ theo Từ trường nhòe. Ví dụ nếu tay người quơ nhanh qua camera bị nhòe thành cả một vệt xám lớn, Guideline phải quyết định rằng: Phải vẽ theo ước lượng khung xương khuỷu tay ngầm trong vết nhòe đó; hoặc là vẽ một vòng tròn khổng lồ lợp lấy toàn bộ cái vết mây xám chuyển động kia.
         - Đối với `Vùng tiếp xúc` guideline quyết định dựa vào đâu để chẻ viền tách đôi, đa giác của vật 1 và đa giác của vật 2 có được phép bấm trùng điểm chạm (dính line) và đè mép lên nhau 1-2 pixel không? Hay bắt buộc đường chia tách phải có khoảng hở ở giữa tuyệt đối không dính vấp?
         - Đối với `Vùng che khuất`, cho ví dụ một chú chó dầm mình xuống nước, chỉ lòi cái đầu khỏi mặt nước và cái đuôi ngoe nguẩy phía sau nhô lên khỏi mặt nước. Con chó đã bị tách làm 2 mảnh Đa giác vật lý hoàn toàn cách biệt (méo mó), guideline phải chỉ định: Hai mảnh Đa giác rời rạc này phải được gộp lại với nhau (Merge Attributes) dưới cùng chung một mã instance_id duy nhất, để dạy mô hình AI hiểu rằng: Hai cục tách biệt đó trên màn hình vẫn là da thịt của cùng 1 con chó duy nhất bên dưới mặt nước. Không được tách đôi tính là 2 con chó. 

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn/ID duy nhất đại diện cho toàn ảnh (`class_id`). | Ảnh có nhiều chủ thể gây tranh chấp (VD: Cả chó và người). | Chọn nhãn theo luật ranh giới ưu tiên (Dominant / Priority rule) trong Guideline. | Đối chiếu xem Annotator đã chọn đúng chủ thể được ưu tiên chưa. |
| Phát hiện vật thể | Tập hợp khung chữ nhật Bounding Box (`bbox_xyxy`). | Vật bị che lấp (occlusion) làm vỡ làm đôi hoặc cắt mép ảnh. | Ước lượng vẽ trùm luôn phần che khuất (amodal), hoặc bỏ qua nếu mất >70%. | Xem mép hộp (box) có bao khít pixel không, hay bị hở dư khoảng đệm quá mức. |
| Instance segmentation | Tập hợp đa giác tọa độ viền biên (`polygon_xy`) và `instance_id`. | Ánh sáng nhòe (motion blur) mờ biên, hoặc 2 vật thể dính sát rạt. | Tuân thủ luật vẽ khối cơ bắp cứng và nhớ gộp chung 1 `instance_id` nếu bị cắt khúc. | Xem viền đa giác đục lỗ có nét không, có bị gộp lầm 2 `instance_id` đè nhau không. |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Báo cho Quản lý dự án hoặc các kỹ sư dữ liệu hoặc kỹ sư AI liên quan.

## 6. Danh sách bằng chứng

- [X] `classification_predictions.json`
- [X] `detection_predictions.json`
- [X] `segmentation_predictions.json`
- [X] `IMAGE_ATTRIBUTION.md`
- [X] `visuals/classification_top5.png`
- [X] `visuals/detection_predictions.png`
- [X] `visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`.
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
