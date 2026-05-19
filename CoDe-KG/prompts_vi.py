# prompts_vi.py
# Vietnamese LLM prompts for domain-agnostic KG extraction.
# JSON keys stay in English for downstream parsers; entity/relation text may be Vietnamese.

COREf_FICL_SYSTEM = r"""Bạn là hệ thống phân giải đồng tham chiếu (coreference) cho văn bản tiếng Việt thuộc bất kỳ lĩnh vực nào.
Dưới đây là một đoạn văn được biểu diễn dưới dạng chuỗi token kèm chỉ số (token, index).
Nhiệm vụ: tìm các biểu thức đồng tham chiếu (đại từ, tên viết tắt, cụm danh từ lặp lại) và ánh xạ về thực thể hoặc khái niệm gốc.

Quy tắc:
- Chỉ ghi các biểu thức tham chiếu ngược về một thực thể, khái niệm hoặc sự kiện đã xuất hiện trước đó trong đoạn.
- "StartToken" và "EndToken" lấy đúng chỉ số token trong đoạn đã cho.
- "RefersTo" là cụm danh từ đầy đủ hoặc tên riêng của thực thể gốc (tiếng Việt hoặc ngoại ngữ giữ nguyên trong văn bản).
- Không tạo tham chiếu cho thông tin mới chưa xuất hiện.

Định dạng mỗi mục (một JSON object trên một dòng hoặc trong mảng):
{
"Expression": "chuỗi biểu thức",
"StartToken": số_nguyên,
"EndToken": số_nguyên,
"RefersTo": "thực thể gốc"
}

Chỉ trả về các object JSON hợp lệ, không giải thích thêm.
"""

COREf_FICL_USER = r"""Xử lý đoạn văn đã token hóa sau:

{tokenized_text}
"""


SIMPLIFY_COMPLEX_SYSTEM = r"""Bạn tách câu phức (câu chứa mệnh đề phụ, quan hệ từ hoặc đại từ quan hệ) thành các câu đơn, mỗi câu một ý hoặc quan hệ rõ ràng.

Ví dụ 1:
Input:
"Harald Kaas, người học tại Học viện Bách khoa Munich, là kiến trúc sư người Na Uy."
Output:
S1 -> Harald Kaas học tại Học viện Bách khoa Munich.
S2 -> Harald Kaas là kiến trúc sư người Na Uy.

Ví dụ 2:
Input:
"Đỉnh Coburg, vốn nằm trong dãy Erul Heights, cao 783 m trên bán đảo Trinity."
Output:
S1 -> Đỉnh Coburg nằm trong dãy Erul Heights.
S2 -> Đỉnh Coburg cao 783 m trên bán đảo Trinity.

Áp dụng cùng quy tắc cho câu mới. Giữ nguyên tên riêng, số liệu và thuật ngữ chuyên ngành.
***CHỈ in các dòng dạng S1 -> ..., S2 -> ..., không thêm lời dẫn.***
"""

SIMPLIFY_COMPLEX_USER = r"""Input: "{sentence}"
"""


SIMPLIFY_COMPOUND_SYSTEM = r"""Bạn tách câu ghép (nhiều mệnh đề độc lập nối bằng và, hoặc, nhưng,…) thành câu đơn.

Ví dụ 1:
Input:
"Lung cancer là nguyên nhân tử vong hàng đầu do ung thư ở Mỹ, và tỷ lệ mắc bệnh vẫn cao ở nhiều quốc gia."
Output:
S1 -> Lung cancer là nguyên nhân tử vong hàng đầu do ung thư ở Mỹ.
S2 -> Tỷ lệ mắc lung cancer vẫn cao ở nhiều quốc gia.

Ví dụ 2:
Input:
"Khí hậu thay đổi làm tan băng cực nhanh hơn, và mực nước biển dâng đe dọa các cộng đồng ven biển."
Output:
S1 -> Khí hậu thay đổi làm tan băng cực nhanh hơn.
S2 -> Mực nước biển dâng đe dọa các cộng đồng ven biển.

CHỈ in các dòng S1 -> ..., S2 -> ..., không giải thích.
"""

SIMPLIFY_COMPOUND_USER = r"""Input: "{sentence}"
"""


SIMPLIFY_COMPOUND_COMPLEX_SYSTEM = r"""Bạn tách câu ghép-phức (vừa có mệnh đề phụ vừa nhiều mệnh đề chính) thành câu đơn.

Ví dụ 1:
Input:
"Mặc dù nhiệt độ tăng, các cánh đồng vẫn khô hạn, và nông dân lo ngại về hạn hán."
Output:
S1 -> Nhiệt độ tăng.
S2 -> Các cánh đồng vẫn khô hạn.
S3 -> Nông dân lo ngại về hạn hán.

Ví dụ 2:
Input:
"Khi bệnh nhân được điều trị bằng thuốc đã được tổng hợp trong phòng thí nghiệm, chúng tôi đo sự thay đổi huỳnh quang bằng máy quang phổ."
Output:
S1 -> Bệnh nhân được điều trị bằng thuốc.
S2 -> Thuốc đã được tổng hợp trong phòng thí nghiệm.
S3 -> Chúng tôi đo sự thay đổi huỳnh quang bằng máy quang phổ.

CHỈ in các dòng S1 -> ..., S2 -> ..., không giải thích.
"""

SIMPLIFY_COMPOUND_COMPLEX_USER = r"""Input: "{sentence}"
"""


REL_EXTRACT_SYSTEM = r"""Bạn là agent trích xuất quan hệ cho đồ thị tri thức (KG) từ câu đơn tiếng Việt, áp dụng được cho mọi lĩnh vực (khoa học, địa lý, lịch sử, y học, kinh doanh, v.v.).

Nhiệm vụ:
- Nhận diện thực thể: danh từ, cụm danh từ, tên riêng, khái niệm, sự kiện, địa điểm, tổ chức, người, hiện tượng, đại lượng hoặc thuộc tính.
- Xác định quan hệ từ động từ, giới từ và nghĩa câu (ví dụ: là, nằm tại, thuộc, gây ra, có, tham gia, đạt được).
- Xuất bộ ba JSON với khóa tiếng Anh: "Entity 1", "Relationship", "Entity 2".
- Giá trị có thể bằng tiếng Việt; giữ nguyên tên riêng và thuật ngữ như trong câu gốc.
- Một câu có thể có nhiều triple trong mảng JSON.
- Bỏ qua triple không có ý nghĩa ngữ nghĩa.

Ví dụ:

Input:
"Đỉnh Coburg là đỉnh núi đá cao 783 m nằm trong dãy Erul Heights trên bán đảo Trinity."
Output:
[{
"Entity 1": "Đỉnh Coburg",
"Entity 2": "đỉnh núi đá cao 783 m",
"Relationship": "là"
},
{
"Entity 1": "Đỉnh Coburg",
"Entity 2": "dãy Erul Heights",
"Relationship": "nằm trong"
},
{
"Entity 1": "Đỉnh Coburg",
"Entity 2": "bán đảo Trinity",
"Relationship": "nằm trên"
}]

Input:
"Harald Kaas là kiến trúc sư người Na Uy."
Output:
[{
"Entity 1": "Harald Kaas",
"Entity 2": "kiến trúc sư người Na Uy",
"Relationship": "là"
}]

Input:
"Với lựa chọn đặc trưng có ý nghĩa lâm sàng, phương pháp over-sampling đạt kết quả AUC cao nhất."
Output:
[{
"Entity 1": "lựa chọn đặc trưng có ý nghĩa lâm sàng",
"Entity 2": "phương pháp over-sampling",
"Relationship": "Với"
},
{
"Entity 1": "phương pháp over-sampling",
"Entity 2": "kết quả AUC cao nhất",
"Relationship": "đạt"
}]

Chỉ trả về mảng JSON hợp lệ, không markdown, không giải thích.
"""

REL_EXTRACT_USER = r"""Trích xuất quan hệ KG từ câu sau:

{sentence}
"""


def build_prompt(system_text: str, user_text: str, **kwargs) -> str:
    return system_text.strip() + "\n\n" + user_text.format(**kwargs).strip()
