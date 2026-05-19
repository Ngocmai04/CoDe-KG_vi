# prompts_vi.py
# Vietnamese LLM prompts for financial news KG extraction (CafeF, Vietstock, etc.)
# JSON keys stay in English for downstream parsers; content may be Vietnamese.

COREf_FICL_SYSTEM = r"""Bạn là hệ thống phân giải đồng tham chiếu (coreference) cho văn bản tin tài chính tiếng Việt.
Dưới đây là một đoạn tin được biểu diễn dưới dạng chuỗi token kèm chỉ số (token, index).
Nhiệm vụ: tìm các biểu thức đồng tham chiếu (đại từ, tên viết tắt, cụm danh từ lặp lại) và ánh xạ về thực thể gốc.

Quy tắc:
- Chỉ ghi các biểu thức tham chiếu ngược về một thực thể hoặc khái niệm đã xuất hiện trước đó (công ty, cổ phiếu, người, chỉ số, sự kiện M&A, v.v.).
- "StartToken" và "EndToken" lấy đúng chỉ số token trong đoạn đã cho.
- "RefersTo" là cụm danh từ đầy đủ hoặc tên riêng tiếng Việt/Anh của thực thể gốc (ví dụ: "Tập đoàn Vingroup", "mã HPG", "UBCKNN").
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

COREf_FICL_USER = r"""Xử lý đoạn tin đã token hóa sau:

{tokenized_text}
"""


SIMPLIFY_COMPLEX_SYSTEM = r"""Bạn tách câu phức (câu chứa mệnh đề phụ, quan hệ từ/đại từ quan hệ) trong tin tài chính thành các câu đơn, mỗi câu một sự kiện hoặc quan hệ rõ ràng.

Ví dụ 1:
Input:
"Theo báo cáo quý III mà Vingroup công bố hôm qua, doanh thu hợp nhất tăng 15% so với cùng kỳ."
Output:
S1 -> Vingroup công bố báo cáo quý III hôm qua.
S2 -> Doanh thu hợp nhất của Vingroup tăng 15% so với cùng kỳ.

Ví dụ 2:
Input:
"Cổ phiếu HPG, vốn niêm yết trên HOSE, đã tăng trần sau khi công ty công bố kế hoạch mở rộng nhà máy."
Output:
S1 -> Cổ phiếu HPG niêm yết trên HOSE.
S2 -> Cổ phiếu HPG tăng trần.
S3 -> Công ty công bố kế hoạch mở rộng nhà máy.

Áp dụng cùng quy tắc cho câu mới. Giữ tên riêng, mã CK, số liệu.
***CHỈ in các dòng dạng S1 -> ..., S2 -> ..., không thêm lời dẫn.***
"""

SIMPLIFY_COMPLEX_USER = r"""Input: "{sentence}"
"""


SIMPLIFY_COMPOUND_SYSTEM = r"""Bạn tách câu ghép (nhiều mệnh đề độc lập nối bằng và/hoặc/nhưng) trong tin tài chính thành câu đơn.

Ví dụ 1:
Input:
"FPT báo lãi ròng quý II tăng 20% và cổ tức tiền mặt dự kiến 15%."
Output:
S1 -> FPT báo lãi ròng quý II tăng 20%.
S2 -> FPT dự kiến cổ tức tiền mặt 15%.

Ví dụ 2:
Input:
"VNM là thương hiệu sữa hàng đầu Việt Nam, còn MSN đa dạng hóa sang bán lẻ."
Output:
S1 -> VNM là thương hiệu sữa hàng đầu Việt Nam.
S2 -> MSN đa dạng hóa sang bán lẻ.

CHỈ in các dòng S1 -> ..., S2 -> ..., không giải thích.
"""

SIMPLIFY_COMPOUND_USER = r"""Input: "{sentence}"
"""


SIMPLIFY_COMPOUND_COMPLEX_SYSTEM = r"""Bạn tách câu ghép-phức (vừa có mệnh đề phụ vừa nhiều mệnh đề chính) trong tin tài chính thành câu đơn.

Ví dụ 1:
Input:
"Mặc dù thị trường chứng khoán biến động mạnh, Vietcombank vẫn báo lợi nhuận trước thuế tăng và tăng room ngoại."
Output:
S1 -> Thị trường chứng khoán biến động mạnh.
S2 -> Vietcombank vẫn báo lợi nhuận trước thuế tăng.
S3 -> Vietcombank tăng room ngoại.

Ví dụ 2:
Input:
"Khi giá dầu thế giới giảm, PVS ghi nhận doanh thu sụt, song lợi nhuận ròng vẫn dương nhờ cắt giảm chi phí."
Output:
S1 -> Giá dầu thế giới giảm.
S2 -> PVS ghi nhận doanh thu sụt.
S3 -> Lợi nhuận ròng của PVS vẫn dương.
S4 -> PVS cắt giảm chi phí.

CHỈ in các dòng S1 -> ..., S2 -> ..., không giải thích.
"""

SIMPLIFY_COMPOUND_COMPLEX_USER = r"""Input: "{sentence}"
"""


REL_EXTRACT_SYSTEM = r"""Bạn là agent trích xuất quan hệ cho đồ thị tri thức (KG) từ câu đơn tiếng Việt trong lĩnh vực tài chính—chứng khoán, doanh nghiệp, ngân hàng, M&A, niêm yết, kết quả kinh doanh (CafeF, Vietstock, v.v.).

Nhiệm vụ:
- Nhận diện thực thể: công ty, mã CK, người (CEO/CFO), sàn (HOSE/HNX/UPCOM), chỉ số, sản phẩm/dịch vụ, số liệu tài chính, quốc gia/vùng.
- Xác định quan hệ từ động từ, giới từ, cụm nghĩa (ví dụ: sở hữu, mua lại, niêm yết tại, báo cáo, tăng/giảm, đầu tư vào, là đối tác của).
- Xuất bộ ba JSON với khóa tiếng Anh: "Entity 1", "Relationship", "Entity 2".
- Giá trị có thể bằng tiếng Việt; giữ nguyên tên riêng và mã CK.
- Một câu có thể có nhiều triple trong mảng JSON.
- Bỏ qua triple không có ý nghĩa ngữ nghĩa.

Ví dụ:

Input:
"Everest là thương hiệu gia vị lớn nhất Ấn Độ có trụ sở tại Mumbai."
Output:
[{
"Entity 1": "Everest",
"Entity 2": "thương hiệu gia vị lớn nhất Ấn Độ",
"Relationship": "là"
},
{
"Entity 1": "Everest",
"Entity 2": "Mumbai",
"Relationship": "có trụ sở tại"
}]

Input:
"Paul L. Foster là chủ tịch hội đồng quản trị của Western Refining."
Output:
[{
"Entity 1": "Paul L. Foster",
"Entity 2": "Western Refining",
"Relationship": "là chủ tịch hội đồng quản trị của"
}]

Input:
"Theo nghiên cứu năm 2007, doanh thu hợp nhất của FPT tăng 18%."
Output:
[{
"Entity 1": "FPT",
"Entity 2": "doanh thu hợp nhất tăng 18%",
"Relationship": "báo cáo theo nghiên cứu năm 2007"
}]

Chỉ trả về mảng JSON hợp lệ, không markdown, không giải thích.
"""

REL_EXTRACT_USER = r"""Trích xuất quan hệ KG từ câu sau:

{sentence}
"""


def build_prompt(system_text: str, user_text: str, **kwargs) -> str:
    return system_text.strip() + "\n\n" + user_text.format(**kwargs).strip()
