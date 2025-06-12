# Hệ Thống Nhận Dạng Ngôn Ngữ Ký Hiệu

Đây là một hệ thống nhận dạng ngôn ngữ ký hiệu, có khả năng chuyển đổi các cử chỉ ngôn ngữ ký hiệu thành văn bản và giọng nói. Hệ thống sử dụng Raspberry Pi với module camera để ghi lại các cử chỉ và giao diện web để hiển thị văn bản đã nhận dạng và phát âm thanh đầu ra.

## Cấu Trúc Dự Án

```
sign-language-system
├── server
│   ├── app.py
│   ├── models
│   │   └── sign_language_model.h5
│   ├── routes
│   │   └── api.py
│   ├── utils
│   │   └── preprocess.py
│   └── requirements.txt
├── webapp
│   ├── server.js
│   ├── package.json
│   ├── package-lock.json
│   ├── public
│   └── src
├── raspberry_pi
│   ├── camera_module.py
│   ├── audio_output.py
│   └── config.json
├── README.md
└── docs
```

## Tính Năng

- **Nhận Dạng Cử Chỉ**: Hệ thống ghi lại các cử chỉ ngôn ngữ ký hiệu bằng camera và xử lý để nhận dạng văn bản tương ứng.
- **Hiển Thị Văn Bản**: Văn bản đã nhận dạng được hiển thị trên giao diện web để người dùng có thể theo dõi.
- **Đầu Ra Âm Thanh**: Hệ thống chuyển đổi văn bản đã nhận dạng thành giọng nói và phát qua loa được kết nối.
- **Xử Lý Thời Gian Thực**: Camera ghi lại hình ảnh và gửi đến máy chủ để phân tích và phản hồi theo thời gian thực.
- **Cài Đặt Có Thể Tùy Chỉnh**: Các cài đặt cấu hình cho Raspberry Pi, như URL máy chủ và cài đặt camera, có thể được điều chỉnh trong tệp `config.json`.

## Hướng Dẫn Cài Đặt

1. **Cài Đặt Máy Chủ**:
   - Di chuyển đến thư mục `server`.
   - Cài đặt các gói Python cần thiết được liệt kê trong `requirements.txt` bằng pip.
   - Chạy máy chủ bằng lệnh `python app.py`.

2. **Cài Đặt Raspberry Pi**:
   - Kết nối module camera và loa với Raspberry Pi.
   - Cập nhật tệp `config.json` với URL máy chủ.
   - Chạy `camera_module.py` để bắt đầu ghi lại hình ảnh và gửi đến máy chủ.
3. **Cài Đặt Web Server**:
   - Di chuyển đến mục `webapp`
   - Chạy Web bằng lệnh `node server.js`

## Cách Sử Dụng
- Chạy server
- Mở trình duyệt web và truy cập địa chỉ của máy chủ (mặc định là http://localhost:5002)
- Hướng camera về phía cử chỉ ngôn ngữ ký hiệu bạn muốn nhận dạng.
- Văn bản đã nhận dạng sẽ được hiển thị trên giao diện web, và giọng nói tương ứng sẽ được phát qua loa được kết nối với Raspberry Pi.

## Tài Liệu

Để biết thêm thông tin chi tiết về kiến trúc hệ thống và tương tác giữa các thành phần, vui lòng tham khảo tệp `docs/system_architecture.md`.