const express = require('express');
const cors = require('cors');
const path = require('path');
const app = express();
const port = 5002;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Lưu trữ kết quả mới nhất
let latestResult = {
    recognized_text: '',
    confidence: 0,
    timestamp: Date.now()
};

// Lưu trữ video stream mới nhất
let latestStream = {
    image: '',
    timestamp: Date.now()
};

// Lưu trữ 10 frame gần nhất
let recentFrames = [];
const MAX_FRAMES = 10;

// Hàm xử lý kết quả từ 10 frame
function processFrames(newFrame) {
    // Thêm frame mới vào mảng
    recentFrames.push(newFrame);
    
    // Giới hạn số lượng frame
    if (recentFrames.length > MAX_FRAMES) {
        recentFrames.shift();
    }
    
    // Nếu chưa đủ 10 frame, trả về kết quả mới nhất
    if (recentFrames.length < MAX_FRAMES) {
        return newFrame;
    }
    
    // Đếm số lần xuất hiện của mỗi kết quả
    const resultCounts = {};
    let maxCount = 0;
    let mostCommonResult = null;
    
    recentFrames.forEach(frame => {
        if (frame.confidence >= 0.7) { // Chỉ xét các kết quả có confidence >= 70%
            const key = frame.recognized_text;
            resultCounts[key] = (resultCounts[key] || 0) + 1;
            
            if (resultCounts[key] > maxCount) {
                maxCount = resultCounts[key];
                mostCommonResult = frame;
            }
        }
    });
    
    // Nếu có kết quả xuất hiện nhiều nhất, trả về kết quả đó
    if (mostCommonResult) {
        return mostCommonResult;
    }
    
    // Nếu không có kết quả nào đủ điều kiện, trả về kết quả mới nhất
    return newFrame;
}

// API endpoints
app.post('/api/update-result', (req, res) => {
    const result = req.body;
    if (result.error) {
        latestResult = {
            recognized_text: 'Lỗi: ' + result.error,
            confidence: 0,
            timestamp: result.timestamp || Date.now()
        };
    } else {
        latestResult = {
            recognized_text: result.recognized_text,
            confidence: result.confidence,
            timestamp: result.timestamp || Date.now()
        };
    }
    res.json({ success: true });
});

app.get('/api/latest-result', (req, res) => {
    res.json(latestResult);
});

// API endpoint để cập nhật video stream
app.post('/api/update-stream', (req, res) => {
    const stream = req.body;
    if (stream.image) {
        latestStream = {
            image: stream.image,
            timestamp: stream.timestamp || Date.now()
        };
    }
    res.json({ success: true });
});

// API endpoint để lấy video stream mới nhất
app.get('/api/latest-stream', (req, res) => {
    res.json(latestStream);
});

// Serve index.html
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start server
app.listen(port, '0.0.0.0', () => {
    console.log(`Web server is running on port ${port}`);
}); 