import cv2

# Load ảnh từ file hoặc camera
image_path = 'path_to_your_image.jpg'
cap = cv2.VideoCapture(0)  # Dùng camera

# Load classifier cho việc phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load classifier cho việc phát hiện mắt
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    # Đọc frame từ camera hoặc ảnh
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi ảnh sang ảnh xám vì detector khuôn mặt thường hoạt động trên ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật cho khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Xác định vị trí của mắt dựa trên vị trí của khuôn mặt
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            # Tính toán tọa độ trung tâm và trục dài, trục nhỏ cho elip
            center_x = x + ex + ew // 2
            center_y = y + ey + eh // 2
            major_axis = ew // 2
            minor_axis = eh // 2
            
            # Vẽ elip cho mắt
            cv2.ellipse(frame, (center_x, center_y), (major_axis, minor_axis), 0, 0, 360, (0, 0, 255), 2)

    # Hiển thị frame với khuôn mặt và mắt đã được đánh dấu
    cv2.imshow('Face Detection', frame)

    # Dừng khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()


