import cv2
import torch
import numpy as np

from PIL import Image
from facenet_pytorch import MTCNN


device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备：", device)

mtcnn = MTCNN(
    keep_all=True,
    device=device
)

camera = cv2.VideoCapture(0, cv2.CAP_V4L2)

camera.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

if not camera.isOpened():
    print("无法打开摄像头")
    exit()
    
while True:
    success, frame = camera.read()

    if not success:
        print("无法读取摄像头画面")
        break

    # OpenCV 默认是 BGR，MTCNN 使用 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 转成 Pillow 图片
    image = Image.fromarray(rgb_frame)

    # 检测当前这一帧中的人脸
    boxes, probabilities = mtcnn.detect(image)

    if boxes is not None:
        
        for box, probability in zip(boxes, probabilities):
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            text = f"Face: {probability:.2f}"

            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    cv2.imshow("Real-time Face Detection", frame)

    # 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()