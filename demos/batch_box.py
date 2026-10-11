import cv2
import torch
import numpy as np

from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN


# 输入和输出文件夹
input_dir = Path("samples")
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

# 使用 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备：", device)

# 创建 MTCNN 人脸检测模型
mtcnn = MTCNN(
    keep_all=True,
    device=device
)

# 支持的图片格式
image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

image_paths = sorted(
    path for path in input_dir.iterdir()
    if path.is_file()
    and path.suffix.lower() in image_extensions
)

print("找到图片数量：", len(image_paths))

for image_path in image_paths:
    print("正在处理：", image_path.name)

    # 读取图片
    pil_image = Image.open(image_path).convert("RGB")

    # 检测人脸
    boxes, probabilities = mtcnn.detect(pil_image)

    # 转换为 OpenCV 格式
    result = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR
    )

    face_count = 0

    if boxes is not None:
        for box, probability in zip(boxes, probabilities):
            x1, y1, x2, y2 = map(int, box)

            # 画人脸框
            cv2.rectangle(
                result,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # 写置信度
            text = f"Face {probability:.2f}"

            cv2.putText(
                result,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            face_count += 1

    # 统一保存为 JPG
    output_path = output_dir / f"{image_path.stem}_result.jpg"
    cv2.imwrite(str(output_path), result)

    print("检测到人脸：", face_count)
    print("结果保存到：", output_path)

print("全部处理完成")
