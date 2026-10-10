
from pathlib import Path
from itertools import product
import time

import cv2
import numpy as np
import onnxruntime as ort

SIZE = 320
THRESHOLD = 0.65
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models" / "RetinaFace" / "RetinaFace_mobile320.onnx"
)


def make_priors():
    """
    生成 RetinaFace 的 4200 个先验框
    """
    priors = []
    steps = (8, 16, 32)
    sizes = ((16, 32), (64, 128), (256, 512))

    for step, level_sizes in zip(steps, sizes):
        for y, x in product(range(SIZE // step),
                            range(SIZE // step)):
            for s in level_sizes:
                priors.append([
                    (x + 0.5) * step / SIZE,
                    (y + 0.5) * step / SIZE,
                    s / SIZE,
                    s / SIZE,
                ])

    return np.asarray(priors, dtype=np.float32)


PRIORS = make_priors()


def preprocess(frame):
    """
    等比例缩放、填充、RGB 转换、均值处理
    """
    h, w = frame.shape[:2]
    scale = min(SIZE / w, SIZE / h)

    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))

    canvas = np.full(
        (SIZE, SIZE, 3), 114, dtype=np.uint8
    )
    px = (SIZE - nw) // 2
    py = (SIZE - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    data = rgb.astype(np.float32)
    data -= np.array([104, 117, 123], dtype=np.float32)

    data = data.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(data), scale, px, py


def nms(boxes, scores, threshold=0.4):
    """
    非极大值抑制，去除重叠检测框
    """
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        others = order[1:]

        if others.size == 0:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[others, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[others, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[others, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[others, 3])

        inter = (
            np.maximum(0, xx2 - xx1)
            * np.maximum(0, yy2 - yy1)
        )

        area_a = (
            (boxes[i, 2] - boxes[i, 0])
            * (boxes[i, 3] - boxes[i, 1])
        )
        area_b = (
            (boxes[others, 2] - boxes[others, 0])
            * (boxes[others, 3] - boxes[others, 1])
        )

        iou = inter / (area_a + area_b - inter + 1e-6)
        order = others[iou <= threshold]

    return keep


def detect(frame, session):
    tensor, scale, px, py = preprocess(frame)
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: tensor})
    loc, conf, _ = outputs

    loc = np.asarray(loc).reshape(-1, 4)
    conf = np.asarray(conf).reshape(-1, 2)

    if len(loc) != len(PRIORS):
        raise ValueError(
            f"模型输出 {len(loc)} 个先验框，"
            f"预期 {len(PRIORS)}，请检查模型版本"
        )

    scores = conf[:, 1]
    indices = np.where(scores >= THRESHOLD)[0]

    if len(indices) == 0:
        return []

    p = PRIORS[indices]
    l = loc[indices]
    scores = scores[indices]

    centers = p[:, :2] + 0.1 * l[:, :2] * p[:, 2:]
    sizes = p[:, 2:] * np.exp(0.2 * l[:, 2:])

    boxes = np.concatenate(
        [centers - sizes / 2, centers + sizes / 2],
        axis=1
    ) * SIZE

    boxes[:, 0::2] = (boxes[:, 0::2] - px) / scale
    boxes[:, 1::2] = (boxes[:, 1::2] - py) / scale

    h, w = frame.shape[:2]
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w - 1)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h - 1)

    keep = nms(boxes, scores)

    return [
        (boxes[i].astype(int), float(scores[i]))
        for i in keep
    ]


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"找不到模型: {MODEL_PATH}")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("按 Q 退出")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.perf_counter()
            faces = detect(frame, session)

            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2

            cv2.drawMarker(
                frame, (cx, cy), (255, 0, 0),
                cv2.MARKER_CROSS, 20, 2
            )

            for box, score in faces:
                x1, y1, x2, y2 = box
                fx = (x1 + x2) // 2
                fy = (y1 + y2) // 2

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )
                cv2.circle(
                    frame, (fx, fy), 4,
                    (0, 0, 255), -1
                )

                cv2.putText(
                    frame, f"{score:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

            # 暂时选择面积最大的人脸作为目标
            if faces:
                box, _ = max(
                    faces,
                    key=lambda f: (
                        (f[0][2] - f[0][0])
                        * (f[0][3] - f[0][1])
                    )
                )
                fx = (box[0] + box[2]) // 2
                fy = (box[1] + box[3]) // 2

                error_x = fx - cx
                error_y = fy - cy

                cv2.line(
                    frame, (cx, cy), (fx, fy),
                    (0, 255, 255), 2
                )
                cv2.putText(
                    frame,
                    f"dx={error_x} dy={error_y}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2
                )

            fps = 1.0 / max(
                time.perf_counter() - start, 1e-6
            )
            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2
            )

            cv2.imshow("RetinaFace Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
