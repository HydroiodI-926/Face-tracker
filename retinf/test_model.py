
from pathlib import Path
import onnxruntime as ort

# 模型路径相对于当前 Python 文件
model_path = (
    Path(__file__).resolve().parent
    / "models"
    / "RetinaFace_mobile320.onnx"
)

if not model_path.is_file():
    raise FileNotFoundError(f"模型不存在: {model_path}")

session = ort.InferenceSession(
    str(model_path),
    providers=["CPUExecutionProvider"]
)

print("模型加载成功！")
print("推理设备:", session.get_providers())

print("\n模型输入:")
for item in session.get_inputs():
    print(item.name, item.shape, item.type)

print("\n模型输出:")
for item in session.get_outputs():
    print(item.name, item.shape, item.type)
