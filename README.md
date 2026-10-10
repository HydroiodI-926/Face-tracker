文件架构
```txt
face-tracking/
│
├── README.md                  # 项目介绍、使用方法
├── .gitignore                 # Git 忽略文件
├── run.py                     # 统一运行入口（后续实现）
│
├── detectors/                 # 人脸检测算法
│   ├── __init__.py
│   ├── retinaface_detector.py # RetinaFace 实现
│   └── mtcnn_detector.py      # MTCNN 实现
│
├── tracking/                  # 公共跟踪模块
│   ├── __init__.py
│   ├── kalman.py              # 卡尔曼滤波
│   └── face_selector.py       # 最大人脸筛选
│
├── demos/                     # 两套独立 Demo
│   ├── retinaface_demo.py
│   └── mtcnn_demo.py
│
├── models/                    # 模型权重
│   ├── retinaface/
│   └── mtcnn/
│
├── requirements/              # 依赖管理
│   ├── retinaface.txt
│   └── mtcnn.txt
│
├── configs/                   # 配置文件
│   └── default.yaml
│
└── tests/                     # 测试代码
```