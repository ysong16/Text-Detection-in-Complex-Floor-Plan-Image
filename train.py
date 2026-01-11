# simple_train.py
from ultralytics import YOLO

model = YOLO(
    "/root/autodl-tmp/ultralytics-main/ultralytics/cfg/models/11/yolo11-obb_cbam4_head2_3.yaml"
)  # 或 'yolo11s-obb.yaml' 等
model.load("runs/obb/train_eiou_976_872/weights/best.pt")
model.train(
    data="datasets/yourself_datasets/yourself_data.yaml",  # 你的 OBB 数据集配置文件
    epochs=1000,
    imgsz=1024,
    batch=16,  # 指定多卡，自动启用 DDP 多卡训练！
)
