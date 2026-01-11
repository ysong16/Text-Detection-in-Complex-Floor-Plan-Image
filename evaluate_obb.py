"""
使用指令：
python evaluate_obb.py.
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from thop import profile

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


def compute_model_complexity(model, img_size=640):
    """计算模型的FLOPs和参数量."""
    # 创建一个虚拟输入
    dummy_input = torch.randn(1, 3, img_size, img_size).to(model.device)

    # 计算FLOPs和参数量
    flops, params = profile(model.model, inputs=(dummy_input,), verbose=False)
    flops = flops / 1e9  # GFLOPs
    params = params / 1e6  # MParams

    return flops, params


def get_image_paths_from_dataset_dict(dataset_dict, split="val"):
    """从数据集字典中获取图像路径列表."""
    # 获取基础路径
    base_path = Path(dataset_dict.get("path", ""))

    # 获取指定分割的图像路径
    split_path = dataset_dict.get(split, "")
    if not split_path:
        raise ValueError(f"数据集配置中未找到 '{split}' 分割路径")

    # 构建完整路径
    full_path = base_path / split_path

    # 检查路径是文件还是目录
    if full_path.is_file() and full_path.suffix == ".txt":
        # 如果是文本文件，读取其中的图像路径
        with open(full_path) as f:
            image_paths = [line.strip() for line in f.readlines()]
    elif full_path.is_dir():
        # 如果是目录，查找所有支持的图像文件
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(str(full_path / ext)))
    else:
        raise ValueError(f"无法解析路径: {full_path}")

    return image_paths


def compute_inference_speed(model, dataset_dict, img_size=1024, num_images=10, split="val"):
    """计算模型的推理速度."""
    times = []

    # 从数据集字典中获取图像路径
    image_paths = get_image_paths_from_dataset_dict(dataset_dict, split)

    # 随机选择一些图像进行测试
    if len(image_paths) > num_images:
        indices = np.random.choice(len(image_paths), num_images, replace=False)
        selected_paths = [image_paths[i] for i in indices]
    else:
        selected_paths = image_paths

    # 预热模型
    if selected_paths:
        _ = model.predict(selected_paths[0], imgsz=img_size, verbose=False)

    for img_path in selected_paths:
        # 计时推理
        start_time = time.time()
        _ = model.predict(img_path, imgsz=img_size, verbose=False)
        end_time = time.time()

        times.append(end_time - start_time)

    # 计算平均推理时间
    avg_time = np.mean(times) if times else 0
    fps = 1 / avg_time if avg_time > 0 else 0

    return avg_time, fps


def calculate_angle_error(pred_angle, gt_angle):
    """计算角度误差，处理角度周期性."""
    error = abs(pred_angle - gt_angle)
    # 处理角度周期性（0度和180度实际上是相同的）
    return min(error, 180 - error)


def extract_angle_from_obb(points):
    """从OBB点中提取角度."""
    if len(points) < 4:
        return 0

    # 转换为numpy数组
    points = np.array(points).reshape(4, 2)

    # 计算长边向量
    vec1 = points[1] - points[0]
    vec2 = points[2] - points[1]

    # 选择较长的边
    if np.linalg.norm(vec1) > np.linalg.norm(vec2):
        angle = np.degrees(np.arctan2(vec1[1], vec1[0]))
    else:
        angle = np.degrees(np.arctan2(vec2[1], vec2[0]))

    # 规范化角度到0-180度范围
    angle = angle % 180
    if angle < 0:
        angle += 180

    return angle


def evaluate_yolov8_obb(model_path, data_yaml, img_size, batch_size, split, conf_thres, iou_thres):
    """综合评估YOLOv8-OBB模型."""
    # 加载模型
    model = YOLO(model_path)

    # 设置模型参数
    # model.overrides['conf'] = conf_thres
    # model.overrides['iou'] = iou_thres
    # model.overrides['imgsz'] = img_size

    # 加载数据集配置
    dataset_dict = check_det_dataset(data_yaml)

    # 1. 运行标准YOLO验证
    print("运行标准验证...")
    metrics = model.val(data=data_yaml, imgsz=img_size, batch=batch_size, conf=conf_thres, iou=iou_thres)

    # 2. 计算模型复杂度
    print("计算模型复杂度...")
    flops, params = compute_model_complexity(model, img_size)

    # 3. 计算推理速度
    print("计算推理速度...")
    avg_time, fps = compute_inference_speed(model, dataset_dict, img_size, split=split)

    # # 4. 计算角度误差 (这里需要您根据实际情况实现)
    # print("计算角度误差...")
    # # 注意：这部分需要您根据数据格式实现角度误差的计算
    # # 这里只是一个示例，您需要替换为实际的实现
    # mean_angle_error = 0.0  # 默认值

    # 5. 准备结果
    results = {
        "model": model_path,
        "dataset": data_yaml,
        "image_size": img_size,
        "conf_threshold": conf_thres,
        "iou_threshold": iou_thres,
        # 标准YOLO指标
        "map50": metrics.box.map50,
        "map50_95": metrics.box.map,
        "P": metrics.box.mp,  # 精度
        "R": metrics.box.mr,  # 召回率
        "F1": 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-16),  # F1分数
        # 模型复杂度指标
        "FLOPs": flops,  # FLOPs (G)
        "params": params,  # 参数量 (M)
        "inference_time": avg_time * 1000,  # 推理时间 (秒/图像)
        "FPS": fps,  # FPS
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="YOLOv8-OBB模型评估脚本")
    parser.add_argument("--model", type=str, default="runs/obb/train/weights/best.pt", help="模型路径 (.pt文件)")
    parser.add_argument("--data", type=str, default="ultralytics/cfg/datasets/dota8.yaml", help="数据集YAML文件路径")
    parser.add_argument("--img-size", type=int, default=1024, help="图像大小")
    parser.add_argument("--batch_size", type=int, default=16, help="批次")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="使用哪个数据集")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.7, help="iou阈值")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="输出文件路径")

    args = parser.parse_args()

    # 运行评估
    results = evaluate_yolov8_obb(
        args.model, args.data, args.img_size, args.batch_size, args.split, args.conf_thres, args.iou_thres
    )

    # 保存结果
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)

    # 打印结果表格
    print("\n===== 评估结果 =====")
    print(f"{'指标':<20} {'值':<15}")
    print("-" * 35)
    print(f"{'Model':<20} {results['model']:<15}")
    print(f"{'Dataset':<20} {results['dataset']:<15}")
    print(f"{'Precision':<20} {results['P']:.4f}")
    print(f"{'Recall':<20} {results['R']:.4f}")
    print(f"{'F1-Score':<20} {results['F1']:.4f}")

    print(f"{'mAP50':<20} {results['map50']:.4f}")
    print(f"{'mAP50_95':<20} {results['map50_95']:.4f}")

    print(f"{'FLOPs':<20} {results['FLOPs']:.2f} G")
    print(f"{'Parameters':<20} {results['params']:.2f} M")
    print(f"{'Speed':<20} {results['inference_time']:.4f} ms/img")
    print(f"{'FPS':<20} {results['FPS']:.2f}")

    print(f"\n完整结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
