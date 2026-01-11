# from ultralytics import YOLO
# import cv2

# def load_model(model_path):
#     """
#     加载YOLO模型。

#     :param model_path: 模型权重文件路径，例如 'yolov11n.pt'
#     :return: 加载的YOLO模型实例
#     """
#     model = YOLO(model_path)
#     return model

# def detect_and_draw(image_path, model):
#     """
#     使用YOLO模型检测图片中的对象，并绘制边界框（不包含标签）。

#     :param image_path: 输入图片的路径
#     :param model: 已加载的YOLO模型
#     """
#     # 执行预测
#     results = model(image_path)

#     # 读取原始图片用于绘图
#     img = cv2.imread(image_path)
#     print(results)

#     for result in results:
#         # 获取每个检测到的对象的边界框信息
#         boxes = result.boxes.cpu().numpy()
#         for box in boxes:
#             # 绘制边界框
#             r = box.xyxy
#             cv2.rectangle(img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0, 255, 0), 2)

#     # 显示最终图像
#     # cv2.imshow('Detected Objects', img)
#     cv2.imwrite("detect_result.jpg", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # 设置模型路径和待检测图片路径
#     model_path = './runs/obb/train51/weights/best.pt'  # 根据实际情况调整模型路径
#     # model_path = './yolo11n.pt'  # 根据实际情况调整模型路径
#     image_path = './datasets/yourself_datasets/images/test/110-06(test).jpg'  # 替换为你的图片路径

#     # 加载模型
#     model = load_model(model_path)
#     print(model)
#     # 进行检测并绘制结果
#     detect_and_draw(image_path, model)

import cv2
import numpy as np

from ultralytics import YOLO

# 加载模型
model = YOLO(r"./runs/obb/train60/weights/best.pt")

# 推理（关闭自动保存）
results = model.predict(
    r"./datasets/yourself_datasets/images/test/15-07(test).jpg",
    imgsz=1024,
    conf=0.5,
    save=False,  # 我们自己保存
)

# 读取图像
img_path = r"./datasets/yourself_datasets/images/test/15-07(test).jpg"
im = cv2.imread(img_path)

# 设置颜色和线宽（BGR格式）
COLOR = (0, 255, 0)  # 绿色
THICKNESS = 3

for result in results:
    if result.obb is not None:
        obb_data = result.obb.data.cpu().numpy()  # [cx, cy, w, h, angle_rad, conf, cls]

        # 直接读取绝对坐标
        for *xywhr, conf, cls in obb_data:
            cx, cy, w, h, angle_rad = xywhr

            # angle 转为 OpenCV 所需的度数（注意负号调整方向）
            angle_deg = -angle_rad * 180 / np.pi  # 注意：有时需加负号

            rect = ((cx, cy), (w, h), angle_deg)
            box_points = cv2.boxPoints(rect)
            box_points = np.int32(box_points)

            cv2.drawContours(im, [box_points], 0, (0, 0, 139), 2)

            # （可选）添加类别和置信度标签
            # label = f"{int(cls)} {conf:.2f}"
            # cv2.putText(im, label, (box_points[0][0], box_points[0][1] - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR, 2)

# 保存图像
cv2.imwrite("obb_result_custom_color.jpg", im)
