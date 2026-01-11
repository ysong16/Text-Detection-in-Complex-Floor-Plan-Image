# YOLOv11-CFPI: Efficient Text Detection in Complex Floor Plan Images

![image](https://github.com/ysong16/Text-Detection-in-Complex-Floor-Plan-Image/blob/55ec4899e6b58ce1b8df3f272245b0bb1d41c6ac/datasets/yourself_datasets/complex%20floor%20plan%20image.jpg)
## Description

Our proposed model aims to address the challenges of detecting text in complex diagram images, such as small font sizes, extreme aspect ratios and cluttered backgrounds.

## Getting Started

### Dependencies

```
pip install ultralytics
```

### Datasets

Please download the dataset folder (yourself_datasets) from the link and put it in the folder datasets:

https://drive.google.com/drive/folders/1ELDAV9c6mB4gsqtWE0HF41qAwez7thzJ?usp=share_link

### Executing program

Download yolo11s-obb.pt from the link: https://docs.ultralytics.com/tasks/obb/#models

Train:
```
yolo obb train data=./datasets/yourself_datasets/yourself_datasets.yaml model=yolo11s-obb.pt
```
Test:
```
python evaluate_obb.py --model runs/obb/train/weights/best.pt --data datasets/yourself_datasets/yourself_data.yaml --output result.json
```

## Authors

By Song Yuhang, Cheng Li, and Mandal Mrinal
