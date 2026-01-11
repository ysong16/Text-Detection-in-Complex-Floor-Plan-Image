# YOLOv11-CFPI: Efficient Text Detection in Complex Floor Plan Images

Text information detector for complex diagram images

## Description

Our proposed model aims to address the challenges of detecting text in complex diagram images, such as small font sizes, extreme aspect ratios and cluttered back-grounds.

## Getting Started

### Dependencies

```
pip install ultralytics
```

### Datasets

Please download the dataset folder (yourself_datasets) from the link:

https://drive.google.com/drive/folders/1ELDAV9c6mB4gsqtWE0HF41qAwez7thzJ?usp=share_link

### Executing program

Train:
```
yolo obb train data=./datasets/yourself_datasets/yourself_datasets.yaml model=yolo11n-obb.pt epochs=500
```
Test:
```
python evaluate_obb.py --model runs/obb/train/weights/best.pt --data datasets/yourself_datasets/yourself_data.yaml --output result.json
```
## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors



By Yuhang Song, Li Cheng, and Mrinal Mandal



## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)


