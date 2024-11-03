# AS-YOLO: Efficient Real-Time Apple Stem Segmentation

## Overview

AS-YOLO is based on the [YOLOv8 architecture](https://github.com/ultralytics/ultralytics), with efficient and enhanced real-time segmentation of fruits and their stems in agricultural environments. This model integrates Ghost Bottleneck and Global Attention Mechanism (GAM) to improve computational efficiency and accuracy in detecting small objects like stems, all while maintaining real-time processing speeds.

Key Features:

- **Ghost Bottleneck**: Reduces computational cost while maintaining feature richness.
- **Global Attention Mechanism (GAM)**: Improves focus on small objects and important regions within the input images.

## Table of Contents

1. Installation
2. Usage
3. Performance
4. AS-Seg Dataset
5. License

## Installation

Clone this repository and install the required dependencies using `pip`

```bash
pip install ultralytics
```

Ensure that you have a working GPU environment with CUDA support for faster training and inference.

## Usage

### Training

To train the AS-YOLO model :

```python
python train.py 
```

### Inference

To run inference using a trained AS-YOLO model:

```python
python test.py 
```

## Performance

AS-YOLO outperforms existing state-of-the-art models such as Mask R-CNN, YOLOv5, and YOLACT in terms of both speed and segmentation accuracy, especially in small object detection (e.g., stems). The model delivers:

- **mAP@50**: 0.674 (best for stem detection)

A full comparison with other models is available in Table 4 of the paper.

## AS-Seg Dataset

The **AS-Seg** dataset is custom-built for fruit and stem segmentation tasks.  It combines manually labeled portions of the [Fruit Recognition Dataset](https://zenodo.org/records/1310165) with proprietary data collected under various lighting conditions and environments.

- Open table setup
- Enclosed dark chamber
- Transparent box with turntable

If you would like to download the AS-Seg dataset, please read and follow the instructions in the [Agreement.md](./Agreement.md) file.

## Citation

If you use this code or dataset for your research, please cite:

```css
@article{ban2aru2024asyolo,
  title={AS-YOLO:An Improved YOLO by Ghost Bottleneck and Global Attention Mechanism for Apple Stem Segmentation},
  author={Baek, Na Rae Baek and Lee, Yeongwook and Cho, Se Woon and Noh, Dong-hee and Lee, Hea-Min},
  journal={Neural Computing and Applications},
  year={2024}
  note={submitted}
}

```

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](./LICENSE) file for details.
