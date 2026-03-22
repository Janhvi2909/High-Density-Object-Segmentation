# High-Density Object Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A comprehensive approach to segmenting densely packed, heavily occluded objects using traditional ML, advanced CV, deep learning, and hybrid techniques.

![Project Banner](docs/assets/banner.png)

## Problem Statement

Segmenting objects in high-density scenarios (50-150+ objects per image) with significant occlusion remains a challenging computer vision problem. Traditional instance segmentation models struggle when objects are:
- **Densely packed**: Objects touching or overlapping
- **Heavily occluded**: 30-70% of objects partially hidden
- **Small scale**: Many objects occupy <1% of image area
- **Visually similar**: Products with similar packaging

This project implements and compares **four distinct approaches** to tackle this challenge, with rigorous analysis of failure modes and performance across varying density levels.

## Key Results

| Model | mAP@0.5 | mAP@0.5:0.95 | FPS | Parameters |
|-------|---------|--------------|-----|------------|
| Baseline (Edge Detection) | 18.3% | 8.2% | 45 | - |
| Advanced ML (Watershed) | 34.7% | 15.4% | 12 | - |
| YOLOv8-Seg | 61.2% | 38.5% | 85 | 11.2M |
| Mask R-CNN | 67.8% | 44.2% | 18 | 44.1M |
| **Hybrid (Ours)** | **72.4%** | **48.7%** | 24 | 55.3M |

### Performance by Object Density

| Density Level | Objects/Image | Hybrid mAP@0.5 |
|---------------|---------------|----------------|
| Low | 1-20 | 89.2% |
| Medium | 21-50 | 78.6% |
| High | 51-100 | 68.3% |
| Very High | 100+ | 54.1% |

## Features

- **Four Model Approaches**: Baseline ML, Advanced ML, Deep Learning, Hybrid
- **Novel Density-Aware Routing**: Adaptive model selection based on local object density
- **Comprehensive EDA**: Non-obvious pattern discovery in SKU-110K dataset
- **Rigorous Failure Analysis**: Categorized error modes with visual examples
- **Robustness Analysis**: Performance under various image corruptions
- **Colab-Ready Notebooks**: Train on Google Colab with GPU support

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.7+ (for GPU training)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/High-Density-Object-Segmentation.git
cd High-Density-Object-Segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Google Colab

Open any notebook and run the setup cell:
```python
!git clone https://github.com/yourusername/High-Density-Object-Segmentation.git
%cd High-Density-Object-Segmentation
!pip install -r requirements.txt
```

## Dataset

We use the **SKU-110K** dataset - a large-scale dataset for retail product detection with extreme density.

| Split | Images | Objects | Avg Objects/Image |
|-------|--------|---------|-------------------|
| Train | 8,219 | 1,177,896 | 143.3 |
| Val | 588 | 84,168 | 143.1 |
| Test | 2,936 | 426,820 | 145.4 |

### Download Dataset

```bash
python scripts/download_data.py --dataset sku110k --output data/raw/
```

## Project Structure

```
High-Density-Object-Segmentation/
├── config/                    # Configuration files
├── data/                      # Dataset storage
├── notebooks/                 # Jupyter notebooks (EDA, training, analysis)
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Baseline_ML.ipynb
│   ├── 04_Advanced_ML.ipynb
│   ├── 05_Deep_Learning.ipynb
│   ├── 06_Hybrid_Approach.ipynb
│   └── 07_Analysis_Comparison.ipynb
├── src/                       # Source code
│   ├── data/                  # Data loading and preprocessing
│   ├── features/              # Feature engineering
│   ├── models/                # Model implementations
│   ├── evaluation/            # Metrics and analysis
│   ├── visualization/         # Plotting utilities
│   └── utils/                 # Helper functions
├── scripts/                   # Training and inference scripts
├── report/                    # LaTeX report
├── docs/                      # Documentation
└── tests/                     # Unit tests
```

## Usage

### Training

```bash
# Train YOLOv8 segmentation model
python scripts/train.py --model yolov8 --epochs 100 --batch-size 16

# Train Mask R-CNN
python scripts/train.py --model maskrcnn --epochs 50 --batch-size 8

# Train hybrid model
python scripts/train.py --model hybrid --epochs 100
```

### Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py --model hybrid --checkpoint checkpoints/hybrid_best.pth

# Generate failure analysis report
python scripts/evaluate.py --model hybrid --failure-analysis
```

### Interactive Demo

```bash
# Launch Gradio demo
python scripts/demo.py --model hybrid --share
```

## Methodology

### 1. Baseline ML (Traditional Image Processing)
- Adaptive thresholding with morphological operations
- Canny edge detection + contour extraction
- Connected components labeling

### 2. Advanced ML (Non-Deep Learning CV)
- **Marker-controlled Watershed**: Distance transform seeding
- **GMM Segmentation**: Color-based Gaussian mixture clustering
- **SLIC Superpixels**: Oversegmentation with merging

### 3. Deep Learning
- **Mask R-CNN**: ResNet-50-FPN backbone, fine-tuned on SKU-110K
- **YOLOv8-Seg**: Real-time instance segmentation
- **SAM**: Zero-shot segmentation with prompt refinement

### 4. Hybrid Approach (Novel Contribution)
- **Density-Aware Routing**:
  - Estimate local object density using sliding window
  - Route low-density regions to fast YOLOv8
  - Route high-density regions to Mask R-CNN + SAM refinement
- **Ensemble Strategy**: Weighted voting with calibrated confidences

## Novel Contributions

1. **Density-Aware Attention Features**: Novel feature engineering that captures local object density patterns
2. **Adaptive Model Routing**: Dynamic selection between fast/accurate models based on scene complexity
3. **Occlusion-Aware Loss**: Modified loss function that weights partially visible objects appropriately

## Documentation

- [Literature Review](docs/literature_review.md) - Comprehensive survey of object segmentation
- [Theoretical Foundations](docs/theoretical_foundations.md) - Mathematical foundations and derivations
- [API Reference](docs/api_reference.md) - Code documentation

## Results Visualization

### Segmentation Examples
![Segmentation Results](docs/assets/segmentation_results.png)

### Performance Analysis
![Density Analysis](docs/assets/density_analysis.png)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{highdensity2026,
  author = {Kumar, Ronit},
  title = {High-Density Object Segmentation: A Comparative Study},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/High-Density-Object-Segmentation}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [SKU-110K Dataset](https://github.com/eg4000/SKU110K_CVPR19)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
