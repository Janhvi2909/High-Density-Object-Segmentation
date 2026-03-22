# Data Directory

This directory contains the SKU-110K dataset for high-density object segmentation.

## Structure

```
data/
├── raw/                    # Original downloaded data
│   ├── images/            # Image files
│   ├── annotations/       # CSV annotations
│   └── coco_format/       # Converted COCO annotations
└── processed/              # Preprocessed data
```

## Downloading the Dataset

Run the download script:

```bash
python scripts/download_data.py --dataset sku110k --output data/raw/
```

Or manually download from: https://github.com/eg4000/SKU110K_CVPR19

## Dataset Statistics

| Split | Images | Objects | Avg Objects/Image |
|-------|--------|---------|-------------------|
| Train | 8,219 | 1,177,896 | 143.3 |
| Val | 588 | 84,168 | 143.1 |
| Test | 2,936 | 426,820 | 145.4 |

## Citation

```bibtex
@inproceedings{goldman2019precise,
  title={Precise Detection in Densely Packed Scenes},
  author={Goldman, Eran and Herzig, Roei and Eisenschtat, Aviv and Goldberger, Jacob and Hassner, Tal},
  booktitle={CVPR},
  year={2019}
}
```
