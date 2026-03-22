#!/usr/bin/env python3
"""
Download SKU-110K dataset.

Usage:
    python scripts/download_data.py --dataset sku110k --output data/raw/
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.downloader import SKU110KDownloader


def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument(
        "--dataset", type=str, default="sku110k",
        choices=["sku110k"],
        help="Dataset to download"
    )
    parser.add_argument(
        "--output", type=str, default="data/raw",
        help="Output directory"
    )

    args = parser.parse_args()

    output_path = project_root / args.output

    if args.dataset == "sku110k":
        downloader = SKU110KDownloader(output_path)
        downloader.download()

        if downloader.verify_dataset():
            print("\nDataset download complete!")
            print("Converting to COCO format...")
            downloader.create_coco_format()
            print("Done!")
        else:
            print("\nDataset verification failed. Please check the download.")


if __name__ == "__main__":
    main()
