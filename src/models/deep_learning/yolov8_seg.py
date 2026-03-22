"""
YOLOv8 for instance segmentation.

Fast, efficient deep learning model using Ultralytics YOLO.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class YOLOv8Segmenter:
    """
    YOLOv8 instance segmentation wrapper.

    Fast inference suitable for real-time applications.
    Expected performance: High (55-65% mAP) with fast inference.
    """

    def __init__(
        self,
        model_size: str = 'm',
        checkpoint_path: Optional[str] = None,
        device: str = 'auto',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300
    ):
        """
        Initialize YOLOv8 segmenter.

        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            checkpoint_path: Path to custom checkpoint
            device: Device ('auto', 'cpu', '0', '0,1', etc.)
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
            max_det: Maximum detections per image
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics is required for YOLOv8")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.device = device

        # Load model
        if checkpoint_path and Path(checkpoint_path).exists():
            self.model = YOLO(checkpoint_path)
        else:
            # Load pretrained model
            model_name = f'yolov8{model_size}-seg.pt'
            self.model = YOLO(model_name)

    def predict(
        self,
        image: Union[np.ndarray, str, Path],
        verbose: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on a single image.

        Args:
            image: Input image (numpy array or path)
            verbose: Print verbose output

        Returns:
            Dictionary with masks, bboxes, scores, and labels
        """
        # Run inference
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            device=self.device,
            verbose=verbose
        )[0]

        # Extract results
        if results.masks is None:
            return {
                'masks': np.zeros((0, image.shape[0] if isinstance(image, np.ndarray) else 640,
                                  image.shape[1] if isinstance(image, np.ndarray) else 640)),
                'bboxes': np.zeros((0, 4)),
                'scores': np.zeros((0,)),
                'labels': np.zeros((0,), dtype=np.int64)
            }

        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy().astype(np.int64)

        # Resize masks to original image size
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            import cv2
            resized_masks = []
            for mask in masks:
                resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                resized_masks.append(resized)
            masks = np.array(resized_masks)

        return {
            'masks': masks,
            'bboxes': boxes,
            'scores': scores,
            'labels': labels
        }

    def predict_batch(
        self,
        images: List[Union[np.ndarray, str, Path]],
        batch_size: int = 16
    ) -> List[Dict[str, np.ndarray]]:
        """
        Run inference on multiple images.

        Args:
            images: List of images
            batch_size: Batch size for inference

        Returns:
            List of prediction dictionaries
        """
        all_results = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            results = self.model.predict(
                source=batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                device=self.device,
                verbose=False
            )

            for result, img in zip(results, batch):
                if result.masks is None:
                    h = img.shape[0] if isinstance(img, np.ndarray) else 640
                    w = img.shape[1] if isinstance(img, np.ndarray) else 640
                    all_results.append({
                        'masks': np.zeros((0, h, w)),
                        'bboxes': np.zeros((0, 4)),
                        'scores': np.zeros((0,)),
                        'labels': np.zeros((0,), dtype=np.int64)
                    })
                else:
                    masks = result.masks.data.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    labels = result.boxes.cls.cpu().numpy().astype(np.int64)

                    all_results.append({
                        'masks': masks,
                        'bboxes': boxes,
                        'scores': scores,
                        'labels': labels
                    })

        return all_results

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch_size: int = 16,
        project: str = 'runs/segment',
        name: str = 'train',
        pretrained: bool = True,
        **kwargs
    ) -> None:
        """
        Train YOLOv8 segmentation model.

        Args:
            data_yaml: Path to data configuration YAML
            epochs: Number of epochs
            imgsz: Image size
            batch_size: Batch size
            project: Project directory
            name: Run name
            pretrained: Use pretrained weights
            **kwargs: Additional training arguments
        """
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=project,
            name=name,
            pretrained=pretrained,
            device=self.device,
            **kwargs
        )

    def export(
        self,
        format: str = 'onnx',
        imgsz: int = 640,
        half: bool = False
    ) -> str:
        """
        Export model to different format.

        Args:
            format: Export format ('onnx', 'torchscript', 'engine')
            imgsz: Image size
            half: Use FP16

        Returns:
            Path to exported model
        """
        return self.model.export(format=format, imgsz=imgsz, half=half)

    def benchmark(
        self,
        image: Union[np.ndarray, str],
        runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.

        Args:
            image: Test image
            runs: Number of inference runs

        Returns:
            Dictionary with timing statistics
        """
        import time

        # Warmup
        for _ in range(10):
            self.predict(image, verbose=False)

        # Benchmark
        times = []
        for _ in range(runs):
            start = time.time()
            self.predict(image, verbose=False)
            times.append(time.time() - start)

        times = np.array(times)
        return {
            'mean_ms': times.mean() * 1000,
            'std_ms': times.std() * 1000,
            'fps': 1.0 / times.mean()
        }
