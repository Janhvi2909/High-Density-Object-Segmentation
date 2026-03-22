#!/usr/bin/env python3
"""
Interactive Gradio demo for High-Density Object Segmentation.

Launch with: python scripts/demo.py
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio not installed. Run: pip install gradio")


def load_model(model_type: str):
    """Load specified model."""
    print(f"Loading {model_type} model...")

    if model_type == "yolov8":
        from src.models.deep_learning.yolov8_seg import YOLOv8Segmenter
        return YOLOv8Segmenter(model_size='m')
    elif model_type == "maskrcnn":
        from src.models.deep_learning.mask_rcnn import MaskRCNNSegmenter
        return MaskRCNNSegmenter()
    elif model_type == "hybrid":
        from src.models.hybrid.density_aware import DensityAdaptiveSegmenter
        return DensityAdaptiveSegmenter()
    elif model_type == "baseline":
        from src.models.baseline.thresholding import AdaptiveThresholdSegmenter
        return AdaptiveThresholdSegmenter()
    elif model_type == "watershed":
        from src.models.advanced.watershed import MarkerControlledWatershed
        return MarkerControlledWatershed()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_demo(model_type: str = "hybrid", share: bool = False):
    """Create and launch Gradio demo."""
    if not GRADIO_AVAILABLE:
        print("Gradio is required for the demo. Install with: pip install gradio")
        return

    from src.visualization.segmentation_viz import draw_segmentation

    # Load model
    try:
        model = load_model(model_type)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Using placeholder demo...")
        model = None

    def segment_image(image, conf_threshold):
        """Process uploaded image."""
        if image is None:
            return None, "Please upload an image"

        if model is None:
            # Placeholder: draw random boxes
            h, w = image.shape[:2]
            n_objects = np.random.randint(10, 50)
            bboxes = []
            for _ in range(n_objects):
                x1 = np.random.randint(0, w - 50)
                y1 = np.random.randint(0, h - 50)
                x2 = x1 + np.random.randint(30, 100)
                y2 = y1 + np.random.randint(40, 150)
                bboxes.append([x1, y1, x2, y2])

            result = {
                'masks': np.zeros((n_objects, h, w)),
                'bboxes': np.array(bboxes),
                'scores': np.random.uniform(0.5, 1.0, n_objects)
            }
        else:
            # Run model
            result = model.predict(image)

        # Filter by confidence
        if 'scores' in result and len(result['scores']) > 0:
            keep = result['scores'] >= conf_threshold
            result['masks'] = result['masks'][keep] if len(result['masks']) > 0 else result['masks']
            result['bboxes'] = result['bboxes'][keep]
            result['scores'] = result['scores'][keep]

        # Visualize
        output = draw_segmentation(
            image,
            result.get('masks', np.zeros((0, *image.shape[:2]))),
            result.get('bboxes', np.zeros((0, 4))),
            result.get('scores', None),
            alpha=0.4
        )

        info = f"Detected {len(result.get('bboxes', []))} objects"
        return output, info

    # Create Gradio interface
    with gr.Blocks(title="High-Density Object Segmentation") as demo:
        gr.Markdown("""
        # High-Density Object Segmentation Demo

        Upload an image of densely packed objects (e.g., retail shelf) to segment individual items.

        **Model**: {}
        """.format(model_type.upper()))

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="numpy")
                conf_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Confidence Threshold"
                )
                submit_btn = gr.Button("Segment", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Segmentation Result")
                info_text = gr.Textbox(label="Detection Info")

        # Examples
        gr.Markdown("### Examples")
        gr.Examples(
            examples=[
                ["examples/retail_shelf_1.jpg", 0.5],
                ["examples/retail_shelf_2.jpg", 0.5],
            ],
            inputs=[input_image, conf_slider],
            outputs=[output_image, info_text],
            fn=segment_image,
            cache_examples=False
        )

        submit_btn.click(
            fn=segment_image,
            inputs=[input_image, conf_slider],
            outputs=[output_image, info_text]
        )

    # Launch
    demo.launch(share=share)


def main():
    parser = argparse.ArgumentParser(description="High-Density Object Segmentation Demo")
    parser.add_argument(
        "--model", type=str, default="hybrid",
        choices=["hybrid", "yolov8", "maskrcnn", "baseline", "watershed"],
        help="Model to use for segmentation"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create shareable link"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to run server on"
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("High-Density Object Segmentation Demo")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Share: {args.share}")
    print(f"{'='*60}\n")

    create_demo(args.model, args.share)


if __name__ == "__main__":
    main()
