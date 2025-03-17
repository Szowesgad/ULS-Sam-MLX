#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt

# Add parent directory to path to import from uls_sam_mlx
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uls_sam_mlx import MLXSAMEngine, MLXSAMConfig
from uls_sam_mlx.utils import visualize_masks

def main():
    parser = argparse.ArgumentParser(description="Simple example of image segmentation with ULS-Sam-MLX")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", "-m", type=str, default="weights/sam2_base_plus.mlx", help="Path to MLX model weights")
    parser.add_argument("--output", "-o", type=str, help="Path to save visualization (optional)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Mask confidence threshold")
    args = parser.parse_args()
    
    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image {image_path} does not exist")
        return 1
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return 1
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model {model_path} does not exist")
        print("Did you download and convert the model weights? Try running:")
        print("python scripts/download_models.py")
        print("python scripts/convert_sam_weights.py")
        return 1
    
    # Initialize model
    print(f"Initializing model from {model_path}...")
    config = MLXSAMConfig(mask_threshold=args.threshold)
    model = MLXSAMEngine(model_path=model_path, config=config)
    
    # Run segmentation
    print("Running segmentation...")
    results = model.predict(image)
    
    # Print results
    print(f"Found {len(results['masks'])} segments with confidence threshold {args.threshold}")
    for i, score in enumerate(results['scores']):
        print(f"  Segment {i+1}: confidence {score:.4f}")
    
    # Visualize results
    print("Creating visualization...")
    vis_image = visualize_masks(image, results['masks'], results['scores'])
    
    # Save or display visualization
    if args.output:
        output_path = Path(args.output)
        print(f"Saving visualization to {output_path}...")
        cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    else:
        # Display with matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.title(f"SAM2.1 segmentation: {len(results['masks'])} segments")
        plt.tight_layout()
        plt.show()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())