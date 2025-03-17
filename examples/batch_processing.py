#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys
import json

# Add parent directory to path to import from uls_sam_mlx
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uls_sam_mlx import MLXSAMEngine, MLXSAMConfig
from uls_sam_mlx.utils import batch_process_directory

def main():
    parser = argparse.ArgumentParser(description="Batch process images with ULS-Sam-MLX")
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output-dir", "-o", type=str, help="Directory to save results (optional)")
    parser.add_argument("--model", "-m", type=str, default="weights/sam2_base_plus.mlx", help="Path to MLX model weights")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Mask confidence threshold")
    args = parser.parse_args()
    
    # Check input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory {input_dir} does not exist or is not a directory")
        return 1
    
    # Check output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_results"
    
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
    
    # Process images
    print(f"Processing images in {input_dir}...")
    results = batch_process_directory(model, input_dir, output_dir)
    
    # Save results summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    print(f"Summary saved to {summary_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())