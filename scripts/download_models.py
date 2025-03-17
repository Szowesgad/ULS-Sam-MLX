#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys

# Add parent directory to path to import from uls_sam_mlx
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uls_sam_mlx.utils import download_model

# SAM2.1 model URLs from Meta
MODEL_URLS = {
    "sam2_b": "https://dl.fbaipublicfiles.com/segment_anything/sam2/sam2_b.pt",
    "sam2_l": "https://dl.fbaipublicfiles.com/segment_anything/sam2/sam2_l.pt",
    "sam2_h": "https://dl.fbaipublicfiles.com/segment_anything/sam2/sam2_h.pt",
    "sam2_base_plus": "https://dl.fbaipublicfiles.com/segment_anything/sam2/sam2_hiera_base_plus.pt",
}

def main():
    parser = argparse.ArgumentParser(description="Download SAM2.1 model weights")
    parser.add_argument(
        "--model", 
        type=str, 
        default="sam2_base_plus",
        choices=list(MODEL_URLS.keys()),
        help="Model variant to download"
    )
    parser.add_argument(
        "--output-dir", 
        "-o", 
        type=str, 
        default="weights",
        help="Directory to save model weights"
    )
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the model
    url = MODEL_URLS[args.model]
    save_path = output_dir / f"{args.model}.pt"
    
    try:
        download_model(url, save_path)
        print(f"\nModel downloaded successfully to {save_path}")
        print("\nTo convert to MLX format, run:")
        print(f"python -m uls_sam_mlx.convert_weights --input {save_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())