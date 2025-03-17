#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys

# Add parent directory to path to import from uls_sam_mlx
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uls_sam_mlx.convert_weights import convert_torch_to_mlx

def main():
    parser = argparse.ArgumentParser(description="Convert SAM2.1 weights from PyTorch to MLX format")
    parser.add_argument("--weights-dir", "-d", type=str, default="weights", help="Directory containing PyTorch weights")
    parser.add_argument("--model", "-m", type=str, default=None, help="Specific model to convert (e.g., sam2_base_plus.pt)")
    args = parser.parse_args()
    
    weights_dir = Path(args.weights_dir)
    
    if not weights_dir.exists():
        print(f"Error: Weights directory {weights_dir} does not exist")
        return 1
    
    if args.model:
        # Convert specific model
        model_path = weights_dir / args.model
        if not model_path.exists():
            print(f"Error: Model file {model_path} does not exist")
            return 1
        
        # Generate output path
        output_path = model_path.with_suffix(".mlx")
        
        # Convert the model
        success = convert_torch_to_mlx(model_path, output_path)
        
        if not success:
            return 1
    else:
        # Convert all .pt and .pth files in the directory
        pt_files = list(weights_dir.glob("*.pt")) + list(weights_dir.glob("*.pth"))
        
        if not pt_files:
            print(f"No PyTorch weight files found in {weights_dir}")
            return 1
        
        for pt_file in pt_files:
            # Generate output path
            output_path = pt_file.with_suffix(".mlx")
            
            # Convert the model
            success = convert_torch_to_mlx(pt_file, output_path)
            
            if not success:
                print(f"Failed to convert {pt_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())