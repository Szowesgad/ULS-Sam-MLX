import torch
import mlx.core as mx
import numpy as np
from pathlib import Path
import argparse
import os
from tqdm import tqdm

def convert_torch_to_mlx(torch_path: Path, mlx_path: Path):
    """Convert PyTorch SAM2.1 weights to MLX format
    
    Args:
        torch_path: Path to PyTorch weights (.pt or .pth)
        mlx_path: Path to save MLX weights (.mlx)
    """
    print(f"Converting {torch_path} to {mlx_path}")
    
    # Create output directory if it doesn't exist
    mlx_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load PyTorch weights
    try:
        # Use safe_load to avoid RCE vulnerabilities
        # This is crucial for security
        torch_weights = torch.load(
            torch_path, 
            map_location="cpu",
            weights_only=True  # Avoid loading any attached code
        )
        print("PyTorch weights loaded successfully")
    except Exception as e:
        print(f"Error loading PyTorch weights: {e}")
        return False
        
    # Extract state_dict if needed
    if 'model' in torch_weights:
        state_dict = torch_weights['model']
    else:
        state_dict = torch_weights
        
    # Convert to MLX format
    mlx_weights = {}
    
    # Process each tensor in the state dict
    print("Converting tensors...")
    for key, value in tqdm(state_dict.items()):
        # Convert PyTorch tensor to numpy
        np_tensor = value.cpu().numpy()
        
        # Convert numpy array to MLX array
        mlx_tensor = mx.array(np_tensor)
        
        # Store in new dict with same key
        mlx_weights[key] = mlx_tensor
    
    # Save MLX weights
    print(f"Saving MLX weights to {mlx_path}")
    mx.save(str(mlx_path), mlx_weights)
    print("Conversion complete!")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert SAM2.1 weights from PyTorch to MLX format")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to PyTorch weights (.pt)")
    parser.add_argument("--output", "-o", type=str, help="Path to save MLX weights (.mlx)")
    args = parser.parse_args()
    
    torch_path = Path(args.input)
    
    if not args.output:
        # Auto-generate output path if not provided
        mlx_path = torch_path.with_suffix(".mlx")
    else:
        mlx_path = Path(args.output)
    
    # Check if input file exists
    if not torch_path.exists():
        print(f"Error: Input file {torch_path} does not exist")
        return 1
    
    # Perform conversion
    success = convert_torch_to_mlx(torch_path, mlx_path)
    
    if success:
        print(f"Successfully converted {torch_path} to {mlx_path}")
        return 0
    else:
        print(f"Failed to convert {torch_path}")
        return 1

if __name__ == "__main__":
    main()