# ULS-Sam-MLX

High-performance SAM2.1 implementation for Apple Silicon using MLX framework. Optimized for ultrasonography image segmentation.

## Overview

ULS-Sam-MLX is a specialized implementation of Meta's Segment Anything Model 2.1 (SAM2.1) optimized for medical ultrasonography images. It leverages Apple's MLX framework to achieve maximum performance on Apple Silicon devices (M1, M2, M3 series).

## Features

- Native Apple Silicon optimization via MLX framework
- Specialized for medical ultrasonography segmentation
- Significantly faster than PyTorch-based implementations
- Memory-efficient unified architecture (no CPU-GPU transfer overhead)
- Simple API compatible with existing SAM2.1 implementations

## Requirements

- macOS 12.0+
- Apple Silicon Mac (M1, M2, M3 series)
- Python 3.9+
- MLX library (`pip install mlx`)

## Quick Start

```python
from uls_sam_mlx import MLXSAMEngine
from pathlib import Path
import cv2

# Initialize the model
model = MLXSAMEngine(model_path=Path("weights/sam2_base.mlx"))

# Load an ultrasonography image
image = cv2.imread("example.jpg")

# Get segmentation masks
results = model.predict(image)

# Process the masks
for mask, score in zip(results["masks"], results["scores"]):
    print(f"Found segment with confidence {score:.2f}")
    # Further processing...
```

## Performance

Performance comparison on M3 Max (16-core CPU, 40-core GPU, 48GB RAM):

| Model Implementation | Inference Time (ms) | Memory Usage (MB) |
|----------------------|---------------------|-------------------|
| PyTorch + MPS        | 150-200             | 2000-3000         |
| ULS-Sam-MLX          | 30-50               | 800-1200          |

## License

MIT

## Acknowledgements

- [Meta AI's SAM2.1](https://github.com/facebookresearch/sam2)
- [Apple's MLX Framework](https://github.com/ml-explore/mlx)
