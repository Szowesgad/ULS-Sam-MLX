import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import requests
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union

def download_model(url: str, save_path: Path) -> Path:
    """Download model weights from URL
    
    Args:
        url: URL to download from
        save_path: Path to save the file
        
    Returns:
        Path to the downloaded file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.exists():
        print(f"Model already exists at {save_path}")
        return save_path
    
    print(f"Downloading model from {url} to {save_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    file_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)
    
    with open(save_path, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    
    progress_bar.close()
    return save_path

def visualize_masks(image: np.ndarray, masks: List[np.ndarray], scores: List[float] = None) -> np.ndarray:
    """Visualize segmentation masks on the image
    
    Args:
        image: Original image
        masks: List of segmentation masks
        scores: Optional list of confidence scores
        
    Returns:
        Visualization image
    """
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Make sure image is in RGB format
    if len(vis_image.shape) == 2:  # Grayscale
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
    elif vis_image.shape[2] == 3 and vis_image[0,0,0] > vis_image[0,0,2]:  # BGR
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    
    # Normalize image if needed
    if vis_image.max() <= 1.0:
        vis_image = (vis_image * 255).astype(np.uint8)
    
    # Generate random colors for each mask
    colors = generate_colors(len(masks))
    
    # Apply each mask with a different color
    for i, mask in enumerate(masks):
        color = colors[i]
        vis_image[mask] = vis_image[mask] * 0.5 + np.array(color) * 0.5
        
        # Draw contour
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis_image, contours, -1, color, 2)
        
        # Add score text if available
        if scores and i < len(scores):
            # Find centroid of the mask
            M = cv2.moments(mask.astype(np.uint8))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Add score text
                cv2.putText(
                    vis_image,
                    f"{scores[i]:.2f}",
                    (cX, cY),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
    
    return vis_image

def generate_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate n distinct colors
    
    Args:
        n: Number of colors to generate
        
    Returns:
        List of RGB color tuples
    """
    colors = []
    for i in range(n):
        # Use HSV color space to generate evenly distributed colors
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
        value = 0.9
        
        # Convert HSV to RGB
        h = hue * 360
        s = saturation
        v = value
        
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        r = int((r + m) * 255)
        g = int((g + m) * 255)
        b = int((b + m) * 255)
        
        colors.append((r, g, b))
    
    return colors

def batch_process_directory(model, directory: Union[str, Path], output_dir: Union[str, Path] = None):
    """Process all images in a directory with the model
    
    Args:
        model: Initialized MLXSAMEngine
        directory: Directory containing images
        output_dir: Directory to save results (defaults to {directory}_results)
    """
    # Convert to Path objects
    directory = Path(directory)
    if output_dir is None:
        output_dir = directory.parent / f"{directory.name}_results"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(directory.glob(f"*{ext}")))
        image_files.extend(list(directory.glob(f"*{ext.upper()}")))
    
    # Process each image
    results = {}
    for img_path in tqdm(image_files, desc="Processing images"):
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Could not load image: {img_path}")
            continue
        
        # Get segmentation results
        seg_results = model.predict(image)
        
        # Visualize results
        vis_image = visualize_masks(image, seg_results["masks"], seg_results["scores"])
        
        # Save visualization
        output_path = output_dir / f"{img_path.stem}_segmented{img_path.suffix}"
        cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        # Save individual masks
        for i, mask in enumerate(seg_results["masks"]):
            mask_path = output_dir / f"{img_path.stem}_mask_{i}{img_path.suffix}"
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        
        # Store results
        results[str(img_path)] = {
            "num_masks": len(seg_results["masks"]),
            "scores": [float(s) for s in seg_results["scores"]]
        }
    
    # Print summary
    print(f"Processed {len(image_files)} images")
    print(f"Results saved to {output_dir}")
    
    return results