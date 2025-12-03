"""
0. Code that make binary masks per region
"""

import os
import logging
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms

from models.bisenet import BiSeNet
from utils.common import vis_parsing_maps

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)



class Config:
    """Configuration class to replace argparse"""
    def __init__(self):
        # Set your parameters here
        self.model = "resnet34"
        self.weight = "/data2/jiyoon/PAI-Bench/ckpts/resnet34.pt"
        self.input = "/data2/jiyoon/PAI-Bench/data/datasets_final/positive_pair/cropped/2"
        self.colored_segmentation_output = "/data2/jiyoon/PAI-Bench/data/datasets_final/positive_pair/features/fgis/2/colored_segmentation_output"
        self.binary_mask_output = "/data2/jiyoon/PAI-Bench/data/datasets_final/positive_pair/features/fgis/2/binary_mask_output"


def prepare_image(image: Image.Image, input_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
    """
    Prepare an image for inference by resizing and normalizing it.

    Args:
        image: PIL Image to process
        input_size: Target size for resizing

    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input
    """
    # Resize the image
    resized_image = image.resize(input_size, resample=Image.BILINEAR)

    # Define transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Apply transformations
    image_tensor = transform(resized_image)
    image_batch = image_tensor.unsqueeze(0)

    return image_batch


def load_model(model_name: str, num_classes: int, weight_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load and initialize the BiSeNet model.

    Args:
        model_name: Name of the backbone model (e.g., "resnet18")
        num_classes: Number of segmentation classes
        weight_path: Path to the model weights file
        device: Device to load the model onto

    Returns:
        torch.nn.Module: Initialized and loaded model
    """
    model = BiSeNet(num_classes, backbone_name=model_name)
    model.to(device)

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        raise ValueError(f"Weights not found from given path ({weight_path})")

    model.eval()
    return model


def get_files_to_process(input_path: str) -> List[str]:
    """
    Get a list of image files to process based on the input path.

    Args:
        input_path: Path to a single image file or directory of images

    Returns:
        List[str]: List of file paths to process
    """
    if os.path.isfile(input_path):
        return [input_path]

    # Get all files from the directory
    files = [os.path.join(input_path, f) for f in os.listdir(input_path)]

    # Filter for image files only
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    return [f for f in files if os.path.isfile(f) and f.lower().endswith(image_extensions)]

@torch.no_grad()
def inference(params: Config) -> None:
    """
    Run inference on images using the face parsing model.

    Args:
        params: Configuration object containing required parameters
    """
    cs_path = params.colored_segmentation_output
    os.makedirs(cs_path, exist_ok=True) 
    
    bm_path = params.binary_mask_output
    os.makedirs(bm_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    num_classes = 19  # Number of face parsing classes

    # Load the model
    try:
        model = load_model(params.model, num_classes, params.weight, device)
        logger.info(f"Model loaded successfully: {params.model}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Get list of files to process
    files_to_process = get_files_to_process(params.input)
    logger.info(f"Found {len(files_to_process)} files to process")

    # Process each file
    for file_path in tqdm(files_to_process, desc="Processing images"):
        filename = os.path.basename(file_path)

        try:
            # Load and process the image
            image = Image.open(file_path).convert("RGB")

            # Store original image resolution
            original_size = image.size  # (width, height)

            # Prepare image for inference
            image_batch = prepare_image(image).to(device)

            # Run inference
            output = model(image_batch)[0]  # feat_out, feat_out16, feat_out32 -> use feat_out for inference only
            predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)

            # Convert mask to PIL Image for resizing
            mask_pil = Image.fromarray(predicted_mask.astype(np.uint8))

            # Resize mask back to original image resolution
            restored_mask = mask_pil.resize(original_size, resample=Image.NEAREST)

            # Save the raw mask
            # mask_pil.save(save_raw_path)

            # Convert back to numpy array
            predicted_mask = np.array(restored_mask)

            # Visualize and save the results
            vis_parsing_maps(
                image,
                predicted_mask,
                save_image=True,
                cs_path=cs_path,
                bm_path=bm_path,
                filename=filename
            )

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue

    logger.info(f"Processing complete. Results saved to {cs_path} and {bm_path}.")


def validate_config(config: Config) -> None:
    """
    Validate configuration parameters.

    Args:
        config: Configuration object to validate
    """
    # Validate arguments
    if not os.path.exists(config.input):
        raise ValueError(f"Input path does not exist: {config.input}")

    if not os.path.exists(os.path.dirname(config.weight)):
        logger.warning(f"Weight directory does not exist: {os.path.dirname(config.weight)}")


def main() -> None:
    """Main entry point of the script."""
    try:
        config = Config()
        validate_config(config)
        inference(params=config)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
