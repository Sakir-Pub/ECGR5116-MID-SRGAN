import os
import shutil
from pathlib import Path
from tqdm import tqdm
import torch
from torchvision.datasets import ImageFolder
from PIL import Image

def consolidate_images(source_dirs, output_dir, supported_formats=('.jpg', '.jpeg', '.png', '.bmp')):
    """
    Consolidates images from multiple directories into a single output directory.
    
    Args:
        source_dirs (list): List of source directory paths containing images
        output_dir (str): Path to output directory where images will be copied
        supported_formats (tuple): Tuple of supported image file extensions
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Counter for unique filename generation
    file_counter = 0
    
    # Process each source directory
    for source_dir in source_dirs:
        print(f"Processing directory: {source_dir}")
        
        # Walk through all subdirectories
        for root, _, files in os.walk(source_dir):
            for file in tqdm(files):
                if file.lower().endswith(supported_formats):
                    # Generate source and destination paths
                    source_path = os.path.join(root, file)
                    
                    # Extract original extension
                    _, ext = os.path.splitext(file)
                    
                    # Create new filename with counter to avoid conflicts
                    new_filename = f"image_{file_counter}{ext}"
                    dest_path = os.path.join(output_dir, new_filename)
                    
                    # Copy file to destination
                    try:
                        # Verify if it's a valid image file
                        with Image.open(source_path) as img:
                            img.verify()
                        
                        shutil.copy2(source_path, dest_path)
                        file_counter += 1
                    except Exception as e:
                        print(f"Error processing {source_path}: {str(e)}")

    print(f"Successfully consolidated {file_counter} images to {output_dir}")
    return file_counter

if __name__ == "__main__":
    # Define source directories
    dataset_root = "/data1/Sakir/OCT_128x128"  # Replace with your dataset path
    source_dirs = [
        os.path.join(dataset_root, "train"),
        os.path.join(dataset_root, "valid")
    ]
    
    # Define output directory
    output_dir = "oct_aggregated"  # Replace with your desired output path
    
    # Consolidate images
    total_images = consolidate_images(source_dirs, output_dir)
    print(f"Total images processed: {total_images}")