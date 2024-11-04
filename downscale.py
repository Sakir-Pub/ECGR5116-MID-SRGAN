import os
from PIL import Image
import shutil
from pathlib import Path

def downscale_images(source_root, target_root, size=(32, 32)):
    """
    Downscale all JPEG images in source directory and its subdirectories to specified size.
    Maintains the original folder structure in the target directory.
    
    Args:
        source_root (str): Path to source root directory
        target_root (str): Path to target root directory
        size (tuple): Target size for images (width, height)
    """
    # Create target root if it doesn't exist
    Path(target_root).mkdir(parents=True, exist_ok=True)
    
    # Walk through all directories and files
    for root, dirs, files in os.walk(source_root):
        # Create corresponding directory structure in target
        relative_path = os.path.relpath(root, source_root)
        target_dir = os.path.join(target_root, relative_path)
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        # Process each file
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)
                
                try:
                    # Open, resize, and save image
                    with Image.open(source_path) as img:
                        # Convert to RGB if necessary
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        # Use LANCZOS resampling for better quality
                        resized_img = img.resize(size, Image.Resampling.LANCZOS)
                        resized_img.save(target_path, 'JPEG', quality=95)
                        print(f"Processed: {source_path}")
                except Exception as e:
                    print(f"Error processing {source_path}: {str(e)}")

def main():
    # Define source and target directories
    source_root = "/data1/Sakir/OCT/OCT2017 "  # Change this to your dataset path
    target_root = "OCT_32x32"  # Change this to your desired output path
    
    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        source_split_path = os.path.join(source_root, split)
        target_split_path = os.path.join(target_root, split)
        
        if os.path.exists(source_split_path):
            print(f"\nProcessing {split} split...")
            downscale_images(source_split_path, target_split_path)
        else:
            print(f"Warning: {split} directory not found in source")

if __name__ == "__main__":
    main()