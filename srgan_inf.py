import os
import torch
from torchvision import transforms
from PIL import Image
from srgan import SRGAN_g  # Make sure this import matches your model architecture file
import numpy as np

class SRGANInference:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = SRGAN_g().to(device)
        
        # Load the trained generator weights
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def process_image(self, image_path):
        # Load and prepare image
        img = Image.open(image_path).convert('RGB')
        
        # Ensure input image is 32x32
        if img.size != (32, 32):
            img = img.resize((32, 32), Image.BICUBIC)
        
        # Transform image
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Generate high-res image
            output = self.model(img_tensor)
            
            # Denormalize
            output = output * 0.5 + 0.5
            output = output.clamp(0, 1)
            
            # Convert to image
            output = output.squeeze().cpu().numpy()
            output = np.transpose(output, (1, 2, 0)) * 255
            output = output.astype(np.uint8)
            
            return Image.fromarray(output)

def process_folder(input_folder, output_folder, model_path):
    """
    Process all images in the input folder and save results to output folder
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize the SRGAN inference
    srgan = SRGANInference(model_path)
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f'{filename}')
            
            try:
                # Process the image
                output_img = srgan.process_image(input_path)
                
                # Save the result
                output_img.save(output_path)
                print(f"Processed {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    input_folder = "/data1/Sakir/OCT_32x32/test/CNV"  # Folder containing 32x32 images
    output_folder = "/home/anabil/Development/PhD Courses/ECGR 5116/Mid/output_images"  # Folder where 128x128 images will be saved
    model_path = "/home/anabil/Development/PhD Courses/ECGR 5116/Mid/models/g_1.pth"  # Path to your trained generator model
    
    process_folder(input_folder, output_folder, model_path)