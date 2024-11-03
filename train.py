import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import config
import cv2
from srgan import SRGAN_g, SRGAN_d, VGG19FeatureExtractor

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (keeping same as original)
batch_size = 8
n_epoch_init = config.TRAIN.n_epoch_init
n_epoch = config.TRAIN.n_epoch

# Create folders to save results
save_dir = "samples"
checkpoint_dir = "models"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

class TrainDataset(Dataset):
    def __init__(self, hr_img_path):
        self.hr_image_files = [os.path.join(hr_img_path, f) for f in os.listdir(hr_img_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.hr_transforms = transforms.Compose([
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(),
        ])
        
        self.lr_transforms = transforms.Compose([
            transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC)
        ])
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        hr_img = Image.open(self.hr_image_files[index]).convert('RGB')
        
        # Apply transformations
        hr_img = self.hr_transforms(hr_img)
        lr_img = self.lr_transforms(hr_img)
        
        # Convert to tensors and normalize
        hr_tensor = self.normalize(hr_img)
        lr_tensor = self.normalize(lr_img)
        
        return lr_tensor, hr_tensor

    def __len__(self):
        return len(self.hr_image_files)

def train():
    # Initialize models
    G = SRGAN_g().to(device)
    D = SRGAN_d().to(device)
    vgg = VGG19FeatureExtractor().to(device)
    
    # Setup optimizers
    g_optimizer_init = optim.Adam(G.parameters(), lr=config.TRAIN.lr_init, betas=(config.TRAIN.beta1, 0.999))
    g_optimizer = optim.Adam(G.parameters(), lr=config.TRAIN.lr_init, betas=(config.TRAIN.beta1, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=config.TRAIN.lr_init, betas=(config.TRAIN.beta1, 0.999))
    
    # Loss functions
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    
    # Load dataset
    train_ds = TrainDataset(config.TRAIN.hr_img_path)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    
    # Initial training of generator
    print("Initial training of generator...")
    G.train()
    for epoch in range(n_epoch_init):
        for step, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            
            # Update G
            g_optimizer_init.zero_grad()
            fake_imgs = G(lr_imgs)
            g_init_loss = mse_loss(fake_imgs, hr_imgs)
            
            g_init_loss.backward()
            g_optimizer_init.step()
            
            if step % 10 == 0:
                print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f} g_init_loss: {:.3f}".format(
                    epoch, n_epoch_init, step, len(train_loader), time.time(), g_init_loss.item()))
    
    # Adversarial training
    print("Starting adversarial training...")
    for epoch in range(n_epoch):
        G.train()
        D.train()
        for step, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            batch_size = lr_imgs.size(0)
            
            # Update D
            d_optimizer.zero_grad()
            fake_imgs = G(lr_imgs)
            
            d_real, _ = D(hr_imgs)
            d_fake, _ = D(fake_imgs.detach())
            
            d_real_loss = bce_loss(d_real, torch.ones_like(d_real))
            d_fake_loss = bce_loss(d_fake, torch.zeros_like(d_fake))
            d_loss = d_real_loss + d_fake_loss
            
            d_loss.backward()
            d_optimizer.step()
            
            # Update G
            g_optimizer.zero_grad()
            fake_imgs = G(lr_imgs)
            d_fake_for_g, _ = D(fake_imgs)
            
            # Calculate VGG feature maps
            fake_features = vgg((fake_imgs + 1) / 2.0)
            real_features = vgg((hr_imgs + 1) / 2.0).detach()
            
            g_gan_loss = 1e-3 * bce_loss(d_fake_for_g, torch.ones_like(d_fake_for_g))
            g_mse_loss = mse_loss(fake_imgs, hr_imgs)
            g_vgg_loss = 2e-6 * mse_loss(fake_features, real_features)
            
            g_loss = g_mse_loss + g_vgg_loss + g_gan_loss
            
            g_loss.backward()
            g_optimizer.step()
            
            if step % 10 == 0:
                print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f} g_loss: {:.3f} d_loss: {:.3f}".format(
                    epoch, n_epoch, step, len(train_loader), time.time(), g_loss.item(), d_loss.item()))
        
        # Save models every 10 epochs
        if (epoch + 1) % 1 == 0:
            torch.save(G.state_dict(), os.path.join(checkpoint_dir, f'g_{epoch+1}.pth'))
            torch.save(D.state_dict(), os.path.join(checkpoint_dir, f'd_{epoch+1}.pth'))

def evaluate():
    # Load model
    G = SRGAN_g().to(device)
    G.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'g.pth')))
    G.eval()
    
    # Load test image
    valid_hr_img = Image.open(os.path.join(config.VALID.hr_img_path, os.listdir(config.VALID.hr_img_path)[0])).convert('RGB')
    hr_size = valid_hr_img.size
    
    # Prepare LR image
    valid_lr_img = valid_hr_img.resize((hr_size[0] // 4, hr_size[1] // 4), Image.BICUBIC)
    
    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    lr_tensor = transform(valid_lr_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = G(lr_tensor)
    
    # Save results
    def tensor_to_img(tensor):
        tensor = tensor.squeeze().cpu()
        tensor = tensor * 0.5 + 0.5  # Denormalize
        tensor = tensor.clamp(0, 1)
        tensor = tensor.permute(1, 2, 0).numpy() * 255
        return Image.fromarray(tensor.astype('uint8'))
    
    output_img = tensor_to_img(output)
    output_img.save(os.path.join(save_dir, 'valid_gen.png'))
    valid_lr_img.save(os.path.join(save_dir, 'valid_lr.png'))
    valid_hr_img.save(os.path.join(save_dir, 'valid_hr.png'))
    
    # Save bicubic upscaled version
    bicubic_img = valid_lr_img.resize((hr_size[0], hr_size[1]), Image.BICUBIC)
    bicubic_img.save(os.path.join(save_dir, 'valid_hr_cubic.png'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train, eval')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'eval':
        evaluate()
    else:
        raise Exception("Unknown --mode")