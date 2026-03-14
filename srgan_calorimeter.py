"""
Super-Resolution GAN for High Energy Physics Calorimeter Images
"""

from __future__ import annotations
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Tuple

# ==========================================
# GENERATOR (SRGAN Style)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, negative_slope: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.LeakyReLU(negative_slope, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, scale_factor: int = 2, negative_slope: float = 0.2):
        super().__init__()
        out_channels = in_channels * (scale_factor ** 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.LeakyReLU(negative_slope, inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class CalorimeterGenerator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_residual_blocks=8, scale_factor=2, target_size=(125, 125)):
        super().__init__()
        self.scale_factor = scale_factor
        self.target_size = target_size
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_residual_blocks)]
        )
        
        self.post_residual = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
        )
        
        self.upsample_layers = nn.Sequential()
        if scale_factor > 1:
            num_upsample = int(math.log2(scale_factor))
            for _ in range(num_upsample):
                self.upsample_layers.append(UpsampleBlock(base_channels, scale_factor=2))
        
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, kernel_size=9, padding=4),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_upsampled = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        feat = self.initial(x)
        residual = self.residual_blocks(feat)
        feat = feat + self.post_residual(residual)
        feat = self.upsample_layers(feat)
        out = self.final(feat)
        
        if out.shape[2:] != self.target_size:
            out = F.interpolate(out, size=self.target_size, mode='bilinear', align_corners=False)
            
        out = out + x_upsampled
        return torch.clamp(out, -1, 1)


# ==========================================
# DISCRIMINATOR (PatchGAN Style)
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class CalorimeterDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, base_channels, stride=1, use_bn=False),
            ConvBlock(base_channels, base_channels, stride=2, use_bn=True),
            ConvBlock(base_channels, base_channels * 2, stride=1, use_bn=True),
            ConvBlock(base_channels * 2, base_channels * 2, stride=2, use_bn=True),
            ConvBlock(base_channels * 2, base_channels * 4, stride=1, use_bn=True),
            ConvBlock(base_channels * 4, base_channels * 4, stride=2, use_bn=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(base_channels * 4 * 4 * 4, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ==========================================
# TRAINER & LOSSES
# ==========================================
class ContentLoss(nn.Module):
    def __init__(self, l1_weight=1.0, mse_weight=0.1, energy_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.energy_weight = energy_weight
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, generated, target):
        losses = {}
        total = torch.tensor(0.0, device=generated.device)
        if self.l1_weight > 0:
            l1 = self.l1_loss(generated, target)
            total += self.l1_weight * l1
        if self.mse_weight > 0:
            mse = self.mse_loss(generated, target)
            total += self.mse_weight * mse
        if self.energy_weight > 0:
            gen_energy = generated.sum(dim=(1, 2, 3))
            target_energy = target.sum(dim=(1, 2, 3))
            energy_loss = F.l1_loss(gen_energy, target_energy)
            total += self.energy_weight * energy_loss
        losses['total'] = total
        return losses

class SRGANTrainer:
    def __init__(self, generator, discriminator, train_loader, val_loader, dataset, lr_g=1e-4, lr_d=1e-4,
                 content_weight=1.0, adversarial_weight=0.001, perceptual_weight=0.0, 
                 pretrain_epochs=5, total_epochs=25, device="cuda"):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.content_weight = content_weight
        self.adversarial_weight = adversarial_weight
        self.pretrain_epochs = pretrain_epochs
        self.total_epochs = total_epochs
        
        self.optimizer_g = Adam(generator.parameters(), lr=lr_g, betas=(0.9, 0.999))
        self.optimizer_d = Adam(discriminator.parameters(), lr=lr_d, betas=(0.9, 0.999))
        self.content_loss_fn = ContentLoss()
        self.adv_loss_fn = nn.MSELoss() # LSGAN
        
        self.history = {'train_g_loss': [], 'train_d_loss': [], 'val_psnr': []}

    def train(self):
        print(f"Starting Training on {self.device}...")
        for epoch in range(self.total_epochs):
            self.generator.train()
            self.discriminator.train()
            is_adv = epoch >= self.pretrain_epochs
            epoch_g_loss, epoch_d_loss = 0.0, 0.0
            
            for batch_idx, (lr, hr) in enumerate(self.train_loader):
                lr, hr = lr.to(self.device), hr.to(self.device)
                
                # Train Discriminator
                if is_adv:
                    self.optimizer_d.zero_grad()
                    real_validity = self.discriminator(hr)
                    fake_hr = self.generator(lr)
                    fake_validity = self.discriminator(fake_hr.detach())
                    
                    d_loss_real = self.adv_loss_fn(real_validity, torch.ones_like(real_validity))
                    d_loss_fake = self.adv_loss_fn(fake_validity, torch.zeros_like(fake_validity))
                    d_loss = (d_loss_real + d_loss_fake) / 2
                    d_loss.backward()
                    self.optimizer_d.step()
                    epoch_d_loss += d_loss.item()
                
                # Train Generator
                self.optimizer_g.zero_grad()
                fake_hr = self.generator(lr)
                content_loss = self.content_loss_fn(fake_hr, hr)['total'] * self.content_weight
                g_loss = content_loss
                
                if is_adv:
                    fake_validity = self.discriminator(fake_hr)
                    adv_loss = self.adv_loss_fn(fake_validity, torch.ones_like(fake_validity))
                    g_loss += self.adversarial_weight * adv_loss
                    
                g_loss.backward()
                self.optimizer_g.step()
                epoch_g_loss += g_loss.item()
            
            # Validation
            self.generator.eval()
            val_psnr = 0.0
            with torch.no_grad():
                for lr, hr in self.val_loader:
                    lr, hr = lr.to(self.device), hr.to(self.device)
                    fake_hr = self.generator(lr)
                    mse = F.mse_loss(fake_hr, hr)
                    val_psnr += (10 * math.log10(4.0 / (mse.item() + 1e-8)))
            val_psnr /= len(self.val_loader)
            
            avg_g_loss = epoch_g_loss / len(self.train_loader)
            avg_d_loss = epoch_d_loss / len(self.train_loader) if is_adv else 0
            
            self.history['train_g_loss'].append(avg_g_loss)
            self.history['train_d_loss'].append(avg_d_loss)
            self.history['val_psnr'].append(val_psnr)
            
            print(f"Epoch {epoch+1}/{self.total_epochs} | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f} | Val PSNR: {val_psnr:.2f} dB")
            
        return self.history