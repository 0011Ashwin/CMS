"""
Training Script for Calorimeter Super-Resolution using Real Parquet Dataset

This script trains the SRGAN on the QCDToGGQQ_IMGjet_RH1all jet dataset.

Dataset Info:
- LR images: (3, 64, 64) - 3 channels (ECAL, HCAL, TrackPt)
- HR images: (3, 125, 125)
- Scale factor: ~1.95x (64 → 125)
- Very sparse: ~2% non-zero pixels
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import argparse
from datetime import datetime

# Import our modules
from parquet_dataset import CachedParquetDataset, create_dataloader
from srgan_calorimeter import (
    CalorimeterGenerator, 
    CalorimeterDiscriminator,
    ContentLoss,
    AdversarialLoss,
    PerceptualLoss,
)

# Attempt visualization imports
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def setup_device():
    """Setup compute device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU (CUDA not available)")
    return device


def create_models(device, in_channels=3, scale_factor=2, target_size=(125, 125)):
    """Create Generator and Discriminator."""
    # Generator: 64 → (64*2=128) → resize to 125
    generator = CalorimeterGenerator(
        in_channels=in_channels,
        base_channels=64,
        num_residual_blocks=16,
        scale_factor=scale_factor,  # 2x upsampling: 64→128
        upsample_mode='pixelshuffle',
        target_size=target_size,    # Resize to exact 125x125
    ).to(device)
    
    # Discriminator for HR images (125x125)
    discriminator = CalorimeterDiscriminator(
        in_channels=in_channels,
        base_channels=64,
    ).to(device)
    
    return generator, discriminator


def create_losses(device, use_perceptual=False):
    """Setup loss functions."""
    content_loss = ContentLoss(
        l1_weight=1.0,
        mse_weight=1.0,
        energy_weight=0.5,  # Energy conservation
    )
    
    adversarial_loss = AdversarialLoss(gan_type='lsgan')
    
    perceptual_loss = None
    if use_perceptual:
        try:
            perceptual_loss = PerceptualLoss(device=device)
        except Exception as e:
            print(f"Warning: Could not create perceptual loss: {e}")
    
    return content_loss, adversarial_loss, perceptual_loss


def visualize_batch(lr_batch, hr_batch, sr_batch, epoch, save_dir):
    """Save visualization of LR, HR, and SR images."""
    if not HAS_MATPLOTLIB:
        return
    
    n_show = min(4, lr_batch.shape[0])
    fig, axes = plt.subplots(3, n_show, figsize=(4*n_show, 12))
    
    for i in range(n_show):
        # Channel 0 (e.g., ECAL)
        lr_img = lr_batch[i, 0].cpu().numpy()
        hr_img = hr_batch[i, 0].cpu().numpy()
        sr_img = sr_batch[i, 0].detach().cpu().numpy()
        
        vmin = min(lr_img.min(), hr_img.min(), sr_img.min())
        vmax = max(lr_img.max(), hr_img.max(), sr_img.max())
        
        axes[0, i].imshow(lr_img, cmap='hot', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'LR {lr_img.shape}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(sr_img, cmap='hot', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'SR {sr_img.shape}')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(hr_img, cmap='hot', vmin=vmin, vmax=vmax)
        axes[2, i].set_title(f'HR (GT) {hr_img.shape}')
        axes[2, i].axis('off')
    
    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch:03d}.png'), dpi=150)
    plt.close()


def compute_metrics(sr_batch, hr_batch):
    """Compute PSNR and energy conservation metrics."""
    with torch.no_grad():
        # MSE and PSNR
        mse = F.mse_loss(sr_batch, hr_batch)
        psnr = 10 * torch.log10(4.0 / (mse + 1e-8))  # range [-1,1] so max=4
        
        # Energy conservation (sum of all pixels per sample)
        sr_energy = sr_batch.sum(dim=(1, 2, 3))
        hr_energy = hr_batch.sum(dim=(1, 2, 3))
        energy_diff = torch.abs(sr_energy - hr_energy) / (torch.abs(hr_energy) + 1e-8)
        avg_energy_error = energy_diff.mean() * 100
        
    return {
        'psnr': psnr.item(),
        'mse': mse.item(),
        'energy_error_pct': avg_energy_error.item()
    }


def train_epoch(
    generator, discriminator, 
    train_loader, 
    g_optimizer, d_optimizer,
    content_loss_fn, adv_loss_fn, perceptual_loss_fn,
    device, epoch, args
):
    """Train for one epoch."""
    generator.train()
    discriminator.train()
    
    running_loss_g = 0.0
    running_loss_d = 0.0
    metrics_sum = {'psnr': 0.0, 'mse': 0.0, 'energy_error_pct': 0.0}
    n_batches = 0
    
    for batch_idx, (lr, hr) in enumerate(train_loader):
        lr = lr.to(device)
        hr = hr.to(device)
        batch_size = lr.shape[0]
        
        # =====================
        # Train Discriminator
        # =====================
        d_optimizer.zero_grad()
        
        with torch.no_grad():
            sr = generator(lr)
        
        real_pred = discriminator(hr)
        fake_pred = discriminator(sr.detach())
        
        # LSGAN loss for discriminator
        d_loss_real = adv_loss_fn(real_pred, target_is_real=True)
        d_loss_fake = adv_loss_fn(fake_pred, target_is_real=False)
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        d_loss.backward()
        d_optimizer.step()
        
        # =====================
        # Train Generator
        # =====================
        g_optimizer.zero_grad()
        
        sr = generator(lr)
        fake_pred = discriminator(sr)
        
        # Content loss (L1 + MSE + Energy conservation) - returns dict
        g_content_dict = content_loss_fn(sr, hr)
        g_content = g_content_dict['total']
        
        # Adversarial loss (generator wants discriminator to output 1)
        g_adv = adv_loss_fn(fake_pred, target_is_real=True)
        
        # Perceptual loss (if available)
        g_perceptual = 0.0
        if perceptual_loss_fn is not None:
            g_perceptual = perceptual_loss_fn(sr, hr) * args.perceptual_weight
        
        g_loss = g_content + args.adv_weight * g_adv + g_perceptual
        
        g_loss.backward()
        g_optimizer.step()
        
        # Track metrics
        running_loss_g += g_loss.item()
        running_loss_d += d_loss.item()
        
        batch_metrics = compute_metrics(sr, hr)
        for k in metrics_sum:
            metrics_sum[k] += batch_metrics[k]
        
        n_batches += 1
        
        # Progress
        if (batch_idx + 1) % args.log_interval == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"G_loss: {g_loss.item():.4f} D_loss: {d_loss.item():.4f} "
                  f"PSNR: {batch_metrics['psnr']:.2f}")
    
    # Average metrics
    avg_metrics = {k: v / n_batches for k, v in metrics_sum.items()}
    avg_g_loss = running_loss_g / n_batches
    avg_d_loss = running_loss_d / n_batches
    
    return avg_g_loss, avg_d_loss, avg_metrics


def pretrain_generator(
    generator, train_loader, 
    optimizer, content_loss_fn,
    device, epochs, args
):
    """Pre-train generator with content loss only (no adversarial)."""
    print("\n" + "="*60)
    print("Pre-training Generator (content loss only)")
    print("="*60)
    
    generator.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for lr, hr in train_loader:
            lr = lr.to(device)
            hr = hr.to(device)
            
            optimizer.zero_grad()
            sr = generator(lr)
            loss_dict = content_loss_fn(sr, hr)
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Pretrain Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
    
    print("Pre-training complete.\n")


def main():
    parser = argparse.ArgumentParser(description='Train SRGAN on Calorimeter Data')
    
    # Data
    parser.add_argument('--data-path', type=str, 
                        default=r'D:\GSoC kuberflow work',
                        help='Path to parquet files')
    parser.add_argument('--max-samples', type=int, default=10000,
                        help='Max samples to load into memory')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--pretrain-epochs', type=int, default=10,
                        help='Pre-training epochs (generator only)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr-g', type=float, default=1e-4,
                        help='Generator learning rate')
    parser.add_argument('--lr-d', type=float, default=1e-4,
                        help='Discriminator learning rate')
    
    # Loss weights
    parser.add_argument('--adv-weight', type=float, default=1e-3,
                        help='Adversarial loss weight')
    parser.add_argument('--perceptual-weight', type=float, default=0.0,
                        help='Perceptual loss weight (0 to disable)')
    
    # Logging
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Log every N batches')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Find parquet files
    data_path = Path(args.data_path)
    parquet_files = sorted(data_path.glob('*_LR.parquet'))
    
    if not parquet_files:
        print(f"No parquet files found in {data_path}")
        print("Looking for *_LR.parquet files...")
        sys.exit(1)
    
    print(f"\nFound {len(parquet_files)} parquet files:")
    for f in parquet_files:
        print(f"  - {f.name}")
    
    # Create dataset (using first file for now, or all files)
    print(f"\nLoading up to {args.max_samples} samples...")
    train_loader = create_dataloader(
        [str(f) for f in parquet_files],
        batch_size=args.batch_size,
        mode='cached',
        max_samples=args.max_samples,
        normalize=True,
    )
    
    print(f"Dataset size: {len(train_loader.dataset)} samples")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Create models
    # Data is 64→125, use scale_factor=2 (64→128) then resize to 125
    generator, discriminator = create_models(
        device, 
        in_channels=3, 
        scale_factor=2,  # 64 * 2 = 128 > 125
        target_size=(125, 125)  # Resize to exact target
    )
    
    print(f"\nGenerator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Create losses
    content_loss, adv_loss, perceptual_loss = create_losses(
        device, 
        use_perceptual=(args.perceptual_weight > 0)
    )
    
    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.9, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.9, 0.999))
    
    # Pre-train generator
    if args.pretrain_epochs > 0:
        pretrain_generator(
            generator, train_loader,
            g_optimizer, content_loss,
            device, args.pretrain_epochs, args
        )
    
    # Main training loop
    print("\n" + "="*60)
    print("Starting Adversarial Training")
    print("="*60)
    
    best_psnr = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        
        g_loss, d_loss, metrics = train_epoch(
            generator, discriminator,
            train_loader,
            g_optimizer, d_optimizer,
            content_loss, adv_loss, perceptual_loss,
            device, epoch, args
        )
        
        print(f"  G_loss: {g_loss:.4f}, D_loss: {d_loss:.4f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB, MSE: {metrics['mse']:.4f}")
        print(f"  Energy Error: {metrics['energy_error_pct']:.2f}%")
        
        # Visualize
        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():
                sample_lr, sample_hr = next(iter(train_loader))
                sample_lr = sample_lr.to(device)
                sample_hr = sample_hr.to(device)
                sample_sr = generator(sample_lr)
            
            visualize_batch(
                sample_lr, sample_hr, sample_sr, epoch,
                os.path.join(output_dir, 'visualizations')
            )
        
        # Save checkpoint
        if epoch % args.save_interval == 0 or metrics['psnr'] > best_psnr:
            if metrics['psnr'] > best_psnr:
                best_psnr = metrics['psnr']
                save_path = os.path.join(output_dir, 'best_model.pt')
            else:
                save_path = os.path.join(output_dir, f'checkpoint_epoch{epoch}.pt')
            
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'metrics': metrics,
            }, save_path)
            print(f"  Saved: {save_path}")
    
    # Final save
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
    }, os.path.join(output_dir, 'final_model.pt'))
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Models saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
