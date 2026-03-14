
# Task 2b: SRGAN Training for High Energy Physics Calorimeter Images

## Overview
This task successfully implements and trains an SRGAN (Super-Resolution Generative Adversarial Network) model designed to upscale sparse High Energy Physics (HEP) calorimeter jet images from a low resolution (64x64) directly to high resolution (125x125). The training loop, dataset handling, architecture setup, and results were successfully realized on a cloud-based GPU environment (Google Colab with a Tesla T4).

## The Pipeline & Architecture
1. **Generator (`CalorimeterGenerator` in `srgan_calorimeter.py`)**  
   The SRGAN Generator is built around a Residual Network (ResNet) backbone with 16 residual blocks optimized for extreme sparsity (up to 95% zero pixels in HEP data). 
   - Uses **LeakyReLU** to maintain gradient flow on sparse inputs.
   - Upsampling is handled robustly via Sub-pixel convoluting (**PixelShuffle**) with an internal dynamic resizing module (via `target_size`) to scale from the non-power-of-two size mapping (64x64 -> 128x128 -> 125x125).

2. **Discriminator (`CalorimeterDiscriminator` in `srgan_calorimeter.py`)**  
   Employs a **PatchGAN** VGG-style spatial discriminator to ensure patches of the generated calorimeter deposits are indistinguishable from ground-truth HR deposits. This focus on "local realism" preserves small-scale sub-structure within jets instead of blurring them out.

3. **Multi-Component Loss Strategy**  
   - **Content Loss**: Robustly measures exact structural match taking advantage of **L1 Loss** combined with **MSE Loss** to penalize large localized structural gaps. 
   - **Physics-informed Energy Conservation Loss**: Monitors the integral over all pixels, guiding the network to ensure total energy input is accurately conserved within the high-res generated topology.
   - **Adversarial Loss**: Uses LSGAN (Least Squares GAN) objectives to train more stably on sparse inputs.

4. **Data Handling (`train_parquet.py`)**  
   Implemented lazy execution and resilient chunk loading out of massive `.parquet` datasets using `pyarrow`, providing memory-safe sequential loading.

## Training Results
Training was successfully executed over **25 Epochs**. The protocol executed a phased approach:
- Over successive epochs, the model stabilized its geometric mapping structures while injecting nuanced sub-jet localized texture information.

**Performance Metrics Evolution**:  
During validation over the dataset:
- Starting benchmark **PSNR** logged at **27.38 dB**.
- **Generator Loss** rapidly decreased and stabilized from >5.15 down to **0.26**.
- The model successfully achieved a **Peak PSNR of ~31.08 dB** by termination.

## Generated Artifacts
All computational resources have been packaged to allow for reproducible physics validation and inference testing locally or remotely:
- `srgan_calorimeter.py`: Definition models for the Generator and Discriminator Neural Networks.
- `train_parquet.py`: Main execution logic for batched dataset loading, metric calculation, and loss evaluation.
- `train_srgan_calorimeter.ipynb`: The Colab training environment tracking training evolutions, reporting validation metrics, plotting loss arrays, and displaying sample projections.
- `gsoc_srgan_calorimeter.pth`: The final binary checkpoint housing the fully trained 3-channel SRGAN weights (Large file tracking, excluded from direct source tracking where necessary).

