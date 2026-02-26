import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path to allow imports from existing modules
src_path = Path(__file__).parent / "MAEs/Hybrid_Transformer_Thanh_Nguyen/src"
sys.path.append(str(src_path))

try:
    from models.lorentz_gatr import LGATrEncoder
    from models.particle_transformer import ParticleAttentionBlock
except ImportError:
    # Mocking for standalone functionality if dependencies aren't installed/found
    print("Warning: Could not import LGATrEncoder or ParticleAttentionBlock. Using mocks.")
    class LGATrEncoder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.linear = nn.Linear(4, 128) # Projects 4-vector to 128
        def forward(self, x, mask=None):
            return self.linear(x)

    class ParticleAttentionBlock(nn.Module):
        def __init__(self, embed_dim=128, **kwargs):
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
        def forward(self, x, padding_mask=None):
            out, _ = self.attn(x, x, x, key_padding_mask=padding_mask)
            return self.norm(x + out)

class HybridPhysicsMAE(nn.Module):
    def __init__(self, 
                 input_dim=4, 
                 embed_dim=128, 
                 num_heads=8, 
                 num_layers=4, 
                 num_classes=2, 
                 mask_ratio=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        
        # 1. Lorentz Encoder (L-GATr)
        # Input: (B, N, 4) -> Output: (B, N, 128)
        # Assuming LGATrEncoder takes standard args
        self.lorentz_encoder = LGATrEncoder(
            hidden_mv_channels=embed_dim // 16, # Adjust based on LGATr config logic
            hidden_s_channels=embed_dim
        )
        
        # Project to exact embed_dim if LGATr output is different, or assume it matches.
        # Here we assume LGATr is configured to output `embed_dim` features.
        self.embed_proj = nn.Linear(embed_dim, embed_dim) # Optional alignment
        
        # Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # 2. Transformer Backbone (ParT)
        self.transformer_blocks = nn.ModuleList([
            ParticleAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        
        # 3. Heads
        # Reconstruction Head: Predicts 4-vectors (E, px, py, pz)
        self.recon_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, input_dim) 
        )
        
        # Classification Head: Jet Tagging
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
    def random_masking(self, x):
        """
        x: [B, N, D]
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        
        # Sort noise to get random indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first len_keep subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask for loss computation
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        # x: (B, N, 4)
        
        # L-GATr Embedding
        # Note: input might be expected as specific object for LGATr, 
        # but here we pass tensor assuming wrapper handles it or it's a mock
        x_embed = self.lorentz_encoder(x) # (B, N, D)
        x_embed = self.embed_proj(x_embed)
        
        # Masking
        x_masked, mask, ids_restore = self.random_masking(x_embed)
        
        # Append CLS token
        cls_token = self.cls_token.expand(x_masked.shape[0], -1, -1)
        x_cat = torch.cat((cls_token, x_masked), dim=1)
        
        # Transformer Backbone
        for block in self.transformer_blocks:
            # Gradient Checkpointing for efficiency
            if self.training:
                x_cat = torch.utils.checkpoint.checkpoint(block, x_cat)
            else:
                x_cat = block(x_cat)
                
        return x_cat, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # x: [B, N_masked + 1, D] (including CLS)
        
        # Separate CLS
        # x_cls = x[:, :1, :] # Not used for reconstruction usually, but used for classification
        x_vis = x[:, 1:, :]
        
        # Append mask tokens to sequence
        B, _, D = x_vis.shape
        # Total length original N
        # We need to restore full sequence length
        # Get number of masked tokens
        
        # Since we kept len_keep, we need to add back (N - len_keep) mask tokens
        # But wait, ids_restore tells us how to shuffle back.
        # We construct a full sequence where visible parts are x_vis and masked parts are mask_tokens
        
        N_original = ids_restore.shape[1]
        N_keep = x_vis.shape[1]
        
        mask_tokens = self.mask_token.repeat(B, N_original - N_keep + 0, 1) # +0 to be explicit
        x_ = torch.cat([x_vis, mask_tokens], dim=1)
        
        # Unshuffle to original order
        x_restored = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        
        # Reconstruct
        # Apply reconstruction head to ALL tokens (or just masked ones, but typically all)
        pred = self.recon_head(x_restored)
        return pred

    def forward(self, x):
        # x: (B, N, 4) 4-vectors
        
        # Encoder
        z, mask, ids_restore = self.forward_encoder(x)
        
        # Classification (using CLS token output from encoder)
        cls_logits = self.cls_head(z[:, 0])
        
        # Decoder / Reconstruction
        pred = self.forward_decoder(z, ids_restore)
        
        return pred, cls_logits, mask

def mae_loss(pred, target, mask):
    """
    pred: (B, N, 4)
    target: (B, N, 4)
    mask: (B, N), 0 is keep, 1 is remove
    """
    # MSE Loss
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1) # (B, N), mean over 4-vector dim
    
    # Only calculate loss on masked tokens
    loss = (loss * mask).sum() / mask.sum() # Mean over masked tokens
    return loss
