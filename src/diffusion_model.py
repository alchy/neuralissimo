"""
Conditional Diffusion Model (DDPM) for Electric Piano Sound Generation

This module implements a denoising diffusion probabilistic model (DDPM)
conditioned on MIDI note and velocity parameters for generating mel spectrograms.

Based on "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for timestep encoding"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: Timestep tensor [batch]
        Returns:
            embeddings: Sinusoidal embeddings [batch, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time and condition embedding"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 condition_dim: int, dropout: float = 0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        # Condition embedding projection
        self.cond_mlp = nn.Linear(condition_dim, out_channels)

        # Normalization
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor,
                cond_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, channels, height, width]
            time_emb: Time embedding [batch, time_emb_dim]
            cond_emb: Condition embedding [batch, condition_dim]
        Returns:
            Output tensor [batch, out_channels, height, width]
        """
        residual = x

        # First conv
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)

        # Add time embedding
        time_proj = self.time_mlp(time_emb)[:, :, None, None]
        x = x + time_proj

        # Add condition embedding
        cond_proj = self.cond_mlp(cond_emb)[:, :, None, None]
        x = x + cond_proj

        # Second conv
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)

        # Residual
        return x + self.residual_conv(residual)


class AttentionBlock(nn.Module):
    """Self-attention block for UNet"""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, channels, height, width]
        Returns:
            Output tensor [batch, channels, height, width]
        """
        batch, channels, height, width = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)

        # Reshape for multi-head attention
        qkv = qkv.reshape(batch, 3, self.num_heads, channels // self.num_heads, height * width)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, batch, heads, hw, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scale = (channels // self.num_heads) ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.permute(0, 1, 3, 2).reshape(batch, channels, height, width)
        out = self.proj(out)

        return out + residual


class DownBlock(nn.Module):
    """Downsampling block for UNet"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 condition_dim: int, use_attention: bool = False, num_heads: int = 4):
        super().__init__()

        self.res1 = ResidualBlock(in_channels, out_channels, time_emb_dim, condition_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_emb_dim, condition_dim)

        self.attention = AttentionBlock(out_channels, num_heads) if use_attention else nn.Identity()
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor,
                cond_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns output and skip connection"""
        x = self.res1(x, time_emb, cond_emb)
        x = self.res2(x, time_emb, cond_emb)
        x = self.attention(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block for UNet"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 condition_dim: int, use_attention: bool = False, num_heads: int = 4):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)

        # Note: in_channels + in_channels because of skip connection
        self.res1 = ResidualBlock(in_channels * 2, out_channels, time_emb_dim, condition_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_emb_dim, condition_dim)

        self.attention = AttentionBlock(out_channels, num_heads) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor,
                cond_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input from previous layer
            skip: Skip connection from encoder
        """
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, time_emb, cond_emb)
        x = self.res2(x, time_emb, cond_emb)
        x = self.attention(x)
        return x


class UNet(nn.Module):
    """
    UNet architecture for diffusion model

    Processes mel spectrograms with conditioning on timestep and MIDI/velocity
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 128,
        channel_mults: List[int] = [1, 2, 3, 4],
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        condition_dim: int = 128,
        use_attention: List[bool] = [False, False, True, True],
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Condition embedding (MIDI note + velocity)
        self.cond_embedding = nn.Sequential(
            nn.Linear(2, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim)
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            use_attn = use_attention[i] if i < len(use_attention) else False

            self.down_blocks.append(
                DownBlock(channels, out_ch, time_emb_dim, condition_dim, use_attn)
            )
            channels = out_ch

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(channels, channels, time_emb_dim, condition_dim),
            AttentionBlock(channels),
            ResidualBlock(channels, channels, time_emb_dim, condition_dim)
        ])

        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            in_ch = channels
            use_attn = use_attention[len(channel_mults) - 1 - i] if i > 0 else False

            self.up_blocks.append(
                UpBlock(in_ch, out_ch, time_emb_dim, condition_dim, use_attn)
            )
            channels = out_ch

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy input [batch, 1, n_mels, time]
            timesteps: Diffusion timesteps [batch]
            condition: Conditioning [batch, 2] (midi_note, velocity)
        Returns:
            Predicted noise [batch, 1, n_mels, time]
        """
        # Embeddings
        time_emb = self.time_embedding(timesteps)
        cond_emb = self.cond_embedding(condition)

        # Initial conv
        x = self.init_conv(x)

        # Encoder
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, time_emb, cond_emb)
            skips.append(skip)

        # Bottleneck
        for block in self.bottleneck:
            if isinstance(block, ResidualBlock):
                x = block(x, time_emb, cond_emb)
            else:
                x = block(x)

        # Decoder
        for up_block in self.up_blocks:
            skip = skips.pop()
            x = up_block(x, skip, time_emb, cond_emb)

        # Final conv
        x = self.final_conv(x)

        return x


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Process (DDPM)

    Implements the forward diffusion (adding noise) and reverse diffusion
    (denoising) processes.
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear"
    ):
        super().__init__()

        self.model = model
        self.timesteps = timesteps

        # Define beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Pre-compute diffusion parameters
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (not parameters)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion: sample from q(x_t | x_0)

        Args:
            x_start: Original clean data [batch, channels, height, width]
            t: Timesteps [batch]
            noise: Optional noise to add
        Returns:
            Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor,
                                  noise: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor,
                    t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior mean and variance"""
        posterior_mean = (
            self.posterior_mean_coef1[t][:, None, None, None] * x_start +
            self.posterior_mean_coef2[t][:, None, None, None] * x_t
        )
        posterior_variance = self.posterior_variance[t][:, None, None, None]

        return posterior_mean, posterior_variance

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor,
                       condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance of reverse process

        Args:
            x_t: Noisy input at timestep t
            t: Timesteps
            condition: Conditioning vector
        """
        # Predict noise
        pred_noise = self.model(x_t, t, condition)

        # Predict x_0
        x_start = self.predict_start_from_noise(x_t, t, pred_noise)
        x_start = torch.clamp(x_start, -1.0, 1.0)

        # Get posterior
        model_mean, model_variance = self.q_posterior(x_start, x_t, t)

        return model_mean, model_variance

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int, condition: torch.Tensor) -> torch.Tensor:
        """
        Single reverse diffusion step: sample from p(x_{t-1} | x_t)

        Args:
            x_t: Noisy input at timestep t
            t: Timestep (int)
            condition: Conditioning vector
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)

        # Get mean and variance
        model_mean, model_variance = self.p_mean_variance(x_t, t_tensor, condition)

        # No noise when t == 0
        if t == 0:
            return model_mean

        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(model_variance) * noise

    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], condition: torch.Tensor,
               device: str = 'cpu') -> torch.Tensor:
        """
        Generate samples from the model

        Args:
            shape: Shape of output (batch, channels, height, width)
            condition: Conditioning vector [batch, 2]
            device: Device to run on
        Returns:
            Generated samples
        """
        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Reverse diffusion process
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t, condition)

        return x

    def training_loss(self, x_start: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Compute training loss (simple MSE on noise prediction)

        Args:
            x_start: Clean data [batch, channels, height, width]
            condition: Conditioning vector [batch, 2]
        Returns:
            Loss value
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

        # Sample noise
        noise = torch.randn_like(x_start)

        # Get noisy input
        x_t = self.q_sample(x_start, t, noise)

        # Predict noise
        pred_noise = self.model(x_t, t, condition)

        # Simple MSE loss
        loss = F.mse_loss(pred_noise, noise)

        return loss


# Test
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create UNet
    unet = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=128,
        channel_mults=[1, 2, 3, 4],
        time_emb_dim=256,
        condition_dim=128
    ).to(device)

    # Create diffusion model
    diffusion = GaussianDiffusion(
        model=unet,
        timesteps=1000,
        beta_schedule="cosine"
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in unet.parameters()):,}")

    # Test forward pass
    batch_size = 4
    n_mels = 128
    time_frames = 281

    x = torch.randn(batch_size, 1, n_mels, time_frames).to(device)
    condition = torch.rand(batch_size, 2).to(device)

    # Training loss
    loss = diffusion.training_loss(x, condition)
    print(f"\nTraining loss: {loss.item():.4f}")

    # Test sampling
    print("\nTesting sampling...")
    samples = diffusion.sample((2, 1, n_mels, time_frames), condition[:2], device)
    print(f"Generated samples shape: {samples.shape}")
