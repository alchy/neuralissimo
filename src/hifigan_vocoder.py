"""
HiFi-GAN Vocoder wrapper for high-quality mel-to-audio conversion

Supports both 44.1kHz and 48kHz sample rates
Much better quality than Griffin-Lim algorithm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class ResBlock(nn.Module):
    """Residual block for HiFi-GAN"""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5)):
        super().__init__()

        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                     padding=self.get_padding(kernel_size, dilation[0])),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                     padding=self.get_padding(kernel_size, dilation[1])),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                     padding=self.get_padding(kernel_size, dilation[2]))
        ])

        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                     padding=self.get_padding(kernel_size, 1)),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                     padding=self.get_padding(kernel_size, 1)),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                     padding=self.get_padding(kernel_size, 1))
        ])

    def get_padding(self, kernel_size: int, dilation: int = 1) -> int:
        return int((kernel_size * dilation - dilation) / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN Generator for mel-to-waveform conversion

    Simplified version optimized for piano sounds
    """

    def __init__(
        self,
        n_mels: int = 128,
        upsample_rates: list = [8, 8, 2, 2],  # Product should equal hop_length
        upsample_kernel_sizes: list = [16, 16, 4, 4],
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: list = [3, 7, 11],
        resblock_dilation_sizes: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # Initial convolution
        self.conv_pre = nn.Conv1d(n_mels, upsample_initial_channel, 7, 1, padding=3)

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2**(i+1)),
                    k, u, padding=(k-u)//2
                )
            )

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

        # Post convolution
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Mel spectrogram [batch, n_mels, time]
        Returns:
            Waveform [batch, 1, samples]
        """
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class HiFiGANVocoder:
    """
    Wrapper for HiFi-GAN vocoder with multiple sample rate support
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        hop_length: int = 512,
        n_mels: int = 128,
        device: str = 'cpu'
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.device = device

        # Determine upsample rates based on hop_length
        # hop_length = 512 = 8 * 8 * 2 * 2
        # hop_length = 256 = 8 * 8 * 2 * 2 (for 44.1kHz)
        if hop_length == 512:
            upsample_rates = [8, 8, 2, 2]
            upsample_kernel_sizes = [16, 16, 4, 4]
        elif hop_length == 256:
            upsample_rates = [8, 8, 2, 2]
            upsample_kernel_sizes = [16, 16, 4, 4]
        else:
            # Default fallback
            upsample_rates = [8, 8, 2, 2]
            upsample_kernel_sizes = [16, 16, 4, 4]

        # Create generator
        self.generator = HiFiGANGenerator(
            n_mels=n_mels,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes
        ).to(device)

        self.generator.eval()

    def load_checkpoint(self, checkpoint_path: str):
        """Load pre-trained checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'generator' in checkpoint:
            self.generator.load_state_dict(checkpoint['generator'])
        elif 'model_state_dict' in checkpoint:
            self.generator.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.generator.load_state_dict(checkpoint)

        print(f"Loaded HiFi-GAN checkpoint from {checkpoint_path}")

    @torch.no_grad()
    def mel_to_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Convert mel spectrogram to audio waveform

        Args:
            mel_spec: Mel spectrogram [batch, 1, n_mels, time] or [batch, n_mels, time]
        Returns:
            Waveform [batch, 1, samples]
        """
        # Handle different input shapes
        if mel_spec.dim() == 4:
            mel_spec = mel_spec.squeeze(1)  # Remove channel dim
        elif mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)  # Add batch dim

        # Denormalize if in dB scale (typical range: -80 to 0)
        # Assume input is in dB scale, convert to linear scale
        if mel_spec.min() < 0:
            # Convert from dB to power
            mel_spec = torch.pow(10.0, mel_spec / 10.0)

        # Clamp to prevent extreme values
        mel_spec = torch.clamp(mel_spec, min=1e-5, max=1e5)

        # Convert to log scale for vocoder
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

        # Generate waveform
        waveform = self.generator(mel_spec)

        return waveform


def create_hifigan_vocoder(
    sample_rate: int = 48000,
    hop_length: int = 512,
    n_mels: int = 128,
    checkpoint_path: Optional[str] = None,
    device: str = 'cpu'
) -> HiFiGANVocoder:
    """
    Factory function to create HiFi-GAN vocoder

    Args:
        sample_rate: Target sample rate (44100 or 48000)
        hop_length: Hop length used for mel spectrogram
        n_mels: Number of mel bands
        checkpoint_path: Optional path to pre-trained checkpoint
        device: Device to run on

    Returns:
        HiFiGANVocoder instance
    """
    vocoder = HiFiGANVocoder(
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_mels=n_mels,
        device=device
    )

    if checkpoint_path is not None:
        try:
            vocoder.load_checkpoint(checkpoint_path)
            print(f"Loaded pre-trained vocoder from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Using untrained HiFi-GAN (will need training)")

    return vocoder


# Test
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Test 48kHz vocoder
    print("\nTesting 48kHz HiFi-GAN vocoder:")
    vocoder_48k = create_hifigan_vocoder(
        sample_rate=48000,
        hop_length=512,
        n_mels=128,
        device=device
    )

    # Create dummy mel spectrogram
    batch_size = 2
    n_mels = 128
    time_frames = 281  # ~3 seconds at 48kHz with hop=512

    # Random mel spec in dB scale
    mel_spec = torch.randn(batch_size, n_mels, time_frames).to(device) * 20 - 40

    print(f"Input mel spec shape: {mel_spec.shape}")
    print(f"Input mel spec range: [{mel_spec.min():.2f}, {mel_spec.max():.2f}] dB")

    # Convert to audio
    waveform = vocoder_48k.mel_to_audio(mel_spec)

    expected_samples = time_frames * 512  # hop_length
    print(f"Output waveform shape: {waveform.shape}")
    print(f"Expected samples: ~{expected_samples}")
    print(f"Actual samples: {waveform.shape[-1]}")
    print(f"Duration: {waveform.shape[-1] / 48000:.2f}s")
    print(f"Waveform range: [{waveform.min():.3f}, {waveform.max():.3f}]")

    # Test 44.1kHz vocoder
    print("\nTesting 44.1kHz HiFi-GAN vocoder:")
    vocoder_44k = create_hifigan_vocoder(
        sample_rate=44100,
        hop_length=512,
        n_mels=128,
        device=device
    )

    waveform_44k = vocoder_44k.mel_to_audio(mel_spec)
    print(f"Output waveform shape: {waveform_44k.shape}")
    print(f"Duration: {waveform_44k.shape[-1] / 44100:.2f}s")

    print("\nHiFi-GAN vocoder test completed!")
    print("\nNote: Vocoder is untrained. For best quality:")
    print("  1. Train vocoder on your piano dataset")
    print("  2. Or download pre-trained HiFi-GAN checkpoint")
    print("  3. Fine-tune on your specific instrument samples")
