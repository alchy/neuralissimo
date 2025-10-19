"""
Variable-Length Conditional GAN for Electric Piano Sound Generation

Generates mel spectrograms with variable temporal length based on MIDI note.
Low notes have longer decay, high notes have shorter duration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
import numpy as np


# Duration lookup table from dataset analysis
MIDI_DURATION_MAP = {
    33: 21.688, 34: 19.584, 35: 29.134, 36: 22.319, 37: 26.954,
    38: 22.548, 39: 23.470, 40: 20.539, 41: 21.114, 42: 19.922,
    43: 16.900, 44: 16.075, 45: 13.556, 46: 16.504, 47: 14.691,
    48: 14.980, 49: 14.041, 50: 10.295, 51: 10.491, 52: 10.464,
    53: 11.526, 54: 10.237, 55: 9.739, 56: 9.477, 57: 8.131,
    58: 6.311, 59: 9.588, 60: 8.306, 61: 7.179, 62: 7.689,
    63: 6.190, 64: 4.785, 65: 7.976, 66: 5.096, 67: 6.525,
    68: 5.164, 69: 6.950, 70: 5.294, 71: 6.221, 72: 4.938,
    73: 4.673, 74: 3.720, 75: 4.167, 76: 3.826, 77: 4.638,
    78: 5.095, 79: 3.950, 80: 4.676, 81: 5.535, 82: 4.592,
    83: 5.479, 84: 3.544, 85: 2.603, 86: 4.385, 87: 3.121,
    88: 3.991, 89: 3.519, 90: 3.225, 92: 3.140, 94: 2.950,
}


def get_duration_from_midi(midi_note: int, sample_rate: int = 48000, hop_length: int = 512) -> int:
    """
    Get expected duration (in time frames) for a MIDI note

    Args:
        midi_note: MIDI note number
        sample_rate: Audio sample rate
        hop_length: Hop length for mel spectrogram
    Returns:
        Number of time frames for mel spectrogram
    """
    # Get duration in seconds
    duration_sec = MIDI_DURATION_MAP.get(midi_note, 8.0)  # Default to 8s

    # Convert to number of samples
    num_samples = int(duration_sec * sample_rate)

    # Convert to time frames
    time_frames = num_samples // hop_length

    return time_frames


class AdaptiveTemporalUpsampling(nn.Module):
    """
    Adaptive upsampling to target temporal length
    Uses interpolation to reach exact target duration
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Args:
            x: Input [batch, channels, mel_bands, time]
            target_length: Target temporal length
        Returns:
            Upsampled [batch, channels, mel_bands, target_length]
        """
        if x.shape[3] == target_length:
            return x

        # Interpolate along temporal dimension
        x = F.interpolate(x, size=(x.shape[2], target_length), mode='bilinear', align_corners=False)
        return x


class VariableLengthGenerator(nn.Module):
    """
    Generator that produces variable-length mel spectrograms
    based on MIDI note
    """

    def __init__(
        self,
        latent_dim: int = 128,
        condition_dim: int = 2,
        base_channels: int = 256,
        sample_rate: int = 48000,
        hop_length: int = 512
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        # Condition embedding (MIDI + velocity)
        self.cond_embedding = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # Combined latent + condition
        combined_dim = latent_dim + 256

        # Initial projection to small spatial size (8x16)
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, base_channels * 4 * 8 * 16),
            nn.ReLU()
        )

        # Upsampling blocks
        # 8x16 -> 16x32 -> 32x64 -> 64x128 -> 128xT
        self.up1 = self._make_upsample_block(base_channels * 4, base_channels * 4)
        self.up2 = self._make_upsample_block(base_channels * 4, base_channels * 2)
        self.up3 = self._make_upsample_block(base_channels * 2, base_channels)
        self.up4 = self._make_upsample_block(base_channels, base_channels // 2)

        # Adaptive temporal upsampling
        self.adaptive_upsample = AdaptiveTemporalUpsampling()

        # Final convolution
        self.final = nn.Sequential(
            nn.Conv2d(base_channels // 2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh()
        )

    def _make_upsample_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create upsampling block"""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, noise: torch.Tensor, condition: torch.Tensor,
                target_lengths: List[int] = None) -> torch.Tensor:
        """
        Args:
            noise: Random noise [batch, latent_dim]
            condition: [batch, 2] (normalized_midi, normalized_velocity)
            target_lengths: Optional list of target lengths per sample
        Returns:
            Generated mel specs [batch, 1, 128, max_length] (padded)
        """
        batch_size = noise.shape[0]

        # Embed condition
        cond_emb = self.cond_embedding(condition)

        # Combine noise and condition
        x = torch.cat([noise, cond_emb], dim=1)

        # Project and reshape
        x = self.fc(x)
        x = x.view(batch_size, -1, 8, 16)

        # Upsample
        x = self.up1(x)  # 16x32
        x = self.up2(x)  # 32x64
        x = self.up3(x)  # 64x128
        x = self.up4(x)  # 128x256

        # If target lengths provided, generate variable-length outputs
        if target_lengths is not None:
            outputs = []
            for i in range(batch_size):
                sample = x[i:i+1]
                target_len = target_lengths[i]
                sample = self.adaptive_upsample(sample, target_len)
                outputs.append(sample)

            # Pad to max length for batching
            max_len = max(target_lengths)
            padded_outputs = []
            for sample, target_len in zip(outputs, target_lengths):
                if sample.shape[3] < max_len:
                    pad_amount = max_len - sample.shape[3]
                    sample = F.pad(sample, (0, pad_amount))
                padded_outputs.append(sample)

            x = torch.cat(padded_outputs, dim=0)

        # Final conv
        x = self.final(x)

        return x


class VariableLengthDiscriminator(nn.Module):
    """
    Discriminator that handles variable-length inputs
    Uses global average pooling to handle different temporal lengths
    """

    def __init__(
        self,
        condition_dim: int = 2,
        base_channels: int = 64
    ):
        super().__init__()

        # Condition embedding
        self.cond_embedding = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # Downsampling blocks
        # Input: 1 channel + 1 condition channel = 2 channels
        self.down1 = self._make_downsample_block(2, base_channels, use_bn=False)
        self.down2 = self._make_downsample_block(base_channels, base_channels * 2)
        self.down3 = self._make_downsample_block(base_channels * 2, base_channels * 4)
        self.down4 = self._make_downsample_block(base_channels * 4, base_channels * 8)

        # Global pooling to handle variable lengths
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8 * 4 * 4, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def _make_downsample_block(self, in_channels: int, out_channels: int,
                               use_bn: bool = True) -> nn.Module:
        """Create downsampling block"""
        layers = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input mel spec [batch, 1, mel_bands, time]
            condition: [batch, 2]
        Returns:
            Predictions [batch, 1]
        """
        batch_size = x.shape[0]
        mel_bands = x.shape[2]
        time_frames = x.shape[3]

        # Embed condition and broadcast to spatial dimensions
        cond_emb = self.cond_embedding(condition)
        cond_map = cond_emb[:, :, None, None].expand(batch_size, 128, mel_bands, time_frames)

        # Take mean over condition channels to get single channel
        cond_channel = cond_map.mean(dim=1, keepdim=True)

        # Concatenate with input
        x = torch.cat([x, cond_channel], dim=1)

        # Downsample
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        # Global pool (handles variable lengths)
        x = self.global_pool(x)

        # Classify
        x = self.classifier(x)

        return x


class VariableLengthGAN(nn.Module):
    """
    Complete variable-length GAN system
    """

    def __init__(
        self,
        latent_dim: int = 128,
        condition_dim: int = 2,
        sample_rate: int = 48000,
        hop_length: int = 512
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        self.generator = VariableLengthGenerator(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            sample_rate=sample_rate,
            hop_length=hop_length
        )

        self.discriminator = VariableLengthDiscriminator(
            condition_dim=condition_dim
        )

    def get_target_lengths(self, midi_notes: torch.Tensor) -> List[int]:
        """
        Get target lengths for batch of MIDI notes

        Args:
            midi_notes: Normalized MIDI notes [batch]
        Returns:
            List of target time frame lengths
        """
        # Denormalize MIDI notes (assuming normalized to [0, 1] from range [33, 94])
        midi_min, midi_max = 33, 94
        denorm_midi = midi_notes * (midi_max - midi_min) + midi_min
        denorm_midi = denorm_midi.cpu().numpy().astype(int)

        # Get durations
        target_lengths = [
            get_duration_from_midi(midi, self.sample_rate, self.hop_length)
            for midi in denorm_midi
        ]

        return target_lengths

    def generate(self, condition: torch.Tensor, device: str = 'cpu',
                use_variable_length: bool = True) -> torch.Tensor:
        """
        Generate samples

        Args:
            condition: [batch, 2] (midi, velocity)
            device: Device
            use_variable_length: If True, use MIDI-specific lengths
        Returns:
            Generated mel spectrograms
        """
        batch_size = condition.shape[0]
        noise = torch.randn(batch_size, self.latent_dim, device=device)

        if use_variable_length:
            # Get target lengths from MIDI notes
            midi_notes = condition[:, 0]
            target_lengths = self.get_target_lengths(midi_notes)
            return self.generator(noise, condition, target_lengths)
        else:
            return self.generator(noise, condition, None)


# Losses
def hinge_loss_dis(real_output: torch.Tensor, fake_output: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator"""
    return torch.mean(F.relu(1.0 - real_output)) + torch.mean(F.relu(1.0 + fake_output))


def hinge_loss_gen(fake_output: torch.Tensor) -> torch.Tensor:
    """Hinge loss for generator"""
    return -torch.mean(fake_output)


# Test
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create GAN
    gan = VariableLengthGAN(
        latent_dim=128,
        condition_dim=2,
        sample_rate=48000,
        hop_length=512
    ).to(device)

    print(f"Generator parameters: {sum(p.numel() for p in gan.generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in gan.discriminator.parameters()):,}")

    # Test with different MIDI notes
    batch_size = 4

    # Normalized conditions: [0, 1] for MIDI 33-94
    conditions = torch.tensor([
        [0.0, 0.5],   # Low note (MIDI 33)
        [0.25, 0.5],  # Mid-low note
        [0.5, 0.5],   # Mid note (MIDI 63)
        [1.0, 0.5],   # High note (MIDI 94)
    ], device=device)

    # Generate with variable lengths
    print("\nGenerating with variable lengths...")
    fake_mels = gan.generate(conditions, device, use_variable_length=True)
    print(f"Generated shape: {fake_mels.shape}")

    # Get expected lengths
    target_lengths = gan.get_target_lengths(conditions[:, 0])
    for i, (cond, length) in enumerate(zip(conditions, target_lengths)):
        midi_denorm = int(cond[0] * (94 - 33) + 33)
        duration_sec = MIDI_DURATION_MAP.get(midi_denorm, 8.0)
        print(f"Sample {i}: MIDI {midi_denorm}, Expected {length} frames ({duration_sec:.2f}s)")

    # Test discriminator
    print("\nTesting discriminator...")
    real_output = gan.discriminator(fake_mels, conditions)
    print(f"Discriminator output shape: {real_output.shape}")
    print(f"Discriminator output range: [{real_output.min().item():.3f}, {real_output.max().item():.3f}]")
