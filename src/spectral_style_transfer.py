"""
Spectral Style Transfer from MP3 Reference Track

Applies spectral characteristics from a reference track to generated samples
Preserves the timing/envelope from GAN but matches spectral profile of reference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
from typing import Tuple, Optional


class SpectralStyleTransfer:
    """
    Apply spectral characteristics from reference track to generated audio
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        device: str = 'cpu'
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.device = device

        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        ).to(device)

        self.amplitude_to_db = T.AmplitudeToDB().to(device)

        # Reference spectral profile (will be extracted from reference track)
        self.reference_profile = None

    def load_reference_track(self, reference_path: str):
        """
        Load and analyze reference track to extract spectral profile

        Args:
            reference_path: Path to reference MP3/WAV track
        """
        print(f"Loading reference track: {reference_path}")

        # Load audio
        audio_data, sr = sf.read(reference_path, dtype='float32')

        # Convert to mono
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Convert to torch
        waveform = torch.from_numpy(audio_data).unsqueeze(0).to(self.device)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate).to(self.device)
            waveform = resampler(waveform)

        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # Extract spectral profile (average over time)
        self.reference_profile = mel_spec_db.mean(dim=-1, keepdim=True)

        print(f"Reference spectral profile extracted")
        print(f"  Shape: {self.reference_profile.shape}")
        print(f"  Range: [{self.reference_profile.min():.2f}, {self.reference_profile.max():.2f}] dB")

        return self.reference_profile

    def compute_spectral_envelope(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral envelope (average frequency profile)

        Args:
            mel_spec: Mel spectrogram [batch, n_mels, time]
        Returns:
            Spectral envelope [batch, n_mels, 1]
        """
        return mel_spec.mean(dim=-1, keepdim=True)

    def apply_style_transfer(
        self,
        source_mel: torch.Tensor,
        strength: float = 1.0
    ) -> torch.Tensor:
        """
        Apply spectral style from reference to source mel spectrogram

        Args:
            source_mel: Source mel spectrogram [batch, 1, n_mels, time] or [batch, n_mels, time]
            strength: Transfer strength (0=no transfer, 1=full transfer)
        Returns:
            Styled mel spectrogram with same shape as input
        """
        if self.reference_profile is None:
            raise ValueError("Reference profile not loaded. Call load_reference_track() first.")

        # Handle input shape
        had_channel_dim = False
        if source_mel.dim() == 4:
            source_mel = source_mel.squeeze(1)
            had_channel_dim = True

        # Compute source spectral envelope
        source_envelope = self.compute_spectral_envelope(source_mel)

        # Compute transfer ratio
        transfer_ratio = self.reference_profile / (source_envelope + 1e-8)

        # Clamp to reasonable range
        transfer_ratio = torch.clamp(transfer_ratio, 0.5, 2.0)

        # Apply transfer with strength
        transfer_ratio = 1.0 + strength * (transfer_ratio - 1.0)

        # Apply to source
        styled_mel = source_mel * transfer_ratio

        # Restore channel dim if needed
        if had_channel_dim:
            styled_mel = styled_mel.unsqueeze(1)

        return styled_mel

    def match_spectral_statistics(
        self,
        source_mel: torch.Tensor,
        match_mean: bool = True,
        match_std: bool = True
    ) -> torch.Tensor:
        """
        Match spectral statistics (mean/std) to reference

        Args:
            source_mel: Source mel spectrogram
            match_mean: Match mean values
            match_std: Match standard deviation
        Returns:
            Statistically matched mel spectrogram
        """
        if self.reference_profile is None:
            raise ValueError("Reference profile not loaded.")

        # Handle input shape
        had_channel_dim = False
        if source_mel.dim() == 4:
            source_mel = source_mel.squeeze(1)
            had_channel_dim = True

        # Compute statistics
        source_mean = source_mel.mean(dim=-1, keepdim=True)
        source_std = source_mel.std(dim=-1, keepdim=True)

        # Normalize source
        normalized = (source_mel - source_mean) / (source_std + 1e-8)

        # Reference statistics (use reference profile as mean target)
        ref_mean = self.reference_profile
        ref_std = torch.ones_like(ref_mean) * 10.0  # Typical std for dB scale

        # Apply reference statistics
        matched = normalized * ref_std + ref_mean

        # Restore channel dim if needed
        if had_channel_dim:
            matched = matched.unsqueeze(1)

        return matched


class SpectralLoss(nn.Module):
    """
    Loss function for spectral matching during training
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128
    ):
        super().__init__()

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute spectral losses

        Args:
            generated: Generated mel spectrogram [batch, n_mels, time]
            target: Target mel spectrogram [batch, n_mels, time]
        Returns:
            spectral_loss: L1 loss on mel spectrograms
            envelope_loss: L1 loss on spectral envelopes
        """
        # L1 loss on full spectrograms
        spectral_loss = F.l1_loss(generated, target)

        # L1 loss on spectral envelopes
        gen_envelope = generated.mean(dim=-1)
        target_envelope = target.mean(dim=-1)
        envelope_loss = F.l1_loss(gen_envelope, target_envelope)

        return spectral_loss, envelope_loss


# Test and utility functions
def visualize_spectral_profile(
    profile: torch.Tensor,
    title: str = "Spectral Profile",
    save_path: Optional[str] = None
):
    """Visualize spectral profile (requires matplotlib)"""
    try:
        import matplotlib.pyplot as plt

        profile_np = profile.squeeze().cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(profile_np)
        plt.xlabel('Mel Band')
        plt.ylabel('Amplitude (dB)')
        plt.title(title)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved spectral profile plot to {save_path}")
        else:
            plt.show()

        plt.close()
    except ImportError:
        print("Matplotlib not available for visualization")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create style transfer engine
    style_transfer = SpectralStyleTransfer(
        sample_rate=48000,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        device=device
    )

    # Test with dummy data
    print("\nTesting spectral style transfer:")

    # Create dummy source mel spectrogram
    batch_size = 2
    n_mels = 128
    time_frames = 281

    source_mel = torch.randn(batch_size, n_mels, time_frames).to(device) * 20 - 40

    print(f"Source mel shape: {source_mel.shape}")
    print(f"Source mel range: [{source_mel.min():.2f}, {source_mel.max():.2f}] dB")

    # Create dummy reference profile
    dummy_ref = torch.randn(1, n_mels, 1).to(device) * 15 - 30
    style_transfer.reference_profile = dummy_ref

    print(f"\nReference profile shape: {dummy_ref.shape}")
    print(f"Reference profile range: [{dummy_ref.min():.2f}, {dummy_ref.max():.2f}] dB")

    # Apply style transfer
    styled_mel = style_transfer.apply_style_transfer(source_mel, strength=0.7)

    print(f"\nStyled mel shape: {styled_mel.shape}")
    print(f"Styled mel range: [{styled_mel.min():.2f}, {styled_mel.max():.2f}] dB")

    # Test spectral loss
    print("\nTesting spectral loss:")
    loss_fn = SpectralLoss()
    target_mel = torch.randn_like(source_mel).to(device) * 20 - 40

    spectral_loss, envelope_loss = loss_fn(source_mel, target_mel)
    print(f"Spectral loss: {spectral_loss.item():.4f}")
    print(f"Envelope loss: {envelope_loss.item():.4f}")

    print("\nSpectral style transfer test completed!")
