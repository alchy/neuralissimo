"""
Inference script for Variable-Length GAN with HiFi-GAN vocoder

Generates high-quality audio samples with proper naming: mXXX-velX-fXX.wav
Supports both 44.1kHz and 48kHz output
"""

import os
import argparse
import yaml
import torch
import numpy as np
from scipy.io import wavfile
from typing import Optional
from pathlib import Path

from configurepaths import *
from variable_gan_model import VariableLengthGAN
from hifigan_vocoder import create_hifigan_vocoder


class GANInferenceEngine:
    """
    Inference engine for generating piano samples with GAN
    """

    def __init__(
        self,
        gan_checkpoint: str,
        config_path: str,
        vocoder_checkpoint: Optional[str] = None,
        device: str = 'cpu'
    ):
        self.device = device

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load GAN model
        print(f"Loading GAN model from {gan_checkpoint}...")
        self.gan = self._load_gan(gan_checkpoint)
        self.gan.eval()
        print("GAN model loaded successfully!")

        # Setup HiFi-GAN vocoder
        sample_rate = self.config['data']['sample_rate']
        hop_length = self.config['model']['hop_length']
        n_mels = self.config['model']['n_mels']

        print(f"\nInitializing HiFi-GAN vocoder ({sample_rate}Hz)...")
        self.vocoder = create_hifigan_vocoder(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_mels=n_mels,
            checkpoint_path=vocoder_checkpoint,
            device=device
        )
        print("Vocoder ready!")

        # MIDI note normalization parameters
        self.midi_min = self.config['model']['midi_note_range'][0]
        self.midi_max = self.config['model']['midi_note_range'][1]
        self.vel_max = self.config['model']['velocity_levels'] - 1

        # Sample rate for output
        self.sample_rate = sample_rate

    def _load_gan(self, checkpoint_path: str) -> VariableLengthGAN:
        """Load GAN from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create model
        gan = VariableLengthGAN(
            latent_dim=self.config['model']['latent_dim'],
            condition_dim=2,
            sample_rate=self.config['data']['sample_rate'],
            hop_length=self.config['model']['hop_length']
        ).to(self.device)

        # Load weights
        if 'generator_state_dict' in checkpoint:
            gan.generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
        elif 'model_state_dict' in checkpoint:
            gan.generator.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            print("Warning: Unexpected checkpoint format")

        return gan

    def _normalize_midi(self, midi_note: int) -> float:
        """Normalize MIDI note to [0, 1]"""
        return (midi_note - self.midi_min) / (self.midi_max - self.midi_min)

    def _normalize_velocity(self, velocity: int) -> float:
        """Normalize velocity to [0, 1]"""
        return velocity / self.vel_max

    def _mel_to_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Convert mel spectrogram to audio using HiFi-GAN vocoder

        Args:
            mel_spec: Mel spectrogram [1, 1, n_mels, time] or [1, n_mels, time]
        Returns:
            Waveform [1, samples]
        """
        # Use HiFi-GAN vocoder
        waveform = self.vocoder.mel_to_audio(mel_spec)

        # Ensure correct shape [1, samples]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        return waveform

    @torch.no_grad()
    def generate(
        self,
        midi_note: int,
        velocity: int,
        num_samples: int = 1,
        temperature: float = 1.0,
        use_variable_length: bool = True
    ) -> torch.Tensor:
        """
        Generate audio samples

        Args:
            midi_note: MIDI note number (33-94)
            velocity: Velocity level (0-7)
            num_samples: Number of samples to generate
            temperature: Noise temperature (higher = more variation)
            use_variable_length: Use MIDI-appropriate length

        Returns:
            Waveforms [num_samples, 1, audio_samples]
        """
        # Validate inputs
        if not (self.midi_min <= midi_note <= self.midi_max):
            print(f"Warning: MIDI note {midi_note} outside training range "
                  f"[{self.midi_min}, {self.midi_max}]")

        if not (0 <= velocity <= self.vel_max):
            raise ValueError(f"Velocity must be between 0 and {self.vel_max}")

        # Create conditioning vector
        midi_norm = self._normalize_midi(midi_note)
        vel_norm = self._normalize_velocity(velocity)

        condition = torch.tensor(
            [[midi_norm, vel_norm]] * num_samples,
            dtype=torch.float32
        ).to(self.device)

        # Generate mel spectrograms
        mel_specs = self.gan.generate(
            condition,
            device=self.device,
            use_variable_length=use_variable_length
        )

        # Apply temperature to add variation (optional post-processing)
        if temperature != 1.0:
            mel_specs = mel_specs * temperature

        # Convert to audio
        waveforms = []
        for i in range(num_samples):
            waveform = self._mel_to_audio(mel_specs[i:i+1])
            waveforms.append(waveform)

        waveforms = torch.cat(waveforms, dim=0)

        return waveforms

    def save_audio(
        self,
        waveform: torch.Tensor,
        output_path: str,
        normalize: bool = True
    ):
        """
        Save audio waveform to file

        Args:
            waveform: Audio waveform [1, samples] or [samples]
            output_path: Path to save audio file
            normalize: Normalize to prevent clipping
        """
        # Ensure correct shape
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        # Move to CPU and convert to numpy
        waveform_np = waveform.cpu().numpy()

        # Normalize to prevent clipping
        if normalize:
            max_val = np.abs(waveform_np).max()
            if max_val > 0:
                waveform_np = waveform_np / max_val

        # Convert to int16
        waveform_int16 = (waveform_np * 32767).astype(np.int16)

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        wavfile.write(output_path, self.sample_rate, waveform_int16)
        print(f"Audio saved to {output_path}")

    def generate_filename(
        self,
        midi_note: int,
        velocity: int,
        sample_idx: int = 0,
        output_dir: str = "outputs"
    ) -> str:
        """
        Generate proper filename in format: mXXX-velX-fXX.wav

        Args:
            midi_note: MIDI note number
            velocity: Velocity level
            sample_idx: Sample index (if generating multiple)
            output_dir: Output directory

        Returns:
            Full path to output file
        """
        # Sample rate in kHz
        sr_khz = self.sample_rate // 1000

        # Format: mXXX-velX-fXX.wav
        if sample_idx == 0:
            filename = f"m{midi_note:03d}-vel{velocity}-f{sr_khz:02d}.wav"
        else:
            filename = f"m{midi_note:03d}-vel{velocity}-f{sr_khz:02d}_s{sample_idx}.wav"

        return os.path.join(output_dir, filename)


def main():
    parser = argparse.ArgumentParser(
        description='Generate piano samples using Variable-Length GAN'
    )
    parser.add_argument('--gan-checkpoint', type=str, required=True,
                       help='Path to trained GAN checkpoint')
    parser.add_argument('--config', type=str, default=str(CONFIG_FILE),
                       help='Path to configuration file')
    parser.add_argument('--vocoder-checkpoint', type=str, default=None,
                       help='Path to HiFi-GAN vocoder checkpoint (optional)')
    parser.add_argument('--midi', type=int, required=True,
                       help='MIDI note number (33-94)')
    parser.add_argument('--velocity', type=int, required=True,
                       help='Velocity level (0-7)')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUTS_DIR),
                       help='Output directory')
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of samples to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Generation temperature (variation)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--no-variable-length', action='store_true',
                       help='Disable variable-length generation')

    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}\n")

    # Create inference engine
    engine = GANInferenceEngine(
        gan_checkpoint=args.gan_checkpoint,
        config_path=args.config,
        vocoder_checkpoint=args.vocoder_checkpoint,
        device=device
    )

    # Generate samples
    print(f"\nGenerating {args.num_samples} sample(s):")
    print(f"  MIDI note: {args.midi}")
    print(f"  Velocity: {args.velocity}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Variable length: {not args.no_variable_length}\n")

    waveforms = engine.generate(
        midi_note=args.midi,
        velocity=args.velocity,
        num_samples=args.num_samples,
        temperature=args.temperature,
        use_variable_length=not args.no_variable_length
    )

    # Save samples
    for i in range(args.num_samples):
        output_path = engine.generate_filename(
            midi_note=args.midi,
            velocity=args.velocity,
            sample_idx=i if args.num_samples > 1 else 0,
            output_dir=args.output_dir
        )

        engine.save_audio(waveforms[i], output_path)

        # Print sample info
        duration = waveforms[i].shape[-1] / engine.sample_rate
        print(f"  Sample {i+1}: {duration:.2f}s, {waveforms[i].shape[-1]} samples")

    print(f"\nGeneration complete! Files saved to {args.output_dir}")


if __name__ == "__main__":
    main()
