"""
Variable-length dataset loader for Electric Piano samples

Supports variable temporal lengths based on MIDI note.
Does not pad to fixed length - each sample keeps its natural duration.
"""

import os
import re
import glob
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
from typing import Tuple, Dict, List, Optional
from configurepaths import *


class VariableLengthElectricPianoDataset(Dataset):
    """
    Dataset that preserves natural audio lengths
    Each sample has variable temporal dimension based on MIDI note
    """

    def __init__(
        self,
        instrument_dir: str,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        use_mel_spectrogram: bool = True,
        target_sample_rate: int = 48000,
        max_duration: Optional[float] = None,  # Optional max duration for memory
        normalize_audio: bool = True
    ):
        """
        Args:
            instrument_dir: Path to directory containing WAV files
            sample_rate: Original sample rate of audio files
            n_fft: FFT window size for spectrogram
            hop_length: Hop length for spectrogram
            n_mels: Number of mel bands
            use_mel_spectrogram: If True, convert to mel spectrogram
            target_sample_rate: Target sample rate (will resample if different)
            max_duration: Optional max duration in seconds (crops longer samples)
            normalize_audio: If True, normalize audio amplitude to [-1, 1]
        """
        self.instrument_dir = instrument_dir
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.use_mel_spectrogram = use_mel_spectrogram
        self.max_duration = max_duration
        self.normalize_audio = normalize_audio

        # Load file paths
        self.file_paths = self._load_file_paths()

        # Parse all files to get metadata
        self.metadata = self._parse_metadata()

        # Calculate normalization statistics
        self.midi_min = min([m['midi_note'] for m in self.metadata])
        self.midi_max = max([m['midi_note'] for m in self.metadata])
        self.vel_min = 0
        self.vel_max = 7

        # Setup mel spectrogram transform
        if use_mel_spectrogram:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=target_sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0
            )
            self.amplitude_to_db = T.AmplitudeToDB()

        # Resampler if needed
        if sample_rate != target_sample_rate:
            self.resampler = T.Resample(
                orig_freq=sample_rate,
                new_freq=target_sample_rate
            )
        else:
            self.resampler = None

        print(f"Loaded {len(self.file_paths)} samples")
        print(f"MIDI range: {self.midi_min}-{self.midi_max}")
        print(f"Velocity range: {self.vel_min}-{self.vel_max}")
        print(f"Variable-length mode: Preserving natural durations")

    def _load_file_paths(self) -> List[str]:
        """Load all WAV file paths from instrument directory"""
        pattern = os.path.join(self.instrument_dir, "*.wav")
        file_paths = glob.glob(pattern)
        return sorted(file_paths)

    def _parse_filename(self, filepath: str) -> Dict:
        """
        Parse filename to extract MIDI note, velocity, and sample rate
        Expected format: mXXX-velX-fXX.wav
        """
        basename = os.path.basename(filepath)
        match = re.match(r'm(\d+)-vel(\d+)-f(\d+)\.wav', basename)

        if not match:
            raise ValueError(f"Invalid filename format: {basename}")

        return {
            'midi_note': int(match.group(1)),
            'velocity': int(match.group(2)),
            'file_sample_rate': int(match.group(3)) * 1000,
            'filepath': filepath
        }

    def _parse_metadata(self) -> List[Dict]:
        """Parse metadata from all files"""
        metadata = []
        for filepath in self.file_paths:
            try:
                meta = self._parse_filename(filepath)
                metadata.append(meta)
            except ValueError as e:
                print(f"Skipping file: {e}")
        return metadata

    def _normalize_midi(self, midi_note: int) -> float:
        """Normalize MIDI note to [0, 1]"""
        return (midi_note - self.midi_min) / (self.midi_max - self.midi_min)

    def _normalize_velocity(self, velocity: int) -> float:
        """Normalize velocity to [0, 1]"""
        return velocity / self.vel_max

    def _load_and_preprocess_audio(self, filepath: str) -> torch.Tensor:
        """
        Load audio file and preprocess (WITHOUT padding to fixed length)

        Steps:
        1. Load audio
        2. Resample if needed
        3. Convert to mono if needed
        4. Normalize amplitude (optional)
        5. Crop to max_duration if specified
        """
        # Load audio using soundfile
        audio_data, sr = sf.read(filepath, dtype='float32')

        # Convert to torch tensor and add channel dimension
        waveform = torch.from_numpy(audio_data)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if self.resampler is not None:
            waveform = self.resampler(waveform)

        # Crop to max duration if specified
        if self.max_duration is not None:
            max_samples = int(self.max_duration * self.target_sample_rate)
            if waveform.shape[1] > max_samples:
                # Crop from center
                start = (waveform.shape[1] - max_samples) // 2
                waveform = waveform[:, start:start + max_samples]

        # Normalize amplitude
        if self.normalize_audio:
            max_val = torch.abs(waveform).max()
            if max_val > 0:
                waveform = waveform / max_val

        return waveform

    def _waveform_to_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel spectrogram"""
        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)

        # Convert to dB scale
        mel_spec_db = self.amplitude_to_db(mel_spec)

        return mel_spec_db

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a single sample

        Returns:
            audio: Preprocessed audio (spectrogram or waveform) [1, n_mels, variable_time]
            condition: Conditioning vector [midi_note, velocity]
            metadata: Dictionary with additional info (MIDI note, velocity, length)
        """
        # Get metadata
        meta = self.metadata[idx]

        # Load and preprocess audio
        waveform = self._load_and_preprocess_audio(meta['filepath'])

        # Convert to spectrogram if needed
        if self.use_mel_spectrogram:
            audio = self._waveform_to_spectrogram(waveform)
        else:
            audio = waveform

        # Create conditioning vector
        midi_norm = self._normalize_midi(meta['midi_note'])
        vel_norm = self._normalize_velocity(meta['velocity'])
        condition = torch.tensor([midi_norm, vel_norm], dtype=torch.float32)

        # Return metadata for collation
        sample_meta = {
            'midi_note': meta['midi_note'],
            'velocity': meta['velocity'],
            'length': audio.shape[-1],  # Temporal length
            'idx': idx
        }

        return audio, condition, sample_meta


def variable_length_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, Dict]]) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[Dict]]:
    """
    Custom collate function for variable-length samples

    Pads samples to the maximum length in the batch
    Returns mask indicating valid (non-padded) regions

    Args:
        batch: List of (audio, condition, metadata) tuples

    Returns:
        audio_padded: Padded audio batch [batch, 1, n_mels, max_time]
        conditions: Condition batch [batch, 2]
        lengths: List of original lengths
        metadata: List of metadata dicts
    """
    audios, conditions, metadatas = zip(*batch)

    # Get maximum length in batch
    lengths = [audio.shape[-1] for audio in audios]
    max_length = max(lengths)

    # Pad all samples to max length
    padded_audios = []
    for audio in audios:
        if audio.shape[-1] < max_length:
            pad_amount = max_length - audio.shape[-1]
            # Pad on the right (end of sequence)
            audio_padded = torch.nn.functional.pad(audio, (0, pad_amount))
        else:
            audio_padded = audio
        padded_audios.append(audio_padded)

    # Stack into batch
    audio_batch = torch.stack(padded_audios, dim=0)
    condition_batch = torch.stack(list(conditions), dim=0)

    return audio_batch, condition_batch, lengths, list(metadatas)


def create_variable_length_dataloaders(
    config: dict,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders with variable-length support

    Args:
        config: Configuration dictionary
        batch_size: Batch size for training
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create full dataset
    full_dataset = VariableLengthElectricPianoDataset(
        instrument_dir=config['data']['instrument_dir'],
        sample_rate=config['data']['sample_rate'],
        n_fft=config['model']['n_fft'],
        hop_length=config['model']['hop_length'],
        n_mels=config['model']['n_mels'],
        use_mel_spectrogram=config['model']['use_mel_spectrogram'],
        target_sample_rate=config['data']['sample_rate'],
        max_duration=config['data'].get('max_audio_length', None),  # Optional
        normalize_audio=True
    )

    # Split dataset
    total_size = len(full_dataset)
    train_size = int(config['data']['train_split'] * total_size)
    val_size = int(config['data']['val_split'] * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=variable_length_collate_fn,
        drop_last=True  # Drop incomplete batches for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=variable_length_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=variable_length_collate_fn
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader


# Test
if __name__ == "__main__":
    import yaml

    # Load config
    config_path = CONFIG_FILE
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Creating variable-length dataloaders...")
    train_loader, val_loader, test_loader = create_variable_length_dataloaders(
        config=config,
        batch_size=4,
        num_workers=0
    )

    print("\nTesting dataloader:")
    for batch_idx, (audio, condition, lengths, metadata) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Audio shape: {audio.shape}")
        print(f"  Condition shape: {condition.shape}")
        print(f"  Lengths: {lengths}")
        print(f"  MIDI notes: {[m['midi_note'] for m in metadata]}")
        print(f"  Velocities: {[m['velocity'] for m in metadata]}")

        if batch_idx >= 2:
            break

    print("\nDataset test completed!")
