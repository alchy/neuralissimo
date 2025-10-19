"""
configurepaths.py

Centralized path management for the Neurailssimo project.

This module defines all the necessary paths for the project, ensuring that
all other scripts can access them consistently. Paths are defined as absolute
paths using pathlib for robustness.
"""

from pathlib import Path

# The absolute path to the 'src' directory, which is the parent of this file.
SRC_ROOT = Path(__file__).resolve().parent

# The absolute path to the project root directory, which is the parent of 'src'.
PROJECT_ROOT = SRC_ROOT.parent

# --- Core Directories ---

# Checkpoints for saving model states during training.
CHECKPOINTS_DIR = SRC_ROOT / "checkpoints"

# Logs for TensorBoard.
LOGS_DIR = PROJECT_ROOT / "logs"

# Output directory for generated audio samples.
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Data directory for storing dataset-related files.
DATA_DIR = PROJECT_ROOT / "data"

# --- Core Files ---

# Main configuration file for the project.
CONFIG_FILE = PROJECT_ROOT / "config.yaml"

# Cached spectral profile for style transfer.
SPECTRAL_PROFILE_FILE = DATA_DIR / "spectral_profile.pkl"

# --- Style Transfer ---

# Default directory for style transfer reference audio files.
# This is a user-specific path and might be better placed in a local config,
# but is provided here as a default.
DEFAULT_STYLE_REFERENCE_DIR = Path("C:/SoundBanks/IthacaPlayer/instrument-styles")


if __name__ == '__main__':
    # When run directly, print all defined paths for verification purposes.
    print("--- Neurailssimo Path Configuration ---")
    print(f"Project Root:          {PROJECT_ROOT}")
    print(f"Source Root:           {SRC_ROOT}")
    print(f"Checkpoints Directory: {CHECKPOINTS_DIR}")
    print(f"Logs Directory:        {LOGS_DIR}")
    print(f"Outputs Directory:     {OUTPUTS_DIR}")
    print(f"Data Directory:        {DATA_DIR}")
    print(f"Config File:           {CONFIG_FILE}")
    print(f"Spectral Profile File: {SPECTRAL_PROFILE_FILE}")
    print(f"Default Style Ref Dir: {DEFAULT_STYLE_REFERENCE_DIR}")
    print("------------------------------------")
