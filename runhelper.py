"""
Helper script for GAN-based piano sound generation workflow

Simplified interface for:
- Training Variable-Length GAN
- Generating samples with proper naming (mXXX-velX-fXX.wav)
- Testing HiFi-GAN vocoder
"""

import argparse
import subprocess
import sys
from pathlib import Path


def train_gan(args):
    """Train the Variable-Length GAN model"""
    print("=" * 60)
    print("TRAINING VARIABLE-LENGTH GAN")
    print("=" * 60)

    cmd = [
        sys.executable,
        "train_gan.py",
        "--config", "../config.yaml"
    ]

    if args.resume:
        cmd.extend(["--resume", args.resume])

    if args.device:
        cmd.extend(["--device", args.device])

    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {Path.cwd() / 'src'}")
    print("=" * 60)

    result = subprocess.run(
        cmd,
        cwd="src",
        env=None
    )

    if result.returncode != 0:
        print(f"\nTraining failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    else:
        print("\nTraining completed successfully!")


def generate_sample(args):
    """Generate a single sample"""
    print("=" * 60)
    print(f"GENERATING SAMPLE: MIDI {args.midi}, Velocity {args.velocity}")
    print("=" * 60)

    # Check if checkpoint exists
    checkpoint_path = Path("src/checkpoints") / args.checkpoint
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        checkpoints_dir = Path("src/checkpoints")
        if checkpoints_dir.exists():
            for ckpt in checkpoints_dir.glob("*.pth"):
                print(f"  - {ckpt.name}")
        sys.exit(1)

    # Use relative path from src/ directory
    checkpoint_rel = f"checkpoints/{args.checkpoint}"

    cmd = [
        sys.executable,
        "inference_gan.py",
        "--gan-checkpoint", checkpoint_rel,
        "--config", "../config.yaml",
        "--midi", str(args.midi),
        "--velocity", str(args.velocity),
        "--output-dir", "../outputs"
    ]

    if args.vocoder:
        cmd.extend(["--vocoder-checkpoint", args.vocoder])

    if args.num_samples and args.num_samples > 1:
        cmd.extend(["--num-samples", str(args.num_samples)])

    if args.temperature:
        cmd.extend(["--temperature", str(args.temperature)])

    if args.device:
        cmd.extend(["--device", args.device])

    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {Path.cwd() / 'src'}")
    print("=" * 60)

    result = subprocess.run(
        cmd,
        cwd="src",
        env=None
    )

    if result.returncode != 0:
        print(f"\nGeneration failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def generate_dataset(args):
    """Generate full dataset of samples (all MIDI notes and velocities)"""
    print("=" * 60)
    print("GENERATING FULL DATASET")
    print("=" * 60)

    checkpoint_path = Path("src/checkpoints") / args.checkpoint
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # MIDI range
    midi_start = args.midi_start or 33
    midi_end = args.midi_end or 94

    # Velocity range
    vel_start = args.vel_start or 0
    vel_end = args.vel_end or 7

    total_samples = (midi_end - midi_start + 1) * (vel_end - vel_start + 1)
    print(f"Will generate {total_samples} samples")
    print(f"MIDI range: {midi_start} to {midi_end}")
    print(f"Velocity range: {vel_start} to {vel_end}")
    print("=" * 60)

    # Generate each combination
    count = 0
    for midi in range(midi_start, midi_end + 1):
        for vel in range(vel_start, vel_end + 1):
            count += 1
            print(f"\n[{count}/{total_samples}] Generating MIDI {midi}, Velocity {vel}...")

            cmd = [
                sys.executable,
                "inference_gan.py",
                "--gan-checkpoint", str(checkpoint_path),
                "--config", "../config.yaml",
                "--midi", str(midi),
                "--velocity", str(vel),
                "--output-dir", "../outputs"
            ]

            if args.vocoder:
                cmd.extend(["--vocoder-checkpoint", args.vocoder])

            if args.device:
                cmd.extend(["--device", args.device])

            result = subprocess.run(
                cmd,
                cwd="src",
                env=None,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"  ERROR: Generation failed for MIDI {midi}, Velocity {vel}")
                print(result.stderr)
            else:
                print(f"  SUCCESS: m{midi:03d}-vel{vel}-f48.wav")

    print("\n" + "=" * 60)
    print(f"Dataset generation complete! Generated {count} samples")
    print("=" * 60)


def test_vocoder(args):
    """Test HiFi-GAN vocoder"""
    print("=" * 60)
    print("TESTING HIFI-GAN VOCODER")
    print("=" * 60)

    cmd = [
        sys.executable,
        "hifigan_vocoder.py"
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {Path.cwd() / 'src'}")
    print("=" * 60)

    result = subprocess.run(
        cmd,
        cwd="src",
        env=None
    )

    if result.returncode != 0:
        print(f"\nVocoder test failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def analyze_durations(args):
    """Analyze duration distribution in dataset"""
    print("=" * 60)
    print("ANALYZING SAMPLE DURATIONS")
    print("=" * 60)

    cmd = [
        sys.executable,
        "analyze_durations.py"
    ]

    result = subprocess.run(
        cmd,
        cwd="src",
        env=None
    )

    if result.returncode != 0:
        print(f"\nAnalysis failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Helper script for GAN-based piano generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train GAN model
  python runhelper_gan.py train

  # Resume training from checkpoint
  python runhelper_gan.py train --resume checkpoints/gan_checkpoint_epoch_50.pth

  # Generate single sample
  python runhelper_gan.py generate --midi 60 --velocity 5

  # Generate with custom checkpoint
  python runhelper_gan.py generate --midi 60 --velocity 5 --checkpoint best_gan_model.pth

  # Generate full dataset (all MIDI/velocity combinations)
  python runhelper_gan.py generate-dataset

  # Generate subset of dataset
  python runhelper_gan.py generate-dataset --midi-start 50 --midi-end 70

  # Test vocoder
  python runhelper_gan.py test-vocoder

  # Analyze sample durations
  python runhelper_gan.py analyze-durations
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train GAN model')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    train_parser.add_argument('--device', type=str, help='Device (cuda/cpu)')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate single sample')
    gen_parser.add_argument('--midi', type=int, required=True, help='MIDI note (33-94)')
    gen_parser.add_argument('--velocity', type=int, required=True, help='Velocity (0-7)')
    gen_parser.add_argument('--checkpoint', type=str, default='best_gan_model.pth',
                           help='GAN checkpoint filename')
    gen_parser.add_argument('--vocoder', type=str, help='HiFi-GAN vocoder checkpoint')
    gen_parser.add_argument('--num-samples', type=int, help='Number of variations to generate')
    gen_parser.add_argument('--temperature', type=float, help='Generation temperature')
    gen_parser.add_argument('--device', type=str, help='Device (cuda/cpu)')

    # Generate dataset command
    dataset_parser = subparsers.add_parser('generate-dataset',
                                          help='Generate full dataset')
    dataset_parser.add_argument('--checkpoint', type=str, default='best_gan_model.pth',
                               help='GAN checkpoint filename')
    dataset_parser.add_argument('--vocoder', type=str, help='HiFi-GAN vocoder checkpoint')
    dataset_parser.add_argument('--midi-start', type=int, help='Start MIDI note (default: 33)')
    dataset_parser.add_argument('--midi-end', type=int, help='End MIDI note (default: 94)')
    dataset_parser.add_argument('--vel-start', type=int, help='Start velocity (default: 0)')
    dataset_parser.add_argument('--vel-end', type=int, help='End velocity (default: 7)')
    dataset_parser.add_argument('--device', type=str, help='Device (cuda/cpu)')

    # Test vocoder command
    vocoder_parser = subparsers.add_parser('test-vocoder', help='Test HiFi-GAN vocoder')

    # Analyze durations command
    analyze_parser = subparsers.add_parser('analyze-durations',
                                          help='Analyze sample durations')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == 'train':
        train_gan(args)
    elif args.command == 'generate':
        generate_sample(args)
    elif args.command == 'generate-dataset':
        generate_dataset(args)
    elif args.command == 'test-vocoder':
        test_vocoder(args)
    elif args.command == 'analyze-durations':
        analyze_durations(args)


if __name__ == "__main__":
    main()
