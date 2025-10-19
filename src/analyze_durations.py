"""
Analyze audio durations in the dataset to understand length distribution
"""

import os
import glob
import re
import soundfile as sf
import numpy as np
from collections import defaultdict

def parse_filename(filepath):
    """Parse MIDI note and velocity from filename"""
    basename = os.path.basename(filepath)
    match = re.match(r'm(\d+)-vel(\d+)-f(\d+)\.wav', basename)
    if match:
        return {
            'midi_note': int(match.group(1)),
            'velocity': int(match.group(2)),
            'sample_rate': int(match.group(3)) * 1000
        }
    return None

def analyze_durations(instrument_dir):
    """Analyze durations of all samples"""
    pattern = os.path.join(instrument_dir, "*.wav")
    file_paths = glob.glob(pattern)

    durations_by_midi = defaultdict(list)
    durations_by_velocity = defaultdict(list)
    all_durations = []

    print(f"Analyzing {len(file_paths)} files...")

    for filepath in file_paths:
        meta = parse_filename(filepath)
        if not meta:
            continue

        # Load audio to get duration
        try:
            audio_data, sr = sf.read(filepath)
            duration = len(audio_data) / sr

            midi_note = meta['midi_note']
            velocity = meta['velocity']

            durations_by_midi[midi_note].append(duration)
            durations_by_velocity[velocity].append(duration)
            all_durations.append((midi_note, velocity, duration, sr))

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    # Statistics
    print("\n" + "="*60)
    print("DURATION ANALYSIS")
    print("="*60)

    # Overall statistics
    if all_durations:
        durations = [d[2] for d in all_durations]
        print(f"\nOverall Statistics:")
        print(f"  Total samples: {len(durations)}")
        print(f"  Min duration: {min(durations):.3f}s")
        print(f"  Max duration: {max(durations):.3f}s")
        print(f"  Mean duration: {np.mean(durations):.3f}s")
        print(f"  Median duration: {np.median(durations):.3f}s")
        print(f"  Std duration: {np.std(durations):.3f}s")

    # Duration by MIDI note
    print(f"\nDuration by MIDI Note:")
    print(f"{'MIDI':<6} {'Count':<8} {'Min':<8} {'Max':<8} {'Mean':<8} {'Median':<8}")
    print("-" * 60)

    midi_notes = sorted(durations_by_midi.keys())
    for midi in midi_notes:
        durs = durations_by_midi[midi]
        print(f"{midi:<6} {len(durs):<8} {min(durs):<8.3f} {max(durs):<8.3f} "
              f"{np.mean(durs):<8.3f} {np.median(durs):<8.3f}")

    # Duration by velocity
    print(f"\nDuration by Velocity Level:")
    print(f"{'Vel':<6} {'Count':<8} {'Min':<8} {'Max':<8} {'Mean':<8} {'Median':<8}")
    print("-" * 60)

    velocities = sorted(durations_by_velocity.keys())
    for vel in velocities:
        durs = durations_by_velocity[vel]
        print(f"{vel:<6} {len(durs):<8} {min(durs):<8.3f} {max(durs):<8.3f} "
              f"{np.mean(durs):<8.3f} {np.median(durs):<8.3f}")

    # Create lookup table for duration prediction
    print(f"\nDuration Lookup Table (by MIDI note):")
    print("midi_duration_map = {")
    for midi in midi_notes:
        mean_dur = np.mean(durations_by_midi[midi])
        print(f"    {midi}: {mean_dur:.3f},")
    print("}")

    return durations_by_midi, durations_by_velocity, all_durations


if __name__ == "__main__":
    instrument_dir = r"C:\SoundBanks\IthacaPlayer\instrument"

    if os.path.exists(instrument_dir):
        durations_by_midi, durations_by_velocity, all_durations = analyze_durations(instrument_dir)

        # Plot duration vs MIDI note (optional, requires matplotlib)
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            matplotlib.use('Agg')  # Non-interactive backend

            midi_notes = sorted(durations_by_midi.keys())
            mean_durations = [np.mean(durations_by_midi[m]) for m in midi_notes]

            plt.figure(figsize=(12, 6))
            plt.plot(midi_notes, mean_durations, 'o-', markersize=5)
            plt.xlabel('MIDI Note')
            plt.ylabel('Mean Duration (seconds)')
            plt.title('Duration vs MIDI Note')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('duration_analysis.png', dpi=150)
            print(f"\nPlot saved to duration_analysis.png")
        except ImportError:
            print(f"\nMatplotlib not available, skipping plot")
        except Exception as e:
            print(f"\nCould not create plot: {e}")

    else:
        print(f"Directory not found: {instrument_dir}")
