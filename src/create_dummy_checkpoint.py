"""
Create dummy GAN checkpoint for testing inference pipeline
"""

import torch
import yaml
from pathlib import Path
from variable_gan_model import VariableLengthGAN

# Load config
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Creating dummy GAN checkpoint for testing...")

# Create model
gan = VariableLengthGAN(
    latent_dim=config['model']['latent_dim'],
    condition_dim=2,
    sample_rate=config['data']['sample_rate'],
    hop_length=config['model']['hop_length']
)

# Create checkpoint
checkpoint = {
    'epoch': 0,
    'generator_state_dict': gan.generator.state_dict(),
    'discriminator_state_dict': gan.discriminator.state_dict(),
    'best_val_loss': float('inf'),
    'global_step': 0,
    'config': config
}

# Save
checkpoint_dir = Path('checkpoints')
checkpoint_dir.mkdir(exist_ok=True)

checkpoint_path = checkpoint_dir / 'test_gan_model.pth'
torch.save(checkpoint, checkpoint_path)

print(f"[OK] Dummy checkpoint created: {checkpoint_path}")
print(f"  Generator params: {sum(p.numel() for p in gan.generator.parameters()):,}")
print(f"  Discriminator params: {sum(p.numel() for p in gan.discriminator.parameters()):,}")
print("\nYou can now test generation with:")
print(f"  python runhelper.py generate --midi 60 --velocity 5 --checkpoint test_gan_model.pth")
