"""
Training script for Variable-Length Conditional GAN

Stage 1: Train on instrument samples (MIDI + velocity conditioning)
Stage 2: Spectral style transfer from MP3 reference track
"""

import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from configurepaths import *

from variable_dataset import create_variable_length_dataloaders
from variable_gan_model import (
    VariableLengthGAN,
    hinge_loss_dis,
    hinge_loss_gen
)


class GANTrainer:
    """Trainer for Variable-Length Conditional GAN"""

    def __init__(self, config: dict, device: str = 'cuda'):
        self.config = config
        self.device = device

        # Create dataloaders
        print("Creating variable-length dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = create_variable_length_dataloaders(
            config=config,
            batch_size=config['training']['batch_size'],
            num_workers=0,
            pin_memory=True if device == 'cuda' else False
        )

        # Create GAN model
        print("\nInitializing GAN model...")
        self.gan = VariableLengthGAN(
            latent_dim=config['model']['latent_dim'],
            condition_dim=2,
            sample_rate=config['data']['sample_rate'],
            hop_length=config['model']['hop_length']
        ).to(device)

        # Count parameters
        gen_params = sum(p.numel() for p in self.gan.generator.parameters())
        disc_params = sum(p.numel() for p in self.gan.discriminator.parameters())
        print(f"Generator parameters: {gen_params:,}")
        print(f"Discriminator parameters: {disc_params:,}")

        # Create optimizers
        self.g_optimizer = self._create_optimizer(
            self.gan.generator.parameters(),
            lr=config['training']['generator_lr']
        )
        self.d_optimizer = self._create_optimizer(
            self.gan.discriminator.parameters(),
            lr=config['training']['discriminator_lr']
        )

        # Learning rate schedulers
        self.g_scheduler = self._create_scheduler(self.g_optimizer)
        self.d_scheduler = self._create_scheduler(self.d_optimizer)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0

        # Loss tracking
        self.loss_history = {
            'g_loss': [],
            'd_loss': [],
            'd_real': [],
            'd_fake': []
        }

        # Create checkpoint directory
        CHECKPOINTS_DIR.mkdir(exist_ok=True)

        # Create TensorBoard writer
        LOGS_DIR.mkdir(exist_ok=True)
        self.writer = SummaryWriter(str(LOGS_DIR / 'gan'))

    def _create_optimizer(self, parameters, lr: float):
        """Create optimizer"""
        optimizer_name = self.config['training']['optimizer'].lower()

        if optimizer_name == 'adam':
            return optim.Adam(
                parameters,
                lr=lr,
                betas=(0.5, 0.999),  # Standard for GANs
                weight_decay=self.config['training']['weight_decay']
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                parameters,
                lr=lr,
                betas=(0.5, 0.999),
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler"""
        scheduler_name = self.config['training'].get('scheduler', 'reduce_on_plateau').lower()

        if scheduler_name == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=self.config['training']['patience']
            )
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=1e-6
            )
        else:
            return None

    def train_step(self, real_audio: torch.Tensor, condition: torch.Tensor,
                  lengths: list) -> dict:
        """
        Single training step

        Args:
            real_audio: Real mel spectrograms [batch, 1, n_mels, max_time]
            condition: Conditioning [batch, 2]
            lengths: List of actual lengths (for masking padded regions)

        Returns:
            Dictionary of losses
        """
        batch_size = real_audio.shape[0]

        # ==================
        # Train Discriminator
        # ==================
        self.d_optimizer.zero_grad()

        # Real samples
        real_output = self.gan.discriminator(real_audio, condition)

        # Generate fake samples
        noise = torch.randn(batch_size, self.gan.latent_dim, device=self.device)
        fake_audio = self.gan.generator(noise, condition, lengths)

        # Fake samples
        fake_output = self.gan.discriminator(fake_audio.detach(), condition)

        # Discriminator loss
        d_loss = hinge_loss_dis(real_output, fake_output)

        d_loss.backward()
        self.d_optimizer.step()

        # ==================
        # Train Generator
        # ==================
        self.g_optimizer.zero_grad()

        # Generate new fake samples
        noise = torch.randn(batch_size, self.gan.latent_dim, device=self.device)
        fake_audio = self.gan.generator(noise, condition, lengths)

        # Generator wants discriminator to think fake is real
        fake_output = self.gan.discriminator(fake_audio, condition)

        # Generator loss
        g_loss = hinge_loss_gen(fake_output)

        g_loss.backward()
        self.g_optimizer.step()

        # Return losses
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'd_real': real_output.mean().item(),
            'd_fake': fake_output.mean().item()
        }

    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.gan.generator.train()
        self.gan.discriminator.train()

        epoch_losses = {
            'g_loss': 0,
            'd_loss': 0,
            'd_real': 0,
            'd_fake': 0
        }
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (audio, condition, lengths, metadata) in enumerate(progress_bar):
            # Move to device
            audio = audio.to(self.device)
            condition = condition.to(self.device)

            # Training step
            losses = self.train_step(audio, condition, lengths)

            # Accumulate losses
            for key in epoch_losses.keys():
                epoch_losses[key] += losses[key]
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'G': f"{losses['g_loss']:.3f}",
                'D': f"{losses['d_loss']:.3f}",
                'D(real)': f"{losses['d_real']:.3f}",
                'D(fake)': f"{losses['d_fake']:.3f}"
            })

            # Log to TensorBoard
            if batch_idx % self.config['training']['log_every'] == 0:
                for key, val in losses.items():
                    self.writer.add_scalar(f'Train/{key}', val, self.global_step)
                self.global_step += 1

        # Average losses
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the model"""
        self.gan.generator.eval()
        self.gan.discriminator.eval()

        val_losses = {
            'g_loss': 0,
            'd_loss': 0,
            'd_real': 0,
            'd_fake': 0
        }
        num_batches = 0

        for audio, condition, lengths, metadata in tqdm(self.val_loader, desc="Validation"):
            batch_size = audio.shape[0]

            # Move to device
            audio = audio.to(self.device)
            condition = condition.to(self.device)

            # Real samples
            real_output = self.gan.discriminator(audio, condition)

            # Generate fake samples
            noise = torch.randn(batch_size, self.gan.latent_dim, device=self.device)
            fake_audio = self.gan.generator(noise, condition, lengths)
            fake_output = self.gan.discriminator(fake_audio, condition)

            # Losses
            d_loss = hinge_loss_dis(real_output, fake_output)
            g_loss = hinge_loss_gen(fake_output)

            val_losses['g_loss'] += g_loss.item()
            val_losses['d_loss'] += d_loss.item()
            val_losses['d_real'] += real_output.mean().item()
            val_losses['d_fake'] += fake_output.mean().item()
            num_batches += 1

        # Average losses
        for key in val_losses.keys():
            val_losses[key] /= num_batches

        return val_losses

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state_dict': self.gan.generator.state_dict(),
            'discriminator_state_dict': self.gan.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'config': self.config
        }

        if self.g_scheduler is not None:
            checkpoint['g_scheduler_state_dict'] = self.g_scheduler.state_dict()
        if self.d_scheduler is not None:
            checkpoint['d_scheduler_state_dict'] = self.d_scheduler.state_dict()

        filepath = CHECKPOINTS_DIR / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.gan.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.gan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.global_step = checkpoint['global_step']

        if self.g_scheduler is not None and 'g_scheduler_state_dict' in checkpoint:
            self.g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
        if self.d_scheduler is not None and 'd_scheduler_state_dict' in checkpoint:
            self.d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])

        print(f"Checkpoint loaded: {filepath}")
        print(f"Resuming from epoch {self.current_epoch}")

    def train(self, num_epochs: int):
        """Main training loop"""
        print(f"\nStarting GAN training for {num_epochs} epochs...")
        print("=" * 60)

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Train
            train_losses = self.train_epoch()

            # Validate
            val_losses = self.validate()

            # Log epoch metrics
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train - G: {train_losses['g_loss']:.4f}, "
                  f"D: {train_losses['d_loss']:.4f}, "
                  f"D(real): {train_losses['d_real']:.4f}, "
                  f"D(fake): {train_losses['d_fake']:.4f}")
            print(f"  Val   - G: {val_losses['g_loss']:.4f}, "
                  f"D: {val_losses['d_loss']:.4f}, "
                  f"D(real): {val_losses['d_real']:.4f}, "
                  f"D(fake): {val_losses['d_fake']:.4f}")

            # TensorBoard logging
            for key in train_losses.keys():
                self.writer.add_scalar(f'Epoch/Train_{key}', train_losses[key], epoch)
                self.writer.add_scalar(f'Epoch/Val_{key}', val_losses[key], epoch)

            self.writer.add_scalar('Epoch/G_LR', self.g_optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('Epoch/D_LR', self.d_optimizer.param_groups[0]['lr'], epoch)

            # Update learning rate
            if self.g_scheduler is not None:
                if isinstance(self.g_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.g_scheduler.step(val_losses['g_loss'])
                else:
                    self.g_scheduler.step()

            if self.d_scheduler is not None:
                if isinstance(self.d_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.d_scheduler.step(val_losses['d_loss'])
                else:
                    self.d_scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_every'] == 0:
                self.save_checkpoint(f'gan_checkpoint_epoch_{epoch+1}.pth')

            # Save best model (based on generator loss)
            if val_losses['g_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['g_loss']
                self.save_checkpoint('best_gan_model.pth')
                print(f"  New best model! Val G loss: {self.best_val_loss:.4f}")

            print("=" * 60)

        print("\nTraining complete!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Variable-Length Conditional GAN')
    parser.add_argument('--config', type=str, default=str(CONFIG_FILE),
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Determine device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    # Create trainer
    trainer = GANTrainer(config, device=device)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    num_epochs = config['training']['num_epochs']
    trainer.train(num_epochs)


if __name__ == "__main__":
    main()
