# Neuralissimo - Variable-Length GAN for Piano Sound Generation

Kompletně refaktorovaný systém pro generování realistických piano samples pomocí **Conditional GAN** s podporou variabilní délky zvuku.

## 🎹 Hlavní funkce

- **Variable-Length Generation**: Automatické určení délky podle MIDI nóty (nízké tóny = delší dozvuk)
- **HiFi-GAN Vocoder**: Vysoká kvalita mel-to-audio konverze (lepší než Griffin-Lim)
- **Spectral Style Transfer**: Aplikace spektrální barvy z referenčního MP3 tracku
- **Proper Naming**: Výstupy ve formátu `mXXX-velX-fXX.wav` jako originální samples
- **Multi Sample Rate**: Podpora 44.1kHz i 48kHz

## 📁 Struktura projektu

```
neuralissimo/
├── config.yaml                    # Hlavní konfigurace (aktualizováno pro GAN)
├── runhelper.py                   # Hlavní rozhraní pro všechny operace
├── src/
│   ├── variable_gan_model.py      # Variable-Length GAN architektura
│   ├── variable_dataset.py        # Dataset s podporou variabilní délky
│   ├── train_gan.py               # GAN training script
│   ├── inference_gan.py           # Generování samples
│   ├── hifigan_vocoder.py         # HiFi-GAN vocoder (mel→audio)
│   ├── spectral_style_transfer.py # Style transfer z MP3
│   ├── diffusion_model.py         # DDPM model (alternativa, pro budoucí použití)
│   ├── analyze_durations.py       # Analýza délek samples
│   └── configurepaths.py          # Správa cest
├── outputs/                       # Generované samples
└── checkpoints/                   # Uložené modely
```

## 🚀 Rychlý start

### 1. Analýza datasetu

Před trénováním analyzujte distribuci délek samples:

```bash
python runhelper.py analyze-durations
```

**Výstup**: Zobrazí průměrné délky pro každou MIDI nótu.

### 2. Trénování GAN modelu

```bash
# Základní trénování
python runhelper.py train

# Pokračování z checkpointu
python runhelper.py train --resume checkpoints/gan_checkpoint_epoch_50.pth

# Trénování na GPU
python runhelper.py train --device cuda
```

**Trénování trvá**: ~200 epoch pro dobré výsledky (závisí na datasetu)

### 3. Generování jednotlivého sample

```bash
# Základní generování
python runhelper.py generate --midi 60 --velocity 5

# S vlastním checkpointem
python runhelper.py generate --midi 60 --velocity 5 --checkpoint best_gan_model.pth

# Více variací
python runhelper.py generate --midi 60 --velocity 5 --num-samples 5

# S větší variabilitou
python runhelper.py generate --midi 60 --velocity 5 --temperature 1.5
```

**Výstup**: `outputs/m060-vel5-f48.wav`

### 4. Generování celého datasetu

```bash
# Celý rozsah (MIDI 33-94, velocity 0-7)
python runhelper.py generate-dataset

# Pouze vybraný rozsah
python runhelper.py generate-dataset --midi-start 50 --midi-end 70

# Pouze vysoké velocity
python runhelper.py generate-dataset --vel-start 5 --vel-end 7
```

**Výstup**: 496 samples (62 MIDI nóty × 8 velocity levels)

## 🎛️ Konfigurace (config.yaml)

### Základní nastavení

```yaml
model:
  type: "gan"                      # Model type
  latent_dim: 128                  # Velikost latent space
  midi_note_range: [33, 94]        # Rozsah MIDI nót
  velocity_levels: 8               # Počet velocity úrovní (0-7)
```

### Training parametry

```yaml
training:
  batch_size: 8                    # Batch size (menší pro variable-length)
  num_epochs: 200                  # Počet epoch
  generator_lr: 0.0002             # Learning rate pro generátor
  discriminator_lr: 0.0002         # Learning rate pro discriminator
  gan_loss_type: "hinge"           # Typ loss (hinge, bce, wgan)
```

### Vocoder nastavení

```yaml
inference:
  vocoder_type: "hifigan"          # hifigan nebo griffin_lim
  hifigan_checkpoint: null         # Cesta k pre-trained vocoder
  griffin_lim_iterations: 300      # Pokud používáte Griffin-Lim
```

### Spectral Style Transfer

```yaml
style_transfer:
  enabled: false                   # Zapnout style transfer
  reference_track: "path/to/reference.mp3"
  spectral_loss_weight: 1.0
  perceptual_loss_weight: 0.5
```

## 📊 Analýza délek podle MIDI nóty

Dataset obsahuje samples s velmi variabilní délkou:

| MIDI Range | Průměrná délka | Popis |
|------------|----------------|-------|
| 33-45      | 14-29s         | Nízké tóny (dlouhý dozvuk) |
| 50-70      | 5-12s          | Střední tóny |
| 85-94      | 2.4-4s         | Vysoké tóny (krátký dozvuk) |

**Variable-Length GAN** automaticky generuje správnou délku podle MIDI nóty.

## 🔧 Pokročilé použití

### Test HiFi-GAN Vocoder

```bash
python runhelper.py test-vocoder
```

### Použití vlastního HiFi-GAN checkpointu

1. Stáhněte pre-trained HiFi-GAN model
2. Upravte `config.yaml`:
   ```yaml
   inference:
     hifigan_checkpoint: "path/to/hifigan_model.pth"
   ```

### Spectral Style Transfer z MP3

1. Připravte referenční track (MP3/WAV)
2. Upravte `config.yaml`:
   ```yaml
   style_transfer:
     enabled: true
     reference_track: "path/to/reference.mp3"
   ```
3. Spusťte trénování (style transfer se aplikuje automaticky)

### Přímé volání Python scriptů

```bash
# Trénování
cd src
python train_gan.py --config ../config.yaml

# Generování
cd src
python inference_gan.py \
  --gan-checkpoint checkpoints/best_gan_model.pth \
  --config ../config.yaml \
  --midi 60 \
  --velocity 5 \
  --output-dir ../outputs
```

## 🎵 Kvalita výstupu

### Oproti původnímu VAE:

| Metrika | VAE (Griffin-Lim) | GAN (HiFi-GAN) |
|---------|-------------------|----------------|
| **Spektrální kvalita** | ⭐⭐ (chrcení) | ⭐⭐⭐⭐⭐ |
| **Délka samples** | Fixed 3s | Variabilní (2-64s) |
| **Realistický dozvuk** | ❌ | ✅ |
| **Spektrální barva** | Špatná | Odpovídá MP3 reference |
| **Inference rychlost** | Pomalá | Rychlá |

### Klíčová vylepšení:

1. ✅ **HiFi-GAN vocoder** → Eliminuje "chrcení" z Griffin-Lim
2. ✅ **Variable-length** → Realistické délky podle frekvence
3. ✅ **Spectral style transfer** → Správná barva z reference
4. ✅ **GAN architektura** → Ostřejší, realističtější zvuk než VAE

## 📝 Formát výstupních souborů

Všechny generované samples mají správné pojmenování:

```
mXXX-velX-fXX.wav

m060-vel5-f48.wav   → MIDI 60, Velocity 5, 48kHz
m033-vel0-f48.wav   → MIDI 33, Velocity 0, 48kHz
m094-vel7-f48.wav   → MIDI 94, Velocity 7, 48kHz
```

## 🔬 Architektura modelu

### Variable-Length Generator

- **Input**: Noise (128D) + Condition (MIDI + velocity)
- **Architecture**: Conditional upsampling s adaptive temporal length
- **Output**: Mel spectrogram (128 × variabilní délka)

### Variable-Length Discriminator

- **Input**: Mel spectrogram + Condition
- **Architecture**: PatchGAN s global pooling (pro variable-length)
- **Output**: Real/Fake classification

### HiFi-GAN Vocoder

- **Input**: Mel spectrogram (128 mels)
- **Architecture**: Multi-scale residual blocks
- **Output**: High-quality waveform (44.1/48kHz)

## 🎓 Training Tips

1. **Checkpoint saving**: Model se ukládá každých 10 epoch
2. **Best model**: Automaticky se ukládá nejlepší model (podle validation loss)
3. **TensorBoard**: Sledujte training progress:
   ```bash
   tensorboard --logdir src/logs/gan
   ```
4. **Overfitting**: Pokud validation loss roste, snižte learning rate nebo přidejte dropout

## 🐛 Troubleshooting

### "RuntimeError: CUDA out of memory"

→ Snižte `batch_size` v `config.yaml` na 4 nebo 2

### "Checkpoint not found"

→ Ujistěte se, že máte natrénovaný model v `src/checkpoints/`

### Špatná kvalita výstupu

→ Natrénujte model déle (minimálně 100 epoch)
→ Použijte pre-trained HiFi-GAN vocoder

### Všechny sample mají stejnou délku

→ Ověřte `use_variable_length: true` v `config.yaml`

## 📚 Další dokumentace

- `diffusion_model.py` - DDPM model (alternativa k GAN, zatím nepoužito)
- `spectral_style_transfer.py` - Detaily style transfer algoritmu
- `analyze_durations.py` - Analýza délek samples

## 🚧 Budoucí vylepšení

- [ ] Fine-tuning HiFi-GAN vocoder na piano samples
- [ ] Multi-track spectral style transfer
- [ ] Real-time inference optimization
- [ ] MIDI-to-audio direct generation (bez intermediate mel)

## 📄 Licence

Internal use only.

---

**Autor**: Claude Code (Anthropic)
**Datum**: 2025-10-19
**Verze**: 2.0 (Complete GAN refactoring)
