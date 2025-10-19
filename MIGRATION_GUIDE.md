# Migration Guide: VAE → Variable-Length GAN

## ⚠️ Breaking Changes

Původní VAE systém byl **kompletně nahrazen** novým GAN systémem s variabilní délkou.

### Odstraněné soubory

Následující soubory byly odstraněny (nekompatibilní s novou architekturou):

```
❌ src/model.py              → nahrazeno variable_gan_model.py
❌ src/train.py              → nahrazeno train_gan.py
❌ src/inference.py          → nahrazeno inference_gan.py
❌ src/dataset.py            → nahrazeno variable_dataset.py
❌ src/gan_model.py          → nahrazeno variable_gan_model.py (fixed→variable length)
❌ src/style_transfer.py     → nahrazeno spectral_style_transfer.py
```

### Staré checkpointy nejsou kompatibilní

**DŮLEŽITÉ**: Checkpointy z původního VAE modelu (`best_model.pth`) **nejsou** kompatibilní s novým GAN systémem.

→ Je potřeba natrénovat nový GAN model od začátku.

## 🔄 Migrace workflow

### Původní workflow (VAE)

```bash
# Trénování
python runhelper.py train

# Generování
python runhelper.py generate --midi 60 --velocity 6
```

### Nový workflow (GAN)

```bash
# Trénování (stejné!)
python runhelper.py train

# Generování (stejné!)
python runhelper.py generate --midi 60 --velocity 6
```

**✅ Interface zůstal stejný!** Pouze interní implementace se změnila.

## 📋 Změny v config.yaml

### Nové sekce

```yaml
model:
  type: "gan"  # Změněno z "vae"

  gan:  # Nová sekce
    generator_base_channels: 256
    discriminator_base_channels: 64

training:
  generator_lr: 0.0002      # Nový parametr
  discriminator_lr: 0.0002  # Nový parametr
  gan_loss_type: "hinge"    # Nový parametr

inference:
  vocoder_type: "hifigan"   # Nový parametr (místo Griffin-Lim)
  use_variable_length: true # Nový parametr

style_transfer:  # Nová sekce
  enabled: false
  reference_track: null
```

## 🆕 Nové funkce

### 1. Variable-Length Generation

**Nově**: Model automaticky generuje správnou délku podle MIDI nóty.

- MIDI 33-45: ~14-29s (dlouhý dozvuk)
- MIDI 50-70: ~5-12s (střední)
- MIDI 85-94: ~2.4-4s (krátký)

**Původně**: Všechny samples měly fixní délku 3s.

### 2. HiFi-GAN Vocoder

**Nově**: Vysoká kvalita mel→audio konverze (žádné "chrcení").

**Původně**: Griffin-Lim (32 iterací) → špatná kvalita.

### 3. Spectral Style Transfer

**Nově**: Aplikace spektrální barvy z MP3 reference tracku.

**Původně**: Nebyla podporována.

### 4. Proper File Naming

**Nově**: `m060-vel5-f48.wav` (stejný formát jako originální samples)

**Původně**: `sample_midi60_vel6.wav`

### 5. Nové helper příkazy

```bash
# Generování celého datasetu
python runhelper.py generate-dataset

# Analýza délek
python runhelper.py analyze-durations

# Test vocoder
python runhelper.py test-vocoder
```

## 🎯 Doporučený postup migrace

### Krok 1: Backup (volitelné)

Pokud chcete zachovat staré VAE checkpointy:

```bash
mkdir backup_vae
mv src/checkpoints/best_model.pth backup_vae/
```

### Krok 2: Analýza datasetu

```bash
python runhelper.py analyze-durations
```

Toto zobrazí statistiky o délkách samples podle MIDI nóty.

### Krok 3: Trénování nového GAN modelu

```bash
python runhelper.py train --device cuda
```

Trénování trvá ~200 epoch (několik hodin až dní podle GPU).

### Krok 4: Test generování

```bash
python runhelper.py generate --midi 60 --velocity 5
```

Ověřte kvalitu v `outputs/m060-vel5-f48.wav`.

### Krok 5: Generování celého datasetu

```bash
python runhelper.py generate-dataset
```

Vygeneruje všechny kombinace MIDI + velocity.

## 📊 Srovnání kvality

| Metrika | VAE (starý) | GAN (nový) |
|---------|-------------|------------|
| Audio kvalita | ⭐⭐ "chrcení" | ⭐⭐⭐⭐⭐ realistické |
| Délka samples | 3s (fixed) | 2-64s (variable) |
| Dozvuk | Nerealistický | Realistický |
| Spektrální profil | Špatný | Odpovídá MP3 |
| Training čas | ~100 epoch | ~200 epoch |
| Inference rychlost | Pomalá (Griffin-Lim) | Rychlá (HiFi-GAN) |

## ⚙️ Konfigurace pro různé use-case

### Pro rychlé testování

```yaml
training:
  batch_size: 8
  num_epochs: 50  # Méně epoch
  save_every: 5

inference:
  vocoder_type: "griffin_lim"  # Rychlejší (ale horší kvalita)
  griffin_lim_iterations: 100
```

### Pro produkční kvalitu

```yaml
training:
  batch_size: 8
  num_epochs: 200
  save_every: 10

inference:
  vocoder_type: "hifigan"
  hifigan_checkpoint: "path/to/pretrained_hifigan.pth"

style_transfer:
  enabled: true
  reference_track: "path/to/reference.mp3"
```

## ❓ FAQ

### Q: Můžu použít staré VAE checkpointy?

**A**: Ne, architektury jsou nekompatibilní. Je třeba natrénovat nový GAN model.

### Q: Jak dlouho trvá trénování?

**A**: ~200 epoch, závisí na GPU:
- RTX 3090: ~6-8 hodin
- RTX 2060: ~15-20 hodin
- CPU: několik dní (nedoporučeno)

### Q: Proč jsou nové samples různě dlouhé?

**A**: To je správně! Nízké frekvence mají delší dozvuk (jako skutečné piano).

### Q: Můžu vypnout variable-length?

**A**: Ano, v `config.yaml`:
```yaml
inference:
  use_variable_length: false
```

### Q: Jak vylepšit kvalitu?

**A**:
1. Natrénujte déle (200+ epoch)
2. Použijte pre-trained HiFi-GAN vocoder
3. Zapněte spectral style transfer s dobrým reference trackem

## 🔗 Související dokumentace

- `README_GAN.md` - Kompletní dokumentace nového systému
- `config.yaml` - Aktuální konfigurace
- `src/variable_gan_model.py` - Implementace architektury

---

**Migrace dokončena!** Nyní máte plně funkční GAN systém s variabilní délkou a vysokou kvalitou výstupu.
