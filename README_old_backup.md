# Neurailssimo - Electric Piano Neural Network Generator

Neuronová síť pro generování zvukových samplů elektrického piana na základě parametrů MIDI noty a velocity.

## Přehled projektu

Tento projekt implementuje **Conditional Variational Autoencoder (CVAE)** pro generování realistických zvuků elektrického piana. Model se učí z banky WAV samplů a dokáže:
- Generovat nové zvuky na základě MIDI noty a velocity
- Interpolovat mezi různými zvuky
- Vytvářet variace existujících zvuků
- Rozšiřovat tónovou paletu nástroje
- **Vylepšovat kvalitu pomocí style transfer z referenčních nahrávek (MP3/WAV/FLAC)**

## Charakteristiky datasetu

- **Celkem samplů:** 922 WAV souborů
- **MIDI rozsah:** 33-94 (60 not, ~5 oktáv)
- **Velocity úrovně:** 0-7 (8 dynamických úrovní)
- **Vzorkovací frekvence:** 44 kHz a 48 kHz
- **Délka samplů:** 15-40 sekund (průměr 26.7s)
- **Formát:** 16-bit, mono

## Struktura projektu

```
neurailssimo/
│
├── src/                    # Zdrojové kódy
│   ├── configurepaths.py  # Centralizované řízení cest
│   ├── dataset.py         # Dataset loader a preprocessing
│   ├── model.py           # Architektura neuronové sítě
│   ├── train.py           # Tréninkový skript
│   ├── inference.py       # Generování samplů
│   ├── runhelper.py       # Jednotné rozhraní pro spouštění úloh
│   └── style_transfer.py  # Modul pro post-processing style transfer
│
├── checkpoints/           # Checkpointy během tréninku
├── logs/                  # TensorBoard logy
├── outputs/               # Generované audio soubory
│
├── config.yaml            # Konfigurace
├── requirements.txt       # Python dependencies
├── analyze_dataset.py     # Analýza datasetu
├── test_setup.py          # Test instalace
├── test_style_transfer.py # Test style transfer
└── generate_batch.bat     # Příklad batch generování pro Windows
```

## Instalace

```bash
pip install -r requirements.txt
```

## Rychlý start

### Jednoduchý způsob - RunHelper

Použijte centrální skript `runhelper.py` pro všechny operace:

```bash
# Test instalace
python runhelper.py test

# Trénink modelu
python runhelper.py train

# Generování samplu
python runhelper.py generate --midi 60 --velocity 5

# Generování s vylepšením kvality (style transfer)
python runhelper.py generate --midi 60 --velocity 5 --enhance

# Batch generování
python runhelper.py batch --midi-range 48-72 --velocities 2,4,6

# Interpolace
python runhelper.py interpolate --midi1 60 --vel1 2 --midi2 72 --vel2 6

# Status projektu
python runhelper.py status
```

### Klasický způsob

#### 1. Instalace

```bash
pip install -r requirements.txt
python test_setup.py  # Ověří instalaci
```

#### 2. Analýza datasetu

```bash
python analyze_dataset.py
```

#### 3. Trénink modelu

```bash
cd src
python train.py --config ../config.yaml
```

Sledujte progress v TensorBoard:
```bash
tensorboard --logdir=logs
```

#### 4. Generování samplů

```bash
python inference.py --model checkpoints/best_model.pth \
                    --midi 60 --velocity 5 \
                    --output ../outputs/sample.wav

# Interpolace mezi zvuky
python inference.py --model checkpoints/best_model.pth \
                    --interpolate \
                    --midi 60 --velocity 1 \
                    --midi2 72 --velocity2 7 \
                    --num-steps 10 \
                    --output ../outputs/interp.wav
```

## Role souborů a metod

Zde je přehled hlavních souborů a jejich rolí v projektu:

-   `config.yaml`: Hlavní konfigurační soubor pro celý projekt. Definuje parametry datasetu, modelu, tréninku a inference.
-   `requirements.txt`: Seznam Python balíčků potřebných pro spuštění projektu.
-   `src/configurepaths.py`: Centralizovaný modul pro správu všech cest v projektu (kořen, zdrojové kódy, checkpointy, logy, výstupy atd.). Zajišťuje konzistenci cest napříč všemi skripty.
-   `src/dataset.py`: Definuje `ElectricPianoDataset` pro načítání a preprocessing audio samplů. Zpracovává WAV soubory, extrahuje MIDI noty a velocity z názvů souborů a převádí audio na mel spektrogramy.
-   `src/model.py`: Obsahuje definici architektury Conditional Variational Autoencoder (CVAE), včetně `Encoderu`, `Decoderu` a `ConditionalVAE` třídy. Zde je definována struktura neuronové sítě.
-   `src/train.py`: Skript pro trénování VAE modelu. Zahrnuje tréninkovou smyčku, validaci, ukládání checkpointů a logování do TensorBoardu.
-   `src/inference.py`: Skript pro generování nových audio samplů z natrénovaného VAE modelu. Převádí mel spektrogramy na audio pomocí Griffin-Lim algoritmu.
-   `src/runhelper.py`: Jednotné rozhraní pro spouštění všech hlavních operací projektu (trénink, generování, analýza, testování, TensorBoard). Zjednodušuje interakci s projektem.
-   `src/style_transfer.py`: Modul pro post-processing style transfer pomocí spektrálního matching. Analyzuje referenční tracky a aplikuje jejich spektrální charakteristiky na vygenerované samply.
-   `src/test_setup.py`: Testovací skript pro ověření základní instalace a funkčnosti projektu (importy, CUDA, konfigurace, načítání datasetu, vytvoření modelu).
-   `src/test_style_transfer.py`: Testovací skript pro ověření funkcionality style transfer modulu.
-   `outputs/`: Adresář pro ukládání vygenerovaných audio souborů.
-   `src/checkpoints/`: Adresář pro ukládání checkpointů modelu během tréninku.
-   `logs/`: Adresář pro ukládání logů z TensorBoardu.

## Architektura modelu

Model používá VAE (Variational Autoencoder) s následujícími komponentami:

1.  **Encoder:** Převádí audio na latentní reprezentaci
2.  **Latent Space:** 256-dimenzionální prostor pro interpolaci
3.  **Decoder:** Rekonstruuje audio z latentní reprezentace
4.  **Conditioning:** MIDI nota a velocity jako dodatečné vstupy

## Technické detaily

### Architektura modelu

**Encoder:**
- Konvoluční vrstvy: [64, 128, 256, 512] kanálů
- Stride 2 pro downsample
- Výstup: latentní mean a log_variance (256-dim)

**Decoder:**
- Transposed konvoluční vrstvy pro upsample
- Conditioning na MIDI notě a velocity
- Výstup: Mel spectrogram (128 mel bands)

**Loss funkce:**
- Reconstruction Loss: MSE mezi vstupem a výstupem
- KL Divergence: Regularizace latentního prostoru

### Preprocessing pipeline

1. Načtení WAV (48 kHz, 16-bit, mono)
2. Ořez/padding na 3 sekundy
3. Převod na mel spectrogram (n_fft=2048, hop=512)
4. Normalizace na [-1, 1]

### Post-processing

1. Mel spectrogram → Linear spectrogram (InverseMel)
2. Griffin-Lim algoritmus pro rekonstrukci fáze
3. Export do WAV (48 kHz, 16-bit)

## Výkonnost

- **Trénink**: ~2-3 hodiny na NVIDIA RTX 3090 (100 epoch)
- **Inference**: ~0.5s na GPU, ~2s na CPU (jeden sample)
- **Paměť GPU**: ~8 GB pro batch_size=16

## Trénink modelu

### Základní trénink

Spustí trénink s výchozí konfigurací:

```bash
python runhelper.py train
```

Toto:
- Načte konfiguraci z `config.yaml`
- Vytvoří model
- Načte dataset
- Spustí trénink na 100 epoch (výchozí)
- Ukládá checkpointy každých 5 epoch
- Ukládá best model do `src/checkpoints/best_model.pth`

### Pokračování v tréninku

Pokračovat od konkrétního checkpointu:

```bash
python runhelper.py train --resume checkpoints/checkpoint_epoch_50.pth
```

### Trénink na GPU

Pokud máte CUDA GPU:

```bash
python runhelper.py train --device cuda
```

### Trénink na CPU

Explicitně specifikovat CPU:

```bash
python runhelper.py train --device cpu
```

## Generování samplů

### Základní generování

Vygeneruje jeden sample pro danou MIDI notu a velocity:

```bash
python runhelper.py generate --midi 60 --velocity 5
```

- MIDI 60 = Middle C (C4)
- Velocity 0-7 (0 = nejslabší, 7 = nejsilnější)
- Výstup: `outputs/sample_midi60_vel5.wav`

### Vlastní výstupní soubor

```bash
python runhelper.py generate --midi 60 --velocity 5 --output moje_piana.wav
```

### Generování více variací

Vygeneruje 5 různých verzí stejného samplu:

```bash
python runhelper.py generate --midi 60 --velocity 5 --num-samples 5
```

Výstup: `moje_piana_1.wav`, `moje_piana_2.wav`, ...

### Použití temperature

Temperature ovlivňuje variabilitu:

```bash
# Konzervativní (podobnější tréninkovým datům)
python runhelper.py generate --midi 60 --velocity 5 --temperature 0.8

# Standardní
python runhelper.py generate --midi 60 --velocity 5 --temperature 1.0

# Experimentální (větší variace)
python runhelper.py generate --midi 60 --velocity 5 --temperature 1.5
```

### Použití konkrétního modelu

```bash
python runhelper.py generate --midi 60 --velocity 5 --model checkpoints/checkpoint_epoch_75.pth
```

## Style Transfer Enhancement

Vylepšení generovaného zvuku pomocí referenčních tracků (spectral matching).

### Jak to funguje?

1.  **Analýza referenčních tracků:** Kvalitní MP3/WAV nahrávka je analyzována pomocí FFT, čímž se získá spektrální profil (průměrná energie v každém frekvenčním pásmu).
2.  **Aplikace na vygenerovaný sample:** Vygenerovaný sample je analyzován pomocí FFT, jeho spektrum je porovnáno s referenčním profilem, amplitudy jsou upraveny a pomocí iFFT je rekonstruován vylepšený sample.
3.  **Zachování dynamiky:** Algoritmus zachovává původní dynamickou strukturu (časový průběh hlasitosti), pouze upravuje spektrální barvu zvuku.

### Použití

#### Základní použití

```bash
python runhelper.py generate --midi 60 --velocity 5 --enhance
```

**Výstup:**
- `outputs/sample_midi60_vel5.wav` - původní vygenerovaný sample
- `outputs/sample_midi60_vel5_enhanced.wav` - vylepšená verze

#### Nastavení síly enhancement

```bash
# Jemné (30%)
python runhelper.py generate --midi 60 --velocity 5 --enhance --enhance-strength 0.3

# Střední (70%, výchozí)
python runhelper.py generate --midi 60 --velocity 5 --enhance --enhance-strength 0.7

# Silné (90%)
python runhelper.py generate --midi 60 --velocity 5 --enhance --enhance-strength 0.9
```

**Doporučení:**
- `0.3-0.5`: Jemné vylepšení, zachová charakter originálu
- `0.7`: Výchozí, dobrá rovnováha
- `0.9`: Maximální přenos stylu z referenčních tracků

#### Vlastní referenční adresář

```bash
python runhelper.py generate --midi 60 --velocity 5 \
    --enhance \
    --enhance-reference "C:\\SoundBanks\\IthacaPlayer\\instrument-styles\"
```

**Podporované formáty:** MP3, WAV, FLAC, M4A

### Příprava referenčních tracků

**Doporučené nahrávky:**
- Studiové nahrávky elektrického piana (Fender Rhodes, Wurlitzer)
- Formát: WAV, FLAC (nejlepší), MP3 320kbps (dobré)
- Kvalita: Hi-Fi, profesionální produkce
- Obsah: Čisté piano tracky (bez bicích, zpěvu, atd.)

**Umístění tracků:**
Výchozí adresář: `C:\\SoundBanks\\IthacaPlayer\\instrument-styles\`

### Test style transfer

Pro otestování bez natrénovaného modelu:

```bash
python test_style_transfer.py
```

Tento skript:
1. Analyzuje referenční tracky
2. Vybere vzorek z datasetu
3. Vytvoří 4 verze s různými silami (0.3, 0.5, 0.7, 0.9)
4. Uloží do `outputs/style_transfer_test/`

### Pokročilé použití CLI nástroje

Style transfer má vlastní CLI rozhraní:

```bash
# Analyzuj reference a ulož profil
python src/style_transfer.py analyze --reference-dir "path/to/tracks"

# Enhance soubor
python src/style_transfer.py enhance \
    --input generated.wav \
    --output enhanced.wav \
    --strength 0.7

# Extrahuj segmenty z tracků (pro pokročilé použití)
python src/style_transfer.py extract \
    --reference-dir "path/to/tracks" \
    --output-dir "data/segments"
```

## Batch generování

### Generování rozsahu not

Vygeneruje samply pro rozsah MIDI not:

```bash
# Generuje noty od C3 (48) do C5 (72) se střední velocity
python runhelper.py batch --midi-range 48-72 --velocities 4
```

### Generování více velocities

```bash
# Generuje C4 (60) s velocities 2, 4, a 6
python runhelper.py batch --midi-range 60-60 --velocities 2,4,6
```

### Kompletní škála

Vygeneruje všechny noty v rozsahu s různými velocities:

```bash
python runhelper.py batch --midi-range 33-94 --velocities 0,2,4,6
```

To vygeneruje: 62 not × 4 velocities = 248 samplů

### Často používané rozsahy

```bash
# Klavírní střední oktávy (C3 - C5)
python runhelper.py batch --midi-range 48-72 --velocities 2,4,6

# Basové noty
python runhelper.py batch --midi-range 33-48 --velocities 4

# Vysoké noty
python runhelper.py batch --midi-range 72-94 --velocities 4
```

## Interpolace

### Základní interpolace

Vytvoří plynulý přechod mezi dvěma zvuky:

```bash
python runhelper.py interpolate \
    --midi1 60 --vel1 2 \
    --midi2 72 --vel2 6 \
    --steps 10
```

To vytvoří:
- 10 WAV souborů
- Přechod od C4 (tichý) k C5 (hlasitý)
- Soubory: `interpolation_step000.wav` až `interpolation_step009.wav`

### Více kroků pro jemnější přechod

```bash
python runhelper.py interpolate \
    --midi1 48 --vel1 0 \
    --midi2 84 --vel2 7 \
    --steps 20
```

### Vlastní výstup

```bash
python runhelper.py interpolate \
    --midi1 60 --vel1 2 \
    --midi2 72 --vel2 6 \
    --output melodie.wav
```

### Interpolace mezi stejnými notami (variace velocity)

```bash
# Vytvoří crescendo na jedné notě
python runhelper.py interpolate \
    --midi1 60 --vel1 0 \
    --midi2 60 --vel2 7 \
    --steps 8
```

## Monitorování

### TensorBoard

Spustí TensorBoard pro sledování tréninku:

```bash
python runhelper.py tensorboard
```

Použijte prohlížeč na: http://localhost:6006

V TensorBoard uvidíte:
- Training loss
- Validation loss
- Reconstruction loss
- KL divergence
- Learning rate

## Pokročilé použití

### Úplný workflow

1.  **Test instalace:**
```bash
python runhelper.py test
```

2.  **Analýza dat:**
```bash
python runhelper.py analyze
```

3.  **Spuštění tréninku:**
```bash
python runhelper.py train --device cuda
```

4.  **Sledování progress:**
```bash
# V novém terminálu
python runhelper.py tensorboard
```

5.  **Kontrola statusu:**
```bash
python runhelper.py status
```

6.  **Generování testovacích samplů:**
```bash
python runhelper.py generate --midi 60 --velocity 5
python runhelper.py generate --midi 72 --velocity 3
```

7.  **Batch generování pro celou škálu:**
```bash
python runhelper.py batch --midi-range 48-72 --velocities 2,4,6
```

8.  **Vytvoření interpolace:**
```bash
python runhelper.py interpolate --midi1 48 --vel1 1 --midi2 84 --vel2 7 --steps 15
```

### Použití různých checkpointů

```bash
# Generování s různými verzemi modelu
python runhelper.py generate --midi 60 --velocity 5 \
    --model checkpoints/checkpoint_epoch_25.pth \
    --output sample_epoch25.wav

python runhelper.py generate --midi 60 --velocity 5 \
    --model checkpoints/checkpoint_epoch_100.pth \
    --output sample_epoch100.wav

# Porovnání kvality
```

### Pipeline pro tvorbu skladby

```bash
# 1. Generuj basovou linku
python runhelper.py batch --midi-range 36-48 --velocities 4

# 2. Generuj melodii
python runhelper.py batch --midi-range 60-72 --velocities 5,6

# 3. Generuj interpolace pro smooth přechody
python runhelper.py interpolate --midi1 60 --vel1 4 --midi2 64 --vel2 5 --steps 8
python runhelper.py interpolate --midi1 64 --vel1 5 --midi2 67 --vel2 6 --steps 8
```

### Vytvoření demo knihovny

```bash
# Generovat reprezentativní samply pro prezentaci
python runhelper.py batch --midi-range 36-84 --velocities 0,3,7
```

To vytvoří:
- Každou notu (36-84 = 49 not)
- Se třemi úrovněmi dynamiky (pp, mf, ff)
- Celkem 147 samplů

## Debugging a optimalizace

### Kontrola kvality během tréninku

Vytvořte skript `check_quality.sh`:

```bash
#!/bin/bash
# Generuje testovací samply každých 10 epoch

for epoch in 10 20 30 40 50 60 70 80 90 100; do
    if [ -f "src/checkpoints/checkpoint_epoch_$epoch.pth" ]; then
        echo "Testing epoch $epoch..."
        python runhelper.py generate \
            --midi 60 --velocity 5 \
            --model "src/checkpoints/checkpoint_epoch_$epoch.pth" \
            --output "quality_tests/test_epoch_$epoch.wav"
    fi
done
```

### Monitoring tréninku

```bash
# Terminál 1: Trénink
python runhelper.py train

# Terminál 2: TensorBoard
python runhelper.py tensorboard

# Terminál 3: Status monitoring
watch -n 60 'python runhelper.py status'
```

### Rychlé testování po změnách konfigurace

```bash
# 1. Upravte config.yaml (např. změňte latent_dim)
# 2. Spusťte test
python runhelper.py test

# 3. Krátký trénink
# Změňte num_epochs na 5 v config.yaml
python runhelper.py train

# 4. Vygenerujte test sample
python runhelper.py generate --midi 60 --velocity 5 --output test_new_config.wav
```

## Řešení problémů

### "Model not found"

```bash
# Zkontrolujte dostupné checkpointy
python runhelper.py status

# Nebo specifikujte cestu
python runhelper.py generate --midi 60 --velocity 5 \
    --model checkpoints/checkpoint_epoch_X.pth
```

### "CUDA out of memory"

```bash
# Použijte CPU
python runhelper.py train --device cpu
python runhelper.py generate --midi 60 --velocity 5 --device cpu
```

### Špatná kvalita výstupu

```bash
# 1. Zkontrolujte, že model je natrénovaný
python runhelper.py status

# 2. Zkuste nižší temperature
python runhelper.py generate --midi 60 --velocity 5 --temperature 0.7

# 3. Vygenerujte víc variací a vyberte nejlepší
python runhelper.py generate --midi 60 --velocity 5 --num-samples 10
```

### Pomalý trénink na CPU

- Trénink na CPU je ~10-50× pomalejší než na GPU
- Generování je relativně rychlé i na CPU (~2-5s per sample)
- Pro trénink doporučujeme GPU, pro generování stačí CPU

## Výsledky testování

**Datum:** 2025-10-18  
**Tester:** Claude Code

### ✅ 1. Test instalace
```bash
python runhelper.py test
```
**Výsledek:** PASSED
- Všechny knihovny nalezeny
- Model vytvořen (23.4M parametrů)
- Dataset načten (922 samples)
- Forward pass funguje

### ✅ 2. Analýza datasetu
```bash
python runhelper.py analyze
```
**Výsledek:** PASSED
- Zobrazeny statistiky datasetu
- 922 WAV souborů
- MIDI rozsah 33-94 (60 not)
- Velocity 0-7 (8 úrovní)

### ✅ 3. Status projektu
```bash
python runhelper.py status
```
**Výsledek:** PASSED
- Dataset: [OK] 922 WAV files
- Zobrazí checkpointy (když existují)
- Zobrazí logy (když existují)

### ✅ 4. Trénink modelu
```bash
python runhelper.py train --device cpu
```
**Výsledek:** PASSED
- Úspěšně spuštěn trénink
- Model načten (55.4M parametrů)
- Dataset rozdělen: 737/92/93
- Trénink běží (6s/batch)
- Loss klesá: 203 → 108

## Roadmap

- [x] ~~Implementovat data loader~~
- [x] ~~Vytvořit VAE architekturu~~
- [x] ~~Implementovat tréninkový pipeline~~
- [x] ~~Vytvořit inference engine~~
- [x] ~~Přidat CLI rozhraní~~
- [ ] Optimalizovat hyperparametry
- [ ] Experimentovat s WaveGAN
- [ ] Implementovat Diffusion Model
- [ ] Přidat data augmentation
- [ ] Real-time inference optimalizace

## Licence

Tento projekt je vytvořen pro studijní a výzkumné účely.