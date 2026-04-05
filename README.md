# open-bible-models

TTS models trained on [Open Bible](https://open.bible/) datasets for low-resource languages.

## Getting Started

This repo uses [Git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) (e.g. F5-TTS), which are not fetched by default when cloning. You need to initialize them explicitly

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/davidguzmanr/open-bible-models.git

# Or, if already cloned without submodules
git submodule update --init --recursive

# To pull the latest changes
git pull --recurse-submodules
```

## F5-TTS

We use [F5-TTS](https://github.com/SWivid/F5-TTS) to train text-to-speech models from scratch on per-language Bible audio data.

### Setup

```bash
cd F5-TTS
conda env create -f environment.yaml
conda activate F5-TTS
pip install -e .
```

### Data Preparation

The `prepare_data.py` script automates the full pipeline: metadata creation, audio preprocessing, vocabulary generation, training parameter estimation, and config generation.

```bash
cd F5-TTS

# Single language (downloads from Hugging Face)
python prepare_data.py --languages Yoruba

# Multiple languages, each trained separately
python prepare_data.py --languages Yoruba Ewe Hausa

# Multilingual (combine into one dataset and one config)
python prepare_data.py --languages Yoruba Ewe Hausa --multilingual

# Filter to a specific testament for one or more languages
python prepare_data.py --languages Yoruba --filter-testament '{"Yoruba": "New Testament"}'
python prepare_data.py --languages Yoruba Ewe Hausa --multilingual --filter-testament '{"Yoruba": "New Testament"}'
```

By default, audio data is downloaded from [davidguzmanr/open-bible-resources](https://huggingface.co/datasets/davidguzmanr/open-bible-resources) on Hugging Face.

For each language it:

1. Downloads wav files from Hugging Face.
2. Runs `prepare_csv_wavs.py` to produce `raw.arrow`, `duration.json`, and `vocab.txt`
3. Verifies the generated vocabulary
4. Estimates training epochs based on dataset size and target updates
5. Generates a Hydra config at `src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_{Lang}.yaml`

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--hf-repo` | `davidguzmanr/open-bible-resources` | Hugging Face dataset repo |
| `--max-duration` | 15 | Discard audio files longer than this many seconds; 0 to disable |
| `--target-updates` | 500,000 | Total training updates to target |
| `--batch-size-per-gpu` | 28,000 | Frame budget per GPU batch |
| `--max-samples` | 32 | Max sequences per batch |
| `--num-gpus` | 1 | Number of GPUs (affects epoch calculation) |
| `--workers` | 4 | Threads for audio preprocessing |
| `--skip-preprocess` | off | Skip `prepare_csv_wavs.py` if already done |
| `--multilingual` | off | Combine all languages into one dataset and config |
| `--filter-testament` | `null` | JSON dict mapping language names to testament strings (e.g. `'{"Yoruba": "New Testament"}'`); languages not listed use all data |

### Training

After data preparation, launch training with the generated config:

```bash
cd F5-TTS

# Single GPU
accelerate launch --mixed_precision bf16 \
    src/f5_tts/train/train.py \
    --config-name F5TTS_v1_Base_Open_Bible_Yoruba.yaml

# Multi-GPU (e.g. 2 GPUs)
accelerate launch --num_processes 2 --mixed_precision bf16 \
    src/f5_tts/train/train.py \
    --config-name F5TTS_v1_Base_Open_Bible_Yoruba.yaml
```

#### Training steps in single vs multi-GPU

The generated config (e.g. Yoruba) sets `optim.epochs=575`, which is calibrated so that a single-GPU run reaches 500k optimizer updates. The `prepare_data.py` script always computes epochs relative to 1-GPU update counts, so the **same config works for any number of GPUs** with equivalent total training data:

| | 1 GPU | 2 GPUs | 4 GPUs |
|---|---|---|---|
| Effective batch size | 28,000 frames | 56,000 frames | 112,000 frames |
| Updates per epoch | ~870 | ~435 | ~218 |
| Total updates (575 epochs) | ~500k | ~250k | ~125k |
| Total data processed | same | same | same |

With DDP (Accelerate's default `split_batches=False`), each GPU processes `batch_size_per_gpu` frames independently, then gradients are averaged. More GPUs means a larger effective batch size per update, fewer updates per epoch, but the same number of passes over the dataset. Wall-clock time scales roughly linearly with the number of GPUs.

### Finetuning from a Multilingual Checkpoint

To finetune an existing multilingual model, pass `--finetune` with the checkpoint and its vocabulary. The language argument accepts a testament suffix (e.g. `Kikuyu:NT`) to restrict data to one testament:

```bash
cd F5-TTS

python prepare_data.py --languages Maori:NT --finetune \
    --pretrain-ckpt ckpts/F5TTS_v1_Base_vocos_custom_open-bible-hiligaynon-maori-nt-vietnamese/model_last.pt \
    --pretrain-vocab data/open-bible-hiligaynon-maori-nt-vietnamese_custom/vocab.txt \
    --num-gpus 2 --target-updates 200000
```

Then launch training with the finetune command that `prepare_data.py` outputs
```
accelerate launch --num_processes 2 --mixed_precision bf16 \
    src/f5_tts/train/finetune_cli.py \
    --exp_name F5TTS_v1_Base \
    --dataset_name open-bible-maori-nt-finetune \
    --finetune \
    --pretrain ckpts/F5TTS_v1_Base_vocos_custom_open-bible-hiligaynon-maori-nt-vietnamese/model_last.pt \
    --tokenizer custom \
    --tokenizer_path data/open-bible-hiligaynon-maori-nt-vietnamese_custom/vocab.txt \
    --epochs 1243 \
    --learning_rate 1e-5 \
    --batch_size_per_gpu 28000 \
    --max_samples 32 \
    --num_warmup_updates 2000 \
    --save_per_updates 10000 \
    --logger tensorboard
```

## EveryVoice (FastSpeech2)

We also train TTS models using [EveryVoice](https://github.com/EveryVoiceTTS/EveryVoice), which uses a FastSpeech2-based feature prediction model followed by a HiFi-GAN vocoder.

### Setup

Follow the [EveryVoice installation guide](https://docs.everyvoice.ca/latest/install/). Once installed, activate the environment:

```bash
conda activate EveryVoice
```

### Data Preparation

The data is the same as what `F5-TTS/prepare_data.py` produces. EveryVoice expects a filelist (`.psv`) and audio files; preprocessing is handled by `everyvoice preprocess` (see below).

### Project Setup

Create a new project using the interactive wizard with default parameters:

```bash
everyvoice new-project
```

This generates the config files under `config/` (`everyvoice-shared-data.yaml`, `everyvoice-text-to-spec.yaml`, `everyvoice-spec-to-wav.yaml`).

### Training

Training involves three stages: preprocessing, feature prediction (text → mel spectrogram), and vocoder fine-tuning (mel → waveform). See [`jobs/EveryVoice-Open-Bible-Yoruba-NT.sh`](jobs/EveryVoice-Open-Bible-Yoruba-NT.sh) for the exact parameters used.

```bash
# 1. Preprocess
everyvoice preprocess config/everyvoice-text-to-spec.yaml \
    --config-args preprocessing.audio.max_audio_length=15 \
    --config-args preprocessing.train_split=1.0 \
    --overwrite

# 2. Train feature prediction model (2 GPUs, DDP)
everyvoice train text-to-spec config/everyvoice-text-to-spec.yaml \
    --devices 2 --strategy ddp \
    --config-args training.max_steps=250000 \
    --config-args training.batch_size=32

# 3. Fine-tune vocoder
everyvoice train spec-to-wav config/everyvoice-spec-to-wav.yaml \
    --devices 2 --strategy ddp \
    --config-args training.finetune=True \
    --config-args training.max_steps=50000
```

## Evaluation

Once a model is trained, evaluate it on the test split by first synthesizing audio with the trained checkpoint and then running `evaluate-tts.py`.

```bash
python evaluate-tts.py \
    --ground_truth_dir /path/to/ground-truth \
    --synthesized_dir /path/to/generated \
    --metadata_csv /path/to/test.csv \
    --output_csv /path/to/results.csv \
    --metrics utmos wer \
    --asr-lang yor_Latn \
    --system-name everyvoice
```

`evaluate-tts.py` supports four metrics:

| Metric | Type | Direction |
|--------|------|-----------|
| `mcd` | Mel Cepstral Distortion | lower is better |
| `speechbertscore` | WavLM-based similarity | higher is better |
| `utmos` | Predicted MOS (UTMOSv2) | higher is better |
| `wer` | Word Error Rate via ASR | lower is better |