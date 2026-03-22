# open-bible-models

TTS models trained on [Open Bible](https://open.bible/) datasets for low-resource languages.

## Getting Started

This repo uses [Git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) (e.g. F5-TTS), which are not fetched by default when cloning. You need to initialize them explicitly

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/davidguzmanr/open-bible-models.git

# Or, if already cloned without submodules
git submodule update --init --recursive
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