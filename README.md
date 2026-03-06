# open-bible-models

TTS models trained on [Open Bible](https://open.bible/) datasets for low-resource languages.

## F5-TTS

We use [F5-TTS](https://github.com/SWivid/F5-TTS) to train text-to-speech models from scratch on per-language Bible audio data.

### Setup

```bash
cd F5-TTS
conda env create -f environment.yaml
conda activate F5-TTS
```

### Data Preparation

The `prepare_data.py` script automates the full pipeline: metadata creation, audio preprocessing, vocabulary generation, training parameter estimation, and config generation.

```bash
cd F5-TTS

# Single language (downloads from Hugging Face)
python prepare_data.py --languages Yoruba

# Multiple languages
python prepare_data.py --languages Yoruba Ewe Hausa

# Use a local dataset directory instead of Hugging Face
python prepare_data.py --dataset-base /path/to/bible-tts-resources --languages Yoruba
```

By default, audio data is downloaded from [davidguzmanr/open-bible-resources](https://huggingface.co/datasets/davidguzmanr/open-bible-resources) on Hugging Face. Alternatively, use `--dataset-base` to point to a local directory containing one subdirectory per language, each with a `train.tsv` and a `wav/` folder.

For each language it:

1. Downloads wav files from Hugging Face (or creates metadata from local `train.tsv`)
2. Runs `prepare_csv_wavs.py` to produce `raw.arrow`, `duration.json`, and `vocab.txt`
3. Verifies the generated vocabulary
4. Estimates training epochs based on dataset size and target updates
5. Generates a Hydra config at `src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_{Lang}.yaml`

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-base` | *(none)* | Local data path; if omitted, downloads from HF |
| `--hf-repo` | `davidguzmanr/open-bible-resources` | Hugging Face dataset repo |
| `--max-duration` | 15 | Discard audio files longer than this many seconds; 0 to disable |
| `--target-updates` | 500,000 | Total training updates to target |
| `--batch-size-per-gpu` | 28,000 | Frame budget per GPU batch |
| `--max-samples` | 32 | Max sequences per batch |
| `--num-gpus` | 1 | Number of GPUs (affects epoch calculation) |
| `--workers` | 4 | Threads for audio preprocessing |
| `--skip-preprocess` | off | Skip `prepare_csv_wavs.py` if already done |

### Training

After data preparation, launch training with the generated config:

```bash
cd F5-TTS

accelerate launch --mixed_precision bf16 \
    src/f5_tts/train/train.py \
    --config-name F5TTS_v1_Base_Open_Bible_Yoruba.yaml
```