# VITS-TTS

Train a multilingual multispeaker VITS model using [Coqui TTS](https://github.com/coqui-ai/TTS).
Vocabulary is built automatically from the metadata file — no phonemizer required.

## Dataset format

A pipe-separated `metadata.csv` with a header row:

```
audio_file|text|speaker_id
/abs/path/to/wavs/utt_001.wav|Hello world|SPEAKER_00_Lang
/abs/path/to/wavs/utt_002.wav|Another sentence|SPEAKER_01_Lang
```

`audio_file` must be an absolute path to a `.wav` file.

## Usage

**Single GPU**

```bash
python train_vits.py \
    --metadata /path/to/metadata.csv \
    --language yo \
    --output_path /path/to/output
```

**Multi-GPU** (uses Coqui's `trainer.distribute`, not `torchrun`)

```bash
python -m trainer.distribute \
    --script train_vits.py \
    --gpus "0,1" \
    --metadata /path/to/metadata.csv \
    --language yo \
    --output_path /path/to/output \
    --num_gpus 2
```

**Resume training**

Add `--restore_path /path/to/checkpoint_XXXXX.pth` to either command above.

## Key arguments

| Argument | Default | Description |
|---|---|---|
| `--metadata` | required | Path to `metadata.csv` |
| `--language` | required | ISO 639-1/3 language code (e.g. `ha`, `yo`, `sw`) |
| `--output_path` | required | Directory for checkpoints and logs |
| `--global_batch_size` | `32` | Total batch size across all GPUs |
| `--num_gpus` | `1` | Number of GPUs (adjusts per-GPU batch size) |
| `--target_steps` | `500000` | Target optimizer steps; epochs are derived from this |
| `--sample_rate` | `22050` | Audio sample rate |
| `--save_step` | `5000` | Save a checkpoint every N steps |
| `--eval_split_size` | `0.01` | Fraction of data held out for evaluation |
| `--no_eval` | off | Disable evaluation and use all data for training |
| `--restore_path` | `None` | Checkpoint to resume from |

## Monitoring

```bash
tensorboard --logdir /path/to/output_path
```
