"""
Upload a trained F5-TTS Open Bible checkpoint to HuggingFace.

Uploads three model files (model_last.pt, vocab.txt, config YAML) plus a
generated README to multilingual-tts/F5-TTS-OpenBible-{Language}.

Example usage:
    python utils/upload_f5.py \
        --language "Yoruba NT" \
        --language_code yo \
        --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-yoruba-nt/model_last.pt \
        --vocab_file F5-TTS/data/open-bible-yoruba-nt_custom/vocab.txt \
        --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Yoruba-Nt.yaml \
        --metadata_path F5-TTS/data/open-bible-yoruba-nt/metadata.csv
"""

import argparse
import os
import textwrap

import pandas as pd
from huggingface_hub import HfApi


README_TEMPLATE = """\
---
language:
  - {language_code}
license: cc-by-sa-4.0
library_name: f5-tts
tags:
  - text-to-speech
  - tts
  - f5-tts
  - open-bible
  - {language_slug}
pipeline_tag: text-to-speech
base_model: SWivid/F5-TTS
datasets:
  - davidguzmanr/open-bible-resources
inference: false
---

# F5-TTS Open Bible — {language_name}

A zero-shot text-to-speech model for **{language_name}**, trained from scratch on
the [Open Bible](https://huggingface.co/datasets/davidguzmanr/open-bible-resources)
corpus using the [F5-TTS](https://github.com/SWivid/F5-TTS) architecture
(diffusion transformer with vocos vocoder, 24 kHz output).

The model takes a short reference audio clip (5–10 seconds) and a target text,
and synthesises the target text in the voice of the reference speaker. No
fine-tuning per voice is required.

## Files

| File | Purpose |
|------|---------|
| `model_last.pt` | Trained model weights. |
| `vocab.txt` | Character vocabulary built from the training transcripts. |
| `{config_filename}` | Hydra training/inference config (architecture, mel spec settings, tokenizer). |

## Intended use

- Zero-shot TTS for {language_name}, controlled by a user-supplied reference clip.
- Research on multilingual TTS, low-resource TTS evaluation, and listening
  studies on Open Bible–style read-speech.

## How to use

Install F5-TTS:

```bash
pip install git+https://github.com/SWivid/F5-TTS.git
```

Download the checkpoint and run inference:

```python
import torch
from huggingface_hub import hf_hub_download
from hydra.utils import get_class
from omegaconf import OmegaConf
from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text

repo_id = "{repo_id}"
ckpt   = hf_hub_download(repo_id, "model_last.pt")
vocab  = hf_hub_download(repo_id, "vocab.txt")
config = hf_hub_download(repo_id, "{config_filename}")

device = "cuda" if torch.cuda.is_available() else "cpu"

model_cfg = OmegaConf.load(config)
model_cls = get_class(f"f5_tts.model.{{model_cfg.model.backbone}}")

vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)
model   = load_model(
    model_cls, model_cfg.model.arch, ckpt,
    mel_spec_type="vocos", vocab_file=vocab, use_ema=True, device=device,
)

# Supply your own clean reference clip — 5–10 s, single speaker and its transcription.
ref_audio = "/path/to/your-{language_slug}-clip.wav"
ref_text  = "Exact transcription of the clip"
gen_text  = "..."   # text to synthesise in {language_name}

ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(ref_audio, ref_text)
wav, sr, _ = infer_process(
    ref_audio_proc, ref_text_proc, gen_text, model, vocoder,
    mel_spec_type="vocos", device=device,
)
```

## Training data

- **Source:** `davidguzmanr/open-bible-resources`, config `{dataset_config}`
- **Size:** approximately {utterance_count} utterances
- **Speakers:** multispeaker; speaker identity is supplied at inference time
  via the reference clip, not by a fixed speaker id
- **Sample rate:** 24 kHz
- **Maximum utterance duration during training:** 15 s

## Training procedure

- Base architecture: F5-TTS v1 Base (DiT, 1024 dim, 22 layers, 16 heads,
  text dim 512, 4 convolutional layers).
- Tokenizer: custom character-level, built from the training transcripts.
- Vocoder: vocos.
- Mel spectrogram: 100 channels, hop 256, win 1024, n_fft 1024.
- Optimizer: AdamW, learning rate 7.5e-5, 20 000 warmup updates.
- Training budget: 500,000 optimizer updates on 4 GPUs with mixed precision
  (bf16), global batch ≈ 112,000 frames.

Audio preprocessing, vocab generation, and config sizing are reproducible via
the upstream
[open-bible-models](https://github.com/davidguzmanr/open-bible-models) repo.

## Evaluation

Evaluated alongside other Open-Bible TTS systems on character/word error rate
(via Meta's Omnilingual ASR) and UTMOSv2 naturalness scores. See the
[open-bible-models](https://github.com/davidguzmanr/open-bible-models) repository
for the evaluation pipeline and the
[open-bible-surveys](https://github.com/davidguzmanr/open-bible-surveys) repository
for the human-listening survey methodology.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload a trained F5-TTS Open Bible model to HuggingFace."
    )
    parser.add_argument(
        "--language",
        required=True,
        help='Display name of the language, e.g. "Yoruba NT" or "Assamese".',
    )
    parser.add_argument(
        "--language_code",
        required=True,
        help="BCP 47 / ISO 639 language code, e.g. yo or as.",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to model_last.pt.",
    )
    parser.add_argument(
        "--vocab_file",
        required=True,
        help="Path to vocab.txt.",
    )
    parser.add_argument(
        "--model_cfg",
        required=True,
        help="Path to the model config YAML file.",
    )
    parser.add_argument(
        "--metadata_path",
        default=None,
        help="Path to metadata.csv (pipe-separated). Used to compute utterance count.",
    )
    parser.add_argument(
        "--dataset_config",
        default=None,
        help=(
            "Config name in the HuggingFace dataset davidguzmanr/open-bible-resources. "
            "Defaults to --language."
        ),
    )
    parser.add_argument(
        "--org",
        default="multilingual-tts",
        help="HuggingFace organisation (default: multilingual-tts).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be uploaded without actually uploading.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    language_name   = args.language
    language_slug   = language_name.lower().replace(" ", "-")
    language_hf_id  = language_name.replace(" ", "-")
    repo_id         = f"{args.org}/F5-TTS-OpenBible-{language_hf_id}"
    config_filename = f"F5-TTS_OpenBible_{language_hf_id}.yaml"
    dataset_config  = args.dataset_config or language_name

    # Verify input files exist
    for label, path in [
        ("checkpoint", args.ckpt_path),
        ("vocab file", args.vocab_file),
        ("model config", args.model_cfg),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    # Count utterances from metadata if provided
    if args.metadata_path:
        if not os.path.isfile(args.metadata_path):
            raise FileNotFoundError(f"metadata not found: {args.metadata_path}")
        meta = pd.read_csv(args.metadata_path, sep="|")
        n_utterances = len(meta)
        utterance_count = f"{n_utterances:,}"
    else:
        utterance_count = "N/A"

    readme = README_TEMPLATE.format(
        language_code=args.language_code,
        language_slug=language_slug,
        language_name=language_name,
        config_filename=config_filename,
        repo_id=repo_id,
        dataset_config=dataset_config,
        utterance_count=utterance_count,
    )

    print(f"Repository : {repo_id}")
    print(f"Config file: {config_filename}")
    print(f"Utterances : {utterance_count}")
    print()

    if args.dry_run:
        print("=== DRY RUN — nothing uploaded ===")
        print(f"  {args.ckpt_path}  →  model_last.pt")
        print(f"  {args.vocab_file}  →  vocab.txt")
        print(f"  {args.model_cfg}  →  {config_filename}")
        print("  <generated>  →  README.md")
        return

    api = HfApi()

    # Create the repo if it does not already exist
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=False,
    )
    print(f"Repository ready: https://huggingface.co/{repo_id}")

    # Upload README
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add README for {language_name}",
    )
    print("Uploaded README.md")

    # Upload vocab
    api.upload_file(
        path_or_fileobj=args.vocab_file,
        path_in_repo="vocab.txt",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add vocab.txt for {language_name}",
    )
    print(f"Uploaded vocab.txt  ←  {args.vocab_file}")

    # Upload config YAML (renamed to the canonical HF filename)
    api.upload_file(
        path_or_fileobj=args.model_cfg,
        path_in_repo=config_filename,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add model config for {language_name}",
    )
    print(f"Uploaded {config_filename}  ←  {args.model_cfg}")

    # Upload checkpoint (largest file — do last)
    api.upload_file(
        path_or_fileobj=args.ckpt_path,
        path_in_repo="model_last.pt",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add model_last.pt for {language_name}",
    )
    print(f"Uploaded model_last.pt  ←  {args.ckpt_path}")

    print(f"\nDone! https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
