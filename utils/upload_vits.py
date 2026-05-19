"""
Upload a trained VITS (CoquiTTS) Open Bible checkpoint to HuggingFace.

Uploads model_last.pth, config.json, speakers.pth, and a generated README to
multilingual-tts/VITS-OpenBible-{Language}.

config.json and speakers.pth are auto-detected from the same directory as
--ckpt_path unless overridden explicitly.

Example usage:
    python utils/upload_vits.py \
        --language "Igbo" \
        --language_code ig \
        --ckpt_path VITS-TTS/outputs/igbo/vits_igbo-May-14-2026_10+52AM-106eab5/checkpoint_250000.pth \
        --metadata_path VITS-TTS/data/open-bible-igbo/metadata.csv
"""

import argparse
import os
from pathlib import Path

import re

import pandas as pd
from huggingface_hub import HfApi


def sanitize_config(path: Path) -> bytes:
    """Return config.json contents with bare Infinity replaced by null.

    Coqui TTS serialises float('inf') as the bare token Infinity, which is
    valid Python/JS but not valid JSON.  HuggingFace rejects it with a parse
    warning.  We replace it in-memory so the local file is left unchanged.
    """
    text = path.read_text(encoding="utf-8")
    sanitized = re.sub(r"\bInfinity\b", "null", text)
    return sanitized.encode("utf-8")


README_TEMPLATE = """\
---
language:
  - {language_code}
license: cc-by-sa-4.0
library_name: coqui-tts
tags:
  - text-to-speech
  - tts
  - vits
  - open-bible
  - {language_slug}
pipeline_tag: text-to-speech
datasets:
  - davidguzmanr/open-bible-resources
inference: false
---

# VITS Open Bible — {language_name}

A multispeaker text-to-speech model for **{language_name}**, trained from scratch on
the [Open Bible](https://huggingface.co/datasets/davidguzmanr/open-bible-resources)
corpus using the [VITS](https://arxiv.org/abs/2106.06103) architecture
(end-to-end TTS with adversarial learning, 22,050 Hz output) via the
[Coqui TTS](https://github.com/coqui-ai/TTS) framework.

Unlike zero-shot TTS models, VITS is conditioned on speaker embeddings learned
during training. A speaker name from the training set must be supplied at
inference time.

## Files

| File | Purpose |
|------|---------|
| `model_last.pth` | Trained model weights. |
| `config.json` | Coqui TTS model configuration. |
| `speakers.pth` | Speaker ID → embedding mapping. |

## Intended use

- Multispeaker TTS for {language_name} using one of the training-set speaker voices.
- Research on multilingual TTS, low-resource TTS evaluation, and listening
  studies on Open Bible–style read-speech.

## How to use

Install Coqui TTS:

```bash
pip install TTS
```

Download the checkpoint and run inference:

```python
import torch
from huggingface_hub import hf_hub_download
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.synthesizer import Synthesizer

repo_id  = "{repo_id}"
ckpt     = hf_hub_download(repo_id, "model_last.pth")
config   = hf_hub_download(repo_id, "config.json")
speakers = hf_hub_download(repo_id, "speakers.pth")

use_cuda = torch.cuda.is_available()
synthesizer = Synthesizer(
    tts_checkpoint=ckpt,
    tts_config_path=config,
    tts_speakers_file=speakers,
    use_cuda=use_cuda,
)

# Coqui's Synthesizer may not inject the speakers file into the model config
# automatically — restore the SpeakerManager manually when needed.
if synthesizer.tts_model.speaker_manager is None:
    synthesizer.tts_model.speaker_manager = SpeakerManager(
        speaker_id_file_path=speakers
    )

# List available speaker names
print(sorted(synthesizer.tts_model.speaker_manager.speaker_names))

wav = synthesizer.tts(
    text="...",          # text to synthesise in {language_name}
    speaker_name="...",  # one of the speaker names printed above
    split_sentences=True,
)
```

## Training data

- **Source:** `davidguzmanr/open-bible-resources`, config `{dataset_config}`
- **Size:** approximately {utterance_count} utterances
- **Speakers:** multispeaker; speaker identity is fixed to one of the training-set
  voices and selected by name at inference time
- **Sample rate:** 22,050 Hz

## Training procedure

- Architecture: VITS (Conditional Variational Autoencoder + adversarial training).
- Grapheme-level tokenizer, built from the training transcripts.
- Optimizer: AdamW, learning rate 2e-4.
- Training budget: 500,000 optimizer updates on 2 GPUs with mixed precision
  (bf16).

Audio preprocessing and training are reproducible via the upstream
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
        description="Upload a trained VITS CoquiTTS Open Bible model to HuggingFace."
    )
    parser.add_argument(
        "--language",
        required=True,
        help='Display name of the language, e.g. "Igbo" or "Haitian Creole".',
    )
    parser.add_argument(
        "--language_code",
        required=True,
        help="BCP 47 / ISO 639 language code, e.g. ig or ht.",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to the checkpoint (.pth file), typically checkpoint_250000.pth.",
    )
    parser.add_argument(
        "--config_path",
        default=None,
        help="Path to config.json. Defaults to config.json in the same dir as --ckpt_path.",
    )
    parser.add_argument(
        "--speakers_path",
        default=None,
        help="Path to speakers.pth. Defaults to speakers.pth in the same dir as --ckpt_path.",
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

    language_name  = args.language
    language_slug  = language_name.lower().replace(" ", "-")
    language_hf_id = language_name.replace(" ", "-")
    repo_id        = f"{args.org}/VITS-OpenBible-{language_hf_id}"
    dataset_config = args.dataset_config or language_name

    ckpt_dir = Path(args.ckpt_path).parent
    config_path   = Path(args.config_path)   if args.config_path   else ckpt_dir / "config.json"
    speakers_path = Path(args.speakers_path) if args.speakers_path else ckpt_dir / "speakers.pth"

    # Verify input files exist
    for label, path in [
        ("checkpoint",  args.ckpt_path),
        ("config.json", config_path),
        ("speakers.pth", speakers_path),
    ]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    # Count utterances from metadata if provided
    if args.metadata_path:
        if not os.path.isfile(args.metadata_path):
            raise FileNotFoundError(f"metadata not found: {args.metadata_path}")
        meta = pd.read_csv(args.metadata_path, sep="|")
        utterance_count = f"{len(meta):,}"
    else:
        utterance_count = "N/A"

    readme = README_TEMPLATE.format(
        language_code=args.language_code,
        language_slug=language_slug,
        language_name=language_name,
        repo_id=repo_id,
        dataset_config=dataset_config,
        utterance_count=utterance_count,
    )

    print(f"Repository : {repo_id}")
    print(f"Utterances : {utterance_count}")
    print(f"Config     : {config_path}")
    print(f"Speakers   : {speakers_path}")
    print()

    if args.dry_run:
        print("=== DRY RUN — nothing uploaded ===")
        print(f"  {args.ckpt_path}  →  model_last.pth")
        print(f"  {config_path}  →  config.json")
        print(f"  {speakers_path}  →  speakers.pth")
        print("  <generated>  →  README.md")
        return

    api = HfApi()

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=False,
    )
    print(f"Repository ready: https://huggingface.co/{repo_id}")

    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add README for {language_name}",
    )
    print("Uploaded README.md")

    api.upload_file(
        path_or_fileobj=sanitize_config(config_path),
        path_in_repo="config.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add config.json for {language_name}",
    )
    print(f"Uploaded config.json  ←  {config_path}")

    api.upload_file(
        path_or_fileobj=str(speakers_path),
        path_in_repo="speakers.pth",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add speakers.pth for {language_name}",
    )
    print(f"Uploaded speakers.pth  ←  {speakers_path}")

    api.upload_file(
        path_or_fileobj=args.ckpt_path,
        path_in_repo="model_last.pth",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add model_last.pth for {language_name}",
    )
    print(f"Uploaded model_last.pth  ←  {args.ckpt_path}")

    print(f"\nDone! https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
