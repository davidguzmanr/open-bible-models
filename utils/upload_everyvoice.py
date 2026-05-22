"""
Upload a trained EveryVoice Open Bible checkpoint to HuggingFace.

Uploads the feature-prediction checkpoint (last.ckpt), the config directory,
the training filelist (PSV), an optional vocoder checkpoint, and a generated
README to multilingual-tts/EveryVoice-OpenBible-{Language}.

The config directory is auto-detected from the grandparent of --ckpt_path
(i.e. the project root: .../logs_and_checkpoints/../../../config) unless
overridden with --config_dir.

Example usage:
    python utils/upload_everyvoice.py \\
        --language "Igbo" \\
        --language_code ig \\
        --ckpt_path Open-Bible-Igbo/logs_and_checkpoints/FeaturePredictionExperiment/base/checkpoints/last.ckpt \\
        --filelist_path EveryVoice/open-bible-igbo-filelist.psv \\
        --vocoder_ckpt_path /path/to/hifigan_universal_v1_everyvoice.ckpt
"""

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi


README_TEMPLATE = """\
---
language:
  - {language_code}
license: cc-by-sa-4.0
library_name: everyvoice
tags:
  - text-to-speech
  - tts
  - everyvoice
  - fastspeech2
  - open-bible
  - {language_slug}
pipeline_tag: text-to-speech
datasets:
  - davidguzmanr/open-bible-resources
inference: false
---

# EveryVoice Open Bible — {language_name}

A multispeaker text-to-speech model for **{language_name}**, trained from scratch on
the [Open Bible](https://huggingface.co/datasets/davidguzmanr/open-bible-resources)
corpus using the [EveryVoice](https://github.com/EveryVoiceTTS/EveryVoice) TTS toolkit
(FastSpeech2 acoustic model + HiFi-GAN vocoder, 22,050 Hz output).

The model is conditioned on speaker embeddings learned during training. A speaker
name from the training set must be supplied at inference time.

## Files

| File | Purpose |
|------|---------|
| `feature_prediction.ckpt` | Trained FastSpeech2 feature-prediction weights. |
| `vocoder.ckpt` | HiFi-GAN vocoder checkpoint (optional — can be replaced with a universal vocoder). |
| `config/` | EveryVoice YAML config files (shared data, text, feature-prediction, spec-to-wav). |
| `filelist.psv` | Pipe-separated training filelist (`basename|language|speaker|characters|phones`). |

## Intended use

- Multispeaker TTS for {language_name} using one of the training-set speaker voices.
- Research on multilingual TTS, low-resource TTS evaluation, and listening
  studies on Open Bible–style read-speech.

## How to use

Install EveryVoice:

```bash
pip install everyvoice
```

Download the checkpoint and run inference:

```python
import torch
from pathlib import Path
from huggingface_hub import snapshot_download

from everyvoice.config.type_definitions import DatasetTextRepresentation
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.synthesize import (
    get_global_step,
    synthesize_helper,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import FastSpeech2
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions import (
    SynthesizeOutputFormats,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
    load_hifigan_from_checkpoint,
)
from everyvoice.utils.heavy import get_device_from_accelerator

repo_id  = "{repo_id}"
local    = Path(snapshot_download(repo_id))

ckpt_path    = local / "feature_prediction.ckpt"
vocoder_path = local / "vocoder.ckpt"

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
device = get_device_from_accelerator(accelerator)

model = FastSpeech2.load_from_checkpoint(str(ckpt_path)).to(device)
model.eval()
global_step = get_global_step(ckpt_path)

vocoder_ckpt = torch.load(str(vocoder_path), map_location=device, weights_only=True)
vocoder_model, vocoder_config = load_hifigan_from_checkpoint(vocoder_ckpt, device)
vocoder_global_step = get_global_step(vocoder_path)

# Pick any speaker from the model
speaker = next(iter(model.speaker2id.keys()))
language = next(iter(model.lang2id.keys()))
print(f"Available speakers: {{list(model.speaker2id.keys())}}")

filelist_data = [
    {{
        "basename":         "sample-0",
        "characters":       "...",   # text to synthesise in {language_name}
        "language":         language,
        "speaker":          speaker,
        "duration_control": 1.0,
    }}
]

output_dir = Path("everyvoice_output")
output_dir.mkdir(exist_ok=True)

synthesize_helper(
    model=model,
    texts=None,
    style_reference=None,
    language=None,
    speaker=None,
    duration_control=1.0,
    global_step=global_step,
    output_type=[SynthesizeOutputFormats.wav],
    text_representation=DatasetTextRepresentation.characters,
    accelerator=accelerator,
    devices="auto",
    device=device,
    batch_size=1,
    num_workers=1,
    filelist=None,
    filelist_data=filelist_data,
    output_dir=output_dir,
    teacher_forcing_directory=None,
    vocoder_model=vocoder_model,
    vocoder_config=vocoder_config,
    vocoder_global_step=vocoder_global_step,
)
# Generated WAVs land in output_dir/wav/
```

## Training data

- **Source:** `davidguzmanr/open-bible-resources`, config `{dataset_config}`
- **Size:** approximately {utterance_count} utterances
- **Speakers:** multispeaker; speaker identity is fixed to one of the training-set
  voices and selected by name at inference time
- **Sample rate:** 22,050 Hz

## Training procedure

- Acoustic model: FastSpeech2 (non-autoregressive, duration-prediction based).
- Vocoder: HiFi-GAN (iSTFT variant).
- Character-level tokenizer built from the training transcripts.
- Trained with the [EveryVoice](https://github.com/EveryVoiceTTS/EveryVoice) toolkit.

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


def _find_config_dir(ckpt_path: Path) -> Path:
    """Walk up from the checkpoint to find the EveryVoice project config/ directory.

    Expected layout:
        <project_root>/
            config/                          ← target
            logs_and_checkpoints/
                FeaturePredictionExperiment/
                    base/
                        checkpoints/
                            last.ckpt        ← ckpt_path

    So the project root is 5 levels above last.ckpt.
    """
    project_root = ckpt_path.resolve()
    for _ in range(5):
        project_root = project_root.parent
    candidate = project_root / "config"
    return candidate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload a trained EveryVoice Open Bible model to HuggingFace."
    )
    parser.add_argument(
        "--language",
        required=True,
        help='Display name of the language, e.g. "Igbo" or "Yoruba NT".',
    )
    parser.add_argument(
        "--language_code",
        required=True,
        help="BCP 47 / ISO 639 language code, e.g. ig or yo.",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to the feature-prediction checkpoint (last.ckpt).",
    )
    parser.add_argument(
        "--filelist_path",
        required=True,
        help=(
            "Path to the EveryVoice training filelist (pipe-separated PSV with columns "
            "basename|language|speaker|characters|phones)."
        ),
    )
    parser.add_argument(
        "--vocoder_ckpt_path",
        default=None,
        help="Path to the HiFi-GAN vocoder checkpoint (.ckpt). Skipped if not provided.",
    )
    parser.add_argument(
        "--config_dir",
        default=None,
        help=(
            "Path to the EveryVoice config/ directory. "
            "Auto-detected from --ckpt_path if not provided."
        ),
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
    repo_id        = f"{args.org}/EveryVoice-OpenBible-{language_hf_id}"
    dataset_config = args.dataset_config or language_name

    ckpt_path    = Path(args.ckpt_path)
    filelist_path = Path(args.filelist_path)
    vocoder_path = Path(args.vocoder_ckpt_path) if args.vocoder_ckpt_path else None

    # Auto-detect config directory from checkpoint path
    if args.config_dir:
        config_dir = Path(args.config_dir)
    else:
        config_dir = _find_config_dir(ckpt_path)

    # Verify required input files
    for label, path in [
        ("feature-prediction checkpoint", ckpt_path),
        ("filelist PSV", filelist_path),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    if vocoder_path is not None and not vocoder_path.is_file():
        raise FileNotFoundError(f"vocoder checkpoint not found: {vocoder_path}")

    if not config_dir.is_dir():
        raise FileNotFoundError(
            f"config directory not found: {config_dir}\n"
            "Pass --config_dir explicitly if your project layout differs from the default."
        )

    config_yamls = sorted(config_dir.glob("*.yaml"))
    if not config_yamls:
        raise FileNotFoundError(f"No YAML files found in config directory: {config_dir}")

    # Count utterances from the filelist (already a required argument)
    filelist_df = pd.read_csv(filelist_path, sep="|")
    utterance_count = f"{len(filelist_df):,}"

    readme = README_TEMPLATE.format(
        language_code=args.language_code,
        language_slug=language_slug,
        language_name=language_name,
        repo_id=repo_id,
        dataset_config=dataset_config,
        utterance_count=utterance_count,
    )

    print(f"Repository  : {repo_id}")
    print(f"Checkpoint  : {ckpt_path}")
    print(f"Config dir  : {config_dir}  ({len(config_yamls)} YAML files)")
    print(f"Filelist    : {filelist_path}")
    if vocoder_path:
        print(f"Vocoder     : {vocoder_path}")
    print(f"Utterances  : {utterance_count}")
    print()

    if args.dry_run:
        print("=== DRY RUN — nothing uploaded ===")
        print(f"  {ckpt_path}  →  feature_prediction.ckpt")
        for yaml_file in config_yamls:
            print(f"  {yaml_file}  →  config/{yaml_file.name}")
        print(f"  {filelist_path}  →  filelist.psv")
        if vocoder_path:
            print(f"  {vocoder_path}  →  vocoder.ckpt")
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

    # Upload README
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add README for {language_name}",
    )
    print("Uploaded README.md")

    # Upload config YAML files into config/
    for yaml_file in config_yamls:
        api.upload_file(
            path_or_fileobj=str(yaml_file),
            path_in_repo=f"config/{yaml_file.name}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add config/{yaml_file.name} for {language_name}",
        )
        print(f"Uploaded config/{yaml_file.name}  ←  {yaml_file}")

    # Upload filelist
    api.upload_file(
        path_or_fileobj=str(filelist_path),
        path_in_repo="filelist.psv",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add training filelist for {language_name}",
    )
    print(f"Uploaded filelist.psv  ←  {filelist_path}")

    # Upload vocoder (optional)
    if vocoder_path:
        api.upload_file(
            path_or_fileobj=str(vocoder_path),
            path_in_repo="vocoder.ckpt",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add vocoder checkpoint for {language_name}",
        )
        print(f"Uploaded vocoder.ckpt  ←  {vocoder_path}")

    # Upload feature-prediction checkpoint last (largest file)
    api.upload_file(
        path_or_fileobj=str(ckpt_path),
        path_in_repo="feature_prediction.ckpt",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add feature_prediction.ckpt for {language_name}",
    )
    print(f"Uploaded feature_prediction.ckpt  ←  {ckpt_path}")

    print(f"\nDone! https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
