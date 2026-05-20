"""
Batch TTS inference with EveryVoice over a HuggingFace dataset split for a given language/checkpoint.

Example usage:
    python inference-EveryVoice-open-bible.py \\
        --language Igbo \\
        --output_dir synthesis_output/igbo-everyvoice \\
        --ckpt_path Open-Bible-Igbo/logs_and_checkpoints/FeaturePredictionExperiment/base/checkpoints/last.ckpt \\
        --vocoder_ckpt_path /path/to/hifigan_universal_v1_everyvoice.ckpt \\
        --filelist_path EveryVoice/open-bible-igbo-filelist.psv
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset

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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch EveryVoice TTS inference over a local test CSV."
    )
    parser.add_argument(
        "--language",
        required=True,
        help="Language name as used in the HuggingFace dataset (e.g. 'Igbo').",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where generated WAVs and logs are saved.",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to the EveryVoice feature-prediction checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--vocoder_ckpt_path",
        required=True,
        help="Path to the HiFi-GAN vocoder checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--filelist_path",
        required=True,
        help=(
            "Path to the EveryVoice training filelist (pipe-separated PSV with columns "
            "basename|language|speaker|characters|phones) used to pick the majority speaker."
        ),
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        metavar="N",
        help="Only synthesize the first N samples (useful for quick tests).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # EveryVoice always writes WAVs into {output_dir}/wav/; we rename and move
    # them directly into output_dir so names match the F5 layout.
    everyvoice_wav_dir = output_dir / "wav"

    # ── Load test set ─────────────────────────────────────────────────────────
    print(f"Loading test set for language: {args.language}")
    ds = load_dataset(
        "parquet",
        data_files={
            "test": f"hf://datasets/davidguzmanr/open-bible-resources/{args.language}/test-*.parquet"
        },
        split="test",
    )
    n = min(500, len(ds)) if args.head is None else min(args.head, len(ds))
    subset = ds.select(range(n))
    print(f"Test samples (total): {len(ds)}, synthesizing: {n}")

    # ── Pick the majority speaker from the EveryVoice training filelist ──────
    # The PSV has columns: basename|language|speaker|characters|phones.
    # Speaker names already match model.speaker2id, so no remapping is needed.
    # We use utterance count as a proxy for total audio (no audio paths in PSV).
    print(f"Loading EveryVoice filelist from: {args.filelist_path}")
    filelist = pd.read_csv(args.filelist_path, sep="|")
    speaker_counts = filelist["speaker"].value_counts()
    best_speaker   = speaker_counts.idxmax()
    print(
        f"Speaker with most utterances: {best_speaker} "
        f"({speaker_counts[best_speaker]} utterances)"
    )

    # ── Load models ───────────────────────────────────────────────────────────
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    device = get_device_from_accelerator(accelerator)
    print(f"Using device: {device} (accelerator: {accelerator})")

    feature_prediction_checkpoint = Path(args.ckpt_path)
    vocoder_base_checkpoint        = Path(args.vocoder_ckpt_path)

    print("Loading feature prediction model…")
    model = FastSpeech2.load_from_checkpoint(str(feature_prediction_checkpoint)).to(device)
    model.eval()
    global_step = get_global_step(feature_prediction_checkpoint)

    print("Loading vocoder…")
    vocoder_ckpt = torch.load(
        str(vocoder_base_checkpoint), map_location=device, weights_only=True
    )
    vocoder_model, vocoder_config = load_hifigan_from_checkpoint(vocoder_ckpt, device)
    vocoder_global_step = get_global_step(vocoder_base_checkpoint)

    print("Models loaded successfully!")

    # ── Map the majority speaker to model.speaker2id ──────────────────────────
    # model.speaker2id is a dict {speaker_name: int}.  If the metadata speaker_id
    # matches directly, use it; otherwise fall back to the first available key.
    if best_speaker in model.speaker2id:
        synthesis_speaker = best_speaker
    else:
        synthesis_speaker = next(iter(model.speaker2id.keys()), None)
        print(
            f"Warning: '{best_speaker}' not found in model.speaker2id "
            f"{list(model.speaker2id.keys())[:5]}…  "
            f"Falling back to '{synthesis_speaker}'."
        )

    default_language = next(iter(model.lang2id.keys()), None)
    print(f"Synthesis speaker : {synthesis_speaker}")
    print(f"Synthesis language: {default_language}")

    # ── Build filelist ────────────────────────────────────────────────────────
    # Skip files that have already been generated.
    filelist_data = []
    skipped = []
    for row in subset:
        basename = f"{row['testament']}-{row['book']}-{row['chapter']}-{row['verse']}"
        out_path = output_dir / f"{basename}.wav"
        if out_path.exists():
            skipped.append(str(out_path))
            continue
        filelist_data.append(
            {
                "basename":         basename,
                "characters":       row["text"],
                "language":         default_language,
                "speaker":          synthesis_speaker,
                "duration_control": 1.0,
            }
        )

    if skipped:
        print(f"Skipping {len(skipped)} already-generated files.")

    if not filelist_data:
        print("All files already generated. Nothing to do.")
        return

    print(f"Prepared {len(filelist_data)} entries for synthesis")

    # ── Batch synthesis ───────────────────────────────────────────────────────
    print("Starting batch synthesis…")
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
        batch_size=16,
        num_workers=4,
        filelist=None,
        filelist_data=filelist_data,
        output_dir=output_dir,
        teacher_forcing_directory=None,
        vocoder_model=vocoder_model,
        vocoder_config=vocoder_config,
        vocoder_global_step=vocoder_global_step,
    )
    print("Batch synthesis complete!")

    # ── Rename and move generated files to plain <stem>.wav ───────────────────
    # EveryVoice writes: {output_dir}/wav/basename--speaker--lang--ckpt=N--v_ckpt=M--pred.wav
    # We rename to basename.wav directly in output_dir.
    wav_files = list(everyvoice_wav_dir.glob("*.wav")) if everyvoice_wav_dir.exists() else []
    print(f"Found {len(wav_files)} generated wav files in {everyvoice_wav_dir}")

    moved = 0
    for wav_path in wav_files:
        stem        = wav_path.name.split("--")[0]
        target_path = output_dir / f"{stem}.wav"
        os.rename(wav_path, target_path)
        moved += 1

    print(f"Moved and renamed {moved} files to {output_dir}")

    # ── Verify all expected files are present ─────────────────────────────────
    missing = []
    for entry in filelist_data:
        expected_path = output_dir / f"{entry['basename']}.wav"
        if not expected_path.exists():
            missing.append(entry["basename"])

    if missing:
        missing_list = "\n  - ".join(missing)
        raise RuntimeError(
            f"{len(missing)} expected output file(s) were not generated:\n  - {missing_list}"
        )

    total = len(filelist_data) + len(skipped)
    print(f"\nDone! {total} files present ({len(filelist_data)} newly generated, {len(skipped)} skipped).")

    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
