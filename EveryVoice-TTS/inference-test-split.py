"""
Inference script for EveryVoice TTS on the test split of an Open-Bible language.

Loads models once, saves ground-truth WAVs, then synthesises all test sentences
in batches.  Generated and ground-truth files share the same stem so that any
paired evaluation metric (e.g. MOS, UTMOS, speaker-similarity) can be run
directly on the two directories afterwards.

Usage example:
    python inference-test-split.py \
        --language Yoruba \
        --ckpt_path Open-Bible-Yoruba-NT/logs_and_checkpoints/FeaturePredictionExperiment/base/checkpoints/last.ckpt \
        --vocoder_ckpt_path /path/to/hifigan_universal_v1_everyvoice.ckpt \
        --output_dir synthesis_output/yoruba-nt
"""

import argparse
import io
import os
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

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
        description="EveryVoice TTS inference on the Open-Bible test split."
    )
    parser.add_argument(
        "--language",
        required=True,
        help="Language name as it appears in the HuggingFace dataset (e.g. 'Yoruba').",
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
        "--output_dir",
        required=True,
        help="Root directory where outputs (ground-truth WAVs, generated WAVs, test.csv) are saved.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_audio_filename(example):
    example["filename"] = example["audio"]["path"].split("/")[-1]
    return example


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_dir = output_dir / "ground-truth"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    # EveryVoice always writes WAVs to {output_dir}/wav/; we rename and move
    # them into {output_dir}/generated/ so names match the ground-truth files.
    everyvoice_wav_dir = output_dir / "wav"
    wav_dir = output_dir / "generated"

    test_csv_path = output_dir / "test.csv"

    # ── Load test split ───────────────────────────────────────────────────────
    print(f"Loading test split for language: {args.language}")
    ds = load_dataset(
        "parquet",
        data_files={
            "test": f"hf://datasets/davidguzmanr/open-bible-resources/{args.language}/test-*.parquet"
        },
        split="test",
    )
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.map(get_audio_filename)

    test_df = ds.remove_columns("audio").to_pandas()
    print(f"Test samples: {len(test_df)}")

    test_df.to_csv(test_csv_path, index=False)
    print(f"Saved test dataframe to: {test_csv_path}")

    # ── Save ground-truth WAVs ────────────────────────────────────────────────
    print("Saving ground-truth WAV files…")
    for example in tqdm(ds, desc="Saving ground-truth WAVs"):
        filename = example["filename"]
        stem = os.path.splitext(filename)[0]
        out_path = ground_truth_dir / f"{stem}.wav"

        with io.BytesIO(example["audio"]["bytes"]) as buf:
            audio_array, sample_rate = sf.read(buf)

        sf.write(str(out_path), audio_array, sample_rate)

    print(f"Saved {len(ds)} WAV files to {ground_truth_dir}")

    # ── Prepare test DataFrame for synthesis ──────────────────────────────────
    test_df = pd.read_csv(test_csv_path)
    test_df = test_df.rename(columns={"filename": "file"})
    test_df = test_df[["file", "text"]]
    print(f"Total sentences to synthesise: {len(test_df)}")

    # ── Load models ───────────────────────────────────────────────────────────
    device = get_device_from_accelerator("gpu")
    print(f"Using device: {device}")

    feature_prediction_checkpoint = Path(args.ckpt_path)
    vocoder_base_checkpoint = Path(args.vocoder_ckpt_path)

    print("Loading feature prediction model…")
    model = FastSpeech2.load_from_checkpoint(str(feature_prediction_checkpoint)).to(device)
    model.eval()
    global_step = get_global_step(feature_prediction_checkpoint)

    print("Loading vocoder…")
    vocoder_ckpt = torch.load(str(vocoder_base_checkpoint), map_location=device, weights_only=True)
    vocoder_model, vocoder_config = load_hifigan_from_checkpoint(vocoder_ckpt, device)
    vocoder_global_step = get_global_step(vocoder_base_checkpoint)

    print("Models loaded successfully!")

    # ── Build filelist ────────────────────────────────────────────────────────
    default_language = next(iter(model.lang2id.keys()), None)
    default_speaker = next(iter(model.speaker2id.keys()), None)

    filelist_data = [
        {
            "basename": os.path.splitext(row["file"])[0],
            "characters": row["text"],
            "language": default_language,
            "speaker": default_speaker,
            "duration_control": 1.0,
        }
        for _, row in test_df.iterrows()
    ]
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
        accelerator="gpu",
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

    # ── Rename and move generated files to plain <stem>.wav ──────────────────
    # EveryVoice writes: {output_dir}/wav/basename--speaker--lang--ckpt=N--v_ckpt=M--pred.wav
    # We rename to plain basename.wav and move into {output_dir}/generated/.
    wav_dir.mkdir(parents=True, exist_ok=True)

    wav_files = list(everyvoice_wav_dir.glob("*.wav")) if everyvoice_wav_dir.exists() else []
    print(f"Found {len(wav_files)} generated wav files in {everyvoice_wav_dir}")

    moved = 0
    for wav_path in wav_files:
        stem = wav_path.name.split("--")[0]
        target_path = wav_dir / f"{stem}.wav"
        os.rename(wav_path, target_path)
        moved += 1

    print(f"Moved and renamed {moved} files to {wav_dir}")

    # ── Verify all expected files are present ─────────────────────────────────
    missing = []
    for _, row in test_df.iterrows():
        stem = os.path.splitext(row["file"])[0]
        expected_path = wav_dir / f"{stem}.wav"
        if not expected_path.exists():
            missing.append(row["file"])

    if missing:
        print(f"\nMissing {len(missing)} generated files:")
        for f in missing[:10]:
            print(f"  - {f}")
        if len(missing) > 10:
            print(f"  … and {len(missing) - 10} more")
    else:
        print(f"\nAll {len(test_df)} expected WAV files are present in {wav_dir}")

    print("\nDone.")
    print(f"  Ground-truth WAVs : {ground_truth_dir}")
    print(f"  Generated WAVs    : {wav_dir}")


if __name__ == "__main__":
    main()
