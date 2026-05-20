"""
Batch TTS inference over a HuggingFace dataset split for a given language/checkpoint.

Example usage:
    python inference-test-VITS.py \
        --language      Igbo \
        --output_dir    synthesis_output/igbo-vits \
        --ckpt_path     outputs/igbo/vits_igbo-April-29-2026_11+52PM-fd8dd03/checkpoint_250000.pth \
        --metadata_path data/open-bible-igbo/metadata.csv
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Monkey-patch: guard rational_quadratic_spline against empty-tensor inputs.
# Identical to the patch applied in train_vits.py so inference uses the same
# corrected code path.
# ---------------------------------------------------------------------------
import TTS.tts.layers.vits.transforms as _vits_transforms

_orig_rqs = _vits_transforms.rational_quadratic_spline


def _patched_rqs(inputs, *args, **kwargs):
    if inputs.numel() == 0:
        return inputs, torch.zeros_like(inputs)
    return _orig_rqs(inputs, *args, **kwargs)


_vits_transforms.rational_quadratic_spline = _patched_rqs


from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.synthesizer import Synthesizer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch Coqui VITS TTS inference over a local test CSV."
    )
    parser.add_argument(
        "--language",
        required=True,
        help="Language name as used in the HuggingFace dataset (e.g. 'Igbo').",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where generated WAVs are saved.",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to the VITS checkpoint (.pth file).",
    )
    parser.add_argument(
        "--config_path",
        default=None,
        help=(
            "Path to config.json. Defaults to config.json in the same directory "
            "as --ckpt_path."
        ),
    )
    parser.add_argument(
        "--speakers_path",
        default=None,
        help=(
            "Path to speakers.pth. Defaults to speakers.pth in the same directory "
            "as --ckpt_path."
        ),
    )
    parser.add_argument(
        "--metadata_path",
        required=True,
        help=(
            "Path to the training metadata CSV (pipe-separated: audio_file|text|speaker_id, "
            "with header) used to pick the majority speaker by utterance count."
        ),
    )
    parser.add_argument(
        "--speaker_name",
        default=None,
        help=(
            "Force a specific speaker name instead of auto-selecting the majority speaker "
            "from --metadata_path."
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

    ckpt_path = Path(args.ckpt_path)
    ckpt_dir  = ckpt_path.parent

    config_path   = Path(args.config_path)   if args.config_path   else ckpt_dir / "config.json"
    speakers_path = Path(args.speakers_path) if args.speakers_path else ckpt_dir / "speakers.pth"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate derived paths ─────────────────────────────────────────────────
    for p, label in [(config_path, "config.json"), (speakers_path, "speakers.pth")]:
        if not p.exists():
            raise FileNotFoundError(
                f"{label} not found at {p}. Pass --config_path / --speakers_path explicitly."
            )

    print(f"Checkpoint : {ckpt_path}")
    print(f"Config     : {config_path}")
    print(f"Speakers   : {speakers_path}")

    # ── Load test set ──────────────────────────────────────────────────────────
    print(f"\nLoading test set for language: {args.language}")
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

    # ── Save metadata CSV ──────────────────────────────────────────────────────
    csv_path = output_dir / "test.csv"
    df_meta = subset.to_pandas().drop(columns=["audio"])
    df_meta["filename"] = (
        df_meta["testament"].astype(str) + "-" +
        df_meta["book"].astype(str) + "-" +
        df_meta["chapter"].astype(str) + "-" +
        df_meta["verse"].astype(str) + ".wav"
    )
    df_meta.to_csv(csv_path, index=False)
    print(f"Metadata CSV saved to: {csv_path}")

    # ── Pick majority speaker from training metadata ───────────────────────────
    print(f"\nLoading training metadata from: {args.metadata_path}")
    metadata = pd.read_csv(args.metadata_path, sep="|")
    speaker_counts = metadata["speaker_id"].value_counts()
    majority_speaker = speaker_counts.idxmax()
    print(
        f"Speaker with most utterances: {majority_speaker} "
        f"({speaker_counts[majority_speaker]} utterances)"
    )

    synthesis_speaker = args.speaker_name if args.speaker_name else majority_speaker
    print(f"Using speaker: {synthesis_speaker}")

    # ── Load Synthesizer ───────────────────────────────────────────────────────
    use_cuda = torch.cuda.is_available()
    print(f"\nCUDA available: {use_cuda}")
    print("Loading Synthesizer…")

    synthesizer = Synthesizer(
        tts_checkpoint=str(ckpt_path),
        tts_config_path=str(config_path),
        tts_speakers_file=str(speakers_path),
        use_cuda=use_cuda,
    )

    # Coqui's Synthesizer stores tts_speakers_file but does not inject it into
    # the config before building the model, so SpeakerManager.init_from_config
    # returns None.  Manually restore from the saved .pth file when needed.
    if synthesizer.tts_model.speaker_manager is None:
        synthesizer.tts_model.speaker_manager = SpeakerManager(
            speaker_id_file_path=str(speakers_path)
        )

    sm = synthesizer.tts_model.speaker_manager
    print(f"Model loaded! Speakers ({sm.num_speakers}): {sorted(sm.speaker_names)}")

    # Validate speaker name against the model
    if synthesis_speaker not in sm.speaker_names:
        fallback = sorted(sm.speaker_names)[0]
        print(
            f"Warning: '{synthesis_speaker}' not found in model speakers "
            f"{sorted(sm.speaker_names)[:5]}…  Falling back to '{fallback}'."
        )
        synthesis_speaker = fallback

    sample_rate = synthesizer.output_sample_rate
    print(f"Output sample rate: {sample_rate} Hz")

    # ── Batch inference ────────────────────────────────────────────────────────
    skipped = []
    generated = []

    for row in tqdm(subset, total=n, desc="Synthesizing"):
        out_filename = f"{row['testament']}-{row['book']}-{row['chapter']}-{row['verse']}.wav"
        out_path     = output_dir / out_filename

        if out_path.exists():
            skipped.append(str(out_path))
            continue

        wav_data = synthesizer.tts(
            text=row["text"],
            speaker_name=synthesis_speaker,
            split_sentences=True,
        )
        waveform = np.array(wav_data, dtype=np.float32)
        sf.write(str(out_path), waveform, sample_rate)
        generated.append(str(out_path))

    if skipped:
        print(f"\nSkipped {len(skipped)} already-generated files.")

    # ── Verify all expected files are present ─────────────────────────────────
    missing = []
    for row in subset:
        out_filename  = f"{row['testament']}-{row['book']}-{row['chapter']}-{row['verse']}.wav"
        expected_path = output_dir / out_filename
        if not expected_path.exists():
            missing.append(out_filename)

    if missing:
        missing_list = "\n  - ".join(missing)
        raise RuntimeError(
            f"{len(missing)} expected output file(s) were not generated:\n  - {missing_list}"
        )

    total = len(generated) + len(skipped)
    print(
        f"\nDone! {total} files present "
        f"({len(generated)} newly generated, {len(skipped)} skipped)."
    )
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
