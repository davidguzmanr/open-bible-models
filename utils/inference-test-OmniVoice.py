"""
Batch synthesis with the k2-fsa/OmniVoice zero-shot TTS model.

Loads the test split from hf://datasets/davidguzmanr/open-bible-resources/{language}/test-*.parquet
and writes one WAV per row to output_dir. A single reference audio file and its transcription are
used as the voice prompt for every row, so the entire test set is rendered in a consistent voice.
Existing output files are skipped on subsequent invocations, making the script safe to re-run.

The reference audio and text can be supplied directly via --ref_audio /
--ref_text, or derived automatically from a training metadata CSV via
--metadata_path (pipe-separated, with columns: audio_file, text,
speaker_id). When --metadata_path is given the script selects the
speaker with the most total audio, then picks that speaker's shortest
clip whose duration falls between 6 and 10 seconds.

If --ref_text is omitted and --metadata_path is not used, OmniVoice
auto-transcribes the reference clip with its built-in Whisper ASR. For
low-resource languages where Whisper accuracy is poor, supply an
explicit transcription.

Examples:
    # explicit reference
    python inference-OmniVoice-open-bible.py \\
        --language Igbo \\
        --output_dir audios/open-bible/OmniVoice/Igbo \\
        --ref_audio /path/to/igbo-ref.wav \\
        --ref_text "Exact transcription of the clip"

    # reference derived from metadata
    python inference-OmniVoice-open-bible.py \\
        --language Igbo \\
        --output_dir audios/open-bible/OmniVoice/Igbo \\
        --metadata_path data/open-bible-igbo/metadata.csv
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--language", required=True,
        help="Language name as used in the HuggingFace dataset (e.g. 'Igbo').",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Directory where generated WAVs are saved.",
    )
    p.add_argument(
        "--metadata_path", default=None,
        help="Path to the training metadata CSV (pipe-separated, columns: "
             "audio_file, text, speaker_id). When supplied, the reference audio "
             "and text are selected automatically (speaker with most audio, "
             "shortest clip between 6–10 s). Mutually exclusive with --ref_audio.",
    )
    p.add_argument(
        "--ref_audio", default=None,
        help="Path to a single reference WAV used for all rows. "
             "Required when --metadata_path is not given.",
    )
    p.add_argument(
        "--ref_text", default=None,
        help="Transcription of --ref_audio (recommended for low-resource langs). "
             "Omit to let OmniVoice auto-transcribe via Whisper.",
    )
    p.add_argument(
        "--model_card", default="k2-fsa/OmniVoice",
        help="HF repo id for the OmniVoice checkpoint.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype", default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    p.add_argument(
        "--sample_rate", type=int, default=24000,
        help="Output sample rate (OmniVoice default: 24000).",
    )
    p.add_argument(
        "--head", type=int, default=None, metavar="N",
        help="Only synthesize the first N samples (useful for quick tests).",
    )
    return p.parse_args()


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve reference audio and text ──────────────────────────────────────
    if args.metadata_path and args.ref_audio:
        raise SystemExit("ERROR: --metadata_path and --ref_audio are mutually exclusive.")
    if not args.metadata_path and not args.ref_audio:
        raise SystemExit("ERROR: one of --metadata_path or --ref_audio is required.")

    if args.metadata_path:
        print(f"Loading training metadata from: {args.metadata_path}")
        train = pd.read_csv(args.metadata_path, sep="|")
        train["duration_seconds"] = train["audio_file"].apply(
            lambda path: sf.info(path).duration
        )

        speaker_totals = train.groupby("speaker_id")["duration_seconds"].sum()
        best_speaker   = speaker_totals.idxmax()
        print(f"Speaker with most audio: {best_speaker} "
              f"({speaker_totals[best_speaker]:.1f}s total)")

        speaker_rows = train[train["speaker_id"] == best_speaker]
        candidates   = speaker_rows[speaker_rows["duration_seconds"].between(6, 10)]
        if candidates.empty:
            raise SystemExit(
                f"ERROR: No clips between 6–10 s found for speaker {best_speaker}."
            )
        ref_row   = candidates.loc[candidates["duration_seconds"].idxmin()]
        ref_audio = ref_row["audio_file"]
        ref_text  = ref_row["text"]
        print(f"Reference audio : {ref_audio}")
        print(f"Reference text  : {ref_text}")
        print(f"Reference dur   : {ref_row['duration_seconds']:.2f}s")
    else:
        ref_audio = args.ref_audio
        ref_text  = args.ref_text
        if not Path(ref_audio).is_file():
            raise SystemExit(f"ERROR: --ref_audio not found: {ref_audio}")

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

    print(f"Reference: {ref_audio}")
    print(f"  ref_text: " + (ref_text[:80] if ref_text else "(auto-transcribed via Whisper)"))
    print(f"Output -> {output_dir}")

    # Local imports
    from omnivoice import OmniVoice

    print(f"Loading OmniVoice from {args.model_card} on {args.device} ({args.dtype})…")
    model = OmniVoice.from_pretrained(
        args.model_card,
        device_map=args.device,
        dtype=DTYPE_MAP[args.dtype],
    )
    print("Model ready.")

    completed = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    for row in tqdm(subset, total=n, desc="Synthesising"):
        out_filename = f"{row['testament']}-{row['book']}-{row['chapter']}-{row['verse']}.wav"
        out_path = output_dir / out_filename
        if out_path.exists():
            skipped += 1
            continue

        try:
            gen_kwargs = {
                "text": row["text"],
                "ref_audio": ref_audio,
            }
            if ref_text:
                gen_kwargs["ref_text"] = ref_text
            audio = model.generate(**gen_kwargs)
        except Exception as exc:
            failed.append((out_filename, str(exc)))
            continue

        # OmniVoice returns list-of-tensor-or-numpy. Normalise to [1, T] torch.
        wav = audio[0] if isinstance(audio, (list, tuple)) else audio
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        wav = wav.squeeze()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        torchaudio.save(str(out_path), wav.cpu(), args.sample_rate)
        completed += 1

    print(f"\nDone. {completed} generated, {skipped} already existed, {len(failed)} failed.")
    if failed:
        print("First few failures:")
        for stem, err in failed[:10]:
            print(f"  - {stem}: {err}")


if __name__ == "__main__":
    main()

