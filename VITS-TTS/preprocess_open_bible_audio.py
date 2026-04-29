#!/usr/bin/env python3
"""
Resample Open Bible-style TTS metadata WAVs to a target sample rate for VITS training.

Expected CSV format (same as train_vits.py open_bible_formatter):

    audio_file|text|speaker_id

with a header row. Each audio_file is loaded from disk (typically an absolute path),
converted to mono if needed, resampled with band-limited interpolation when the
native rate differs from the target, and written as WAV under output_dir/wavs/.
A new metadata.csv is written under output_dir with absolute
paths pointing at the processed files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metadata",
        required=True,
        help="Path to pipe-separated metadata.csv (audio_file|text|speaker_id, with header)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to create (wavs/ + metadata.csv with absolute paths)",
    )
    p.add_argument(
        "--target-sample-rate",
        type=int,
        default=22050,
        help="Output sample rate in Hz (default: 22050)",
    )
    return p.parse_args()


def load_rows(meta_path: Path) -> tuple[str, list[tuple[str, str, str]]]:
    rows: list[tuple[str, str, str]] = []
    with meta_path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n\r")
        for line_no, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            audio_file = parts[0].strip()
            text = parts[1].strip()
            speaker = parts[2].strip() if len(parts) > 2 else "default"
            rows.append((audio_file, text, speaker))
    return header, rows


def process_wav(in_path: Path, out_path: Path, target_sr: int) -> None:
    wav, sr = torchaudio.load(str(in_path))
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = torch.clamp(wav, -1.0, 1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), wav, target_sr)


def main() -> None:
    args = parse_args()
    meta_in = Path(args.metadata).resolve()
    out_root = Path(args.output_dir).resolve()
    wav_dir = out_root / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    if not meta_in.is_file():
        print(f"error: metadata file not found: {meta_in}", file=sys.stderr)
        sys.exit(1)

    header, rows = load_rows(meta_in)
    if not rows:
        print("error: no data rows in metadata", file=sys.stderr)
        sys.exit(1)

    meta_out = out_root / "metadata.csv"
    out_lines: list[str] = [header]

    for i, (audio_file, text, speaker) in enumerate(
        tqdm(rows, desc="Preprocessing", unit="file", total=len(rows))
    ):
        in_path = Path(audio_file)
        if not in_path.is_file():
            print(f"error: missing audio file (row {i + 2}): {in_path}", file=sys.stderr)
            sys.exit(1)

        out_name = in_path.name
        if not out_name.lower().endswith(".wav"):
            out_name = f"{out_name}.wav"
        out_path = (wav_dir / out_name).resolve()

        process_wav(in_path, out_path, args.target_sample_rate)
        new_audio = str(out_path)
        line = "|".join([new_audio, text, speaker])
        out_lines.append(line)

    meta_out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"wrote {meta_out}")
    print(f"wrote {len(rows)} wav files under {wav_dir}")


if __name__ == "__main__":
    main()
