"""
Save ground-truth WAVs and test.csv from the Open-Bible test split.

Produces the same filename convention and CSV structure as the inference scripts
(e.g. inference-test-F5.py), so synthesized outputs can be evaluated directly
against the saved ground-truth directory.

Example usage:
    python utils/save-ground-truth.py \
        --language Hausa \
        --output_dir ground-truth/hausa
"""

import argparse
import io
import os
from pathlib import Path

import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save ground-truth WAVs and test.csv for an Open-Bible language."
    )
    parser.add_argument(
        "--language",
        required=True,
        help="Language name as used in the HuggingFace dataset (e.g. 'Hausa').",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory where ground-truth WAVs and test.csv are saved. "
             "Defaults to ground-truth/<language-lowercase>.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        metavar="N",
        help="Only save the first N samples (defaults to min(500, test size)).",
    )
    return parser.parse_args()


def make_filename(row) -> str:
    return (
        f"{row['testament']}-{row['book']}-"
        f"{row['chapter']}-{row['verse']}.wav"
    )


def main():
    args = parse_args()

    language = args.language
    output_dir = Path(
        args.output_dir if args.output_dir else f"ground-truth/{language.lower()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading test set for language: {language}")
    ds = load_dataset(
        "parquet",
        data_files={
            "test": f"hf://datasets/davidguzmanr/open-bible-resources/{language}/test-*.parquet"
        },
        split="test",
    )
    print(f"Test samples (total): {len(ds)}")

    n = min(500, len(ds)) if args.head is None else min(args.head, len(ds))
    subset = ds.select(range(n))
    subset = subset.cast_column("audio", Audio(decode=False))
    print(f"Saving {n} samples")

    csv_path = output_dir / "test.csv"
    df_meta = subset.remove_columns("audio").to_pandas()
    df_meta["filename"] = df_meta.apply(make_filename, axis=1)
    df_meta.to_csv(csv_path, index=False)
    print(f"Metadata CSV saved to: {csv_path}")

    saved_files = []
    for row in tqdm(subset, total=n, desc="Saving ground-truth WAVs"):
        out_filename = make_filename(row)
        out_path = output_dir / out_filename

        if out_path.exists():
            saved_files.append(str(out_path))
            continue

        with io.BytesIO(row["audio"]["bytes"]) as buf:
            audio_array, sample_rate = sf.read(buf)

        sf.write(str(out_path), audio_array, sample_rate)
        saved_files.append(str(out_path))

    print(f"\nDone! {len(saved_files)} files saved.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
