"""
Batch TTS inference over the HuggingFace test split for a given language/checkpoint.

Example usage:
    python inference-test-split.py \
        --language Yoruba \
        --ckpt_path ckpts/F5TTS_v1_Base_vocos_custom_open-bible-yoruba-nt/model_last.pt \
        --vocab_file data/open-bible-yoruba-nt_custom/vocab.txt \
        --model_cfg src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Yoruba-Nt.yaml \
        --metadata_path data/open-bible-yoruba-nt/metadata.csv
"""

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, load_dataset
from tqdm import tqdm


# ── Fix: torchaudio 2.10 defaults to torchcodec which needs FFmpeg libs ───────
def _torchaudio_load_sf(filepath, *args, **kwargs):
    data, samplerate = sf.read(filepath, dtype="float32")
    tensor = torch.from_numpy(data.T if data.ndim > 1 else data[None, :])
    return tensor, samplerate

torchaudio.load = _torchaudio_load_sf
print("Patched torchaudio.load → soundfile backend")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch TTS inference over the HuggingFace test split."
    )
    parser.add_argument(
        "--language",
        required=True,
        help="Language name as it appears in the HuggingFace dataset path "
             "(e.g. 'Yoruba', 'Ewe', 'Igbo').",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to the model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--vocab_file",
        required=True,
        help="Path to the vocabulary file (vocab.txt).",
    )
    parser.add_argument(
        "--model_cfg",
        required=True,
        help="Path to the model config YAML file.",
    )
    parser.add_argument(
        "--metadata_path",
        required=True,
        help="Path to the training metadata CSV (pipe-separated) used to pick "
             "the reference audio.",
    )
    parser.add_argument(
        "--ref_index",
        type=int,
        default=0,
        help="Index into the filtered reference candidates (default: 0).",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        metavar="N",
        help="Only synthesize the first N samples (useful for quick tests).",
    )
    return parser.parse_args()


def get_audio_filename(example):
    example["filename"] = example["audio"]["path"].split("/")[-1]
    return example


def main():
    args = parse_args()

    LANGUAGE      = args.language
    CKPT_PATH     = args.ckpt_path
    VOCAB_FILE    = args.vocab_file
    MODEL_CFG     = args.model_cfg
    METADATA_PATH = args.metadata_path

    OUTPUT_DIR = f"synthesis_output/{Path(CKPT_PATH).parent.name}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "generated"), exist_ok=True)

    # ── Load test set ──────────────────────────────────────────────────────────
    print(f"Loading test set for language: {LANGUAGE} ...")
    ds = load_dataset(
        "parquet",
        data_files={
            "test": f"hf://datasets/davidguzmanr/open-bible-resources/{LANGUAGE}/test-*.parquet"
        },
        split="test",
    )
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.map(get_audio_filename)

    test = ds.remove_columns("audio").to_pandas()
    print(f"Test samples: {len(test)}")

    # ── Save ground-truth audio files ─────────────────────────────────────────
    GROUND_TRUTH_DIR = f"{OUTPUT_DIR}/ground-truth"
    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)

    for example in tqdm(ds, desc="Saving ground-truth WAVs"):
        filename = example["filename"]
        stem     = os.path.splitext(filename)[0]
        out_path = f"{GROUND_TRUTH_DIR}/{stem}.wav"

        if os.path.exists(out_path):
            continue

        audio_bytes = example["audio"]["bytes"]
        with io.BytesIO(audio_bytes) as buf:
            audio_array, sample_rate = sf.read(buf)
        sf.write(out_path, audio_array, sample_rate)

    print(f"Ground-truth WAVs saved to: {GROUND_TRUTH_DIR}")

    # ── Pick a reference audio from the training set ───────────────────────────
    print(f"Loading training metadata from: {METADATA_PATH}")
    train = pd.read_csv(METADATA_PATH, sep="|")
    train["language"] = train["audio_file"].apply(lambda x: x.split("/")[-3])
    train = train[train["language"].str.lower().str.contains(LANGUAGE.lower())]
    train["duration_seconds"] = train["audio_file"].apply(
        lambda path: sf.info(path).duration
    )

    candidates = train[
        (train["duration_seconds"] >= 6) & (train["duration_seconds"] <= 10)
    ]
    candidates = (
        candidates[candidates["audio_file"].str.contains("New-Testament", case=True)]
        .sort_values("audio_file")
    )

    ref_row   = candidates.iloc[args.ref_index]
    REF_AUDIO = ref_row["audio_file"]
    REF_TEXT  = ref_row["text"]

    print(f"Reference audio : {REF_AUDIO}")
    print(f"Reference text  : {REF_TEXT}")
    print(f"Reference dur   : {ref_row['duration_seconds']:.2f}s")

    # ── Load model & vocoder ───────────────────────────────────────────────────
    from hydra.utils import get_class
    from omegaconf import OmegaConf

    from f5_tts.infer.utils_infer import (
        infer_process,
        load_model,
        load_vocoder,
        preprocess_ref_audio_text,
    )

    model_cfg = OmegaConf.load(MODEL_CFG)
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    vocoder = load_vocoder(vocoder_name="vocos", is_local=False)

    ema_model = load_model(
        model_cls,
        model_arc,
        CKPT_PATH,
        mel_spec_type="vocos",
        vocab_file=VOCAB_FILE,
    )
    print("Model loaded ✓")

    # ── Preprocess reference audio ─────────────────────────────────────────────
    ref_audio, ref_text = preprocess_ref_audio_text(REF_AUDIO, REF_TEXT)
    print(f"Preprocessed ref_audio: {ref_audio}")
    print(f"Preprocessed ref_text : {ref_text}")

    # ── Batch inference ────────────────────────────────────────────────────────
    rows = test.head(args.head) if args.head is not None else test

    generated_files = []
    errors = []

    for idx, row in tqdm(rows.iterrows(), total=len(rows), desc="Synthesizing"):
        gen_text     = row["text"]
        stem         = os.path.splitext(row["filename"])[0]
        out_filename = f"{stem}.wav"
        out_path     = os.path.join(OUTPUT_DIR, "generated", out_filename)

        if os.path.exists(out_path):
            generated_files.append(out_path)
            continue

        try:
            audio_segment, final_sample_rate, _ = infer_process(
                ref_audio,
                ref_text,
                gen_text,
                ema_model,
                vocoder,
                mel_spec_type="vocos",
            )
            sf.write(out_path, audio_segment, final_sample_rate)
            generated_files.append(out_path)
        except Exception as e:
            print(f"Error on {out_filename}: {e}")
            errors.append({"filename": out_filename, "original_filename": row["filename"], "error": str(e)})

    print(f"\nDone! {len(generated_files)} files generated, {len(errors)} errors.")
    print(f"Output directory: {OUTPUT_DIR}")

    csv_path = os.path.join(OUTPUT_DIR, "test.csv")
    test.to_csv(csv_path, index=False)
    print(f"Saved test dataframe to: {csv_path}")

    if errors:
        errors_df = pd.DataFrame(errors)
        errors_path = os.path.join(OUTPUT_DIR, "errors.csv")
        errors_df.to_csv(errors_path, index=False)
        print(f"Saved error log to: {errors_path}")


if __name__ == "__main__":
    main()
