"""
Batch TTS inference over the HuggingFace test split for a multilingual model
trained with the --prepend flag (language-ID tokens such as <|Yoruba|>).

Each text fed to the model is prefixed with the language token so that the
model receives the same format it saw during training.

Example usage (single language from a multilingual checkpoint):
    python inference-test-split-tagged.py \
        --languages Yoruba \
        --ckpt_path ckpts/F5TTS_v1_Base_vocos_custom_open-bible-ewe-hausa-yoruba/model_last.pt \
        --vocab_file data/open-bible-ewe-hausa-yoruba_custom/vocab.txt \
        --model_cfg src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Ewe-Hausa-Yoruba.yaml \
        --metadata_path data/open-bible-ewe-hausa-yoruba/metadata.csv

Example usage (all languages in one shot):
    python inference-test-split-tagged.py \
        --languages Yoruba Ewe Hausa \
        --ckpt_path ckpts/F5TTS_v1_Base_vocos_custom_open-bible-ewe-hausa-yoruba/model_last.pt \
        --vocab_file data/open-bible-ewe-hausa-yoruba_custom/vocab.txt \
        --model_cfg src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Ewe-Hausa-Yoruba.yaml \
        --metadata_path data/open-bible-ewe-hausa-yoruba/metadata.csv
"""

import argparse
import io
import os
import re
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


# Matches an existing <|...|> tag at the start of a string (with optional space).
_LEADING_TAG_RE = re.compile(r"^<\|[^|>]+\|>\s*")


def strip_tag(text: str) -> str:
    """Remove a leading language-ID token if present."""
    return _LEADING_TAG_RE.sub("", text).strip()


def with_tag(language: str, text: str) -> str:
    """Ensure text starts with <|language|>, replacing any existing tag."""
    return f"<|{language}|> {strip_tag(text)}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch TTS inference for a multilingual model trained with --prepend. "
            "Runs over the HuggingFace test split for each specified language."
        )
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        help=(
            "One or more language names as they appear in the HuggingFace dataset "
            "(e.g. 'Yoruba', 'Ewe Hausa'). Each language is processed in sequence."
        ),
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to the multilingual model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--vocab_file",
        required=True,
        help="Path to the vocabulary file (vocab.txt) for the multilingual model.",
    )
    parser.add_argument(
        "--model_cfg",
        required=True,
        help="Path to the Hydra model config YAML for the multilingual model.",
    )
    parser.add_argument(
        "--metadata_path",
        required=True,
        help=(
            "Path to the combined training metadata CSV (pipe-separated). "
            "Used to select a reference audio for each language. "
            "This is the metadata.csv produced by prepare_data.py --multilingual --prepend."
        ),
    )
    parser.add_argument(
        "--ref_index",
        type=int,
        default=0,
        help="Index into the filtered reference candidates per language (default: 0).",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        metavar="N",
        help="Only synthesize the first N samples per language (useful for quick tests).",
    )
    parser.add_argument(
        "--prepend",
        action="store_true",
        default=False,
        help=(
            "Prepend <|Language|> to the reference text. Use this when the model was "
            "trained with --prepend. Omit for models trained without language-ID tokens."
        ),
    )
    return parser.parse_args()


def get_audio_filename(example):
    example["filename"] = example["audio"]["path"].split("/")[-1]
    return example


def pick_reference(train: pd.DataFrame, language: str, ref_index: int):
    """Select a reference audio row for *language* from the combined metadata.

    Strategy: prefer a New-Testament audio between 6 and 10 seconds, matching
    the same heuristic used in inference-test-split.py.
    """
    lang_rows = train[
        train["audio_file"].apply(
            lambda p: language.lower() in Path(p).parts[-3].lower()
        )
    ].copy()

    if lang_rows.empty:
        # Fallback: search the text column for the language tag
        lang_rows = train[
            train["text"].str.startswith(f"<|{language}|>", na=False)
        ].copy()

    if lang_rows.empty:
        raise ValueError(
            f"No training rows found for language '{language}' in {len(train)} metadata rows. "
            f"Check that the metadata CSV was built with --prepend and contains this language."
        )

    lang_rows["duration_seconds"] = lang_rows["audio_file"].apply(
        lambda p: sf.info(p).duration
    )

    candidates = lang_rows[
        (lang_rows["duration_seconds"] >= 6) & (lang_rows["duration_seconds"] <= 10)
    ]
    candidates = (
        candidates[candidates["audio_file"].str.contains("New-Testament", case=True)]
        .sort_values("audio_file")
    )

    if candidates.empty:
        # Relax filters if nothing matched
        candidates = lang_rows.sort_values("duration_seconds")

    ref_row = candidates.iloc[ref_index]
    return ref_row


def run_language(
    language: str,
    train: pd.DataFrame,
    ema_model,
    vocoder,
    output_base: str,
    ref_index: int,
    head: int | None,
    prepend: bool = False,
):
    print(f"\n{'='*60}")
    print(f"Language: {language}")
    print(f"{'='*60}")

    OUTPUT_DIR = os.path.join(output_base, language)
    GROUND_TRUTH_DIR = os.path.join(OUTPUT_DIR, "ground-truth")
    GENERATED_DIR = os.path.join(OUTPUT_DIR, "generated")
    os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)

    # ── Load test set ──────────────────────────────────────────────────────────
    print(f"Loading test set for {language} ...")
    ds = load_dataset(
        "parquet",
        data_files={
            "test": f"hf://datasets/davidguzmanr/open-bible-resources/{language}/test-*.parquet"
        },
        split="test",
    )
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.map(get_audio_filename)

    test = ds.remove_columns("audio").to_pandas()
    print(f"Test samples: {len(test)}")

    # ── Save ground-truth WAVs ─────────────────────────────────────────────────
    for example in tqdm(ds, desc=f"Saving ground-truth WAVs ({language})"):
        filename = example["filename"]
        stem     = os.path.splitext(filename)[0]
        out_path = os.path.join(GROUND_TRUTH_DIR, f"{stem}.wav")
        if os.path.exists(out_path):
            continue
        audio_bytes = example["audio"]["bytes"]
        with io.BytesIO(audio_bytes) as buf:
            audio_array, sample_rate = sf.read(buf)
        sf.write(out_path, audio_array, sample_rate)
    print(f"Ground-truth WAVs saved to: {GROUND_TRUTH_DIR}")

    # ── Pick a reference audio ─────────────────────────────────────────────────
    ref_row = pick_reference(train, language, ref_index)
    REF_AUDIO = ref_row["audio_file"]

    # Strip any existing tag then re-add only if the model was trained with --prepend.
    if prepend:
        REF_TEXT = with_tag(language, ref_row["text"])
    else:
        REF_TEXT = strip_tag(ref_row["text"])

    print(f"Reference audio : {REF_AUDIO}")
    print(f"Reference text  : {REF_TEXT}")
    print(f"Reference dur   : {ref_row.get('duration_seconds', '?'):.2f}s" if "duration_seconds" in ref_row else "")

    # ── Preprocess reference audio ─────────────────────────────────────────────
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    ref_audio, ref_text_processed = preprocess_ref_audio_text(REF_AUDIO, REF_TEXT)
    # preprocess_ref_audio_text may normalise whitespace; re-apply tag to be safe.
    if prepend:
        ref_text_final = with_tag(language, ref_text_processed)
    else:
        ref_text_final = strip_tag(ref_text_processed)
    print(f"Final ref_text  : {ref_text_final}")

    # ── Batch inference ────────────────────────────────────────────────────────
    rows = test.head(head) if head is not None else test

    generated_files = []
    errors = []

    for idx, row in tqdm(rows.iterrows(), total=len(rows), desc=f"Synthesizing ({language})"):
        stem         = os.path.splitext(row["filename"])[0]
        out_path     = os.path.join(GENERATED_DIR, f"{stem}.wav")

        if os.path.exists(out_path):
            generated_files.append(out_path)
            continue

        # ref_text_final already carries the language tag; gen_text must not add
        # a second one so that the combined ref_text + gen_text has exactly one
        # <|Language|> token at the very beginning, matching the training format.
        gen_text = strip_tag(row["text"])

        try:
            audio_segment, final_sample_rate, _ = infer_process(
                ref_audio,
                ref_text_final,
                gen_text,
                ema_model,
                vocoder,
                mel_spec_type="vocos",
            )
            sf.write(out_path, audio_segment, final_sample_rate)
            generated_files.append(out_path)
        except Exception as e:
            print(f"Error on {stem}.wav: {e}")
            errors.append({
                "filename": f"{stem}.wav",
                "original_filename": row["filename"],
                "error": str(e),
            })

    print(f"Done ({language}): {len(generated_files)} generated, {len(errors)} errors.")

    csv_path = os.path.join(OUTPUT_DIR, "test.csv")
    test.to_csv(csv_path, index=False)
    print(f"Saved test dataframe: {csv_path}")

    if errors:
        errors_df = pd.DataFrame(errors)
        errors_path = os.path.join(OUTPUT_DIR, "errors.csv")
        errors_df.to_csv(errors_path, index=False)
        print(f"Saved error log: {errors_path}")

    return len(generated_files), len(errors)


def main():
    args = parse_args()

    CKPT_PATH  = args.ckpt_path
    VOCAB_FILE = args.vocab_file
    MODEL_CFG  = args.model_cfg

    # Output sits under synthesis_output/<checkpoint-dir>/<language>/
    OUTPUT_BASE = os.path.join("synthesis_output", Path(CKPT_PATH).parent.name)
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # ── Load combined training metadata once ───────────────────────────────────
    print(f"Loading training metadata from: {args.metadata_path}")
    train = pd.read_csv(args.metadata_path, sep="|")
    print(f"Total training rows: {len(train)}")

    # ── Load model & vocoder once, reuse across all languages ─────────────────
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from f5_tts.infer.utils_infer import load_model, load_vocoder

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
    print(f"Model loaded ✓  (vocab: {VOCAB_FILE})")
    print(f"Languages to process: {args.languages}")

    # ── Run inference per language ─────────────────────────────────────────────
    summary = {}
    for language in args.languages:
        n_ok, n_err = run_language(
            language=language,
            train=train,
            ema_model=ema_model,
            vocoder=vocoder,
            output_base=OUTPUT_BASE,
            ref_index=args.ref_index,
            head=args.head,
            prepend=args.prepend,
        )
        summary[language] = {"generated": n_ok, "errors": n_err}

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for lang, counts in summary.items():
        print(f"  {lang:20s}  generated={counts['generated']:5d}  errors={counts['errors']:4d}")
    print(f"\nAll outputs under: {OUTPUT_BASE}/")


if __name__ == "__main__":
    main()
