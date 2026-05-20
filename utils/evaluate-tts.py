#!/usr/bin/env python3
"""Evaluate a TTS model against ground truth audio on multiple metrics.

Usage
-----
    python evaluate-tts.py \\
        /path/to/ground_truth \\
        /path/to/synthesized \\
        /path/to/test.csv \\
        /path/to/results.csv \\
        --metrics mcd speechbertscore utmos wer

The script is fully resumable: re-running it will skip rows that are already
scored in OUTPUT_CSV.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import gc

import numpy as np
import pandas as pd
from tqdm import tqdm


VALID_METRICS = ["mcd", "speechbertscore", "utmos", "wer"]


# ── Memory management ────────────────────────────────────────────────

def free_memory() -> None:
    """Release Python objects and flush the CUDA/MPS allocator cache.

    Call this between metrics so that large models (e.g. SpeechBERTScore's
    WavLM, UTMOSv2, or the 7B ASR model) do not accumulate in memory and
    trigger the OOM killer when the next model tries to load.
    """
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ── Audio utilities ──────────────────────────────────────────────────

def load_and_resample(path: str, target_sr: int = 16000):
    """Load a waveform and resample to *target_sr* if necessary."""
    import torchaudio

    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze().numpy()


# ── File validation ──────────────────────────────────────────────────

def check_files(
    df: pd.DataFrame,
    ground_truth_dir: str,
    synthesized_dir: str,
    filename_column: str = "filename",
) -> None:
    """Exit with an error if any listed filename is missing from either directory."""
    missing_gt: list[str] = []
    missing_syn: list[str] = []

    for fname in df[filename_column]:
        if not os.path.isfile(os.path.join(ground_truth_dir, fname)):
            missing_gt.append(fname)
        if not os.path.isfile(os.path.join(synthesized_dir, fname)):
            missing_syn.append(fname)

    def _report(label: str, files: list[str]) -> None:
        print(f"ERROR: {len(files)} file(s) missing from {label}:")
        for f in files[:5]:
            print(f"  {f}")
        if len(files) > 5:
            print(f"  … and {len(files) - 5} more")

    if missing_gt:
        _report(f"ground-truth dir ({ground_truth_dir})", missing_gt)
    if missing_syn:
        _report(f"synthesized dir ({synthesized_dir})", missing_syn)

    if missing_gt or missing_syn:
        sys.exit(1)

    print(f"File check passed: all {len(df)} files present in both directories.")


def check_synthesized_files(
    df: pd.DataFrame,
    synthesized_dir: str,
    filename_column: str = "filename",
) -> None:
    """Warn (but do not abort) about synthesized files that cannot be found."""
    missing = [
        fname
        for fname in df[filename_column]
        if not os.path.isfile(os.path.join(synthesized_dir, fname))
    ]
    if missing:
        print(f"WARNING: {len(missing)} synthesized file(s) not found in {synthesized_dir}.")
        for f in missing[:5]:
            print(f"  {f}")
        if len(missing) > 5:
            print(f"  … and {len(missing) - 5} more")


# ── Metric functions ─────────────────────────────────────────────────

def evaluate_mcd(
    df: pd.DataFrame,
    ground_truth_dir: str,
    synthesized_dir: str,
    system_name: str = "tts",
    target_sr: int = 16000,
    output_csv: str | None = None,
    filename_column: str = "filename",
) -> pd.DataFrame:
    """Mel Cepstral Distortion (lower is better, reference-based)."""
    from discrete_speech_metrics import MCD

    df = df.copy()
    mcd_scorer = MCD(sr=target_sr)
    col = f"{system_name}_mcd"

    if col not in df.columns:
        df[col] = np.nan

    remaining = df.index[df[col].isna()].tolist()
    if not remaining:
        print("MCD: all rows already scored — nothing to do.")
        return df
    print(
        f"MCD: scoring {len(remaining)} / {len(df)} rows "
        f"(skipping {len(df) - len(remaining)} already scored)"
    )

    for idx in tqdm(remaining, desc=f"MCD [{system_name}]"):
        fname = df.at[idx, filename_column]
        wav_gt = load_and_resample(os.path.join(ground_truth_dir, fname), target_sr)
        wav_syn = load_and_resample(os.path.join(synthesized_dir, fname), target_sr)
        df.at[idx, col] = mcd_scorer.score(wav_gt, wav_syn)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"MCD results saved to {output_csv}")
    return df


def evaluate_speechbertscore(
    df: pd.DataFrame,
    ground_truth_dir: str,
    synthesized_dir: str,
    system_name: str = "tts",
    target_sr: int = 16000,
    model_type: str = "wavlm-large",
    layer: int | None = None,
    use_gpu: bool = True,
    output_csv: str | None = None,
    filename_column: str = "filename",
) -> pd.DataFrame:
    """SpeechBERTScore precision (higher is better, reference-based)."""
    import torch
    from discrete_speech_metrics import SpeechBERTScore

    df = df.copy()
    scorer = SpeechBERTScore(
        sr=target_sr, model_type=model_type, layer=layer, use_gpu=use_gpu,
    )
    col = f"{system_name}_bertscore_{model_type}"

    if col not in df.columns:
        df[col] = np.nan

    remaining = df.index[df[col].isna()].tolist()
    if not remaining:
        print("SpeechBERTScore: all rows already scored — nothing to do.")
        return df
    print(
        f"SpeechBERTScore: scoring {len(remaining)} / {len(df)} rows "
        f"(skipping {len(df) - len(remaining)} already scored)"
    )

    with torch.inference_mode():
        for idx in tqdm(remaining, desc=f"BERTScore [{system_name}]"):
            fname = df.at[idx, filename_column]
            wav_gt = load_and_resample(os.path.join(ground_truth_dir, fname), target_sr)
            wav_syn = load_and_resample(os.path.join(synthesized_dir, fname), target_sr)
            precision, _, _ = scorer.score(wav_gt, wav_syn)
            df.at[idx, col] = precision

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"SpeechBERTScore results saved to {output_csv}")
    return df


def evaluate_utmos(
    df: pd.DataFrame,
    synthesized_dir: str,
    system_name: str = "tts",
    fold: int = 0,
    num_repetitions: int = 1,
    predict_dataset: str = "sarulab",
    device: str = "cuda:0",
    batch_size: int = 16,
    num_workers: int = 4,
    output_csv: str | None = None,
    filename_column: str = "filename",
) -> pd.DataFrame:
    """UTMOSv2 predicted MOS (higher is better, non-reference, batched)."""
    import utmosv2

    df = df.copy()
    col = f"{system_name}_utmos"

    if col not in df.columns:
        df[col] = np.nan

    remaining = df.index[df[col].isna()].tolist()
    if not remaining:
        print(f"UTMOS [{system_name}]: all rows already scored — nothing to do.")
        return df
    print(
        f"UTMOS [{system_name}]: scoring {len(remaining)} / {len(df)} rows "
        f"(skipping {len(df) - len(remaining)} already scored)"
    )

    model = utmosv2.create_model(pretrained=True, fold=fold)

    remaining_fnames = [
        df.at[idx, filename_column].replace(".wav", "")
        for idx in remaining
    ]

    results = model.predict(
        input_dir=synthesized_dir,
        val_list=remaining_fnames,
        predict_dataset=predict_dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        num_repetitions=num_repetitions,
    )

    mos_by_fname = {
        Path(r["file_path"]).name: r["predicted_mos"]
        for r in results
    }

    for idx in remaining:
        fname = df.at[idx, filename_column]
        if fname in mos_by_fname:
            df.at[idx, col] = mos_by_fname[fname]

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"UTMOS [{system_name}]: results saved to {output_csv}")
    return df


def evaluate_wer(
    df: pd.DataFrame,
    synthesized_dir: str,
    system_name: str = "tts",
    lang: str = "yor_Latn",
    batch_size: int = 8,
    asr_model_card: str = "omniASR_LLM_7B_v2",
    output_csv: str | None = None,
    filename_column: str = "filename",
    text_column: str = "text",
) -> pd.DataFrame:
    """Word Error Rate via omnilingual ASR (lower is better, non-reference, batched)."""
    import jiwer
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    df = df.copy()
    hyp_col = f"{system_name}_asr_hyp"
    wer_col = f"{system_name}_wer"

    for col in (hyp_col, wer_col):
        if col not in df.columns:
            df[col] = np.nan

    remaining = df.index[df[wer_col].isna()].tolist()
    if not remaining:
        print(f"ASR-WER [{system_name}]: all rows already scored — nothing to do.")
        return df
    print(
        f"ASR-WER [{system_name}]: scoring {len(remaining)} / {len(df)} rows "
        f"(skipping {len(df) - len(remaining)} already scored)"
    )

    pipeline = ASRInferencePipeline(model_card=asr_model_card)
    print("ASR pipeline ready.")

    for batch_start in tqdm(
        range(0, len(remaining), batch_size),
        desc=f"ASR-WER [{system_name}]",
    ):
        batch_indices = remaining[batch_start : batch_start + batch_size]
        batch_paths = [
            str(Path(synthesized_dir) / df.at[idx, filename_column])
            for idx in batch_indices
        ]

        missing = [p for p in batch_paths if not os.path.isfile(p)]
        if missing:
            print(f"  Warning: {len(missing)} file(s) not found in this batch, skipping:")
            for m in missing[:3]:
                print(f"    {m}")
            for idx in batch_indices:
                df.at[idx, hyp_col] = ""
                df.at[idx, wer_col] = np.nan
            continue

        hypotheses = pipeline.transcribe(
            batch_paths,
            lang=[lang] * len(batch_paths),
            batch_size=batch_size,
        )

        for idx, hyp in zip(batch_indices, hypotheses):
            ref = df.at[idx, text_column]
            df.at[idx, hyp_col] = hyp
            df.at[idx, wer_col] = jiwer.wer(str(ref), str(hyp))

        if output_csv:
            df.to_csv(output_csv, index=False)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"ASR-WER [{system_name}]: results saved to {output_csv}")
    return df


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TTS model outputs against ground truth audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required named arguments
    parser.add_argument(
        "--ground_truth_dir",
        required=True,
        help="Directory containing ground-truth waveforms.",
    )
    parser.add_argument(
        "--synthesized_dir",
        required=True,
        help="Directory containing synthesized waveforms.",
    )
    parser.add_argument(
        "--metadata_csv",
        required=True,
        help=(
            "CSV (or TSV) file with at minimum a filename column and a text column. "
            "Example: test.csv with columns text, filename, …"
        ),
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to write (or resume) the results CSV.",
    )

    # Metric selection
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=VALID_METRICS,
        default=VALID_METRICS,
        metavar="METRIC",
        help=f"Metrics to compute. Choices: {VALID_METRICS}. Default: all.",
    )

    # General options
    parser.add_argument(
        "--system-name",
        default="tts",
        help="Prefix used for output column names.",
    )
    parser.add_argument(
        "--filename-column",
        default="filename",
        help="Column in METADATA_CSV that holds the .wav filenames.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column in METADATA_CSV that holds the reference transcription (used for WER).",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Target sample rate (Hz) for audio resampling.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device string (e.g. cuda:0, cpu). Auto-detected if omitted.",
    )

    # UTMOS-specific
    utmos_group = parser.add_argument_group("UTMOS options")
    utmos_group.add_argument("--utmos-fold", type=int, default=0, help="UTMOSv2 fold index (0–4).")
    utmos_group.add_argument("--utmos-batch-size", type=int, default=32)
    utmos_group.add_argument("--utmos-num-workers", type=int, default=4)
    utmos_group.add_argument(
        "--utmos-num-repetitions",
        type=int,
        default=1,
        help="TTA repetitions for more stable UTMOS scores.",
    )

    # WER-specific
    wer_group = parser.add_argument_group("WER / ASR options")
    wer_group.add_argument(
        "--asr-lang",
        required=True,
        help="BCP-47 language code for the ASR model (e.g. yor_Latn, ewe_Latn, ibo_Latn).",
    )
    wer_group.add_argument("--asr-batch-size", type=int, default=16)
    wer_group.add_argument(
        "--asr-model-card",
        default="omniASR_LLM_7B_v2",
        help="Model card identifier for omnilingual ASR.",
    )

    # SpeechBERTScore-specific
    bert_group = parser.add_argument_group("SpeechBERTScore options")
    bert_group.add_argument(
        "--bertscore-model-type",
        default="wavlm-large",
        help="Backbone model for SpeechBERTScore.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # Load metadata (or resume from an existing output CSV)
    if os.path.exists(args.output_csv):
        print(f"Resuming from existing results: {args.output_csv}")
        df = pd.read_csv(args.output_csv)
    else:
        sep = "\t" if args.metadata_csv.endswith(".tsv") else ","
        df = pd.read_csv(args.metadata_csv, sep=sep)
        print(f"Loaded {len(df)} rows from {args.metadata_csv}")

    # Auto-detect device
    device: str = args.device or ""
    if not device:
        import torch

        if torch.cuda.is_available():
            device = "cuda:0"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    # Validate file presence
    reference_metrics = {"mcd", "speechbertscore"}
    if any(m in reference_metrics for m in args.metrics):
        check_files(df, args.ground_truth_dir, args.synthesized_dir, args.filename_column)
    else:
        check_synthesized_files(df, args.synthesized_dir, args.filename_column)

    use_gpu = device.startswith("cuda") or device == "mps"

    # Run selected metrics in a fixed order (cheapest first).
    # free_memory() is called between metrics to release GPU/CPU allocations
    # from the previous model before the next (potentially larger) one loads.
    if "mcd" in args.metrics:
        print(f"\n{'='*60}\n  Metric: MCD\n{'='*60}")
        df = evaluate_mcd(
            df=df,
            ground_truth_dir=args.ground_truth_dir,
            synthesized_dir=args.synthesized_dir,
            system_name=args.system_name,
            target_sr=args.target_sr,
            output_csv=args.output_csv,
            filename_column=args.filename_column,
        )
        free_memory()

    if "speechbertscore" in args.metrics:
        print(f"\n{'='*60}\n  Metric: SpeechBERTScore ({args.bertscore_model_type})\n{'='*60}")
        df = evaluate_speechbertscore(
            df=df,
            ground_truth_dir=args.ground_truth_dir,
            synthesized_dir=args.synthesized_dir,
            system_name=args.system_name,
            target_sr=args.target_sr,
            model_type=args.bertscore_model_type,
            use_gpu=use_gpu,
            output_csv=args.output_csv,
            filename_column=args.filename_column,
        )
        free_memory()

    if "utmos" in args.metrics:
        print(f"\n{'='*60}\n  Metric: UTMOS\n{'='*60}")
        df = evaluate_utmos(
            df=df,
            synthesized_dir=args.synthesized_dir,
            system_name=args.system_name,
            fold=args.utmos_fold,
            num_repetitions=args.utmos_num_repetitions,
            predict_dataset="sarulab",
            device=device,
            batch_size=args.utmos_batch_size,
            num_workers=args.utmos_num_workers,
            output_csv=args.output_csv,
            filename_column=args.filename_column,
        )
        free_memory()

    if "wer" in args.metrics:
        print(f"\n{'='*60}\n  Metric: WER (lang={args.asr_lang})\n{'='*60}")
        df = evaluate_wer(
            df=df,
            synthesized_dir=args.synthesized_dir,
            system_name=args.system_name,
            lang=args.asr_lang,
            batch_size=args.asr_batch_size,
            asr_model_card=args.asr_model_card,
            output_csv=args.output_csv,
            filename_column=args.filename_column,
            text_column=args.text_column,
        )

    # Final save
    df.to_csv(args.output_csv, index=False)
    print(f"\nDone. Final results saved to: {args.output_csv}")

    # Summary statistics
    prefix = args.system_name
    metric_cols = [
        c for c in df.columns
        if c in (f"{prefix}_mcd", f"{prefix}_utmos", f"{prefix}_wer")
        or c.startswith(f"{prefix}_bertscore_")
    ]
    if metric_cols:
        print("\n── Summary ──")
        print(df[metric_cols].describe().loc[["count", "mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
