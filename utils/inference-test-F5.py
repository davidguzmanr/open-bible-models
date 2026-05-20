"""
Batch TTS inference over a HuggingFace dataset split for a given language/checkpoint.

Example usage:
    python inference-F5-open-bible.py \
        --language Igbo \
        --output_dir synthesis_output/igbo \
        --ckpt_path ckpts/F5TTS_v1_Base_vocos_custom_open-bible-igbo/model_last.pt \
        --vocab_file data/open-bible-igbo_custom/vocab.txt \
        --model_cfg src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Igbo.yaml \
        --metadata_path data/open-bible-igbo/metadata.csv
"""

import argparse
import os
import tempfile

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset
from tqdm import tqdm


# ── Fix: torchaudio 2.10 defaults to torchcodec which needs FFmpeg libs ───────
def _torchaudio_load_sf(filepath, *args, **kwargs):
    data, samplerate = sf.read(filepath, dtype="float32")
    tensor = torch.from_numpy(data.T if data.ndim > 1 else data[None, :])
    return tensor, samplerate

torchaudio.load = _torchaudio_load_sf
print("Patched torchaudio.load → soundfile backend")


PUNCTUATION = set(".。!！?？,，;；:")


def ensure_punctuation(text: str) -> str:
    """Ensure the text ends with punctuation followed by a space.

    F5-TTS requires a trailing space after sentence-ending punctuation so the
    chunker recognises the sentence boundary correctly (see infer/README.md).
    """
    text = text.strip()
    if not text:
        return text
    if text[-1] not in PUNCTUATION:
        text = text + ". "
    elif not text.endswith(" "):
        text = text + " "
    return text


def append_silence(audio_path: str, duration_s: float = 1.0) -> str:
    """Write a new temp WAV with `duration_s` seconds of silence appended."""
    data, sr = sf.read(audio_path, dtype="float32")
    silence_samples = int(sr * duration_s)
    if data.ndim == 1:
        silence = np.zeros(silence_samples, dtype=data.dtype)
    else:
        silence = np.zeros((silence_samples, data.shape[1]), dtype=data.dtype)
    padded = np.concatenate([data, silence], axis=0)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, padded, sr)
    tmp.close()
    return tmp.name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch TTS inference over a local test CSV."
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
        "--head",
        type=int,
        default=None,
        metavar="N",
        help="Only synthesize the first N samples (useful for quick tests).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    LANGUAGE      = args.language
    OUTPUT_DIR    = args.output_dir
    CKPT_PATH     = args.ckpt_path
    VOCAB_FILE    = args.vocab_file
    MODEL_CFG     = args.model_cfg
    METADATA_PATH = args.metadata_path

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load test set ──────────────────────────────────────────────────────────
    print(f"Loading test set for language: {LANGUAGE}")
    ds = load_dataset(
        "parquet",
        data_files={
            "test": f"hf://datasets/davidguzmanr/open-bible-resources/{LANGUAGE}/test-*.parquet"
        },
        split="test",
    )
    print(f"Test samples (total): {len(ds)}")

    # ── Pick a reference audio from the training set ───────────────────────────
    print(f"Loading training metadata from: {METADATA_PATH}")
    train = pd.read_csv(METADATA_PATH, sep="|")
    train["duration_seconds"] = train["audio_file"].apply(
        lambda path: sf.info(path).duration
    )

    # Find the speaker with the most total audio
    speaker_totals = train.groupby("speaker_id")["duration_seconds"].sum()
    best_speaker   = speaker_totals.idxmax()
    print(f"Speaker with most audio: {best_speaker} "
          f"({speaker_totals[best_speaker]:.1f}s total)")

    # From that speaker, pick the shortest clip between 6 and 10 seconds
    speaker_rows = train[train["speaker_id"] == best_speaker]
    candidates   = speaker_rows[
        speaker_rows["duration_seconds"].between(6, 10)
    ]
    if candidates.empty:
        raise ValueError(
            f"No clips between 6–10s found for speaker {best_speaker}."
        )
    ref_row = candidates.loc[candidates["duration_seconds"].idxmin()]
    REF_AUDIO    = ref_row["audio_file"]
    REF_TEXT     = ensure_punctuation(ref_row["text"])

    print(f"Reference audio : {REF_AUDIO}")
    print(f"Reference text  : {REF_TEXT}")
    print(f"Reference dur   : {ref_row['duration_seconds']:.2f}s")

    print(f"Reference audio (original): {REF_AUDIO}")

    # ── Load model & vocoder ───────────────────────────────────────────────────
    from hydra.utils import get_class
    from omegaconf import OmegaConf

    from f5_tts.infer.utils_infer import (
        infer_process,
        load_model,
        load_vocoder,
        preprocess_ref_audio_text,
    )

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model_cfg = OmegaConf.load(MODEL_CFG)
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

    ema_model = load_model(
        model_cls,
        model_arc,
        CKPT_PATH,
        mel_spec_type="vocos",
        vocab_file=VOCAB_FILE,
        use_ema=True,
        device=device,
    )
    print("Model loaded ✓")

    # ── Preprocess reference audio ─────────────────────────────────────────────
    # preprocess_ref_audio_text internally calls remove_silence_edges(), which
    # strips trailing silence, then adds only 50 ms back.  We therefore append
    # our 1-second boundary silence AFTER preprocessing so it is preserved.
    ref_audio, ref_text = preprocess_ref_audio_text(REF_AUDIO, REF_TEXT)
    ref_audio = append_silence(ref_audio, duration_s=1.0)
    print(f"Preprocessed ref_audio (+ 1 s silence): {ref_audio}")
    print(f"Preprocessed ref_text                 : {ref_text}")

    # ── Batch inference ────────────────────────────────────────────────────────
    n = min(500, len(ds)) if args.head is None else min(args.head, len(ds))
    subset = ds.select(range(n))
    print(f"Synthesizing {n} samples")

    generated_files = []

    for row in tqdm(subset, total=n, desc="Synthesizing"):
        gen_text     = ensure_punctuation(row["text"])
        out_filename = f"{row['testament']}-{row['book']}-{row['chapter']}-{row['verse']}.wav"
        out_path     = os.path.join(OUTPUT_DIR, out_filename)

        if os.path.exists(out_path):
            generated_files.append(out_path)
            continue

        audio_segment, final_sample_rate, _ = infer_process(
            ref_audio,
            ref_text,
            gen_text,
            ema_model,
            vocoder,
            mel_spec_type="vocos",
            device=device,
        )
        sf.write(out_path, audio_segment, final_sample_rate)
        generated_files.append(out_path)

    print(f"\nDone! {len(generated_files)} files generated.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
