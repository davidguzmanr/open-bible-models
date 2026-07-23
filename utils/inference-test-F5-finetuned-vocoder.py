"""
Batch TTS inference over a HuggingFace dataset split for a given language/checkpoint,
using a fine-tuned Vocos vocoder loaded from a PyTorch-Lightning checkpoint.

Example usage:
    python inference-test-F5-finetuned-vocoder.py \
        --language Yoruba \
        --output_dir synthesis_output/yoruba-finetuned-vocoder \
        --ckpt_path ckpts/F5TTS_v1_Base_vocos_custom_open-bible-yoruba/model_last.pt \
        --vocab_file data/open-bible-yoruba_custom/vocab.txt \
        --model_cfg src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Yoruba.yaml \
        --metadata_path data/open-bible-yoruba/metadata.csv \
        --vocoder_ckpt vocos/logs/lightning_logs/version_0/checkpoints/last.ckpt \
        --vocoder_cfg  vocos/logs/lightning_logs/version_0/config.yaml
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import yaml
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


def load_finetuned_vocoder(vocoder_ckpt: str, vocoder_cfg: str, device: str):
    """Load a fine-tuned Vocos model from a PyTorch-Lightning checkpoint.

    The Lightning checkpoint stores the full training state (generator +
    discriminators). Only the generator weights (feature_extractor, backbone,
    head) are extracted and loaded into a plain ``Vocos`` inference object,
    which exposes the same ``.decode(mel)`` interface expected by F5-TTS.

    Args:
        vocoder_ckpt: Path to the Lightning checkpoint (.ckpt).
        vocoder_cfg:  Path to the training config.yaml saved alongside the
                      checkpoint (e.g. version_0/config.yaml).
        device:       Torch device string.

    Returns:
        A ``vocos.Vocos`` instance in eval mode on ``device``.
    """
    sys.path.insert(0, "vocos")
    from vocos import Vocos
    from vocos.pretrained import instantiate_class

    with open(vocoder_cfg) as f:
        train_cfg = yaml.safe_load(f)

    model_args = train_cfg["model"]["init_args"]
    feature_extractor = instantiate_class(args=(), init=model_args["feature_extractor"])
    backbone          = instantiate_class(args=(), init=model_args["backbone"])
    head              = instantiate_class(args=(), init=model_args["head"])
    vocoder = Vocos(feature_extractor=feature_extractor, backbone=backbone, head=head)

    ckpt = torch.load(vocoder_ckpt, map_location="cpu", weights_only=False)
    gen_prefixes = {"feature_extractor", "backbone", "head"}
    state_dict = {
        k: v for k, v in ckpt["state_dict"].items()
        if k.split(".")[0] in gen_prefixes
    }
    vocoder.load_state_dict(state_dict, strict=True)
    vocoder.eval().to(device)
    print(f"Fine-tuned vocoder loaded from {vocoder_ckpt}  ({len(state_dict)} tensors)")
    return vocoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch TTS inference over a local test CSV using a fine-tuned Vocos vocoder."
    )
    parser.add_argument(
        "--language",
        required=True,
        help="Language name as used in the HuggingFace dataset (e.g. 'Yoruba').",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where generated WAVs and logs are saved.",
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to the F5-TTS model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--vocab_file",
        required=True,
        help="Path to the vocabulary file (vocab.txt).",
    )
    parser.add_argument(
        "--model_cfg",
        required=True,
        help="Path to the F5-TTS model config YAML file.",
    )
    parser.add_argument(
        "--metadata_path",
        required=True,
        help="Path to the training metadata CSV (pipe-separated) used to pick "
             "the reference audio.",
    )
    parser.add_argument(
        "--vocoder_ckpt",
        required=True,
        help="Path to the fine-tuned Vocos Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--vocoder_cfg",
        default=None,
        help="Path to the training config.yaml for the fine-tuned vocoder. "
             "Defaults to config.yaml two directories above the checkpoint "
             "(i.e. <version_dir>/config.yaml).",
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
    VOCODER_CKPT  = args.vocoder_ckpt

    # Derive the vocoder config path from the checkpoint location if not given:
    # checkpoints/last.ckpt  →  ../config.yaml  (version_N/config.yaml)
    if args.vocoder_cfg is not None:
        VOCODER_CFG = args.vocoder_cfg
    else:
        VOCODER_CFG = str(Path(VOCODER_CKPT).parent.parent / "config.yaml")
        print(f"--vocoder_cfg not set, using: {VOCODER_CFG}")

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

    speaker_totals = train.groupby("speaker_id")["duration_seconds"].sum()
    best_speaker   = speaker_totals.idxmax()
    print(f"Speaker with most audio: {best_speaker} "
          f"({speaker_totals[best_speaker]:.1f}s total)")

    speaker_rows = train[train["speaker_id"] == best_speaker]
    candidates   = speaker_rows[
        speaker_rows["duration_seconds"].between(6, 10)
    ]
    if candidates.empty:
        raise ValueError(
            f"No clips between 6–10s found for speaker {best_speaker}."
        )
    ref_row   = candidates.loc[candidates["duration_seconds"].idxmin()]
    REF_AUDIO = ref_row["audio_file"]
    REF_TEXT  = ensure_punctuation(ref_row["text"])

    print(f"Reference audio : {REF_AUDIO}")
    print(f"Reference text  : {REF_TEXT}")
    print(f"Reference dur   : {ref_row['duration_seconds']:.2f}s")

    # ── Load F5 model & fine-tuned vocoder ─────────────────────────────────────
    sys.path.insert(0, "F5-TTS/src")

    from hydra.utils import get_class
    from omegaconf import OmegaConf

    from f5_tts.infer.utils_infer import (
        infer_process,
        load_model,
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

    vocoder = load_finetuned_vocoder(VOCODER_CKPT, VOCODER_CFG, device)

    ema_model = load_model(
        model_cls,
        model_arc,
        CKPT_PATH,
        mel_spec_type="vocos",
        vocab_file=VOCAB_FILE,
        use_ema=True,
        device=device,
    )
    print("F5 model loaded ✓")

    # ── Preprocess reference audio ─────────────────────────────────────────────
    ref_audio, ref_text = preprocess_ref_audio_text(REF_AUDIO, REF_TEXT)
    ref_audio = append_silence(ref_audio, duration_s=1.0)
    print(f"Preprocessed ref_audio (+ 1 s silence): {ref_audio}")
    print(f"Preprocessed ref_text                 : {ref_text}")

    # ── Batch inference ────────────────────────────────────────────────────────
    n = min(500, len(ds)) if args.head is None else min(args.head, len(ds))
    subset = ds.select(range(n))
    print(f"Synthesizing {n} samples")

    # ── Save metadata CSV ──────────────────────────────────────────────────────
    csv_path = Path(OUTPUT_DIR) / "test.csv"
    df_meta = subset.to_pandas().drop(columns=["audio"])
    df_meta["filename"] = (
        df_meta["testament"].astype(str) + "-" +
        df_meta["book"].astype(str) + "-" +
        df_meta["chapter"].astype(str) + "-" +
        df_meta["verse"].astype(str) + ".wav"
    )
    df_meta.to_csv(csv_path, index=False)
    print(f"Metadata CSV saved to: {csv_path}")

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
