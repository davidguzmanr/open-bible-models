"""
EveryVoice TTS inference on an Open-Bible test split for multilingual,
multispeaker checkpoints.

Same pipeline as inference-test-split.py, but each synthesis row uses explicit
``language`` and ``speaker`` ids that must exist in the trained FastSpeech2
checkpoint (``model.lang2id`` / ``model.speaker2id``).

Note: models trained with ``multilingual: false`` in the EveryVoice config use
``und`` as their single language tag (or may have an empty lang2id altogether).
Language information is then encoded exclusively through speaker IDs.

Usage (Yoruba test split, Igbo-Ewe-Yoruba-NT checkpoint):
    python inference-test-split-multilingual.py \\
        --language Yoruba \\
        --ev-language und \\
        --speaker SPEAKER_00_Yoruba \\
        --ckpt_path Igbo-Ewe-Yoruba-NT/logs_and_checkpoints/FeaturePredictionExperiment/base/checkpoints/last.ckpt \\
        --vocoder_ckpt_path /path/to/hifigan_universal_v1_everyvoice.ckpt \\
        --output_dir synthesis_output/yoruba-nt-speaker00

If you are unsure which ids the checkpoint expects, run:
    python inference-test-split-multilingual.py \\
        --list-model-ids \\
        --ckpt_path path/to/last.ckpt
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, load_dataset
from tqdm import tqdm

from everyvoice.config.type_definitions import DatasetTextRepresentation
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.cli.synthesize import (
    get_global_step,
    synthesize_helper,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.model import FastSpeech2
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.prediction_writing_callback import (
    PredictionWritingWavCallback,
)
from everyvoice.model.feature_prediction.FastSpeech2_lightning.fs2.type_definitions import (
    SynthesizeOutputFormats,
)
from everyvoice.model.vocoder.HiFiGAN_iSTFT_lightning.hfgl.utils import (
    load_hifigan_from_checkpoint,
)
from everyvoice.utils.heavy import get_device_from_accelerator


def _patched_wav_on_predict_batch_end(
    self,
    _trainer,
    _pl_module,
    outputs,
    batch,
    _batch_idx,
    _dataloader_idx=0,
):
    """Replacement for PredictionWritingWavCallback.on_predict_batch_end that
    uses batch["basename"] for the output filename instead of slugifying the
    text.  The default callback derives the filename from the text content,
    which causes collisions when multiple utterances share a common prefix.
    Applying this patch before calling synthesize_helper ensures every
    generated WAV is named after our desired testament-book-chapter-verse
    basename."""
    wavs, sr = self.synthesize_audio(outputs)
    assert "tgt_lens" in outputs and outputs["tgt_lens"] is not None

    basenames = batch["basename"]
    speakers = batch["speaker"]
    languages = batch["language"]
    is_last_input_chunk = batch["is_last_input_chunk"]
    texts = batch["raw_text"]
    unmasked_lens = list(outputs["tgt_lens"])

    for i, wav in enumerate(wavs):
        trimmed_wav = wav[:, : (unmasked_lens[i] * self.output_hop_size)]
        self.full_wav = torch.cat((self.full_wav, trimmed_wav), -1)
        self.full_text += texts[i]

        if is_last_input_chunk[i]:
            filename = self.get_filename(basenames[i], speakers[i], languages[i])
            torchaudio.save(
                filename,
                self.full_wav,
                sr,
                format="wav",
                encoding="PCM_S",
                bits_per_sample=16,
            )
            self.full_wav = torch.tensor(())
            self.full_text = ""
            self.last_file_written = filename


# Apply patch at import time so it is in effect when synthesize_helper
# instantiates the callback internally.
PredictionWritingWavCallback.on_predict_batch_end = _patched_wav_on_predict_batch_end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "EveryVoice TTS on the Open-Bible test split with explicit "
            "multilingual language id and multispeaker id."
        )
    )
    parser.add_argument(
        "--list-model-ids",
        action="store_true",
        help="Load the feature-prediction checkpoint and print language/speaker "
        "ids, then exit (no synthesis).",
    )
    parser.add_argument(
        "--language",
        help=(
            "Language name for the HuggingFace Open-Bible parquet "
            "(e.g. 'Yoruba'). Not used with --list-model-ids."
        ),
    )
    parser.add_argument(
        "--ev-language",
        dest="ev_language",
        default=None,
        help=(
            "Language id stored in the checkpoint (training filelist ``language`` "
            "column). If omitted, uses the first key in model.lang2id."
        ),
    )
    parser.add_argument(
        "--speaker",
        default=None,
        help=(
            "Speaker id stored in the checkpoint (training filelist ``speaker`` "
            "column). If omitted, uses the first key in model.speaker2id."
        ),
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="Path to the EveryVoice feature-prediction checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--vocoder_ckpt_path",
        help="Path to the HiFi-GAN vocoder checkpoint (.ckpt). Not used with --list-model-ids.",
    )
    parser.add_argument(
        "--output_dir",
        help="Root directory for ground-truth WAVs, generated WAVs, and test.csv.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        metavar="N",
        help="Only synthesize the first N samples (useful for quick tests). "
        "Default: up to 500.",
    )
    args = parser.parse_args()

    if args.list_model_ids:
        return args

    if not args.language:
        parser.error("--language is required unless --list-model-ids is set.")
    if not args.vocoder_ckpt_path:
        parser.error("--vocoder_ckpt_path is required unless --list-model-ids is set.")
    if not args.output_dir:
        parser.error("--output_dir is required unless --list-model-ids is set.")

    return args


def list_checkpoint_language_speaker_ids(ckpt_path: Path) -> None:
    print(f"Loading checkpoint (CPU, ids only): {ckpt_path}")
    model = FastSpeech2.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model.eval()
    langs = sorted(model.lang2id.keys())
    speakers = sorted(model.speaker2id.keys())
    if langs:
        print("\nLanguages (use one as --ev-language):")
        for k in langs:
            print(f"  {k!r}")
    else:
        print(
            "\nLanguages: (none — model trained with multilingual=false; "
            "omit --ev-language or use the language column from the training filelist)"
        )
    print("\nSpeakers (use one as --speaker):")
    for k in speakers:
        print(f"  {k!r}")
    print(f"\nTotal: {len(langs)} language(s), {len(speakers)} speaker(s).")


def resolve_language_speaker(
    model: FastSpeech2,
    ev_language: str | None,
    speaker: str | None,
) -> tuple[str | None, str]:
    lang_keys = list(model.lang2id.keys())
    spk_keys = list(model.speaker2id.keys())

    if not spk_keys:
        raise RuntimeError("Checkpoint has empty speaker2id. Cannot perform synthesis.")

    # When multilingual=false, lang2id may be empty; language conditioning is unused.
    if not lang_keys:
        if ev_language is not None:
            print(
                f"Warning: --ev-language {ev_language!r} was specified but "
                f"model.lang2id is empty (model trained with multilingual=false). "
                f"Proceeding with language=None."
            )
        lang = None
    else:
        lang = ev_language if ev_language is not None else lang_keys[0]
        if lang not in model.lang2id:
            raise ValueError(
                f"Unknown --ev-language {lang!r}. "
                f"Valid languages: {sorted(model.lang2id)}"
            )

    spk = speaker if speaker is not None else spk_keys[0]
    if spk not in model.speaker2id:
        raise ValueError(
            f"Unknown --speaker {spk!r}. Valid speakers: {sorted(model.speaker2id)}"
        )
    return lang, spk


def main() -> None:
    args = parse_args()

    if args.list_model_ids:
        list_checkpoint_language_speaker_ids(Path(args.ckpt_path))
        return

    device = get_device_from_accelerator("gpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_dir = output_dir / "ground-truth"
    ground_truth_dir.mkdir(parents=True, exist_ok=True)

    everyvoice_wav_dir = output_dir / "wav"
    wav_dir = output_dir / "generated"

    test_csv_path = output_dir / "test.csv"

    print(f"Loading test split for HuggingFace language: {args.language}")
    ds = load_dataset(
        "parquet",
        data_files={
            "test": f"hf://datasets/davidguzmanr/open-bible-resources/{args.language}/test-*.parquet"
        },
        split="test",
    )
    ds = ds.cast_column("audio", Audio(decode=False))

    n = min(500, len(ds)) if args.head is None else min(args.head, len(ds))
    ds = ds.select(range(n))
    print(f"Test samples (total): {len(ds)}, synthesizing: {n}")

    # Build basenames the same way as the monolingual script: testament-book-chapter-verse
    test_df = ds.remove_columns("audio").to_pandas()
    test_df["basename"] = (
        test_df["testament"].astype(str) + "-" +
        test_df["book"].astype(str) + "-" +
        test_df["chapter"].astype(str) + "-" +
        test_df["verse"].astype(str)
    )
    test_df["filename"] = test_df["basename"] + ".wav"

    test_df.to_csv(test_csv_path, index=False)
    print(f"Saved test dataframe to: {test_csv_path}")

    print("Saving ground-truth WAV files…")
    for example, basename in tqdm(
        zip(ds, test_df["basename"]), total=n, desc="Saving ground-truth WAVs"
    ):
        out_path = ground_truth_dir / f"{basename}.wav"
        with io.BytesIO(example["audio"]["bytes"]) as buf:
            audio_array, sample_rate = sf.read(buf)
        sf.write(str(out_path), audio_array, sample_rate)

    print(f"Saved {n} WAV files to {ground_truth_dir}")
    print(f"Total sentences to synthesise: {len(test_df)}")

    print(f"Using device: {device}")

    feature_prediction_checkpoint = Path(args.ckpt_path)
    vocoder_base_checkpoint = Path(args.vocoder_ckpt_path)

    print("Loading feature prediction model…")
    model = FastSpeech2.load_from_checkpoint(str(feature_prediction_checkpoint)).to(device)
    model.eval()
    global_step = get_global_step(feature_prediction_checkpoint)

    synth_language, synth_speaker = resolve_language_speaker(
        model, args.ev_language, args.speaker
    )
    print(
        f"Synthesis conditioning: ev-language={synth_language!r}, "
        f"speaker={synth_speaker!r}"
    )

    print("Loading vocoder…")
    vocoder_ckpt = torch.load(
        str(vocoder_base_checkpoint), map_location=device, weights_only=True
    )
    vocoder_model, vocoder_config = load_hifigan_from_checkpoint(vocoder_ckpt, device)
    vocoder_global_step = get_global_step(vocoder_base_checkpoint)

    print("Models loaded successfully!")

    filelist_data = [
        {
            "basename": row["basename"],
            "characters": row["text"],
            "language": synth_language,
            "speaker": synth_speaker,
            "duration_control": 1.0,
            # Required by the dataset __getitem__ during inference; each entry is a
            # single (non-chunked) utterance, so it is always the last chunk.
            "is_last_input_chunk": True,
        }
        for _, row in test_df.iterrows()
    ]
    print(f"Prepared {len(filelist_data)} entries for synthesis")

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

    missing = []
    for _, row in test_df.iterrows():
        expected_path = wav_dir / f"{row['basename']}.wav"
        if not expected_path.exists():
            missing.append(row["basename"])

    if missing:
        print(f"\nMissing {len(missing)} generated files:", file=sys.stderr)
        for f in missing[:10]:
            print(f"  - {f}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  … and {len(missing) - 10} more", file=sys.stderr)
    else:
        print(f"\nAll {len(test_df)} expected WAV files are present in {wav_dir}")

    print("\nDone.")
    print(f"  Ground-truth WAVs : {ground_truth_dir}")
    print(f"  Generated WAVs    : {wav_dir}")


if __name__ == "__main__":
    main()
