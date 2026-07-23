#!/usr/bin/env python3
"""
Generate teacher-forced Mel spectrograms from a trained F5-TTS model for
vocoder matching (fine-tuning Vocos on the model's own output distribution).

This is the F5 analogue of EveryVoice's ``everyvoice synthesize ... --output-type
spec --teacher-forcing-directory`` step. Because F5 is a flow-matching model with
no explicit duration predictor, we cannot force per-token durations the way
FastSpeech2 does. Instead we use **SDEdit-style reconstruction**: we seed the
flow ODE with the *noised* ground-truth Mel at time ``strength`` and integrate to
``t=1``. Because the seed carries the ground-truth Mel's per-frame content, the
output is exactly length- AND frame-aligned to the ground-truth audio, while
still passing through F5's learned vector field (so it acquires F5's
characteristic output distribution). ``strength`` trades off fidelity to the
ground truth (higher) versus more F5 character (lower).

For each utterance we write ``<mel_dir>/<idx>.pt`` holding a ``[n_mels, T]`` log-Mel
tensor, and emit two filelists (``train_filelist.txt`` / ``val_filelist.txt``)
whose lines are ``<mel_path>|<audio_path>``. The paired ground-truth ``audio_path``
is what Vocos reconstructs against during fine-tuning (see ``vocos.finetune``).

Example:
    python src/f5_tts/train/generate_vocoder_mels.py \
        --config-name F5TTS_v1_Base_Open_Bible_Yoruba \
        --ckpt ckpts/F5TTS_v1_Base_vocos_custom_open-bible-yoruba/model_last.pt \
        --output-dir data/open-bible-yoruba_custom/vocoder_matching \
        --strength 0.5 --steps 16 --val-size 256
"""

import argparse
import sys
from importlib.resources import files
from pathlib import Path

import torch
import torchaudio
from datasets import Dataset as HFDataset_
from datasets import load_from_disk
from hydra.utils import get_class
from omegaconf import OmegaConf
from torchdiffeq import odeint
from tqdm import tqdm

from f5_tts.infer.utils_infer import load_model
from f5_tts.model.utils import list_str_to_idx


# Deterministic peak normalization to a fixed dBFS. Applied identically here (before
# computing the ground-truth Mel that seeds the reconstruction) and in the Vocos
# fine-tune dataset (to the target waveform), so the generated Mel and the Vocos
# reconstruction target stay at a consistent level. Implemented without sox so it
# behaves the same across the F5-TTS and Vocos conda environments.
NORM_DBFS = -3.0


def peak_normalize(audio: torch.Tensor, dbfs: float = NORM_DBFS) -> torch.Tensor:
    peak = audio.abs().max()
    if peak < 1e-8:
        return audio
    target = 10.0 ** (dbfs / 20.0)
    return audio * (target / peak)


def load_config(config_name: str) -> OmegaConf:
    # Accept either a bare name or a path; resolve against f5_tts/configs by default.
    p = Path(config_name)
    if not p.exists():
        p = Path(str(files("f5_tts").joinpath(f"configs/{config_name}.yaml")))
    if not p.exists():
        sys.exit(f"ERROR: config not found: {config_name}")
    return OmegaConf.load(p)


def resolve_data_dir(cfg: OmegaConf) -> Path:
    dataset_name = cfg.datasets.name
    tokenizer = cfg.model.tokenizer
    return Path(str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}")))


def load_raw_dataset(data_dir: Path) -> HFDataset_:
    try:
        return load_from_disk(str(data_dir / "raw"))
    except Exception:  # noqa: BLE001 - fall back to the loose arrow file
        return HFDataset_.from_file(str(data_dir / "raw.arrow"))


@torch.no_grad()
def sdedit_reconstruct(
    model,
    audio: torch.Tensor,
    text: str,
    vocab_char_map: dict,
    strength: float,
    steps: int,
    device: str,
) -> torch.Tensor:
    """Return an SDEdit reconstruction of ``audio``'s Mel, shape ``[n_mels, T]``.

    ``audio`` is mono ``[1, num_samples]`` at the model sampling rate, already
    level-normalized. The output is frame-aligned with ``audio`` by construction.
    """
    # Ground-truth Mel, x1 in flow-matching notation: [1, n_mels, T] -> [1, T, n_mels]
    x1 = model.mel_spec(audio).permute(0, 2, 1).to(device)

    text_ids = list_str_to_idx([text], vocab_char_map).to(device)  # [1, nt], -1 padded

    # SDEdit seed: phi_{t0} = (1 - t0) * noise + t0 * x1
    t0 = float(strength)
    x_init = (1.0 - t0) * torch.randn_like(x1) + t0 * x1

    # No reference audio: the ground-truth information enters only through the seed,
    # so the model must denoise toward a Mel consistent with both the text and the
    # (leaked) ground truth -> stays frame-aligned. Text conditioning stays on.
    cond = torch.zeros_like(x1)

    def fn(t, x):
        return model.transformer(
            x=x,
            cond=cond,
            text=text_ids,
            time=t,
            mask=None,
            drop_audio_cond=True,
            drop_text=False,
            cache=False,
        )

    t = torch.linspace(t0, 1.0, steps + 1, device=device, dtype=x1.dtype)
    trajectory = odeint(fn, x_init, t, **model.odeint_kwargs)
    x_gen = trajectory[-1]  # [1, T, n_mels]

    return x_gen.squeeze(0).permute(1, 0).contiguous().cpu()  # [n_mels, T]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", required=True, help="F5 training config (name under f5_tts/configs or a path)")
    parser.add_argument("--ckpt", required=True, help="Path to the trained F5 checkpoint (.pt)")
    parser.add_argument("--output-dir", required=True, help="Where to write generated Mels and filelists")
    parser.add_argument("--strength", type=float, default=0.5, help="SDEdit seed time t0 in (0,1); lower = more F5 character, less GT fidelity (default: 0.5)")
    parser.add_argument("--steps", type=int, default=16, help="ODE integration steps over [strength, 1] (default: 16)")
    parser.add_argument("--val-size", type=int, default=256, help="Number of utterances reserved for the validation filelist (default: 256)")
    parser.add_argument("--max-utterances", type=int, default=0, help="Cap on utterances processed, for quick tests (0 = all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config_name)
    data_dir = resolve_data_dir(cfg)
    vocab_file = cfg.model.tokenizer_path
    model_cls = get_class(f"f5_tts.model.{cfg.model.backbone}")
    model_arch = cfg.model.arch
    mel_spec_type = cfg.model.mel_spec.mel_spec_type

    print(f"Config:     {args.config_name}")
    print(f"Data dir:   {data_dir}")
    print(f"Vocab:      {vocab_file}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"SDEdit strength (t0): {args.strength}   ODE steps: {args.steps}")

    model = load_model(
        model_cls,
        OmegaConf.to_container(model_arch, resolve=True),
        args.ckpt,
        mel_spec_type=mel_spec_type,
        vocab_file=vocab_file,
        device=args.device,
    )
    # load_model casts to fp16 on capable GPUs; force fp32 for this offline pass so the
    # fp32 SDEdit inputs (and the Mel STFT) match the model dtype.
    model = model.float()
    model.eval()
    vocab_char_map = model.vocab_char_map

    dataset = load_raw_dataset(data_dir)
    n_total = len(dataset)
    if args.max_utterances > 0:
        n_total = min(n_total, args.max_utterances)
    print(f"Utterances: {n_total}")

    out_dir = Path(args.output_dir)
    mel_dir = out_dir / "synthesized_spec"
    mel_dir.mkdir(parents=True, exist_ok=True)

    target_sr = int(cfg.model.mel_spec.target_sample_rate)
    resamplers: dict[int, torchaudio.transforms.Resample] = {}
    pairs: list[tuple[str, str]] = []

    for idx in tqdm(range(n_total), desc="Generating teacher-forced Mels"):
        row = dataset[idx]
        audio_path = row["audio_path"]
        text = row["text"]

        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != target_sr:
            if sr not in resamplers:
                resamplers[sr] = torchaudio.transforms.Resample(sr, target_sr)
            audio = resamplers[sr](audio)
        audio = peak_normalize(audio).to(args.device)

        mel = sdedit_reconstruct(
            model, audio, text, vocab_char_map, args.strength, args.steps, args.device
        )  # [n_mels, T]

        mel_path = mel_dir / f"{idx:07d}.pt"
        torch.save(mel, mel_path)
        pairs.append((str(mel_path.resolve()), str(Path(audio_path).resolve())))

    # Last `val_size` utterances become the validation split (deterministic).
    val_size = min(args.val_size, max(0, len(pairs) - 1))
    train_pairs = pairs[: len(pairs) - val_size] if val_size else pairs
    val_pairs = pairs[len(pairs) - val_size :] if val_size else pairs[:1]

    def write_filelist(path: Path, rows: list[tuple[str, str]]):
        with open(path, "w") as f:
            for mel_path, audio_path in rows:
                f.write(f"{mel_path}|{audio_path}\n")

    train_fl = out_dir / "train_filelist.txt"
    val_fl = out_dir / "val_filelist.txt"
    write_filelist(train_fl, train_pairs)
    write_filelist(val_fl, val_pairs)

    print(f"\nWrote {len(pairs)} Mels to {mel_dir}")
    print(f"  train filelist: {train_fl} ({len(train_pairs)} lines)")
    print(f"  val filelist:   {val_fl} ({len(val_pairs)} lines)")


if __name__ == "__main__":
    main()
