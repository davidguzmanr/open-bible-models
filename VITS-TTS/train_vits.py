"""
Multilingual Multispeaker VITS Training Script
===============================================

Train a VITS model on any language dataset using Coqui TTS.
Vocabulary is built automatically from the metadata file.
Supports multiple speakers via learned speaker embeddings.

DATASET FORMAT:
    metadata.csv — pipe-separated, with a header row:

        audio_file|text|speaker_id
        /abs/path/to/wavs/utt_001.wav|Hello world|SPEAKER_00_Lang
        /abs/path/to/wavs/utt_002.wav|Another sentence|SPEAKER_01_Lang

    The audio_file column must contain absolute paths to wav files.

SINGLE-GPU USAGE:
    python train_vits.py \\
        --metadata /path/to/metadata.csv \\
        --language ha \\
        --output_path /path/to/output

MULTI-GPU USAGE (Coqui distribute.py — NOT torchrun):
    Coqui Trainer uses its own DDP launcher. Pass your training args after
    --script; distribute.py forwards them to each worker process.

    python -m trainer.distribute \\
        --script train_vits.py \\
        --gpus "0,1" \\
        --metadata /path/to/metadata.csv \\
        --language ha \\
        --output_path /path/to/output \\
        --num_gpus 2

    With --num_gpus N, the per-GPU batch size is set to
    global_batch_size // N so the total data seen per step
    remains identical to a single-GPU run.

RESUME TRAINING:
    Preemption / requeue: the script automatically scans output_path for the
    most recent checkpoint and resumes via continue_path if neither
    --restore_path nor --continue_path is supplied.

    To resume manually, pass --continue_path /path/to/run_dir (preferred,
    restores full training state including optimizer) or
    --restore_path /path/to/checkpoint_XXXXX.pth (weights only).

TENSORBOARD:
    tensorboard --logdir /path/to/output_path
"""

import argparse
import math
import os
import unicodedata

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from trainer import Trainer, TrainerArgs

# ---------------------------------------------------------------------------
# Monkey-patch: guard rational_quadratic_spline against empty-tensor inputs.
#
# When all inputs in a batch fall outside the spline interval,
# inside_interval_mask is all-False and inputs[inside_interval_mask] is an
# empty tensor.  The original code then calls torch.min(inputs) / torch.max()
# which raises "RuntimeError: min(): Expected reduction dim to be specified
# for input.numel() == 0".  We replace the function with a version that
# short-circuits for empty inputs — semantically correct because the
# assignment back in the caller is a no-op for an empty mask.
# ---------------------------------------------------------------------------
import TTS.tts.layers.vits.transforms as _vits_transforms

_orig_rqs = _vits_transforms.rational_quadratic_spline


def _patched_rqs(inputs, *args, **kwargs):
    if inputs.numel() == 0:
        return inputs, torch.zeros_like(inputs)
    return _orig_rqs(inputs, *args, **kwargs)


_vits_transforms.rational_quadratic_spline = _patched_rqs
# Python resolves the call inside unconstrained_rational_quadratic_spline via
# the transforms module's own __dict__, so updating the module attribute is
# sufficient to intercept every call path.

from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


# ---------------------------------------------------------------------------
# Length-coordinated DDP sampler
# ---------------------------------------------------------------------------

class LengthCoordinatedSampler(Sampler):
    """Eliminate DDP barrier wait caused by variable-length audio sequences.

    Coqui TTS calls dataset.preprocess_samples() before get_sampler(), which
    sorts the dataset indices from shortest to longest audio. We exploit that
    ordering by grouping indices into super-batches of
    (num_replicas × batch_size). Within each super-batch every sample has a
    similar duration, so rank 0 and rank 1 always finish their forward pass at
    roughly the same time and waste no time at the DDP barrier.

    Super-batches are shuffled in a different order each epoch (via
    auto-incrementing seed) so the model does not always see the same
    length curriculum.
    """

    def __init__(self, dataset_size: int, batch_size: int, num_replicas: int, rank: int,
                 shuffle: bool = True, seed: int = 54321):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        super_batch_size = num_replicas * batch_size
        remainder = dataset_size % super_batch_size
        self._total_size = dataset_size + (super_batch_size - remainder) % super_batch_size
        self._num_samples = self._total_size // num_replicas  # samples this rank sees per epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        # Indices 0..N-1 are already sorted shortest→longest by preprocess_samples().
        indices = list(range(self.dataset_size))
        # Pad to make total size divisible by super_batch_size.
        indices += indices[: self._total_size - self.dataset_size]

        super_batch_size = self.num_replicas * self.batch_size
        num_super_batches = self._total_size // super_batch_size
        super_batches = [
            indices[i * super_batch_size : (i + 1) * super_batch_size]
            for i in range(num_super_batches)
        ]

        # Shuffle the *order* of super-batches each epoch so the model sees
        # length groups in a different sequence, while within-step coordination
        # (all GPUs same length bucket) is preserved.
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self._epoch)
            perm = torch.randperm(num_super_batches, generator=g).tolist()
            super_batches = [super_batches[i] for i in perm]

        # Interleaved (stride) assignment: rank r takes positions r, r+num_replicas,
        # r+2*num_replicas, ... from each sorted super-batch.  Both ranks end up with
        # nearly identical average sequence lengths per step (alternating short/long),
        # so the DDP barrier wait shrinks to ~1 sample-length apart instead of
        # half the super-batch width apart (as with contiguous slicing).
        rank_indices = []
        for sb in super_batches:
            rank_indices.extend(sb[self.rank :: self.num_replicas])

        self._epoch += 1  # auto-increment so each __iter__ call uses a fresh seed
        return iter(rank_indices)

    def __len__(self) -> int:
        return self._num_samples


class VitsCoordinated(Vits):
    """Vits subclass that swaps in LengthCoordinatedSampler for DDP training."""

    def get_sampler(self, config, dataset, num_gpus=1, is_eval=False):
        if num_gpus > 1 and not is_eval:
            return LengthCoordinatedSampler(
                dataset_size=len(dataset),
                batch_size=config.batch_size,
                num_replicas=num_gpus,
                rank=dist.get_rank(),
                shuffle=True,
                seed=54321,
            )
        return super().get_sampler(config, dataset, num_gpus, is_eval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a multilingual multispeaker VITS model with Coqui TTS"
    )

    # --- Required ---
    p.add_argument(
        "--metadata", required=True,
        help="Path to pipe-separated metadata.csv (audio_file|text|speaker_id, with header)",
    )
    p.add_argument(
        "--language", required=True,
        help="ISO 639-1/3 language code used by the dataset (e.g. ha, hi, yo, sw)",
    )
    p.add_argument(
        "--output_path", required=True,
        help="Directory for checkpoints, logs, and tensorboard events",
    )

    # --- Scale ---
    p.add_argument(
        "--global_batch_size", type=int, default=32,
        help="Total batch size across all GPUs (default: 32). "
             "Per-GPU batch = global_batch_size // num_gpus.",
    )
    p.add_argument(
        "--num_gpus", type=int, default=1,
        help="Number of GPUs used for training (default: 1). "
             "Used only to compute per-GPU batch size; launch with torchrun for actual DDP.",
    )
    p.add_argument(
        "--target_steps", type=int, default=500_000,
        help="Target number of optimizer steps (default: 500 000). "
             "Epochs are computed from this and the dataset size.",
    )

    # --- Audio ---
    p.add_argument("--sample_rate", type=int, default=22050, help="Audio sample rate (default: 22050)")
    p.add_argument("--win_length", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--num_mels", type=int, default=80)

    # --- Checkpointing / logging ---
    p.add_argument("--save_step", type=int, default=5000, help="Save a checkpoint every N steps")
    p.add_argument("--save_n_checkpoints", type=int, default=5, help="Keep N most recent checkpoints")
    p.add_argument("--print_step", type=int, default=100)
    p.add_argument("--restore_path", default=None, help="Checkpoint path to resume training from")
    p.add_argument("--continue_path", default=None, help="Run directory to continue training from")

    # --- Eval split ---
    p.add_argument(
        "--no_eval", action="store_true",
        help="Use the entire dataset for training; disable evaluation entirely.",
    )
    p.add_argument(
        "--eval_split_size", type=float, default=0.01,
        help="Fraction of data used for evaluation (default: 0.01 = 1%%). Ignored when --no_eval is set.",
    )
    p.add_argument(
        "--eval_split_max_size", type=int, default=None,
        help="Hard cap on number of eval samples (default: no cap). Ignored when --no_eval is set.",
    )

    # --- Misc ---
    p.add_argument(
        "--run_name", default=None,
        help="Experiment name shown in logs (default: vits_<language>)",
    )

    # --- Injected by trainer.distribute (do not set manually) ---
    p.add_argument("--use_ddp", type=lambda x: str(x).lower() in ("true", "1"), default=False, help=argparse.SUPPRESS)
    p.add_argument("--rank", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--group_id", type=str, default="", help=argparse.SUPPRESS)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Preemption / checkpoint auto-resume
# ---------------------------------------------------------------------------

def find_latest_run_dir(output_path):
    """Scan output_path for the run directory containing the newest checkpoint.

    Coqui Trainer writes checkpoints under:
        output_path/<run_name>-<timestamp>/checkpoint_<step>.pth

    Returns (run_dir, checkpoint_path) of the most recently modified .pth
    file found, or (None, None) if output_path does not exist or is empty.
    """
    if not os.path.isdir(output_path):
        return None, None

    latest_mtime = -1
    latest_ckpt = None
    latest_run_dir = None

    for entry in os.scandir(output_path):
        if not entry.is_dir():
            continue
        for f in os.scandir(entry.path):
            if f.name.endswith(".pth") and f.stat().st_mtime > latest_mtime:
                latest_mtime = f.stat().st_mtime
                latest_ckpt = f.path
                latest_run_dir = entry.path

    return latest_run_dir, latest_ckpt


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

def open_bible_formatter(root_path, meta_file, ignored_speakers=None, **kwargs):
    """Parse metadata with format: audio_file|text|speaker_id (with header row).

    audio_file contains an absolute path; root_path is ignored for path
    resolution but kept to satisfy the Coqui TTS formatter contract.
    """
    items = []
    with open(meta_file, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            audio_file = parts[0].strip()
            text = parts[1].strip()
            speaker_name = parts[2].strip() if len(parts) > 2 else "default"
            if ignored_speakers and speaker_name in ignored_speakers:
                continue
            items.append(
                {
                    "text": text,
                    "audio_file": audio_file,
                    "speaker_name": speaker_name,
                    "root_path": root_path,
                }
            )
    return items


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocab(metadata_path):
    """Scan metadata and collect every Unicode letter and combining mark.

    Keeps letters (L*) and combining diacritics (M*); excludes punctuation,
    digits, and symbols. Space is handled via CharactersConfig.punctuations.
    """
    chars = set()
    with open(metadata_path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            for ch in parts[1].lower():
                cat = unicodedata.category(ch)
                if cat.startswith("L") or cat.startswith("M"):
                    chars.add(ch)
    return "".join(sorted(chars))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # --- Preemption auto-resume ---
    # If the job was preempted and requeued, pick up from the latest checkpoint
    # saved in output_path so training continues seamlessly without any manual
    # intervention. Explicit --restore_path / --continue_path always take precedence.
    if args.restore_path is None and args.continue_path is None:
        run_dir, ckpt = find_latest_run_dir(args.output_path)
        if ckpt is not None:
            print(f" > Preemption resume detected — continuing from run dir : {run_dir}")
            print(f" > Latest checkpoint                                     : {ckpt}")
            args.continue_path = run_dir
        else:
            print(" > No previous checkpoint found — starting training from scratch.")

    # TF32 on Ampere (A100/A6000/…): matmul and cuDNN ops use 19-bit mantissa
    # instead of full FP32 at ~20% higher throughput with negligible accuracy loss.
    # The Trainer ORs its config flag with the current value, so pre-setting here
    # works even if VitsConfig doesn't expose allow_tf32.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Per-GPU batch size keeps total samples-per-step == global_batch_size
    per_gpu_batch = args.global_batch_size // args.num_gpus
    if per_gpu_batch < 1:
        raise ValueError(
            f"global_batch_size ({args.global_batch_size}) must be >= num_gpus ({args.num_gpus})"
        )

    run_name = args.run_name or f"vits_{args.language}"

    # --- Vocabulary ---
    vocab = build_vocab(args.metadata)
    print(f" > Vocabulary: {len(vocab)} unique characters")
    print(f" > Characters: {vocab}")
    print(
        f" > Unicode codepoints: "
        f"{[f'U+{ord(c):04X} ({unicodedata.name(c, repr(c))})' for c in vocab]}"
    )

    # --- Dataset config ---
    # meta_file_train receives the full metadata path; root_path is the
    # directory containing metadata (used by Coqui internals for logging).
    dataset_config = BaseDatasetConfig(
        formatter="open_bible",          # matched by the formatter kwarg below
        meta_file_train=args.metadata,
        path=os.path.dirname(os.path.abspath(args.metadata)),
        language=args.language,
    )

    # --- Audio config ---
    audio_config = VitsAudioConfig(
        sample_rate=args.sample_rate,
        win_length=args.win_length,
        hop_length=args.hop_length,
        num_mels=args.num_mels,
        mel_fmin=0,
        mel_fmax=None,
    )

    # --- Character config ---
    characters_config = CharactersConfig(
        characters=vocab,
        punctuations="!\"'(),-.:;? ",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        is_unique=True,
        is_sorted=True,
    )

    # --- VITS config ---
    config = VitsConfig(
        audio=audio_config,
        run_name=run_name,
        batch_size=per_gpu_batch,
        eval_batch_size=max(1, per_gpu_batch // 2),
        batch_group_size=5,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=not args.no_eval,
        test_delay_epochs=-1,
        epochs=1,              # overwritten below after dataset is loaded

        # Text / tokenisation
        use_phonemes=False,
        text_cleaner="basic_cleaners",
        characters=characters_config,

        # Multispeaker — learned speaker embedding lookup table
        use_speaker_embedding=True,
        num_speakers=0,        # overwritten below after speakers are counted

        # Logging
        dashboard_logger="tensorboard",
        print_step=args.print_step,
        plot_step=args.print_step,
        print_eval=True,

        # Checkpointing
        save_step=args.save_step,
        save_checkpoints=True,
        save_n_checkpoints=args.save_n_checkpoints,
        save_all_best=False,

        # Efficiency
        mixed_precision=True,
        cudnn_benchmark=False,

        output_path=args.output_path,
        datasets=[dataset_config],
    )

    # --- Load samples ---
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=not args.no_eval,
        formatter=open_bible_formatter,
        eval_split_max_size=args.eval_split_max_size,
        eval_split_size=args.eval_split_size,
    )
    if eval_samples is None:
        eval_samples = []

    print(f" > Training samples  : {len(train_samples)}")
    print(f" > Evaluation samples: {len(eval_samples)}")

    # --- Compute epochs from target steps ---
    # VITS uses DistributedSampler when num_gpus > 1, which partitions the
    # dataset across GPUs (each GPU sees N/num_gpus samples per epoch).
    # So optimizer steps per epoch = ceil(N / (per_gpu_batch * num_gpus))
    #                              = ceil(N / global_batch_size)
    # This is identical to the single-GPU formula, so num_epochs is the same
    # regardless of how many GPUs are used — only wall-clock time changes.
    steps_per_epoch = math.ceil(len(train_samples) / args.global_batch_size)
    num_epochs = math.ceil(args.target_steps / steps_per_epoch)
    config.epochs = num_epochs

    print(f" > Per-GPU batch size : {per_gpu_batch}  (global {args.global_batch_size}, {args.num_gpus} GPU(s))")
    print(f" > Steps per epoch   : {steps_per_epoch}")
    print(f" > Target steps      : {args.target_steps:,}")
    print(f" > Computed epochs   : {num_epochs}")

    # --- Audio processor ---
    ap = AudioProcessor.init_from_config(config)

    # --- Tokenizer ---
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # --- Speaker manager ---
    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")  # eval_samples is [] when --no_eval
    config.num_speakers = speaker_manager.num_speakers
    print(f" > Speakers ({speaker_manager.num_speakers}): {speaker_manager.speaker_names}")

    # --- Model ---
    model = VitsCoordinated(config, ap, tokenizer, speaker_manager=speaker_manager)

    # --- Trainer ---
    trainer = Trainer(
        TrainerArgs(
            restore_path=args.restore_path,
            continue_path=args.continue_path,
            use_ddp=args.use_ddp,
            rank=args.rank,
            group_id=args.group_id,
        ),
        config,
        args.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
