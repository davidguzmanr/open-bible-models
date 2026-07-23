#!/usr/bin/env bash
#SBATCH --job-name=F5-Open-Bible-Yoruba-vocoder-matching
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# Vocoder matching for F5-TTS: fine-tune Vocos on F5-generated (teacher-forced) Mels.
#
# This is the F5 + Vocos analogue of the vocoder-matching block in
# jobs/FastSpeech/EveryVoice-Open-Bible-Haitian-Creole.sh. It runs AFTER F5 has been
# trained (see F5-Open-Bible-Yoruba.sh). Two stages:
#   1. Generate teacher-forced Mels from the trained F5 model (F5-TTS conda env).
#   2. Fine-tune a pretrained Vocos on those Mels vs. the ground-truth audio (vocos env).
#
# NOTE: stage 2 needs a separate `vocos` conda env, because Vocos pins
# pytorch_lightning==1.8.6 which conflicts with the F5-TTS environment. Create it once:
#   conda create -n vocos python=3.10 -y
#   conda activate vocos
#   pip install -e /home/mila/g/guzmand/scratch/Repositories/open-bible-models/vocos
#   pip install -r /home/mila/g/guzmand/scratch/Repositories/open-bible-models/vocos/requirements-train.txt
#   # Vocos + PL 1.8.6 need pkg_resources, which setuptools>=81 removed:
#   pip install "setuptools<81"
#
# Env compatibility patches already applied in-repo (needed for modern torch/matplotlib):
#   - vocos/vocos/helpers.py: save_figure_to_numpy() uses buffer_rgba() instead of the
#     removed canvas.tostring_rgb() / np.fromstring() (matplotlib>=3.10, numpy>=2.0).
#   - vocos/vocos/experiment.py: forward() accepts precomputed `features` (fine-tuning).
#   - vocos/vocos/finetune.py: VocosFinetune{Dataset,DataModule,Exp} for vocoder matching.

START_TIME=$(date +%s)
echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

module load miniconda/3
module load gcc/9.3.0
module load cuda/12.3.2

export HF_HOME=$SCRATCH/huggingface
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/home/mila/g/guzmand/scratch/Repositories/open-bible-models
F5_DIR=$REPO/F5-TTS
VOCOS_DIR=$REPO/vocos

# --- experiment-specific paths (edit these for a different language) ---
CONFIG_NAME=F5TTS_v1_Base_Open_Bible_Yoruba
F5_CKPT=$F5_DIR/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-yoruba/model_last.pt
OUT_DIR=$F5_DIR/data/open-bible-yoruba_custom/vocoder_matching
PRETRAINED_VOCOS=charactr/vocos-mel-24khz   # local .bin path or HF repo id
STRENGTH=0.5        # SDEdit seed time t0; lower = more F5 character, less GT fidelity
STEPS=16            # ODE integration steps

##################################################################
# 1. Generate teacher-forced Mel spectrograms from the trained F5 model
##################################################################
conda activate F5-TTS
echo "NVIDIA SMI:"; nvidia-smi
cd "$F5_DIR"

python src/f5_tts/train/generate_vocoder_mels.py \
    --config-name "$CONFIG_NAME" \
    --ckpt "$F5_CKPT" \
    --output-dir "$OUT_DIR" \
    --strength "$STRENGTH" \
    --steps "$STEPS" \
    --val-size 256

conda deactivate

##################################################################
# 2. Fine-tune Vocos on the generated Mels (vocoder matching)
##################################################################
conda activate vocos
cd "$VOCOS_DIR"

TRAIN_FL=$OUT_DIR/train_filelist.txt
VAL_FL=$OUT_DIR/val_filelist.txt

# Fill the config template placeholders (use | as sed delimiter since paths contain /).
TMP_CONFIG=$(mktemp --suffix=.yaml)
sed -e "s|@TRAIN_FILELIST@|$TRAIN_FL|" \
    -e "s|@VAL_FILELIST@|$VAL_FL|" \
    -e "s|@PRETRAINED@|$PRETRAINED_VOCOS|" \
    configs/vocos-finetune-f5.yaml >| "$TMP_CONFIG"

echo "Using generated config: $TMP_CONFIG"
cat "$TMP_CONFIG"

python train.py -c "$TMP_CONFIG"

rm -f "$TMP_CONFIG"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))
echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
