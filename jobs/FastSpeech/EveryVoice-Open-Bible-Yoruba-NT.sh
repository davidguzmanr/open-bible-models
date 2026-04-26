#!/usr/bin/env bash
#SBATCH --job-name=EveryVoice-Open-Bible-Yoruba-NT
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

START_TIME=$(date +%s)
echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

##################################################################
# Activate the environment by loading Python and required packages
##################################################################
module load miniconda/3
module load gcc/9.3.0
module load cuda/12.3.2

export HF_HOME=$SCRATCH/huggingface
export WANDB_MODE=disabled

conda activate EveryVoice

echo "NVCC version:"
nvcc --version
echo "NVIDIA SMI:"
nvidia-smi
echo $HF_HOME

cd /home/mila/g/guzmand/scratch/Repositories/open-bible-models/EveryVoice-TTS/Open-Bible-Yoruba-NT

HIFIGAN_UNIVERSAL_V1_EVERYVOICE_CKPT="/home/mila/g/guzmand/scratch/checkpoints/hifigan_universal_v1_everyvoice.ckpt"

##################################################################
# TODO list before running the experiment
# - Change the max_audio_length and train_split in everyvoice-shared-data.yaml

# - Change the val_check_interval, max_steps and batch_size in everyvoice-text-to-spec.yaml
# - Change vocoder_path in everyvoice-text-to-spec.yaml
#     - /home/mila/g/guzmand/scratch/checkpoints/hifigan_universal_v1_everyvoice.ckpt
# - Change the finetune_checkpoint in everyvoice-spec-to-wav.yaml if finetuning the vocoder
# - Change validation_filelist in case of using very little data
#
# NOTE on max_steps with 2 GPUs (DDP):
#   With DDP, each GPU processes its own mini-batch per step, so the effective
#   batch size doubles. To process the same total amount of data as 500k steps
#   on 1 GPU, set max_steps=250000 (passed via --config-args below).
##################################################################

##################################################################
# 1. Preprocess the data (if not done already)
##################################################################
everyvoice preprocess config/everyvoice-text-to-spec.yaml \
    --config-args preprocessing.audio.max_audio_length=15 \
    --config-args preprocessing.train_split=1.0 \
    --cpus=8 \
    --overwrite

# Validation is required by the training loop; use a subset of training data as a proxy since we used all data for training
head -n 512 preprocessed/training_filelist.psv >| preprocessed/validation_filelist.psv

# ##################################################################
# # Feature prediction (2 GPUs, DDP — equivalent to 500k steps on 1 GPU)
# ##################################################################
everyvoice train text-to-spec config/everyvoice-text-to-spec.yaml \
    --devices 2 \
    --strategy ddp \
    --config-args training.max_steps=100000 \
    --config-args training.batch_size=32 \
    --config-args training.val_check_interval=5000 \
    --config-args "training.vocoder_path=${HIFIGAN_UNIVERSAL_V1_EVERYVOICE_CKPT}"


# ##################################################################
# # Vocoder matching 
# ##################################################################
# Generate a folder full of Mel spectrograms from the training set.
# We split the filelist into chunks of 5000 lines and synthesize each chunk
# separately to avoid OOM. The root cause is that `everyvoice synthesize`
# calls `trainer.predict(..., return_predictions=True)`, which makes PyTorch
# Lightning accumulate every predict_step output tensor in CPU RAM for the
# entire run — even though the PredictionWritingSpecCallback already writes
# each batch to disk immediately. With 24k training samples this exhausts
# available memory before synthesis completes. Chunking caps the in-memory
# accumulation to 5k tensors per call and RAM is freed between chunks.
HEADER=$(head -n 1 preprocessed/training_filelist.psv)
CHUNK_DIR=$(mktemp -d)
tail -n +2 preprocessed/training_filelist.psv | split -l 5000 - "$CHUNK_DIR/chunk_"
for chunk in "$CHUNK_DIR"/chunk_*; do
    { echo "$HEADER"; cat "$chunk"; } > "$CHUNK_DIR/current_chunk.psv"
    everyvoice synthesize from-text \
        logs_and_checkpoints/FeaturePredictionExperiment/base/checkpoints/last.ckpt \
        --output-type spec \
        --filelist "$CHUNK_DIR/current_chunk.psv" \
        --teacher-forcing-directory preprocessed \
        --output-dir preprocessed \
        --accelerator gpu \
        --devices 1 \
        --batch-size 32
done
rm -rf "$CHUNK_DIR"

# # # Generate a folder full of Mel spectrograms from the validation set
everyvoice synthesize from-text \
    logs_and_checkpoints/FeaturePredictionExperiment/base/checkpoints/last.ckpt \
    --output-type spec \
    --filelist preprocessed/validation_filelist.psv \
    --teacher-forcing-directory preprocessed \
    --output-dir preprocessed \
    --accelerator gpu \
    --devices 1 \
    --batch-size 32

# # Fine-tune the vocoder on the generated Mel spectrograms
everyvoice train spec-to-wav config/everyvoice-spec-to-wav.yaml  \
    --devices 2 \
    --strategy ddp \
    --config-args "training.finetune_checkpoint=${HIFIGAN_UNIVERSAL_V1_EVERYVOICE_CKPT}" \
    --config-args training.finetune=True \
    --config-args training.optimizer.learning_rate=0.00001 \
    --config-args training.max_steps=100000 \
    --config-args training.batch_size=32 \
    --config-args training.val_check_interval=5000


END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
