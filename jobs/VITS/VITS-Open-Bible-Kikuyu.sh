#!/usr/bin/env bash
#SBATCH --job-name=VITS-Open-Bible-Kikuyu
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

conda activate CoquiTTS

echo "NVCC version:"
nvcc --version
echo "NVIDIA SMI:"
nvidia-smi
echo $HF_HOME

##################################################################
# Preprocess and stage dataset into $SLURM_TMPDIR for fast local I/O
##################################################################
DATA_SRC=/network/scratch/g/guzmand/Repositories/open-bible-models/F5-TTS/data/open-bible-kikuyu
DATA_DST=$SLURM_TMPDIR/data/open-bible-kikuyu

mkdir -p "$DATA_DST"

echo "Preprocessing audio (resample to 22050 Hz, mono WAV) into \$SLURM_TMPDIR..."

cd /home/mila/g/guzmand/scratch/Repositories/open-bible-models/VITS-TTS

python preprocess_open_bible_audio.py \
  --metadata "$DATA_SRC/metadata.csv" \
  --output-dir "$DATA_DST" \
  --target-sample-rate 22050

echo "Preprocessed dataset ready at $DATA_DST/metadata.csv"

##################################################################
# Training
##################################################################
python -m trainer.distribute \
  --script train_vits.py \
  --gpus "0,1" \
  --metadata $DATA_DST/metadata.csv \
  --language ki \
  --output_path outputs/kikuyu \
  --global_batch_size 64 \
  --num_gpus 2 \
  --target_steps 250000 \
  --sample_rate 22050 \
  --win_length 1024 \
  --hop_length 256 \
  --num_mels 80 \
  --save_step 5000 \
  --save_n_checkpoints 5 \
  --print_step 5000 \
  --grad_clip 1.0 \
  --no_mixed_precision \
  --no_eval \
  --run_name vits_kikuyu


END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
