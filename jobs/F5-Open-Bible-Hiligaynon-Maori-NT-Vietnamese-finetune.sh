#!/usr/bin/env bash
#SBATCH --job-name=F5-Open-Bible-Hiligaynon-Maori-NT-Vietnamese-finetune
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

START_TIME=$SECONDS
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda activate F5-TTS

echo "NVCC version:"
nvcc --version
echo "NVIDIA SMI:"
nvidia-smi
echo $HF_HOME

cd /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS

##################################################################
# Training
##################################################################
python prepare_data.py --languages Maori:NT --finetune \
    --pretrain-ckpt ckpts/F5TTS_v1_Base_vocos_custom_open-bible-hiligaynon-maori-nt-vietnamese/model_last.pt \
    --pretrain-vocab data/open-bible-hiligaynon-maori-nt-vietnamese_custom/vocab.txt \
    --num-gpus 2 --target-updates 200000

accelerate launch --num_processes 2 --mixed_precision bf16 \
    src/f5_tts/train/finetune_cli.py \
    --exp_name F5TTS_v1_Base \
    --dataset_name open-bible-maori-nt-finetune \
    --finetune \
    --pretrain ckpts/F5TTS_v1_Base_vocos_custom_open-bible-hiligaynon-maori-nt-vietnamese/model_last.pt \
    --tokenizer custom \
    --tokenizer_path data/open-bible-hiligaynon-maori-nt-vietnamese_custom/vocab.txt \
    --epochs 1243 \
    --learning_rate 1e-5 \
    --batch_size_per_gpu 28000 \
    --max_samples 32 \
    --num_warmup_updates 2000 \
    --save_per_updates 10000 \
    --logger tensorboard
    
ELAPSED=$(( SECONDS - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECS=$(( ELAPSED % 60 ))
echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECS}s"