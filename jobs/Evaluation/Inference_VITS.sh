#!/usr/bin/env bash
#SBATCH --job-name=Inference_VITS
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
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
# Inference
##################################################################
cd /home/mila/g/guzmand/scratch/Repositories/open-bible-models

python utils/inference-test-VITS.py \
    --language      Ewe \
    --output_dir    synthesis_output/VITS/ewe \
    --ckpt_path     VITS-TTS/outputs/ewe/vits_ewe-May-10-2026_02+40AM-106eab5/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-ewe/metadata.csv

python utils/inference-test-VITS.py \
    --language      Gamo \
    --output_dir    synthesis_output/VITS/gamo \
    --ckpt_path     VITS-TTS/outputs/gamo/vits_gamo-May-10-2026_04+17AM-106eab5/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-gamo/metadata.csv

python utils/inference-test-VITS.py \
    --language      Gofa \
    --output_dir    synthesis_output/VITS/gofa \
    --ckpt_path     VITS-TTS/outputs/gofa/vits_gofa-May-10-2026_02+45AM-106eab5/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-gofa/metadata.csv

python utils/inference-test-VITS.py \
    --language      Gujarati \
    --output_dir    synthesis_output/VITS/gujarati \
    --ckpt_path     VITS-TTS/outputs/gujarati/vits_gujarati-May-11-2026_02+02PM-106eab5/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-gujarati/metadata.csv

python utils/inference-test-VITS.py \
    --language      "Haitian Creole" \
    --output_dir    synthesis_output/VITS/haitian-creole \
    --ckpt_path     VITS-TTS/outputs/haitian-creole/vits_haitian_creole-May-01-2026_09+16AM-2541a19/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-haitian-creole/metadata.csv

python utils/inference-test-VITS.py \
    --language      Hausa \
    --output_dir    synthesis_output/VITS/hausa \
    --ckpt_path     VITS-TTS/outputs/hausa/vits_hausa-May-07-2026_09+34PM-b7ea09e/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-hausa/metadata.csv

python utils/inference-test-VITS.py \
    --language      Hiligaynon \
    --output_dir    synthesis_output/VITS/hiligaynon \
    --ckpt_path     VITS-TTS/outputs/hiligaynon/vits_hiligaynon-May-11-2026_01+59PM-106eab5/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-hiligaynon/metadata.csv

python utils/inference-test-VITS.py \
    --language      Hindi \
    --output_dir    synthesis_output/VITS/hindi \
    --ckpt_path     VITS-TTS/outputs/hindi/vits_hindi-May-02-2026_04+32PM-2541a19/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-hindi/metadata.csv

python utils/inference-test-VITS.py \
    --language      Igbo \
    --output_dir    synthesis_output/VITS/igbo \
    --ckpt_path     VITS-TTS/outputs/igbo/vits_igbo-May-14-2026_10+52AM-106eab5/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-igbo/metadata.csv

python utils/inference-test-VITS.py \
    --language      Kannada \
    --output_dir    synthesis_output/VITS/kannada \
    --ckpt_path     VITS-TTS/outputs/kannada/vits_kannada-May-14-2026_02+32PM-106eab5/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-kannada/metadata.csv

python utils/inference-test-VITS.py \
    --language      Kikuyu \
    --output_dir    synthesis_output/VITS/kikuyu \
    --ckpt_path     VITS-TTS/outputs/kikuyu/vits_kikuyu-May-14-2026_11+17PM-106eab5/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-kikuyu/metadata.csv

python utils/inference-test-VITS.py \
    --language      Lingala \
    --output_dir    synthesis_output/VITS/lingala \
    --ckpt_path     VITS-TTS/outputs/lingala/vits_lingala-May-14-2026_02+30PM-106eab5/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-lingala/metadata.csv

python utils/inference-test-VITS.py \
    --language      Malayalam \
    --output_dir    synthesis_output/VITS/malayalam \
    --ckpt_path     VITS-TTS/outputs/malayalam/vits_malayalam-May-15-2026_12+07AM-106eab5/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-malayalam/metadata.csv

python utils/inference-test-VITS.py \
    --language      Ndebele \
    --output_dir    synthesis_output/VITS/ndebele \
    --ckpt_path     VITS-TTS/outputs/ndebele/vits_ndebele-May-17-2026_12+01AM-106eab5/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-ndebele/metadata.csv

python utils/inference-test-VITS.py \
    --language      Nepali \
    --output_dir    synthesis_output/VITS/nepali \
    --ckpt_path     VITS-TTS/outputs/nepali/vits_nepali-May-18-2026_02+29PM-106eab5/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-nepali/metadata.csv

python utils/inference-test-VITS.py \
    --language      Punjabi \
    --output_dir    synthesis_output/VITS/punjabi \
    --ckpt_path     VITS-TTS/outputs/punjabi/vits_punjabi-May-17-2026_07+57PM-106eab5/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-punjabi/metadata.csv

python utils/inference-test-VITS.py \
    --language      Shona \
    --output_dir    synthesis_output/VITS/shona \
    --ckpt_path     VITS-TTS/outputs/shona/vits_shona-May-02-2026_04+19PM-2541a19/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-shona/metadata.csv

python utils/inference-test-VITS.py \
    --language      Tamil \
    --output_dir    synthesis_output/VITS/tamil \
    --ckpt_path     VITS-TTS/outputs/tamil/vits_tamil-May-17-2026_12+02AM-106eab5/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-tamil/metadata.csv

python utils/inference-test-VITS.py \
    --language      Telugu \
    --output_dir    synthesis_output/VITS/telugu \
    --ckpt_path     VITS-TTS/outputs/telugu/vits_telugu-May-05-2026_05+22PM-2541a19/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-telugu/metadata.csv

python utils/inference-test-VITS.py \
    --language      Turkish \
    --output_dir    synthesis_output/VITS/turkish \
    --ckpt_path     VITS-TTS/outputs/turkish/vits_turkish-May-06-2026_01+06PM-2541a19/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-turkish/metadata.csv

python utils/inference-test-VITS.py \
    --language      Vietnamese \
    --output_dir    synthesis_output/VITS/vietnamese \
    --ckpt_path     VITS-TTS/outputs/vietnamese/vits_vietnamese-May-03-2026_10+38PM-2541a19/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-vietnamese/metadata.csv

python utils/inference-test-VITS.py \
    --language      Yoruba \
    --output_dir    synthesis_output/VITS/yoruba \
    --ckpt_path     VITS-TTS/outputs/yoruba/vits_yoruba-April-29-2026_11+52PM-fd8dd03/checkpoint_250000.pth \
    --metadata_path F5-TTS/data/open-bible-yoruba/metadata.csv


##################################################################
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
