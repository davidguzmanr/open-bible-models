#!/usr/bin/env bash
#SBATCH --job-name=Inference_OmniVoice
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
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

# omnivoice is installed in the F5-TTS environment
conda activate F5-TTS

echo "NVCC version:"
nvcc --version
echo "NVIDIA SMI:"
nvidia-smi
echo $HF_HOME

##################################################################
# Inference
##################################################################
cd /home/mila/g/guzmand/scratch/Repositories/open-bible-models

##################################################################
# Data preparation
##################################################################
# OmniVoice inference points --metadata_path at each language's
# F5-TTS/data/open-bible-<language>/metadata.csv. prepare_data.py creates that
# metadata (and preprocesses the audio), so run it once per language before
# inference. It must run from the F5-TTS directory.
cd /home/mila/g/guzmand/scratch/Repositories/open-bible-models/F5-TTS

# Arabic Standard
# python prepare_data.py --languages "Arabic Standard" --num-gpus 2 --target-updates 500000
# Assamese
# python prepare_data.py --languages Assamese --num-gpus 2 --target-updates 500000
# Bengali
# python prepare_data.py --languages Bengali --num-gpus 2 --target-updates 500000
# Central Kurdish
# python prepare_data.py --languages "Central Kurdish" --num-gpus 2 --target-updates 500000
# Chhattisgarhi
# python prepare_data.py --languages Chhattisgarhi --num-gpus 2 --target-updates 500000
# Chichewa
# python prepare_data.py --languages Chichewa --num-gpus 2 --target-updates 500000
# Dawro
# python prepare_data.py --languages Dawro --num-gpus 2 --target-updates 500000
# Dholuo
# python prepare_data.py --languages Dholuo --num-gpus 2 --target-updates 500000
# Ewe
# python prepare_data.py --languages Ewe --num-gpus 2 --target-updates 500000
# Gamo
# python prepare_data.py --languages Gamo --num-gpus 2 --target-updates 500000
# Gofa
# python prepare_data.py --languages Gofa --num-gpus 2 --target-updates 500000
# Gujarati
# python prepare_data.py --languages Gujarati --num-gpus 2 --target-updates 500000
# Haitian Creole
# python prepare_data.py --languages "Haitian Creole" --num-gpus 2 --target-updates 500000
# Hausa
# python prepare_data.py --languages Hausa --num-gpus 2 --target-updates 500000
# Hiligaynon
# python prepare_data.py --languages Hiligaynon --num-gpus 2 --target-updates 500000
# Hindi
# python prepare_data.py --languages Hindi --num-gpus 2 --target-updates 500000
# Igbo
# python prepare_data.py --languages Igbo --num-gpus 2 --target-updates 500000
# Kannada
# python prepare_data.py --languages Kannada --num-gpus 2 --target-updates 500000
# Kikuyu
# python prepare_data.py --languages Kikuyu --num-gpus 2 --target-updates 500000
# Lingala
# python prepare_data.py --languages Lingala --num-gpus 2 --target-updates 500000
# Luganda
# python prepare_data.py --languages Luganda --num-gpus 2 --target-updates 500000
# Malayalam
# python prepare_data.py --languages Malayalam --num-gpus 2 --target-updates 500000
# Marathi
# python prepare_data.py --languages Marathi --num-gpus 2 --target-updates 500000
# Ndebele
# python prepare_data.py --languages Ndebele --num-gpus 2 --target-updates 500000
# Nepali
# python prepare_data.py --languages Nepali --num-gpus 2 --target-updates 500000
# Oromo
# python prepare_data.py --languages Oromo --num-gpus 2 --target-updates 500000
# Punjabi
# python prepare_data.py --languages Punjabi --num-gpus 2 --target-updates 500000
# Shona
# python prepare_data.py --languages Shona --num-gpus 2 --target-updates 500000
# Swahili
# python prepare_data.py --languages Swahili --num-gpus 2 --target-updates 500000
# Tamil
# python prepare_data.py --languages Tamil --num-gpus 2 --target-updates 500000
# Telugu
# python prepare_data.py --languages Telugu --num-gpus 2 --target-updates 500000
# Turkish
# python prepare_data.py --languages Turkish --num-gpus 2 --target-updates 500000
# Twi (Akuapem)
# python prepare_data.py --languages "Twi (Akuapem)" --num-gpus 2 --target-updates 500000
# Twi (Asante)
# python prepare_data.py --languages "Twi (Asante)" --num-gpus 2 --target-updates 500000
# Urdu
# python prepare_data.py --languages Urdu --num-gpus 2 --target-updates 500000
# Vietnamese
# python prepare_data.py --languages Vietnamese --num-gpus 2 --target-updates 500000
# Yoruba
# python prepare_data.py --languages Yoruba --num-gpus 2 --target-updates 500000

cd /home/mila/g/guzmand/scratch/Repositories/open-bible-models

##################################################################
# Checkpoints
##################################################################
# OmniVoice is zero-shot: a single multilingual checkpoint is used for every
# language, and the voice is set by the reference clip picked from each
# language's training metadata (--metadata_path). Pre-download the checkpoint
# once so the per-language runs below do not each hit the Hub.
python - <<'EOF'
from huggingface_hub import snapshot_download

MODEL_CARD = "k2-fsa/OmniVoice"

print(f"Downloading {MODEL_CARD} ...")
path = snapshot_download(MODEL_CARD)
print(f"  -> {path}")
print("\nDownload complete!")
EOF

##################################################################
# Run inference for all languages
##################################################################
# Arabic Standard
# python utils/inference-test-OmniVoice.py \
#     --language "Arabic Standard" \
#     --output_dir synthesis_output/OmniVoice/arabic-standard \
#     --metadata_path "F5-TTS/data/open-bible-arabic standard/metadata.csv"

# Assamese
# python utils/inference-test-OmniVoice.py \
#     --language Assamese \
#     --output_dir synthesis_output/OmniVoice/assamese \
#     --metadata_path F5-TTS/data/open-bible-assamese/metadata.csv

# Bengali
# python utils/inference-test-OmniVoice.py \
#     --language Bengali \
#     --output_dir synthesis_output/OmniVoice/bengali \
#     --metadata_path F5-TTS/data/open-bible-bengali/metadata.csv

# Central Kurdish
# python utils/inference-test-OmniVoice.py \
#     --language "Central Kurdish" \
#     --output_dir synthesis_output/OmniVoice/central-kurdish \
#     --metadata_path "F5-TTS/data/open-bible-central kurdish/metadata.csv"

# Chhattisgarhi
# python utils/inference-test-OmniVoice.py \
#     --language Chhattisgarhi \
#     --output_dir synthesis_output/OmniVoice/chhattisgarhi \
#     --metadata_path F5-TTS/data/open-bible-chhattisgarhi/metadata.csv

# Chichewa
# python utils/inference-test-OmniVoice.py \
#     --language Chichewa \
#     --output_dir synthesis_output/OmniVoice/chichewa \
#     --metadata_path F5-TTS/data/open-bible-chichewa/metadata.csv

# Dawro
# python utils/inference-test-OmniVoice.py \
#     --language Dawro \
#     --output_dir synthesis_output/OmniVoice/dawro \
#     --metadata_path F5-TTS/data/open-bible-dawro/metadata.csv

# Dholuo
# python utils/inference-test-OmniVoice.py \
#     --language Dholuo \
#     --output_dir synthesis_output/OmniVoice/dholuo \
#     --metadata_path F5-TTS/data/open-bible-dholuo/metadata.csv

# Ewe
# python utils/inference-test-OmniVoice.py \
#     --language Ewe \
#     --output_dir synthesis_output/OmniVoice/ewe \
#     --metadata_path F5-TTS/data/open-bible-ewe/metadata.csv

# Gamo
# python utils/inference-test-OmniVoice.py \
#     --language Gamo \
#     --output_dir synthesis_output/OmniVoice/gamo \
#     --metadata_path F5-TTS/data/open-bible-gamo/metadata.csv

# Gofa
# python utils/inference-test-OmniVoice.py \
#     --language Gofa \
#     --output_dir synthesis_output/OmniVoice/gofa \
#     --metadata_path F5-TTS/data/open-bible-gofa/metadata.csv

# Gujarati
# python utils/inference-test-OmniVoice.py \
#     --language Gujarati \
#     --output_dir synthesis_output/OmniVoice/gujarati \
#     --metadata_path F5-TTS/data/open-bible-gujarati/metadata.csv

# Haitian Creole
python utils/inference-test-OmniVoice.py \
    --language "Haitian Creole" \
    --output_dir synthesis_output/OmniVoice/haitian-creole \
    --metadata_path "F5-TTS/data/open-bible-haitian creole/metadata.csv"

# Hausa
python utils/inference-test-OmniVoice.py \
    --language Hausa \
    --output_dir synthesis_output/OmniVoice/hausa \
    --metadata_path F5-TTS/data/open-bible-hausa/metadata.csv

# Hiligaynon
# python utils/inference-test-OmniVoice.py \
#     --language Hiligaynon \
#     --output_dir synthesis_output/OmniVoice/hiligaynon \
#     --metadata_path F5-TTS/data/open-bible-hiligaynon/metadata.csv

# Hindi
python utils/inference-test-OmniVoice.py \
    --language Hindi \
    --output_dir synthesis_output/OmniVoice/hindi \
    --metadata_path F5-TTS/data/open-bible-hindi/metadata.csv

# Igbo
# python utils/inference-test-OmniVoice.py \
#     --language Igbo \
#     --output_dir synthesis_output/OmniVoice/igbo \
#     --metadata_path F5-TTS/data/open-bible-igbo/metadata.csv

# Kannada
# python utils/inference-test-OmniVoice.py \
#     --language Kannada \
#     --output_dir synthesis_output/OmniVoice/kannada \
#     --metadata_path F5-TTS/data/open-bible-kannada/metadata.csv

# Kikuyu
# python utils/inference-test-OmniVoice.py \
#     --language Kikuyu \
#     --output_dir synthesis_output/OmniVoice/kikuyu \
#     --metadata_path F5-TTS/data/open-bible-kikuyu/metadata.csv

# Lingala
# python utils/inference-test-OmniVoice.py \
#     --language Lingala \
#     --output_dir synthesis_output/OmniVoice/lingala \
#     --metadata_path F5-TTS/data/open-bible-lingala/metadata.csv

# Luganda
# python utils/inference-test-OmniVoice.py \
#     --language Luganda \
#     --output_dir synthesis_output/OmniVoice/luganda \
#     --metadata_path F5-TTS/data/open-bible-luganda/metadata.csv

# Malayalam
# python utils/inference-test-OmniVoice.py \
#     --language Malayalam \
#     --output_dir synthesis_output/OmniVoice/malayalam \
#     --metadata_path F5-TTS/data/open-bible-malayalam/metadata.csv

# Marathi
# python utils/inference-test-OmniVoice.py \
#     --language Marathi \
#     --output_dir synthesis_output/OmniVoice/marathi \
#     --metadata_path F5-TTS/data/open-bible-marathi/metadata.csv

# Ndebele
# python utils/inference-test-OmniVoice.py \
#     --language Ndebele \
#     --output_dir synthesis_output/OmniVoice/ndebele \
#     --metadata_path F5-TTS/data/open-bible-ndebele/metadata.csv

# Nepali
# python utils/inference-test-OmniVoice.py \
#     --language Nepali \
#     --output_dir synthesis_output/OmniVoice/nepali \
#     --metadata_path F5-TTS/data/open-bible-nepali/metadata.csv

# Oromo
python utils/inference-test-OmniVoice.py \
    --language Oromo \
    --output_dir synthesis_output/OmniVoice/oromo \
    --metadata_path F5-TTS/data/open-bible-oromo/metadata.csv

# Punjabi
# python utils/inference-test-OmniVoice.py \
#     --language Punjabi \
#     --output_dir synthesis_output/OmniVoice/punjabi \
#     --metadata_path F5-TTS/data/open-bible-punjabi/metadata.csv

# Shona
python utils/inference-test-OmniVoice.py \
    --language Shona \
    --output_dir synthesis_output/OmniVoice/shona \
    --metadata_path F5-TTS/data/open-bible-shona/metadata.csv

# Swahili
python utils/inference-test-OmniVoice.py \
    --language Swahili \
    --output_dir synthesis_output/OmniVoice/swahili \
    --metadata_path F5-TTS/data/open-bible-swahili/metadata.csv

# Tamil
# python utils/inference-test-OmniVoice.py \
#     --language Tamil \
#     --output_dir synthesis_output/OmniVoice/tamil \
#     --metadata_path F5-TTS/data/open-bible-tamil/metadata.csv

# Telugu
python utils/inference-test-OmniVoice.py \
    --language Telugu \
    --output_dir synthesis_output/OmniVoice/telugu \
    --metadata_path F5-TTS/data/open-bible-telugu/metadata.csv

# Turkish
python utils/inference-test-OmniVoice.py \
    --language Turkish \
    --output_dir synthesis_output/OmniVoice/turkish \
    --metadata_path F5-TTS/data/open-bible-turkish/metadata.csv

# Twi (Akuapem)
# python utils/inference-test-OmniVoice.py \
#     --language "Twi (Akuapem)" \
#     --output_dir synthesis_output/OmniVoice/twi-akuapem \
#     --metadata_path "F5-TTS/data/open-bible-twi (akuapem)/metadata.csv"

# Twi (Asante)
# python utils/inference-test-OmniVoice.py \
#     --language "Twi (Asante)" \
#     --output_dir synthesis_output/OmniVoice/twi-asante \
#     --metadata_path "F5-TTS/data/open-bible-twi (asante)/metadata.csv"

# Urdu
# python utils/inference-test-OmniVoice.py \
#     --language Urdu \
#     --output_dir synthesis_output/OmniVoice/urdu \
#     --metadata_path F5-TTS/data/open-bible-urdu/metadata.csv

# Vietnamese
python utils/inference-test-OmniVoice.py \
    --language Vietnamese \
    --output_dir synthesis_output/OmniVoice/vietnamese \
    --metadata_path F5-TTS/data/open-bible-vietnamese/metadata.csv

# Yoruba
python utils/inference-test-OmniVoice.py \
    --language Yoruba \
    --output_dir synthesis_output/OmniVoice/yoruba \
    --metadata_path F5-TTS/data/open-bible-yoruba/metadata.csv

##################################################################
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
