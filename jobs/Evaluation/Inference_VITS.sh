#!/usr/bin/env bash
#SBATCH --job-name=Inference_VITS
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
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

##################################################################
# Local checkpoints
##################################################################
# None — all VITS Open Bible models are downloaded from HuggingFace.

##################################################################
# HuggingFace checkpoints
##################################################################

# Download all HuggingFace checkpoints (model_last.pth, config.json, speakers.pth)
python - <<'EOF'
import os, shutil
from huggingface_hub import hf_hub_download

BASE     = "/home/mila/g/guzmand/scratch/Repositories/open-bible-models"
CKPT_DIR = os.path.join(BASE, "VITS-TTS/ckpts")

# (hf_slug, ckpt_slug)
languages = [
    ("Arabic-Standard", "arabic-standard"),
    ("Assamese", "assamese"),
    ("Bengali", "bengali"),
    ("Central-Kurdish", "central-kurdish"),
    ("Chhattisgarhi", "chhattisgarhi"),
    ("Chichewa", "chichewa"),
    ("Dawro", "dawro"),
    ("Dholuo", "dholuo"),
    ("Ewe", "ewe"),
    ("Gamo", "gamo"),
    ("Gofa", "gofa"),
    ("Gujarati", "gujarati"),
    ("Haitian-Creole", "haitian-creole"),
    ("Hausa", "hausa"),
    ("Hiligaynon", "hiligaynon"),
    ("Hindi", "hindi"),
    ("Igbo", "igbo"),
    ("Kannada", "kannada"),
    ("Kikuyu", "kikuyu"),
    ("Lingala", "lingala"),
    ("Luganda", "luganda"),
    ("Malayalam", "malayalam"),
    ("Marathi", "marathi"),
    ("Ndebele", "ndebele"),
    ("Nepali", "nepali"),
    ("Oromo", "oromo"),
    ("Punjabi", "punjabi"),
    ("Shona", "shona"),
    ("Swahili", "swahili"),
    ("Tamil", "tamil"),
    ("Telugu", "telugu"),
    ("Turkish", "turkish"),
    ("Twi-Akuapem", "twi-akuapem"),
    ("Twi-Asante", "twi-asante"),
    ("Urdu", "urdu"),
    ("Vietnamese", "vietnamese"),
    ("Yoruba", "yoruba"),
]

files = ("model_last.pth", "config.json", "speakers.pth")

for hf_slug, ckpt_slug in languages:
    repo_id = f"multilingual-tts/VITS-OpenBible-{hf_slug}"
    out_dir = os.path.join(CKPT_DIR, ckpt_slug)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{repo_id} -> {out_dir}")

    for filename in files:
        dst = os.path.join(out_dir, filename)
        if os.path.exists(dst):
            print(f"  {filename} already present, skipping")
            continue
        shutil.copy2(hf_hub_download(repo_id, filename), dst)
        print(f"  {filename} -> {dst}")

print("\nAll downloads complete!")
EOF

##################################################################
# Run inference for all languages
##################################################################
# Arabic Standard
python utils/inference-test-VITS.py \
    --language      "Arabic Standard" \
    --output_dir    synthesis_output/VITS/arabic-standard \
    --ckpt_path     VITS-TTS/ckpts/arabic-standard/model_last.pth \
    --metadata_path "F5-TTS/data/open-bible-arabic standard/metadata.csv"

# Assamese
python utils/inference-test-VITS.py \
    --language      Assamese \
    --output_dir    synthesis_output/VITS/assamese \
    --ckpt_path     VITS-TTS/ckpts/assamese/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-assamese/metadata.csv

# Bengali
python utils/inference-test-VITS.py \
    --language      Bengali \
    --output_dir    synthesis_output/VITS/bengali \
    --ckpt_path     VITS-TTS/ckpts/bengali/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-bengali/metadata.csv

# Central Kurdish
python utils/inference-test-VITS.py \
    --language      "Central Kurdish" \
    --output_dir    synthesis_output/VITS/central-kurdish \
    --ckpt_path     VITS-TTS/ckpts/central-kurdish/model_last.pth \
    --metadata_path "F5-TTS/data/open-bible-central kurdish/metadata.csv"

# Chhattisgarhi
python utils/inference-test-VITS.py \
    --language      Chhattisgarhi \
    --output_dir    synthesis_output/VITS/chhattisgarhi \
    --ckpt_path     VITS-TTS/ckpts/chhattisgarhi/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-chhattisgarhi/metadata.csv

# Chichewa
python utils/inference-test-VITS.py \
    --language      Chichewa \
    --output_dir    synthesis_output/VITS/chichewa \
    --ckpt_path     VITS-TTS/ckpts/chichewa/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-chichewa/metadata.csv

# Dawro
python utils/inference-test-VITS.py \
    --language      Dawro \
    --output_dir    synthesis_output/VITS/dawro \
    --ckpt_path     VITS-TTS/ckpts/dawro/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-dawro/metadata.csv

# Dholuo
python utils/inference-test-VITS.py \
    --language      Dholuo \
    --output_dir    synthesis_output/VITS/dholuo \
    --ckpt_path     VITS-TTS/ckpts/dholuo/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-dholuo/metadata.csv

# Ewe
python utils/inference-test-VITS.py \
    --language      Ewe \
    --output_dir    synthesis_output/VITS/ewe \
    --ckpt_path     VITS-TTS/ckpts/ewe/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-ewe/metadata.csv

# Gamo
python utils/inference-test-VITS.py \
    --language      Gamo \
    --output_dir    synthesis_output/VITS/gamo \
    --ckpt_path     VITS-TTS/ckpts/gamo/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-gamo/metadata.csv

# Gofa
python utils/inference-test-VITS.py \
    --language      Gofa \
    --output_dir    synthesis_output/VITS/gofa \
    --ckpt_path     VITS-TTS/ckpts/gofa/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-gofa/metadata.csv

# Gujarati
python utils/inference-test-VITS.py \
    --language      Gujarati \
    --output_dir    synthesis_output/VITS/gujarati \
    --ckpt_path     VITS-TTS/ckpts/gujarati/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-gujarati/metadata.csv

# Haitian Creole
python utils/inference-test-VITS.py \
    --language      "Haitian Creole" \
    --output_dir    synthesis_output/VITS/haitian-creole \
    --ckpt_path     VITS-TTS/ckpts/haitian-creole/model_last.pth \
    --metadata_path "F5-TTS/data/open-bible-haitian creole/metadata.csv"

# Hausa
python utils/inference-test-VITS.py \
    --language      Hausa \
    --output_dir    synthesis_output/VITS/hausa \
    --ckpt_path     VITS-TTS/ckpts/hausa/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-hausa/metadata.csv

# Hiligaynon
python utils/inference-test-VITS.py \
    --language      Hiligaynon \
    --output_dir    synthesis_output/VITS/hiligaynon \
    --ckpt_path     VITS-TTS/ckpts/hiligaynon/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-hiligaynon/metadata.csv

# Hindi
python utils/inference-test-VITS.py \
    --language      Hindi \
    --output_dir    synthesis_output/VITS/hindi \
    --ckpt_path     VITS-TTS/ckpts/hindi/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-hindi/metadata.csv

# Igbo
python utils/inference-test-VITS.py \
    --language      Igbo \
    --output_dir    synthesis_output/VITS/igbo \
    --ckpt_path     VITS-TTS/ckpts/igbo/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-igbo/metadata.csv

# Kannada
python utils/inference-test-VITS.py \
    --language      Kannada \
    --output_dir    synthesis_output/VITS/kannada \
    --ckpt_path     VITS-TTS/ckpts/kannada/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-kannada/metadata.csv

# Kikuyu
python utils/inference-test-VITS.py \
    --language      Kikuyu \
    --output_dir    synthesis_output/VITS/kikuyu \
    --ckpt_path     VITS-TTS/ckpts/kikuyu/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-kikuyu/metadata.csv

# Lingala
python utils/inference-test-VITS.py \
    --language      Lingala \
    --output_dir    synthesis_output/VITS/lingala \
    --ckpt_path     VITS-TTS/ckpts/lingala/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-lingala/metadata.csv

# Luganda
python utils/inference-test-VITS.py \
    --language      Luganda \
    --output_dir    synthesis_output/VITS/luganda \
    --ckpt_path     VITS-TTS/ckpts/luganda/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-luganda/metadata.csv

# Malayalam
python utils/inference-test-VITS.py \
    --language      Malayalam \
    --output_dir    synthesis_output/VITS/malayalam \
    --ckpt_path     VITS-TTS/ckpts/malayalam/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-malayalam/metadata.csv

# Marathi
python utils/inference-test-VITS.py \
    --language      Marathi \
    --output_dir    synthesis_output/VITS/marathi \
    --ckpt_path     VITS-TTS/ckpts/marathi/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-marathi/metadata.csv

# Ndebele
python utils/inference-test-VITS.py \
    --language      Ndebele \
    --output_dir    synthesis_output/VITS/ndebele \
    --ckpt_path     VITS-TTS/ckpts/ndebele/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-ndebele/metadata.csv

# Nepali
python utils/inference-test-VITS.py \
    --language      Nepali \
    --output_dir    synthesis_output/VITS/nepali \
    --ckpt_path     VITS-TTS/ckpts/nepali/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-nepali/metadata.csv

# Oromo
python utils/inference-test-VITS.py \
    --language      Oromo \
    --output_dir    synthesis_output/VITS/oromo \
    --ckpt_path     VITS-TTS/ckpts/oromo/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-oromo/metadata.csv

# Punjabi
python utils/inference-test-VITS.py \
    --language      Punjabi \
    --output_dir    synthesis_output/VITS/punjabi \
    --ckpt_path     VITS-TTS/ckpts/punjabi/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-punjabi/metadata.csv

# Shona
python utils/inference-test-VITS.py \
    --language      Shona \
    --output_dir    synthesis_output/VITS/shona \
    --ckpt_path     VITS-TTS/ckpts/shona/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-shona/metadata.csv

# Swahili
python utils/inference-test-VITS.py \
    --language      Swahili \
    --output_dir    synthesis_output/VITS/swahili \
    --ckpt_path     VITS-TTS/ckpts/swahili/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-swahili/metadata.csv

# Tamil
python utils/inference-test-VITS.py \
    --language      Tamil \
    --output_dir    synthesis_output/VITS/tamil \
    --ckpt_path     VITS-TTS/ckpts/tamil/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-tamil/metadata.csv

# Telugu
python utils/inference-test-VITS.py \
    --language      Telugu \
    --output_dir    synthesis_output/VITS/telugu \
    --ckpt_path     VITS-TTS/ckpts/telugu/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-telugu/metadata.csv

# Turkish
python utils/inference-test-VITS.py \
    --language      Turkish \
    --output_dir    synthesis_output/VITS/turkish \
    --ckpt_path     VITS-TTS/ckpts/turkish/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-turkish/metadata.csv

# Twi (Akuapem)
python utils/inference-test-VITS.py \
    --language      "Twi (Akuapem)" \
    --output_dir    synthesis_output/VITS/twi-akuapem \
    --ckpt_path     VITS-TTS/ckpts/twi-akuapem/model_last.pth \
    --metadata_path "F5-TTS/data/open-bible-twi (akuapem)/metadata.csv"

# Twi (Asante)
python utils/inference-test-VITS.py \
    --language      "Twi (Asante)" \
    --output_dir    synthesis_output/VITS/twi-asante \
    --ckpt_path     VITS-TTS/ckpts/twi-asante/model_last.pth \
    --metadata_path "F5-TTS/data/open-bible-twi (asante)/metadata.csv"

# Urdu
python utils/inference-test-VITS.py \
    --language      Urdu \
    --output_dir    synthesis_output/VITS/urdu \
    --ckpt_path     VITS-TTS/ckpts/urdu/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-urdu/metadata.csv

# Vietnamese
python utils/inference-test-VITS.py \
    --language      Vietnamese \
    --output_dir    synthesis_output/VITS/vietnamese \
    --ckpt_path     VITS-TTS/ckpts/vietnamese/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-vietnamese/metadata.csv

# Yoruba
python utils/inference-test-VITS.py \
    --language      Yoruba \
    --output_dir    synthesis_output/VITS/yoruba \
    --ckpt_path     VITS-TTS/ckpts/yoruba/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-yoruba/metadata.csv

##################################################################
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
