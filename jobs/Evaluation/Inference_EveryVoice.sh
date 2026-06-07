#!/usr/bin/env bash
#SBATCH --job-name=Inference_EveryVoice
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

conda activate EveryVoice

echo "NVCC version:"
nvcc --version
echo "NVIDIA SMI:"
nvidia-smi
echo $HF_HOME

##################################################################
# Inference
##################################################################
cd /home/mila/g/guzmand/scratch/Repositories/open-bible-models

BASE="EveryVoice-TTS/ckpts"

##################################################################
# Local checkpoints
##################################################################
# None — all EveryVoice Open Bible models were trained on other clusters.

##################################################################
# HuggingFace checkpoints
##################################################################

# Download all HuggingFace checkpoints (feature_prediction.ckpt, vocoder.ckpt, filelist.psv)
python - <<'EOF'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

BASE = "/home/mila/g/guzmand/scratch/Repositories/open-bible-models"
CKPT_DIR = os.path.join(BASE, "EveryVoice-TTS/ckpts")

# (language_name, hf_slug, output_slug)
languages = [
    ("Arabic Standard",  "Arabic-Standard",  "arabic-standard"),
    ("Assamese",         "Assamese",         "assamese"),
    ("Bengali",          "Bengali",          "bengali"),
    ("Central Kurdish",  "Central-Kurdish",  "central-kurdish"),
    ("Chhattisgarhi",    "Chhattisgarhi",    "chhattisgarhi"),
    ("Chichewa",         "Chichewa",         "chichewa"),
    ("Dawro",            "Dawro",            "dawro"),
    ("Dholuo",           "Dholuo",           "dholuo"),
    ("Ewe",              "Ewe",              "ewe"),
    ("Gamo",             "Gamo",             "gamo"),
    ("Gofa",             "Gofa",             "gofa"),
    ("Gujarati",         "Gujarati",         "gujarati"),
    ("Haitian Creole",   "Haitian-Creole",   "haitian-creole"),
    ("Hausa",            "Hausa",            "hausa"),
    ("Hiligaynon",       "Hiligaynon",       "hiligaynon"),
    ("Hindi",            "Hindi",            "hindi"),
    ("Igbo",             "Igbo",             "igbo"),
    ("Kannada",          "Kannada",          "kannada"),
    ("Kikuyu",           "Kikuyu",           "kikuyu"),
    ("Luganda",          "Luganda",          "luganda"),
    ("Malayalam",        "Malayalam",        "malayalam"),
    ("Marathi",          "Marathi",          "marathi"),
    ("Ndebele",          "Ndebele",          "ndebele"),
    ("Nepali",           "Nepali",           "nepali"),
    ("Oromo",            "Oromo",            "oromo"),
    ("Punjabi",          "Punjabi",          "punjabi"),
    ("Shona",            "Shona",            "shona"),
    ("Swahili",          "Swahili",          "swahili"),
    ("Tamil",            "Tamil",            "tamil"),
    ("Telugu",           "Telugu",           "telugu"),
    ("Turkish",          "Turkish",          "turkish"),
    ("Twi (Akuapem)",    "Twi-Akuapem",      "twi-akuapem"),
    ("Twi (Asante)",     "Twi-Asante",       "twi-asante"),
    ("Urdu",             "Urdu",             "urdu"),
    ("Vietnamese",       "Vietnamese",       "vietnamese"),
    ("Yoruba",           "Yoruba",           "yoruba"),
]

for _language_name, hf_slug, output_slug in languages:
    repo_id = f"multilingual-tts/EveryVoice-OpenBible-{hf_slug}"
    local = Path(CKPT_DIR) / output_slug
    marker = local / "feature_prediction.ckpt"

    print(f"\n{repo_id} -> {local}")
    if marker.is_file():
        print("  already present, skipping download")
        continue

    snapshot_download(repo_id, local_dir=str(local))
    print("  download complete")

print("\nAll downloads complete!")
EOF

# Arabic Standard
python utils/inference-test-EveryVoice.py \
    --language "Arabic Standard" \
    --output_dir synthesis_output/EveryVoice/arabic-standard \
    --ckpt_path ${BASE}/arabic-standard/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/arabic-standard/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-arabic standard/metadata.csv"

# Assamese
python utils/inference-test-EveryVoice.py \
    --language Assamese \
    --output_dir synthesis_output/EveryVoice/assamese \
    --ckpt_path ${BASE}/assamese/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/assamese/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-assamese/metadata.csv"

# Bengali
python utils/inference-test-EveryVoice.py \
    --language Bengali \
    --output_dir synthesis_output/EveryVoice/bengali \
    --ckpt_path ${BASE}/bengali/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/bengali/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-bengali/metadata.csv"

# Central Kurdish
python utils/inference-test-EveryVoice.py \
    --language "Central Kurdish" \
    --output_dir synthesis_output/EveryVoice/central-kurdish \
    --ckpt_path ${BASE}/central-kurdish/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/central-kurdish/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-central kurdish/metadata.csv"

# Chhattisgarhi
python utils/inference-test-EveryVoice.py \
    --language Chhattisgarhi \
    --output_dir synthesis_output/EveryVoice/chhattisgarhi \
    --ckpt_path ${BASE}/chhattisgarhi/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/chhattisgarhi/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-chhattisgarhi/metadata.csv"

# Chichewa
python utils/inference-test-EveryVoice.py \
    --language Chichewa \
    --output_dir synthesis_output/EveryVoice/chichewa \
    --ckpt_path ${BASE}/chichewa/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/chichewa/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-chichewa/metadata.csv"

# Dawro
python utils/inference-test-EveryVoice.py \
    --language Dawro \
    --output_dir synthesis_output/EveryVoice/dawro \
    --ckpt_path ${BASE}/dawro/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/dawro/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-dawro/metadata.csv"

# Dholuo
python utils/inference-test-EveryVoice.py \
    --language Dholuo \
    --output_dir synthesis_output/EveryVoice/dholuo \
    --ckpt_path ${BASE}/dholuo/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/dholuo/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-dholuo/metadata.csv"

# Ewe
python utils/inference-test-EveryVoice.py \
    --language Ewe \
    --output_dir synthesis_output/EveryVoice/ewe \
    --ckpt_path ${BASE}/ewe/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/ewe/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-ewe/metadata.csv"

# Gamo
python utils/inference-test-EveryVoice.py \
    --language Gamo \
    --output_dir synthesis_output/EveryVoice/gamo \
    --ckpt_path ${BASE}/gamo/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/gamo/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-gamo/metadata.csv"

# Gofa
python utils/inference-test-EveryVoice.py \
    --language Gofa \
    --output_dir synthesis_output/EveryVoice/gofa \
    --ckpt_path ${BASE}/gofa/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/gofa/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-gofa/metadata.csv"

# Gujarati
python utils/inference-test-EveryVoice.py \
    --language Gujarati \
    --output_dir synthesis_output/EveryVoice/gujarati \
    --ckpt_path ${BASE}/gujarati/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/gujarati/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-gujarati/metadata.csv"

# Haitian Creole
python utils/inference-test-EveryVoice.py \
    --language "Haitian Creole" \
    --output_dir synthesis_output/EveryVoice/haitian-creole \
    --ckpt_path ${BASE}/haitian-creole/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/haitian-creole/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-haitian creole/metadata.csv"

# Hausa
python utils/inference-test-EveryVoice.py \
    --language Hausa \
    --output_dir synthesis_output/EveryVoice/hausa \
    --ckpt_path ${BASE}/hausa/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/hausa/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-hausa/metadata.csv"

# Hiligaynon
python utils/inference-test-EveryVoice.py \
    --language Hiligaynon \
    --output_dir synthesis_output/EveryVoice/hiligaynon \
    --ckpt_path ${BASE}/hiligaynon/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/hiligaynon/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-hiligaynon/metadata.csv"

# Hindi
python utils/inference-test-EveryVoice.py \
    --language Hindi \
    --output_dir synthesis_output/EveryVoice/hindi \
    --ckpt_path ${BASE}/hindi/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/hindi/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-hindi/metadata.csv"

# Igbo
python utils/inference-test-EveryVoice.py \
    --language Igbo \
    --output_dir synthesis_output/EveryVoice/igbo \
    --ckpt_path ${BASE}/igbo/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/igbo/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-igbo/metadata.csv"

# Kannada
python utils/inference-test-EveryVoice.py \
    --language Kannada \
    --output_dir synthesis_output/EveryVoice/kannada \
    --ckpt_path ${BASE}/kannada/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/kannada/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-kannada/metadata.csv"

# Kikuyu
python utils/inference-test-EveryVoice.py \
    --language Kikuyu \
    --output_dir synthesis_output/EveryVoice/kikuyu \
    --ckpt_path ${BASE}/kikuyu/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/kikuyu/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-kikuyu/metadata.csv"

# Luganda
python utils/inference-test-EveryVoice.py \
    --language Luganda \
    --output_dir synthesis_output/EveryVoice/luganda \
    --ckpt_path ${BASE}/luganda/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/luganda/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-luganda/metadata.csv"

# Malayalam
python utils/inference-test-EveryVoice.py \
    --language Malayalam \
    --output_dir synthesis_output/EveryVoice/malayalam \
    --ckpt_path ${BASE}/malayalam/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/malayalam/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-malayalam/metadata.csv"

# Marathi
python utils/inference-test-EveryVoice.py \
    --language Marathi \
    --output_dir synthesis_output/EveryVoice/marathi \
    --ckpt_path ${BASE}/marathi/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/marathi/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-marathi/metadata.csv"

# Ndebele
python utils/inference-test-EveryVoice.py \
    --language Ndebele \
    --output_dir synthesis_output/EveryVoice/ndebele \
    --ckpt_path ${BASE}/ndebele/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/ndebele/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-ndebele/metadata.csv"

# Nepali
python utils/inference-test-EveryVoice.py \
    --language Nepali \
    --output_dir synthesis_output/EveryVoice/nepali \
    --ckpt_path ${BASE}/nepali/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/nepali/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-nepali/metadata.csv"

# Oromo
python utils/inference-test-EveryVoice.py \
    --language Oromo \
    --output_dir synthesis_output/EveryVoice/oromo \
    --ckpt_path ${BASE}/oromo/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/oromo/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-oromo/metadata.csv"

# Punjabi
python utils/inference-test-EveryVoice.py \
    --language Punjabi \
    --output_dir synthesis_output/EveryVoice/punjabi \
    --ckpt_path ${BASE}/punjabi/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/punjabi/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-punjabi/metadata.csv"

# Shona
python utils/inference-test-EveryVoice.py \
    --language Shona \
    --output_dir synthesis_output/EveryVoice/shona \
    --ckpt_path ${BASE}/shona/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/shona/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-shona/metadata.csv"

# Swahili
python utils/inference-test-EveryVoice.py \
    --language Swahili \
    --output_dir synthesis_output/EveryVoice/swahili \
    --ckpt_path ${BASE}/swahili/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/swahili/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-swahili/metadata.csv"

# Tamil
python utils/inference-test-EveryVoice.py \
    --language Tamil \
    --output_dir synthesis_output/EveryVoice/tamil \
    --ckpt_path ${BASE}/tamil/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/tamil/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-tamil/metadata.csv"

# Telugu
python utils/inference-test-EveryVoice.py \
    --language Telugu \
    --output_dir synthesis_output/EveryVoice/telugu \
    --ckpt_path ${BASE}/telugu/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/telugu/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-telugu/metadata.csv"

# Turkish
python utils/inference-test-EveryVoice.py \
    --language Turkish \
    --output_dir synthesis_output/EveryVoice/turkish \
    --ckpt_path ${BASE}/turkish/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/turkish/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-turkish/metadata.csv"

# Twi (Akuapem)
python utils/inference-test-EveryVoice.py \
    --language "Twi (Akuapem)" \
    --output_dir synthesis_output/EveryVoice/twi-akuapem \
    --ckpt_path ${BASE}/twi-akuapem/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/twi-akuapem/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-twi (akuapem)/metadata.csv"

# Twi (Asante)
python utils/inference-test-EveryVoice.py \
    --language "Twi (Asante)" \
    --output_dir synthesis_output/EveryVoice/twi-asante \
    --ckpt_path ${BASE}/twi-asante/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/twi-asante/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-twi (asante)/metadata.csv"

# Urdu
python utils/inference-test-EveryVoice.py \
    --language Urdu \
    --output_dir synthesis_output/EveryVoice/urdu \
    --ckpt_path ${BASE}/urdu/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/urdu/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-urdu/metadata.csv"

# Vietnamese
python utils/inference-test-EveryVoice.py \
    --language Vietnamese \
    --output_dir synthesis_output/EveryVoice/vietnamese \
    --ckpt_path ${BASE}/vietnamese/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/vietnamese/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-vietnamese/metadata.csv"

# Yoruba
python utils/inference-test-EveryVoice.py \
    --language Yoruba \
    --output_dir synthesis_output/EveryVoice/yoruba \
    --ckpt_path ${BASE}/yoruba/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/yoruba/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-yoruba/metadata.csv"

# Lingala — no EveryVoice-OpenBible checkpoint on HuggingFace yet.

##################################################################
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
