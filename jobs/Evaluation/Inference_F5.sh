#!/usr/bin/env bash
#SBATCH --job-name=Inference_F5
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
# Local checkpoints
##################################################################
# Haitian Creole
# Hiligaynon
# Igbo
# Vietnamese
# Yoruba

##################################################################
# HuggingFace checkpoints
##################################################################

# Download all HuggingFace checkpoints (model_last.pt, vocab.txt)
python - <<'EOF'
import os, shutil
from huggingface_hub import hf_hub_download

BASE     = "/home/mila/g/guzmand/scratch/Repositories/open-bible-models"
CKPT_DIR = os.path.join(BASE, "F5-TTS/ckpts")
DATA_DIR = os.path.join(BASE, "F5-TTS/data")

# (hf_slug, data_slug, ckpt_slug)
languages = [
    ("Arabic-Standard", "arabic standard", "arabic-standard"),
    ("Assamese", "assamese", "assamese"),
    ("Bengali", "bengali", "bengali"),
    ("Central-Kurdish", "central kurdish", "central-kurdish"),
    ("Chhattisgarhi", "chhattisgarhi", "chhattisgarhi"),
    ("Chichewa", "chichewa", "chichewa"),
    ("Dawro", "dawro", "dawro"),
    ("Dholuo", "dholuo", "dholuo"),
    ("Ewe", "ewe", "ewe"),
    ("Gamo", "gamo", "gamo"),
    ("Gofa", "gofa", "gofa"),
    ("Gujarati", "gujarati", "gujarati"),
    ("Hausa", "hausa", "hausa"),
    ("Hindi", "hindi", "hindi"),
    ("Kannada", "kannada", "kannada"),
    ("Kikuyu", "kikuyu", "kikuyu"),
    ("Lingala", "lingala", "lingala"),
    ("Luganda", "luganda", "luganda"),
    ("Malayalam", "malayalam", "malayalam"),
    ("Marathi", "marathi", "marathi"),
    ("Ndebele", "ndebele", "ndebele"),
    ("Nepali", "nepali", "nepali"),
    ("Oromo", "oromo", "oromo"),
    ("Punjabi", "punjabi", "punjabi"),
    ("Shona", "shona", "shona"),
    ("Swahili", "swahili", "swahili"),
    ("Tamil", "tamil", "tamil"),
    ("Telugu", "telugu", "telugu"),
    ("Turkish", "turkish", "turkish"),
    ("Twi-Akuapem", "twi (akuapem)", "twi-akuapem"),
    ("Twi-Asante", "twi (asante)", "twi-asante"),
    ("Urdu", "urdu", "urdu"),
]

for hf_slug, data_slug, ckpt_slug in languages:
    repo_id = f"multilingual-tts/F5-TTS-OpenBible-{hf_slug}"
    print(f"\nDownloading {repo_id} ...")

    out_ckpt = os.path.join(CKPT_DIR, f"F5TTS_v1_Base_vocos_custom_open-bible-{ckpt_slug}")
    os.makedirs(out_ckpt, exist_ok=True)

    dst = os.path.join(out_ckpt, "model_last.pt")
    if not os.path.exists(dst):
        shutil.copy2(hf_hub_download(repo_id, "model_last.pt"), dst)
        print(f"  model_last.pt -> {dst}")
    else:
        print(f"  model_last.pt already present, skipping")

    out_vocab = os.path.join(DATA_DIR, f"open-bible-{data_slug}_custom")
    os.makedirs(out_vocab, exist_ok=True)
    dst = os.path.join(out_vocab, "vocab.txt")
    if not os.path.exists(dst):
        shutil.copy2(hf_hub_download(repo_id, "vocab.txt"), dst)
        print(f"  vocab.txt      -> {dst}")
    else:
        print(f"  vocab.txt already present, skipping")

print("\nAll downloads complete!")
EOF

##################################################################
# Run inference for all languages
##################################################################
# Arabic Standard
python utils/inference-test-F5.py \
    --language "Arabic Standard" \
    --output_dir synthesis_output/F5/arabic-standard \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-arabic-standard/model_last.pt \
    --vocab_file "F5-TTS/data/open-bible-arabic standard_custom/vocab.txt" \
    --model_cfg "F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Arabic Standard.yaml" \
    --metadata_path "F5-TTS/data/open-bible-arabic standard/metadata.csv"

# Assamese
python utils/inference-test-F5.py \
    --language Assamese \
    --output_dir synthesis_output/F5/assamese \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-assamese/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-assamese_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Assamese.yaml \
    --metadata_path F5-TTS/data/open-bible-assamese/metadata.csv

# Bengali
python utils/inference-test-F5.py \
    --language Bengali \
    --output_dir synthesis_output/F5/bengali \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-bengali/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-bengali_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Bengali.yaml \
    --metadata_path F5-TTS/data/open-bible-bengali/metadata.csv

# Central Kurdish
python utils/inference-test-F5.py \
    --language "Central Kurdish" \
    --output_dir synthesis_output/F5/central-kurdish \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-central-kurdish/model_last.pt \
    --vocab_file "F5-TTS/data/open-bible-central kurdish_custom/vocab.txt" \
    --model_cfg "F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Central Kurdish.yaml" \
    --metadata_path "F5-TTS/data/open-bible-central kurdish/metadata.csv"

# Chhattisgarhi
python utils/inference-test-F5.py \
    --language Chhattisgarhi \
    --output_dir synthesis_output/F5/chhattisgarhi \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-chhattisgarhi/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-chhattisgarhi_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Chhattisgarhi.yaml \
    --metadata_path F5-TTS/data/open-bible-chhattisgarhi/metadata.csv

# Chichewa
python utils/inference-test-F5.py \
    --language Chichewa \
    --output_dir synthesis_output/F5/chichewa \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-chichewa/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-chichewa_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Chichewa.yaml \
    --metadata_path F5-TTS/data/open-bible-chichewa/metadata.csv

# Dawro
python utils/inference-test-F5.py \
    --language Dawro \
    --output_dir synthesis_output/F5/dawro \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-dawro/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-dawro_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Dawro.yaml \
    --metadata_path F5-TTS/data/open-bible-dawro/metadata.csv

# Dholuo
python utils/inference-test-F5.py \
    --language Dholuo \
    --output_dir synthesis_output/F5/dholuo \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-dholuo/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-dholuo_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Dholuo.yaml \
    --metadata_path F5-TTS/data/open-bible-dholuo/metadata.csv

# Ewe
python utils/inference-test-F5.py \
    --language Ewe \
    --output_dir synthesis_output/F5/ewe \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-ewe/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-ewe_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Ewe.yaml \
    --metadata_path F5-TTS/data/open-bible-ewe/metadata.csv

# Gamo
python utils/inference-test-F5.py \
    --language Gamo \
    --output_dir synthesis_output/F5/gamo \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-gamo/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-gamo_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Gamo.yaml \
    --metadata_path F5-TTS/data/open-bible-gamo/metadata.csv

# Gofa
python utils/inference-test-F5.py \
    --language Gofa \
    --output_dir synthesis_output/F5/gofa \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-gofa/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-gofa_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Gofa.yaml \
    --metadata_path F5-TTS/data/open-bible-gofa/metadata.csv

# Gujarati
python utils/inference-test-F5.py \
    --language Gujarati \
    --output_dir synthesis_output/F5/gujarati \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-gujarati/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-gujarati_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Gujarati.yaml \
    --metadata_path F5-TTS/data/open-bible-gujarati/metadata.csv

# Haitian Creole
python utils/inference-test-F5.py \
    --language "Haitian Creole" \
    --output_dir synthesis_output/F5/haitian-creole \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-haitian-creole/model_last.pt \
    --vocab_file "F5-TTS/data/open-bible-haitian creole_custom/vocab.txt" \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Haitian-Creole.yaml \
    --metadata_path "F5-TTS/data/open-bible-haitian creole/metadata.csv"

# Hausa
python utils/inference-test-F5.py \
    --language Hausa \
    --output_dir synthesis_output/F5/hausa \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-hausa/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-hausa_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Hausa.yaml \
    --metadata_path F5-TTS/data/open-bible-hausa/metadata.csv

# Hiligaynon
python utils/inference-test-F5.py \
    --language Hiligaynon \
    --output_dir synthesis_output/F5/hiligaynon \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-hiligaynon/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-hiligaynon_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Hiligaynon.yaml \
    --metadata_path F5-TTS/data/open-bible-hiligaynon/metadata.csv

# Hindi
python utils/inference-test-F5.py \
    --language Hindi \
    --output_dir synthesis_output/F5/hindi \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-hindi/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-hindi_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Hindi.yaml \
    --metadata_path F5-TTS/data/open-bible-hindi/metadata.csv

# Igbo
python utils/inference-test-F5.py \
    --language Igbo \
    --output_dir synthesis_output/F5/igbo \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-igbo/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-igbo_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Igbo.yaml \
    --metadata_path F5-TTS/data/open-bible-igbo/metadata.csv

# Kannada
python utils/inference-test-F5.py \
    --language Kannada \
    --output_dir synthesis_output/F5/kannada \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-kannada/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-kannada_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Kannada.yaml \
    --metadata_path F5-TTS/data/open-bible-kannada/metadata.csv

# Kikuyu
python utils/inference-test-F5.py \
    --language Kikuyu \
    --output_dir synthesis_output/F5/kikuyu \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-kikuyu/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-kikuyu_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Kikuyu.yaml \
    --metadata_path F5-TTS/data/open-bible-kikuyu/metadata.csv

# Lingala
python utils/inference-test-F5.py \
    --language Lingala \
    --output_dir synthesis_output/F5/lingala \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-lingala/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-lingala_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Lingala.yaml \
    --metadata_path F5-TTS/data/open-bible-lingala/metadata.csv

# Luganda
python utils/inference-test-F5.py \
    --language Luganda \
    --output_dir synthesis_output/F5/luganda \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-luganda/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-luganda_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Luganda.yaml \
    --metadata_path F5-TTS/data/open-bible-luganda/metadata.csv

# Malayalam
python utils/inference-test-F5.py \
    --language Malayalam \
    --output_dir synthesis_output/F5/malayalam \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-malayalam/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-malayalam_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Malayalam.yaml \
    --metadata_path F5-TTS/data/open-bible-malayalam/metadata.csv

# Marathi
python utils/inference-test-F5.py \
    --language Marathi \
    --output_dir synthesis_output/F5/marathi \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-marathi/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-marathi_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Marathi.yaml \
    --metadata_path F5-TTS/data/open-bible-marathi/metadata.csv

# Ndebele
python utils/inference-test-F5.py \
    --language Ndebele \
    --output_dir synthesis_output/F5/ndebele \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-ndebele/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-ndebele_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Ndebele.yaml \
    --metadata_path F5-TTS/data/open-bible-ndebele/metadata.csv

# Nepali
python utils/inference-test-F5.py \
    --language Nepali \
    --output_dir synthesis_output/F5/nepali \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-nepali/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-nepali_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Nepali.yaml \
    --metadata_path F5-TTS/data/open-bible-nepali/metadata.csv

# Oromo
python utils/inference-test-F5.py \
    --language Oromo \
    --output_dir synthesis_output/F5/oromo \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-oromo/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-oromo_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Oromo.yaml \
    --metadata_path F5-TTS/data/open-bible-oromo/metadata.csv

# Punjabi
python utils/inference-test-F5.py \
    --language Punjabi \
    --output_dir synthesis_output/F5/punjabi \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-punjabi/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-punjabi_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Punjabi.yaml \
    --metadata_path F5-TTS/data/open-bible-punjabi/metadata.csv

# Shona
python utils/inference-test-F5.py \
    --language Shona \
    --output_dir synthesis_output/F5/shona \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-shona/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-shona_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Shona.yaml \
    --metadata_path F5-TTS/data/open-bible-shona/metadata.csv

# Swahili
python utils/inference-test-F5.py \
    --language Swahili \
    --output_dir synthesis_output/F5/swahili \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-swahili/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-swahili_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Swahili.yaml \
    --metadata_path F5-TTS/data/open-bible-swahili/metadata.csv

# Tamil
python utils/inference-test-F5.py \
    --language Tamil \
    --output_dir synthesis_output/F5/tamil \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-tamil/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-tamil_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Tamil.yaml \
    --metadata_path F5-TTS/data/open-bible-tamil/metadata.csv

# Telugu
python utils/inference-test-F5.py \
    --language Telugu \
    --output_dir synthesis_output/F5/telugu \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-telugu/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-telugu_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Telugu.yaml \
    --metadata_path F5-TTS/data/open-bible-telugu/metadata.csv

# Turkish
python utils/inference-test-F5.py \
    --language Turkish \
    --output_dir synthesis_output/F5/turkish \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-turkish/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-turkish_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Turkish.yaml \
    --metadata_path F5-TTS/data/open-bible-turkish/metadata.csv

# Twi (Akuapem)
python utils/inference-test-F5.py \
    --language "Twi (Akuapem)" \
    --output_dir synthesis_output/F5/twi-akuapem \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-twi-akuapem/model_last.pt \
    --vocab_file "F5-TTS/data/open-bible-twi (akuapem)_custom/vocab.txt" \
    --model_cfg "F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Twi (Akuapem).yaml" \
    --metadata_path "F5-TTS/data/open-bible-twi (akuapem)/metadata.csv"

# Twi (Asante)
python utils/inference-test-F5.py \
    --language "Twi (Asante)" \
    --output_dir synthesis_output/F5/twi-asante \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-twi-asante/model_last.pt \
    --vocab_file "F5-TTS/data/open-bible-twi (asante)_custom/vocab.txt" \
    --model_cfg "F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Twi (Asante).yaml" \
    --metadata_path "F5-TTS/data/open-bible-twi (asante)/metadata.csv"

# Urdu
python utils/inference-test-F5.py \
    --language Urdu \
    --output_dir synthesis_output/F5/urdu \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-urdu/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-urdu_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Urdu.yaml \
    --metadata_path F5-TTS/data/open-bible-urdu/metadata.csv

# Vietnamese
python utils/inference-test-F5.py \
    --language Vietnamese \
    --output_dir synthesis_output/F5/vietnamese \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_pinyin_open-bible-vietnamese/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-vietnamese_pinyin/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Vietnamese.yaml \
    --metadata_path F5-TTS/data/open-bible-vietnamese/metadata.csv

# Yoruba
python utils/inference-test-F5.py \
    --language Yoruba \
    --output_dir synthesis_output/F5/yoruba \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-yoruba/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-yoruba_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Yoruba.yaml \
    --metadata_path F5-TTS/data/open-bible-yoruba/metadata.csv

##################################################################
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
