#!/usr/bin/env bash
#SBATCH --job-name=Evaluate_F5
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
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

conda activate TTS-Evaluation

echo "NVCC version:"
nvcc --version
echo "NVIDIA SMI:"
nvidia-smi
echo $HF_HOME

cd /home/mila/g/guzmand/scratch/Repositories/open-bible-models/

##################################################################
# Evaluation
##################################################################

BASE_DIR="synthesis_output/F5"
METRICS="utmos wer"

# Arabic Standard (arb_Arab)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/arabic-standard \
    --test_csv ${BASE_DIR}/arabic-standard/test.csv \
    --output_csv ${BASE_DIR}/arabic-standard/results.csv \
    --metrics ${METRICS} \
    --asr-lang arb_Arab

# Assamese (asm_Beng)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/assamese \
    --test_csv ${BASE_DIR}/assamese/test.csv \
    --output_csv ${BASE_DIR}/assamese/results.csv \
    --metrics ${METRICS} \
    --asr-lang asm_Beng

# Bengali (ben_Beng)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/bengali \
    --test_csv ${BASE_DIR}/bengali/test.csv \
    --output_csv ${BASE_DIR}/bengali/results.csv \
    --metrics ${METRICS} \
    --asr-lang ben_Beng

# Central Kurdish (ckb_Arab)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/central-kurdish \
    --test_csv ${BASE_DIR}/central-kurdish/test.csv \
    --output_csv ${BASE_DIR}/central-kurdish/results.csv \
    --metrics ${METRICS} \
    --asr-lang ckb_Arab

# Chhattisgarhi (hne_Deva)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/chhattisgarhi \
    --test_csv ${BASE_DIR}/chhattisgarhi/test.csv \
    --output_csv ${BASE_DIR}/chhattisgarhi/results.csv \
    --metrics ${METRICS} \
    --asr-lang hne_Deva

# Chichewa (nya_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/chichewa \
    --test_csv ${BASE_DIR}/chichewa/test.csv \
    --output_csv ${BASE_DIR}/chichewa/results.csv \
    --metrics ${METRICS} \
    --asr-lang nya_Latn

# Dawro (dwr_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/dawro \
    --test_csv ${BASE_DIR}/dawro/test.csv \
    --output_csv ${BASE_DIR}/dawro/results.csv \
    --metrics ${METRICS} \
    --asr-lang dwr_Latn

# Dholuo (luo_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/dholuo \
    --test_csv ${BASE_DIR}/dholuo/test.csv \
    --output_csv ${BASE_DIR}/dholuo/results.csv \
    --metrics ${METRICS} \
    --asr-lang luo_Latn

# Ewe (ewe_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/ewe \
    --test_csv ${BASE_DIR}/ewe/test.csv \
    --output_csv ${BASE_DIR}/ewe/results.csv \
    --metrics ${METRICS} \
    --asr-lang ewe_Latn

# Gamo (gmv_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/gamo \
    --test_csv ${BASE_DIR}/gamo/test.csv \
    --output_csv ${BASE_DIR}/gamo/results.csv \
    --metrics ${METRICS} \
    --asr-lang gmv_Latn

# Gofa (gof_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/gofa \
    --test_csv ${BASE_DIR}/gofa/test.csv \
    --output_csv ${BASE_DIR}/gofa/results.csv \
    --metrics ${METRICS} \
    --asr-lang gof_Latn

# Gujarati (guj_Gujr)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/gujarati \
    --test_csv ${BASE_DIR}/gujarati/test.csv \
    --output_csv ${BASE_DIR}/gujarati/results.csv \
    --metrics ${METRICS} \
    --asr-lang guj_Gujr

# Haitian Creole (hat_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/haitian-creole \
    --test_csv ${BASE_DIR}/haitian-creole/test.csv \
    --output_csv ${BASE_DIR}/haitian-creole/results.csv \
    --metrics ${METRICS} \
    --asr-lang hat_Latn

# Hausa (hau_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/hausa \
    --test_csv ${BASE_DIR}/hausa/test.csv \
    --output_csv ${BASE_DIR}/hausa/results.csv \
    --metrics ${METRICS} \
    --asr-lang hau_Latn

# Hiligaynon (hil_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/hiligaynon \
    --test_csv ${BASE_DIR}/hiligaynon/test.csv \
    --output_csv ${BASE_DIR}/hiligaynon/results.csv \
    --metrics ${METRICS} \
    --asr-lang hil_Latn

# Hindi (hin_Deva)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/hindi \
    --test_csv ${BASE_DIR}/hindi/test.csv \
    --output_csv ${BASE_DIR}/hindi/results.csv \
    --metrics ${METRICS} \
    --asr-lang hin_Deva

# Igbo (ibo_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/igbo \
    --test_csv ${BASE_DIR}/igbo/test.csv \
    --output_csv ${BASE_DIR}/igbo/results.csv \
    --metrics ${METRICS} \
    --asr-lang ibo_Latn

# Kannada (kan_Knda)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/kannada \
    --test_csv ${BASE_DIR}/kannada/test.csv \
    --output_csv ${BASE_DIR}/kannada/results.csv \
    --metrics ${METRICS} \
    --asr-lang kan_Knda

# Kikuyu (kik_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/kikuyu \
    --test_csv ${BASE_DIR}/kikuyu/test.csv \
    --output_csv ${BASE_DIR}/kikuyu/results.csv \
    --metrics ${METRICS} \
    --asr-lang kik_Latn

# Lingala (lin_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/lingala \
    --test_csv ${BASE_DIR}/lingala/test.csv \
    --output_csv ${BASE_DIR}/lingala/results.csv \
    --metrics ${METRICS} \
    --asr-lang lin_Latn

# Luganda (lug_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/luganda \
    --test_csv ${BASE_DIR}/luganda/test.csv \
    --output_csv ${BASE_DIR}/luganda/results.csv \
    --metrics ${METRICS} \
    --asr-lang lug_Latn

# Malayalam (mal_Mlym)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/malayalam \
    --test_csv ${BASE_DIR}/malayalam/test.csv \
    --output_csv ${BASE_DIR}/malayalam/results.csv \
    --metrics ${METRICS} \
    --asr-lang mal_Mlym

# Marathi (mar_Deva)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/marathi \
    --test_csv ${BASE_DIR}/marathi/test.csv \
    --output_csv ${BASE_DIR}/marathi/results.csv \
    --metrics ${METRICS} \
    --asr-lang mar_Deva

# Ndebele (nde_Latn) — WER skipped, nde_Latn not supported by omniASR
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/ndebele \
    --test_csv ${BASE_DIR}/ndebele/test.csv \
    --output_csv ${BASE_DIR}/ndebele/results.csv \
    --metrics utmos \
    --asr-lang nde_Latn

# Nepali (nep_Deva)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/nepali \
    --test_csv ${BASE_DIR}/nepali/test.csv \
    --output_csv ${BASE_DIR}/nepali/results.csv \
    --metrics ${METRICS} \
    --asr-lang nep_Deva

# Oromo (orm_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/oromo \
    --test_csv ${BASE_DIR}/oromo/test.csv \
    --output_csv ${BASE_DIR}/oromo/results.csv \
    --metrics ${METRICS} \
    --asr-lang orm_Latn

# Punjabi (pan_Guru)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/punjabi \
    --test_csv ${BASE_DIR}/punjabi/test.csv \
    --output_csv ${BASE_DIR}/punjabi/results.csv \
    --metrics ${METRICS} \
    --asr-lang pan_Guru

# Shona (sna_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/shona \
    --test_csv ${BASE_DIR}/shona/test.csv \
    --output_csv ${BASE_DIR}/shona/results.csv \
    --metrics ${METRICS} \
    --asr-lang sna_Latn

# Swahili (swh_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/swahili \
    --test_csv ${BASE_DIR}/swahili/test.csv \
    --output_csv ${BASE_DIR}/swahili/results.csv \
    --metrics ${METRICS} \
    --asr-lang swh_Latn

# Tamil (tam_Taml)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/tamil \
    --test_csv ${BASE_DIR}/tamil/test.csv \
    --output_csv ${BASE_DIR}/tamil/results.csv \
    --metrics ${METRICS} \
    --asr-lang tam_Taml

# Telugu (tel_Telu)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/telugu \
    --test_csv ${BASE_DIR}/telugu/test.csv \
    --output_csv ${BASE_DIR}/telugu/results.csv \
    --metrics ${METRICS} \
    --asr-lang tel_Telu

# Turkish (tur_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/turkish \
    --test_csv ${BASE_DIR}/turkish/test.csv \
    --output_csv ${BASE_DIR}/turkish/results.csv \
    --metrics ${METRICS} \
    --asr-lang tur_Latn

# Twi (Akuapem) (aka_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/twi-akuapem \
    --test_csv ${BASE_DIR}/twi-akuapem/test.csv \
    --output_csv ${BASE_DIR}/twi-akuapem/results.csv \
    --metrics ${METRICS} \
    --asr-lang aka_Latn

# Twi (Asante) (aka_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/twi-asante \
    --test_csv ${BASE_DIR}/twi-asante/test.csv \
    --output_csv ${BASE_DIR}/twi-asante/results.csv \
    --metrics ${METRICS} \
    --asr-lang aka_Latn

# Urdu (urd_Arab)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/urdu \
    --test_csv ${BASE_DIR}/urdu/test.csv \
    --output_csv ${BASE_DIR}/urdu/results.csv \
    --metrics ${METRICS} \
    --asr-lang hin_Deva

# Vietnamese (vie_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/vietnamese \
    --test_csv ${BASE_DIR}/vietnamese/test.csv \
    --output_csv ${BASE_DIR}/vietnamese/results.csv \
    --metrics ${METRICS} \
    --asr-lang vie_Latn

# Yoruba (yor_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/yoruba \
    --test_csv ${BASE_DIR}/yoruba/test.csv \
    --output_csv ${BASE_DIR}/yoruba/results.csv \
    --metrics ${METRICS} \
    --asr-lang yor_Latn

# Yoruba — fine-tuned Vocos vocoder (yor_Latn)
python utils/evaluate-tts.py \
    --synthesized_dir ${BASE_DIR}/yoruba-finetuned-vocoder \
    --test_csv ${BASE_DIR}/yoruba-finetuned-vocoder/test.csv \
    --output_csv ${BASE_DIR}/yoruba-finetuned-vocoder/results.csv \
    --metrics ${METRICS} \
    --asr-lang yor_Latn

##################################################################
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
