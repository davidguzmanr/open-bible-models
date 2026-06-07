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
    ("Bengali", "bengali"),
    ("Dawro", "dawro"),
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
# Bengali
# python utils/inference-test-VITS.py \
#     --language      Bengali \
#     --output_dir    synthesis_output/VITS/bengali \
#     --ckpt_path     VITS-TTS/ckpts/bengali/model_last.pth \
#     --metadata_path F5-TTS/data/open-bible-bengali/metadata.csv

# Dawro
python utils/inference-test-VITS.py \
    --language      Dawro \
    --output_dir    synthesis_output/VITS/dawro \
    --ckpt_path     VITS-TTS/ckpts/dawro/model_last.pth \
    --metadata_path F5-TTS/data/open-bible-dawro/metadata.csv

##################################################################
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
