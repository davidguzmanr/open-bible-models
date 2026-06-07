##################################################################
# HuggingFace checkpoints
##################################################################
BASE="EveryVoice-TTS/ckpts"

# Download all HuggingFace checkpoints (feature_prediction.ckpt, vocoder.ckpt, filelist.psv)
python - <<'EOF'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

BASE = "/home/mila/g/guzmand/scratch/Repositories/open-bible-models"
CKPT_DIR = os.path.join(BASE, "EveryVoice-TTS/ckpts")

# (language_name, hf_slug, output_slug)
languages = [
    ("Lingala",          "Lingala",          "lingala"),
    ("Malayalam",        "Malayalam",        "malayalam"),
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

# Lingala
python utils/inference-test-EveryVoice.py \
    --language Lingala \
    --output_dir synthesis_output/EveryVoice/lingala \
    --ckpt_path ${BASE}/lingala/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/lingala/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-lingala/metadata.csv"

# Malayalam
python utils/inference-test-EveryVoice.py \
    --language Malayalam \
    --output_dir synthesis_output/EveryVoice/malayalam \
    --ckpt_path ${BASE}/malayalam/feature_prediction.ckpt \
    --vocoder_ckpt_path ${BASE}/malayalam/vocoder.ckpt \
    --filelist_path "F5-TTS/data/open-bible-malayalam/metadata.csv"