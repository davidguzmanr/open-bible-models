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
    ("Malayalam", "malayalam", "malayalam"),
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
# Malayalam
python utils/inference-test-F5.py \
    --language Malayalam \
    --output_dir synthesis_output/F5/malayalam \
    --ckpt_path F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_open-bible-malayalam/model_last.pt \
    --vocab_file F5-TTS/data/open-bible-malayalam_custom/vocab.txt \
    --model_cfg F5-TTS/src/f5_tts/configs/F5TTS_v1_Base_Open_Bible_Malayalam.yaml \
    --metadata_path F5-TTS/data/open-bible-malayalam/metadata.csv