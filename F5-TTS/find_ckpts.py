import argparse
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", type=str, required=True)
args = parser.parse_args()

skip_pattern = re.compile(r"model_(?:last|250000)\.pt$")

for pt_file in sorted(Path(args.dir_path).rglob("*.pt")):
    if not skip_pattern.search(pt_file.name):
        print(pt_file)
