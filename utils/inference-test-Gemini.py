"""
Batch synthesis with the Gemini TTS API for a target language.

Authentication: GOOGLE_API_KEY (or GEMINI_API_KEY) must be set in the environment.

Example:
    export GOOGLE_API_KEY=...
    python inference-Gemini-open-bible.py \\
        --language Igbo \\
        --output_dir audios/open-bible/Gemini/Igbo
"""

import argparse
import os
import time
import wave
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


GEMINI_VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--language", required=True,
        help="Language name as used in the HuggingFace dataset and as the 'Read this in <language>:' prompt prefix.",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Directory where generated WAVs are saved.",
    )
    p.add_argument(
        "--model", default="gemini-2.5-pro-preview-tts",
        help="Gemini TTS model id (default: gemini-2.5-pro-preview-tts).",
    )
    p.add_argument(
        "--voice", default="Kore", choices=GEMINI_VOICES,
        help="Prebuilt voice name (default: Kore).",
    )
    p.add_argument(
        "--sample_rate", type=int, default=24000,
        help="Gemini returns 24kHz raw PCM (default: 24000).",
    )
    p.add_argument(
        "--sample_width", type=int, default=2,
        help="16-bit PCM = 2 bytes (default: 2).",
    )
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--backoff_base", type=float, default=2.0)
    p.add_argument(
        "--throttle_s", type=float, default=0.5,
        help="Sleep between successful API calls (quota pacing).",
    )
    p.add_argument(
        "--head", type=int, default=None, metavar="N",
        help="Only synthesize the first N samples (useful for quick tests).",
    )
    return p.parse_args()


def pcm_to_wav(pcm: bytes, out_path: Path, sample_rate: int, sample_width: int) -> None:
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def call_gemini(client, model: str, voice: str, text: str):
    from google.genai import types
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
    )
    resp = client.models.generate_content(model=model, contents=text, config=cfg)
    cand = resp.candidates[0]
    if cand.content is None or not cand.content.parts:
        # Surface why: SAFETY, RECITATION, OTHER, MAX_TOKENS, or LANGUAGE
        reason = getattr(cand, "finish_reason", "UNKNOWN")
        raise RuntimeError(f"Gemini refused (finish_reason={reason})")
    return cand.content.parts[0].inline_data


def synth_with_retry(client, model, voice, text, max_retries, backoff_base):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return call_gemini(client, model, voice, text)
        except Exception as exc:
            last_exc = exc
            time.sleep(backoff_base ** attempt)
    raise last_exc


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: GOOGLE_API_KEY (or GEMINI_API_KEY) env var must be set.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading test set for language: {args.language}")
    ds = load_dataset(
        "parquet",
        data_files={
            "test": f"hf://datasets/davidguzmanr/open-bible-resources/{args.language}/test-*.parquet"
        },
        split="test",
    )
    n = min(500, len(ds)) if args.head is None else min(args.head, len(ds))
    subset = ds.select(range(n))
    print(f"Test samples (total): {len(ds)}, synthesizing: {n}")
    print(f"Model={args.model}  voice={args.voice}  language={args.language}")
    print(f"Output -> {output_dir}")

    from google import genai
    client = genai.Client(api_key=api_key)

    completed = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    for row in tqdm(subset, total=n, desc="Synthesising"):
        out_filename = f"{row['testament']}-{row['book']}-{row['chapter']}-{row['verse']}.wav"
        out_path = output_dir / out_filename
        if out_path.exists():
            skipped += 1
            continue

        prompt = f"Read this in {args.language}: {row['text']}"
        try:
            blob = synth_with_retry(client, args.model, args.voice, prompt,
                                    args.max_retries, args.backoff_base)
        except Exception as exc:
            failed.append((out_filename, str(exc)))
            continue

        pcm_to_wav(blob.data, out_path, args.sample_rate, args.sample_width)
        completed += 1
        if args.throttle_s > 0:
            time.sleep(args.throttle_s)

    print(f"\nDone. {completed} generated, {skipped} already existed, {len(failed)} failed.")
    if failed:
        print("First few failures:")
        for stem, err in failed[:10]:
            print(f"  - {stem}: {err}")


if __name__ == "__main__":
    main()

