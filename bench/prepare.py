"""Build a benchmark set from CoSHE-Eval parquet shards (soketlabs/CoSHE-Eval, CC-BY-NC-4.0).

Each clip is converted to 16 kHz mono FLAC and cut into <=29 s chunks exactly like `minutes`
records them, so models are scored the way they will be used. References are kept whole.

    python prepare.py PARQUET [PARQUET...] --out data/coshe --n 30 --mixed-share 0.7
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

DEVA = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")


def category(text: str) -> str:
    d, l = bool(DEVA.search(text)), bool(LATIN.search(text))
    return "mixed" if d and l else "hindi" if d else "english"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/coshe"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--mixed-share", type=float, default=0.7)
    ap.add_argument("--chunk-seconds", type=int, default=29)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = [r for p in args.parquet for r in pq.read_table(p).to_pylist()]
    by_cat: dict[str, list] = {"mixed": [], "hindi": [], "english": []}
    for r in rows:
        by_cat[category(r["transcription"])].append(r)
    print("available:", {k: len(v) for k, v in by_cat.items()})

    rng = random.Random(args.seed)
    n_mixed = min(len(by_cat["mixed"]), round(args.n * args.mixed_share))
    rest = args.n - n_mixed
    n_hindi = min(len(by_cat["hindi"]), rest // 2 + rest % 2)
    n_english = min(len(by_cat["english"]), rest - n_hindi)
    picked = (rng.sample(by_cat["mixed"], n_mixed) + rng.sample(by_cat["hindi"], n_hindi)
              + rng.sample(by_cat["english"], n_english))

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for r in sorted(picked, key=lambda r: r["audio_file_name"]):
        clip = Path(r["audio_file_name"]).stem
        cdir = args.out / clip
        cdir.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(r["audio"]["bytes"])
            tmp.flush()
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp.name, "-ac", "1", "-ar", "16000",
                            "-c:a", "flac", "-f", "segment", "-segment_time", str(args.chunk_seconds),
                            "-reset_timestamps", "1", str(cdir / "%05d.flac")], check=True)
            # The segment muxer leaves wrong FLAC headers (unknown or whole-file length), which
            # libsndfile trusts; re-encode each chunk so its STREAMINFO is correct.
            for chunk in sorted(cdir.glob("*.flac")):
                fixed = chunk.with_suffix(".tmp.flac")
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(chunk), "-c:a", "flac", str(fixed)], check=True)
                fixed.replace(chunk)
            dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                        "csv=p=0", tmp.name], capture_output=True, text=True).stdout)
        manifest.append({"clip": clip, "category": category(r["transcription"]), "duration_s": round(dur, 2),
                         "chunks": len(list(cdir.glob("*.flac"))), "reference": r["transcription"]})
    with (args.out / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for m in manifest:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    cats = {c: sum(1 for m in manifest if m["category"] == c) for c in by_cat}
    print(f"wrote {len(manifest)} clips, {sum(m['duration_s'] for m in manifest) / 60:.1f} min, {cats} -> {args.out}")


if __name__ == "__main__":
    main()
