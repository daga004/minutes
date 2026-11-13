"""Score every results/hyps/*.jsonl against the manifest: WER, CER, speed, per category.

Normalisation (same for all models and references): Unicode NFC, lowercase, punctuation
(incl. । ॥) removed, whitespace collapsed. A model that writes an English word in Devanagari
("कंप्लेन" for "complain") is counted wrong: the target is mixed-script output.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import jiwer

HERE = Path(__file__).resolve().parent


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    s = "".join(" " if unicodedata.category(c).startswith("P") or c in "।॥" else c for c in s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "data/coshe")
    ap.add_argument("--md", type=Path, default=HERE / "results/scores.md")
    args = ap.parse_args()
    man = {m["clip"]: m for m in map(json.loads, (args.data / "manifest.jsonl").read_text(encoding="utf-8").splitlines())}

    rows = []
    for f in sorted((HERE / "results/hyps").glob("*.jsonl")):
        hyps = {h["clip"]: h for h in map(json.loads, f.read_text(encoding="utf-8").splitlines())}
        clips = [c for c in man if c in hyps]
        if not clips:
            continue
        def score(sel):
            refs = [norm(man[c]["reference"]) for c in sel]
            hs = [norm(hyps[c]["hyp"]) for c in sel]
            return (100 * jiwer.wer(refs, hs), 100 * jiwer.cer(refs, hs)) if sel else (float("nan"),) * 2
        wer, cer = score(clips)
        cat = {k: score([c for c in clips if man[c]["category"] == k])[0] for k in ("mixed", "hindi", "english")}
        audio = sum(hyps[c]["audio_s"] for c in clips)
        wall = sum(hyps[c]["wall_s"] for c in clips)
        rows.append((f.stem, len(clips), wer, cer, cat["mixed"], cat["hindi"], cat["english"], wall / audio))

    head = "| model | clips | WER % | CER % | WER mixed | WER hindi | WER english | RTF |\n|---|---|---|---|---|---|---|---|\n"
    body = "".join(f"| {r[0]} | {r[1]} | {r[2]:.1f} | {r[3]:.1f} | {r[4]:.1f} | {r[5]:.1f} | {r[6]:.1f} | {r[7]:.2f} |\n"
                   for r in sorted(rows, key=lambda r: r[2]))
    table = head + body
    args.md.write_text("# CoSHE-Eval subset scores\n\nRTF = processing time / audio time (lower is faster; <1 is faster than real time).\n\n" + table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
