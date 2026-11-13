"""Convert maya-research/maya-transcribe (transformers-native Qwen3-ASR layout) back to the
original Qwen/Qwen3-ASR checkpoint layout, so optimum-intel (which only exports the original
layout, via the `qwen_asr` package) can produce an OpenVINO model for Intel GPUs.

Mapping (verified: all 611 tensors map, no shape mismatches):
    model.audio_tower.*                    -> thinker.audio_tower.*
    model.multi_modal_projector.linear_1.* -> thinker.audio_tower.proj1.*
    model.multi_modal_projector.linear_2.* -> thinker.audio_tower.proj2.*
    model.language_model.*                 -> thinker.model.*
    (tied)                                 -> thinker.lm_head.weight = embed_tokens

Non-weight files (config, tokenizer, processor, chat template) come from Qwen/Qwen3-ASR-0.6B;
the tokenizer is verified identical to maya's.

    python port/convert.py MAYA_SNAPSHOT_DIR port/maya-orig
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

RULES = [
    (r"^model\.multi_modal_projector\.linear_1\.", "thinker.audio_tower.proj1."),
    (r"^model\.multi_modal_projector\.linear_2\.", "thinker.audio_tower.proj2."),
    (r"^model\.audio_tower\.", "thinker.audio_tower."),
    (r"^model\.language_model\.", "thinker.model."),
]


def rename(key: str) -> str:
    for pat, repl in RULES:
        new, n = re.subn(pat, repl, key)
        if n:
            return new
    raise KeyError(f"no mapping for {key}")


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    sd = load_file(str(src / "model.safetensors"))
    new = {rename(k): v for k, v in sd.items()}
    new["thinker.lm_head.weight"] = new["thinker.model.embed_tokens.weight"].clone()
    out.mkdir(parents=True, exist_ok=True)
    save_file(new, str(out / "model.safetensors"), metadata={"format": "pt"})
    print(f"wrote {len(new)} tensors to {out / 'model.safetensors'}")


if __name__ == "__main__":
    main()
