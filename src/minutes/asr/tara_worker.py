"""Tara (Trelis/tara, Whisper fine-tuned for Hindi-English code-mixing) on OpenVINO.

Runs inside an engine venv, never imported by `minutes`. Self-contained so it can be copied to
another machine (a remote NVIDIA box) and run over ssh:

    python -I tara_worker.py --model-dir DIR --device GPU --cache-dir DIR      # OpenVINO (Intel)
    python -I tara_worker.py --engine torch --device cuda                      # PyTorch (NVIDIA)
    python -I tara_worker.py --model-dir DIR --export                          # export OpenVINO once

Protocol (JSON lines): after loading it prints {"ready": true, "device": ...}; then for each
request {"pcm_b64": <16 kHz mono int16 LE, base64>, "lang": ...} (or {"path": ...}, decoded with
ffmpeg) it prints {"text": ..., "duration": ...} or {"error": ...}.
All library chatter goes to stderr; only protocol lines go to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO = "Trelis/tara"
SR = 16000
WINDOW = 30 * SR  # Whisper sees at most 30 s at a time


def compression_ratio(text: str) -> float:
    """Whisper's hallucination test: a looping transcript compresses far better than speech."""
    import zlib

    data = text.encode("utf-8")
    return len(data) / max(1, len(zlib.compress(data))) if data else 0.0


def load_audio(path: str):
    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, check=True, timeout=120,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def export(model_dir: str) -> None:
    from optimum.intel import OVModelForSpeechSeq2Seq
    from transformers import WhisperProcessor

    m = OVModelForSpeechSeq2Seq.from_pretrained(REPO, export=True, compile=False, load_in_8bit=False)
    m.save_pretrained(model_dir)
    WhisperProcessor.from_pretrained(REPO).save_pretrained(model_dir)


class TaraTorch:
    """Same model and decoding as the OpenVINO path, on PyTorch (CUDA fp16). Weights come from the
    Hugging Face cache (Trelis/tara); bench run tara-cuda: 20.0% WER."""

    def __init__(self, device: str):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.dev = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.processor = WhisperProcessor.from_pretrained(REPO)
        self.model = WhisperForConditionalGeneration.from_pretrained(REPO, dtype=self.dtype).to(device).eval()
        self.tk = self.processor.tokenizer
        self.device = torch.cuda.get_device_name(0) if device.startswith("cuda") else device

    prompt = None  # set below (shared with Tara)

    def generate(self, feats, prompt, **kw):
        from transformers.generation.utils import GenerationMixin

        with self.torch.inference_mode():
            return GenerationMixin.generate(self.model, input_features=feats.to(self.dev, self.dtype),
                                            decoder_input_ids=prompt.to(self.dev), max_new_tokens=440,
                                            do_sample=False, num_beams=1, **kw)


class Tara:
    def __init__(self, model_dir: str, device: str, cache_dir: str | None):
        import torch
        from optimum.intel import OVModelForSpeechSeq2Seq
        from transformers import WhisperProcessor

        self.torch = torch
        self.processor = WhisperProcessor.from_pretrained(model_dir)
        ov_config = {"CACHE_DIR": cache_dir} if cache_dir else {}
        try:
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(model_dir, device=device, ov_config=ov_config)
            self.device = device
        except Exception as e:  # e.g. no GPU driver: fall back to CPU rather than fail the meeting
            if device.upper() == "CPU":
                raise
            print(f"tara: {device} failed ({e}); falling back to CPU", file=sys.stderr)
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(model_dir, device="CPU", ov_config=ov_config)
            self.device = "CPU"
        self.tk = self.processor.tokenizer

    def prompt(self, lang: str | None):
        lang = (lang or "hi").lower()
        tokens = ["<|startoftranscript|>", f"<|{lang}|>"]
        if lang == "hi":
            tokens.append("<|mixedcode|>")  # Tara's Hinglish mode (Hindi in Devanagari, English in Latin)
        tokens += ["<|transcribe|>", "<|notimestamps|>"]
        ids = [self.tk.convert_tokens_to_ids(t) for t in tokens]
        if self.tk.unk_token_id in ids:
            raise ValueError(f"language {lang!r} not supported by Tara")
        return self.torch.tensor([ids])

    def generate(self, feats, prompt, **kw):
        from transformers.generation.utils import GenerationMixin

        return GenerationMixin.generate(self.model, input_features=feats, decoder_input_ids=prompt,
                                        max_new_tokens=440, do_sample=False, num_beams=1, **kw)

    def __call__(self, audio, lang: str | None) -> str:
        prompt = self.prompt(lang)
        parts = []
        for start in range(0, max(len(audio), 1), WINDOW):
            piece = audio[start:start + WINDOW]
            if len(piece) < SR // 10:  # a sliver at the end of a 30.0x s chunk
                continue
            feats = self.processor(piece, sampling_rate=SR, return_tensors="pt").input_features
            out = self.generate(feats, prompt)
            text = self.tk.decode(out[0, prompt.shape[1]:], skip_special_tokens=True).strip()
            if compression_ratio(text) > 2.4:  # stuck in a loop: decode again with repeats blocked
                out = self.generate(feats, prompt, no_repeat_ngram_size=4, repetition_penalty=1.3)
                text = self.tk.decode(out[0, prompt.shape[1]:], skip_special_tokens=True).strip()
            parts.append(text)
        return " ".join(p for p in parts if p)


TaraTorch.prompt = Tara.prompt
TaraTorch.__call__ = Tara.__call__


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["openvino", "torch"], default="openvino")
    ap.add_argument("--model-dir")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--cache-dir")
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()

    # Keep stdout for protocol lines only; everything else (warnings, progress bars) -> stderr.
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def send(obj: dict) -> None:
        proto.write(json.dumps(obj, ensure_ascii=False) + "\n")
        proto.flush()

    if args.export:
        export(args.model_dir)
        send({"exported": args.model_dir})
        return

    import base64

    import numpy as np

    if args.engine == "torch":
        model = TaraTorch(args.device)
    else:
        model = Tara(args.model_dir, args.device, args.cache_dir)
    send({"ready": True, "device": model.device})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if "pcm_b64" in req:
                audio = np.frombuffer(base64.b64decode(req["pcm_b64"]), dtype="<i2").astype(np.float32) / 32768.0
            else:
                audio = load_audio(req["path"])
            send({"text": model(audio, req.get("lang")), "duration": round(len(audio) / SR, 2)})
        except Exception as e:
            send({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
