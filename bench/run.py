"""Transcribe a prepared benchmark set with one model; results are appended (resumable).

    python run.py maya      --device cpu       # maya-research/maya-transcribe, style: mixed
    python run.py tara      --device mps       # Trelis/tara, <|hi|><|mixedcode|>
    python run.py minutes:superwhisper         # any registered `minutes` ASR backend

Output: results/hyps/<model>.jsonl with {clip, hyp, audio_s, wall_s, load_s}.
Runs identically on the laptop (cpu) and the Mac mini (mps).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def pick_dtype(torch, device: str, name: str | None, accel_default: str):
    name = name or ("float32" if device == "cpu" else accel_default)
    return getattr(torch, name)


class Maya:
    repo = "maya-research/maya-transcribe"

    def __init__(self, device: str, dtype_name: str | None = None):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        dtype = pick_dtype(torch, device, dtype_name, "bfloat16")
        self.processor = AutoProcessor.from_pretrained(self.repo, trust_remote_code=False)
        self.model = AutoModelForMultimodalLM.from_pretrained(self.repo, dtype=dtype, trust_remote_code=False).to(device)
        self.model.eval()

    def __call__(self, audio) -> str:
        inputs = self.processor.apply_transcription_request(
            audio=audio, prompt="style: mixed", language="Hindi").to(self.model.device, self.model.dtype)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        return self.processor.decode(out[:, inputs["input_ids"].shape[1]:], return_format="transcription_only")[0]


class Tara:
    repo = "Trelis/tara"

    def __init__(self, device: str, dtype_name: str | None = None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch = torch
        self.device = device
        self.dtype = pick_dtype(torch, device, dtype_name, "float16")
        self.processor = WhisperProcessor.from_pretrained(self.repo)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.repo, dtype=self.dtype).to(device)
        self.model.eval()
        tk = self.processor.tokenizer
        ids = [tk.convert_tokens_to_ids(t) for t in
               ("<|startoftranscript|>", "<|hi|>", "<|mixedcode|>", "<|transcribe|>", "<|notimestamps|>")]
        self.prompt = torch.tensor([ids], device=device)

    def __call__(self, audio) -> str:
        from transformers.generation.utils import GenerationMixin

        feats = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(self.device, self.dtype)
        with self.torch.inference_mode():
            # Plain greedy decoding from our own prompt (Whisper's generate() rebuilds prompts itself).
            out = GenerationMixin.generate(self.model, input_features=feats, decoder_input_ids=self.prompt,
                                           max_new_tokens=440, do_sample=False, num_beams=1)
        return self.processor.tokenizer.decode(out[0, self.prompt.shape[1]:], skip_special_tokens=True)


def _ov_config() -> dict:
    """OV_PRECISION=f32 forces fp32 inference (OpenVINO's GPU default is f16)."""
    prec = os.environ.get("OV_PRECISION")
    return {"INFERENCE_PRECISION_HINT": prec} if prec else {}


class TaraOV:
    """Trelis/tara exported to OpenVINO IR (cached in ov-models/), run on an OpenVINO device (CPU / GPU)."""
    repo = "Trelis/tara"

    def __init__(self, device: str, dtype_name: str | None = None):
        import torch
        from optimum.intel import OVModelForSpeechSeq2Seq
        from transformers import WhisperProcessor

        self.torch = torch
        weights = dtype_name or "fp16"  # fp16 | int8 (weight compression)
        cache = HERE / "ov-models" / f"tara-{weights}"
        self.processor = WhisperProcessor.from_pretrained(self.repo)
        if not (cache / "openvino_encoder_model.xml").exists():
            kw = {"load_in_8bit": True} if weights == "int8" else {"load_in_8bit": False}
            m = OVModelForSpeechSeq2Seq.from_pretrained(self.repo, export=True, compile=False, **kw)
            m.save_pretrained(cache)
            del m
        ov_device = {"cpu": "CPU", "gpu": "GPU", "xpu": "GPU"}.get(device.lower(), device.upper())
        self.model = OVModelForSpeechSeq2Seq.from_pretrained(cache, device=ov_device, ov_config=_ov_config())
        tk = self.processor.tokenizer
        ids = [tk.convert_tokens_to_ids(x) for x in
               ("<|startoftranscript|>", "<|hi|>", "<|mixedcode|>", "<|transcribe|>", "<|notimestamps|>")]
        self.prompt = torch.tensor([ids])

    def __call__(self, audio) -> str:
        from transformers.generation.utils import GenerationMixin

        feats = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        out = GenerationMixin.generate(self.model, input_features=feats, decoder_input_ids=self.prompt,
                                       max_new_tokens=440, do_sample=False, num_beams=1)
        return self.processor.tokenizer.decode(out[0, self.prompt.shape[1]:], skip_special_tokens=True)


class MayaOV:
    """maya-transcribe ported to the original Qwen3-ASR layout (port/convert.py) and exported to
    OpenVINO IR (cached in ov-models/). Needs the .venv-ovmaya env (transformers 4.57.6 + qwen_asr)."""
    PROMPT = ("<|im_start|>system\nstyle: mixed<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>"
              "<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage Hindi<asr_text>")  # = maya's processor

    def __init__(self, device: str, dtype_name: str | None = None):
        import torch
        import qwen_asr  # noqa: F401  registers qwen3_asr with transformers 4.57
        from optimum.intel import OVModelForSpeechSeq2Seq
        from qwen_asr.core.transformers_backend.processing_qwen3_asr import Qwen3ASRProcessor

        self.torch = torch
        src = HERE / "port" / "maya-orig"
        if not (src / "model.safetensors").exists():
            raise SystemExit("run port/convert.py first")
        weights = dtype_name or "fp16"
        cache = HERE / "ov-models" / f"maya-{weights}"
        if not (cache / "openvino_encoder_model.xml").exists():
            m = OVModelForSpeechSeq2Seq.from_pretrained(src, export=True, compile=False, trust_remote_code=True,
                                                        load_in_8bit=(weights == "int8"))
            m.save_pretrained(cache)
            del m
        ov_device = {"cpu": "CPU", "gpu": "GPU", "xpu": "GPU"}.get(device.lower(), device.upper())
        self.model = OVModelForSpeechSeq2Seq.from_pretrained(cache, device=ov_device, trust_remote_code=True, ov_config=_ov_config())
        self.processor = Qwen3ASRProcessor.from_pretrained(src)

    def __call__(self, audio) -> str:
        inp = self.processor(text=[self.PROMPT], audio=[audio], return_tensors="pt", padding=True)
        prompt_ids = inp["input_ids"]
        out = self.model.generate(input_features=inp["input_features"], attention_mask=inp["feature_attention_mask"],
                                  decoder_input_ids=prompt_ids, max_new_tokens=512, do_sample=False, num_beams=1,
                                  # The exported model loses generation_config.json (export warns), so pass
                                  # maya's stop tokens explicitly: <|endoftext|>, <|im_end|>.
                                  eos_token_id=[151643, 151645], pad_token_id=151645)
        text = self.processor.tokenizer.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        return text.split("<asr_text>")[-1].strip()


class MinutesBackend:
    def __init__(self, name: str):
        sys.path.insert(0, str(HERE.parent / "src"))
        from minutes import asr
        from minutes.config import Config

        self.t = asr.get(name, Config(storage_dir=Path("/nonexistent")))

    def __call__(self, path: Path) -> str:
        return " ".join(s.text for s in self.t.transcribe(path, "hi"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="maya | tara | tara-ov | maya-ov | minutes:<backend>")
    ap.add_argument("--data", type=Path, default=HERE / "data/coshe")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dtype", help="float32 | float16 | bfloat16 (tara-ov: fp16 | int8)")
    args = ap.parse_args()

    manifest = [json.loads(l) for l in (args.data / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.limit:
        manifest = manifest[: args.limit]
    tag = args.model.replace(":", "-") + ("" if args.model.startswith("minutes:") else f"-{args.device}")
    if args.dtype:
        tag += f"-{args.dtype}"
    if os.environ.get("OV_PRECISION") and args.model.endswith("-ov"):
        tag += "-" + os.environ["OV_PRECISION"]
    out = HERE / "results/hyps" / f"{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["clip"] for l in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    todo = [m for m in manifest if m["clip"] not in done]
    print(f"{tag}: {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return

    t0 = time.perf_counter()
    if args.model.startswith("minutes:"):
        model, wants_path = MinutesBackend(args.model.split(":", 1)[1]), True
    else:
        if args.threads:
            import torch
            torch.set_num_threads(args.threads)
        model, wants_path = {"maya": Maya, "tara": Tara, "tara-ov": TaraOV, "maya-ov": MayaOV}[args.model](args.device, args.dtype), False
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.1f}s", flush=True)

    if not wants_path:
        import soundfile as sf
    for i, m in enumerate(todo, 1):
        t = time.perf_counter()
        parts = []
        for chunk in sorted((args.data / m["clip"]).glob("*.flac")):
            if wants_path:
                parts.append(model(chunk))
            else:
                audio, sr = sf.read(chunk, dtype="float32")
                assert sr == 16000
                parts.append(model(audio))
        wall = time.perf_counter() - t
        rec = {"clip": m["clip"], "hyp": " ".join(p.strip() for p in parts if p.strip()),
               "audio_s": m["duration_s"], "wall_s": round(wall, 2), "load_s": round(load_s, 1)}
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(todo)}] {m['clip']} {m['duration_s']:.0f}s audio in {wall:.1f}s "
              f"(RTF {wall / m['duration_s']:.2f})", flush=True)


if __name__ == "__main__":
    main()
