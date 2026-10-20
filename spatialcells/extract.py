"""Record MLP neuron activations for every walk frame (and teleport probe).

    python -m spatialcells.extract --model dinov2 [--random]
    python -m spatialcells.extract --model qwen|qwen8b|qwen32b|qwen38|sn8b|qwen8b-ft --mode single|walk|shuffled|probes-only [--random]

DINOv2: post-GELU MLP neurons at the CLS token, all 12 layers.
Qwen VLMs: SwiGLU MLP neurons (the input to down_proj) in 7 language-model layers
spread evenly through depth, read at the <|vision_end|> token that closes each
frame. Modes:
  single       every frame on its own (no memory of the walk)
  walk         the whole walk in one context, frames in order, then the walk's
               teleport probes (random places and headings, random order)
  shuffled     the same walk frames in random order, then the same probes
  probes-only  the probes with no walk before them
Because attention is causal, the walk frames' activations are unaffected by the
probes appended after them. Writes data/acts/<tag>/L<layer>.npy (float16,
n_frames x n_neurons); walk/shuffled also write <model>-probes-<mode>.
"""
from __future__ import annotations

import argparse
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

DINO_ID = "facebook/dinov2-base"
QWEN_IDS = {
    "qwen": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen32b": "Qwen/Qwen3-VL-32B-Instruct",
    "qwen38": "Qwen/Qwen3.8-27B",
    "qwen36": "Qwen/Qwen3.6-35B-A3B",                     # MoE, same family as Qwen3.8 (behavioural tests only)
    "sn8b": "sensenova/SenseNova-SI-1.1-Qwen3-VL-8B",   # Qwen3-VL-8B fine-tuned on 8M spatial QA
}
# Fine-tuned models: name -> (base model, LoRA adapter merged in after loading; finetune.py)
FINETUNED = {"qwen8b-ft": ("Qwen/Qwen3-VL-8B-Instruct", "data/ft/qwen8b-proc")}
# Same relative depths as the 2B model's layers 2, 6, ..., 27 (of 28).
DEPTHS = [2 / 27, 6 / 27, 10 / 27, 14 / 27, 18 / 27, 22 / 27, 1.0]
PROMPT = "These are consecutive first-person views as you walk around a room."
PROBE_PROMPT = "Now you are teleported to random places in the same room. Here is the view from each place:"
PROBE_ONLY_PROMPT = "Here is the view from each of several random places in a room:"


class Recorder:
    """Captures the input of each hooked module at the token positions set in .where."""

    def __init__(self, modules: dict):
        self.where = None
        self.acts = {}
        self.handles = [m.register_forward_pre_hook(self._hook(n)) for n, m in modules.items()]

    def _hook(self, name):
        def hook(_, inp):
            self.acts[name] = inp[0][self.where].float().cpu().numpy().astype(np.float16)
        return hook


def save(out, acts: dict):
    os.makedirs(out, exist_ok=True)
    for name, a in acts.items():
        a = np.concatenate(a) if isinstance(a, list) else a
        bad = ~np.isfinite(a)
        if bad.any():
            print(f"  {name}: {bad.sum()} non-finite values set to 0")
            a[bad] = 0
        np.save(os.path.join(out, f"{name}.npy"), a.astype(np.float16))
    print(f"saved {len(acts)} layers -> {out}")


@torch.no_grad()
def run_dino(frames, random_init, out, batch=256):
    from transformers import AutoConfig, AutoModel
    if random_init:
        torch.manual_seed(0)
        model = AutoModel.from_config(AutoConfig.from_pretrained(DINO_ID))
    else:
        model = AutoModel.from_pretrained(DINO_ID)
    model = model.cuda().eval()
    rec = Recorder({f"L{i + 1}": layer.mlp.fc2 for i, layer in enumerate(model.encoder.layer)})
    rec.where = (slice(None), 0)  # CLS token
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    acts = {f"L{i + 1}": [] for i in range(len(model.encoder.layer))}
    for i in range(0, len(frames), batch):
        x = torch.from_numpy(np.array(frames[i:i + batch])).cuda().permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=224, mode="bicubic", antialias=True)
        model(pixel_values=(x - mean) / std)
        for k, v in rec.acts.items():
            acts[k].append(v)
    save(out, acts)


def load_qwen(model_id, random_init, adapter=None):
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    # Keep every 256x256 frame at exactly 16x16 patches -> 8x8 = 64 visual tokens.
    proc.image_processor.size = {"shortest_edge": 256 * 256, "longest_edge": 256 * 256}
    cfg = AutoConfig.from_pretrained(model_id)
    if random_init:
        torch.manual_seed(0)
        cfg._attn_implementation = "sdpa"
        with torch.device("cuda"):
            model = AutoModelForImageTextToText.from_config(cfg).to(torch.bfloat16)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    n = cfg.text_config.num_hidden_layers
    want = sorted({round(d * (n - 1)) for d in DEPTHS})
    layers = {}
    for name, mod in model.named_modules():
        m = re.search(r"language_model\.layers\.(\d+)\.mlp\.down_proj$", name)
        if m and int(m.group(1)) in want:
            layers[f"L{int(m.group(1))}"] = mod
    assert len(layers) == len(want), list(layers)
    return proc, model, layers


def qwen_text(proc, n_walk, n_probe=0, probes_only=False):
    img = [{"type": "image"}]
    if probes_only:
        content = [{"type": "text", "text": PROBE_ONLY_PROMPT}] + img * n_probe
    else:
        content = [{"type": "text", "text": PROMPT}] + img * n_walk
        if n_probe:
            content += [{"type": "text", "text": PROBE_PROMPT}] + img * n_probe
    return proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                    add_generation_prompt=False)


@torch.no_grad()
def run_qwen(name, frames, meta, probes, pmeta, mode, random_init, batch=128):
    from PIL import Image
    model_id, adapter = FINETUNED.get(name, (QWEN_IDS.get(name), None))
    proc, model, layers = load_qwen(model_id, random_init, adapter)
    print(f"{model_id}{' + ' + adapter if adapter else ''}: recording {list(layers)}")
    rec = Recorder(layers)
    end_id = proc.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    suffix = "-random" if random_init else ""
    new = lambda n: {k: np.zeros((n, m.in_features), np.float16) for k, m in layers.items()}

    def forward(texts, imgs):
        inputs = proc(text=texts, images=[Image.fromarray(np.asarray(f)) for f in imgs],
                      return_tensors="pt", padding=True).to("cuda")
        grid = inputs["image_grid_thw"]
        assert (grid == torch.tensor([1, 16, 16], device="cuda")).all(), grid[:3]
        rec.where = torch.nonzero(inputs["input_ids"] == end_id, as_tuple=True)
        assert len(rec.where[0]) == len(imgs)
        model.model(**inputs, use_cache=False)  # language model only; no LM head needed
        return rec.acts, tuple(inputs["input_ids"].shape)

    t0 = time.time()
    if mode == "single":
        acts = new(len(frames))
        text = qwen_text(proc, 1)
        for i in range(0, len(frames), batch):
            a, shape = forward([text] * min(batch, len(frames) - i), frames[i:i + batch])
            for k, v in a.items():
                acts[k][i:i + batch] = v
        print(f"single: {len(frames)} frames, batch shape {shape}, {time.time() - t0:.0f}s")
        save(f"data/acts/{name}-single{suffix}", acts)
        return
    rng = np.random.default_rng(1)
    walks = np.unique(meta["walk"])
    pacts = new(len(probes))
    acts = new(len(frames)) if mode != "probes-only" else None
    for w in walks:
        pidx = np.flatnonzero(pmeta["walk"] == w)
        pidx = pidx[rng.permutation(len(pidx))]  # probes in random order
        if mode == "probes-only":
            a, shape = forward([qwen_text(proc, 0, len(pidx), probes_only=True)], probes[pidx])
            for k, v in a.items():
                pacts[k][pidx] = v
            continue
        idx = np.flatnonzero(meta["walk"] == w)
        idx = idx[np.argsort(meta["step"][idx])]
        if mode == "shuffled":
            idx = idx[rng.permutation(len(idx))]
        imgs = np.concatenate([frames[idx], probes[pidx]])
        a, shape = forward([qwen_text(proc, len(idx), len(pidx))], imgs)
        for k, v in a.items():
            acts[k][idx] = v[:len(idx)]
            pacts[k][pidx] = v[len(idx):]
    print(f"{mode}: {len(walks)} walks, context {shape[1]} tokens, {time.time() - t0:.0f}s")
    if mode == "probes-only":
        save(f"data/acts/{name}-probes-only{suffix}", pacts)
    else:
        save(f"data/acts/{name}-{mode}{suffix}", acts)
        save(f"data/acts/{name}-probes-{mode}{suffix}", pacts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dinov2", *QWEN_IDS, *FINETUNED], required=True)
    ap.add_argument("--mode", choices=["single", "walk", "shuffled", "probes-only"], default="single")
    ap.add_argument("--random", action="store_true", help="random-weight control")
    ap.add_argument("--data", default="data/walks")
    args = ap.parse_args()
    frames = np.load(os.path.join(args.data, "frames.npy"), mmap_mode="r")
    meta = dict(np.load(os.path.join(args.data, "meta.npz")))
    print(f"== {args.model} {args.mode}{' random' if args.random else ''}: {len(frames)} frames")
    if args.model == "dinov2":
        run_dino(frames, args.random, "data/acts/dinov2" + ("-random" if args.random else ""))
        return
    probes = np.load(os.path.join(args.data, "probes.npy"), mmap_mode="r")
    pmeta = dict(np.load(os.path.join(args.data, "probes_meta.npz")))
    run_qwen(args.model, frames, meta, probes, pmeta, args.mode, args.random)


if __name__ == "__main__":
    main()
