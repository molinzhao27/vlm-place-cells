"""Stage the Hugging Face release in data/hf_release/{model,dataset} with hard links (no extra disk space).

    python3 scripts/hf_release/stage.py

model/   one folder per LoRA run (adapter_config.json, adapter_model.safetensors, train_log.json) + README.md
dataset/ frames/<room>/            the walk frames and probe images every model saw (JPEG, as sent to the APIs),
                                   procedural rooms only, plus trials.json
         two_choice_trials/<room>/ the two-choice trial lists (and their probe images, procedural rooms only)
         walks/roomP5005/          raw frames for the one held-out room the generator cannot re-render exactly
         training/proc/            the 1,200 procedural training rooms (rsync'd from vast-3 to data/hf_release/training/proc)
         results/                  every per-trial answer file and summary
Photoreal renders (Replica, HSSD) are left out: their dataset licenses do not allow redistribution, so
only their trial lists are included and the render scripts regenerate the frames. Human responses and
the world-model pilot are left out too.
"""
import glob
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/data/hf_release"
RUNS = {"run0": "qwen8b-proc", "run1": "qwen8b-proc-s1", "run2": "qwen8b-proc-s2", "run0-gpu-control": "qwen8b-proc-s0-5090",
        **{f"e4-run{k}": f"qwen8b-proc-e4-s{k}" for k in range(3, 8)}}
PROCEDURAL = ["B"] + [f"P50{i:02d}" for i in range(10)]
PHOTOREAL = [f"HSSD{i}" for i in range(1, 9)] + [f"ReplicaApt{i}" for i in range(3)]
RESULTS = ["two_choice", "claude", "api", "std_bench", "behave", "frame_cut"]


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    os.link(src, dst)


def link_tree(src, dst, keep=lambda p: True):
    n = 0
    for p in glob.glob(f"{src}/**/*", recursive=True):
        if os.path.isfile(p) and keep(p):
            link(p, f"{dst}/{os.path.relpath(p, src)}")
            n += 1
    return n


def main():
    for name, ad in RUNS.items():
        src = f"{ROOT}/data/ft/{ad}"
        if not os.path.exists(f"{src}/adapter_model.safetensors"):
            print(f"model: {name} not here yet ({ad})")
            continue
        for f in ("adapter_config.json", "adapter_model.safetensors", "train_log.json"):
            link(f"{src}/{f}", f"{OUT}/model/{name}/{f}")
    exp, ds = f"{ROOT}/results/export", f"{OUT}/dataset"
    counts = {}
    counts["frames/roomA"] = link_tree(f"{exp}/claude", f"{ds}/frames/roomA")
    counts["two_choice_trials/roomA"] = link_tree(f"{exp}/dissimilar", f"{ds}/two_choice_trials/roomA")
    for r in PROCEDURAL:
        counts[f"frames/room{r}"] = link_tree(f"{exp}/claude-room{r}", f"{ds}/frames/room{r}")
        counts[f"two_choice_trials/room{r}"] = link_tree(f"{exp}/dissimilar-room{r}", f"{ds}/two_choice_trials/room{r}")
    for d in sorted(glob.glob(f"{exp}/claude-roomAno*")):
        r = d.split("claude-")[1]
        counts[f"frames/lure_edits/{r}"] = link_tree(d, f"{ds}/frames/lure_edits/{r}")
    for d in sorted(glob.glob(f"{exp}/dissimilar-roomA[nl]*")):
        r = d.split("dissimilar-")[1]
        counts[f"two_choice_trials/lure_edits/{r}"] = link_tree(d, f"{ds}/two_choice_trials/lure_edits/{r}")
    trials_only = lambda p: os.path.basename(p) == "trials.json"
    for r in PHOTOREAL:
        counts[f"frames/room{r}"] = link_tree(f"{exp}/claude-room{r}", f"{ds}/frames/room{r}", trials_only)
        counts[f"two_choice_trials/room{r}"] = link_tree(f"{exp}/dissimilar-room{r}", f"{ds}/two_choice_trials/room{r}", trials_only)
    counts["walks/roomP5005"] = link_tree(f"{ROOT}/data/walks-roomP5005", f"{ds}/walks/roomP5005")
    for r in RESULTS:
        counts[f"results/{r}"] = link_tree(f"{ROOT}/results/{r}", f"{ds}/results/{r}")
    for p in glob.glob(f"{ROOT}/results/*.json"):
        link(p, f"{ds}/results/{os.path.basename(p)}")
    counts["results/*.json"] = len(glob.glob(f"{ROOT}/results/*.json"))
    counts["training/proc"] = link_tree(f"{OUT}/training/proc", f"{ds}/training/proc")   # rsync'd from vast-3
    for k, v in counts.items():
        print(f"{k:40s} {v}")
    for card in ("model", "dataset"):
        if not os.path.exists(f"{OUT}/{card}/README.md"):
            print(f"{card}/README.md missing")


if __name__ == "__main__":
    main()
