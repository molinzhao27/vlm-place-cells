"""Experiment 16's fixed draw of HSSD houses (docs/experiments.md), on the 4070 host through the GPU lock:
    scripts/rtx4070/run.sh --detach logs/render_hssd.log bash scripts/rtx4070/render_hssd.sh

The scene ids of hssd-hab's default scene set (scenes/*.scene_instance.json) are sorted, shuffled with
random.Random(16) and taken in that order. Each drawn house is checked and rendered as in experiment 12
(habitat_scene.py --hssd --walks-out: 32 walks of 256 steps, 128 probes per walk, 0.15 m clearance, seed
1600 + its place in the draw), and its lure-free trials are built (dissimilar_trials.py --walks 32
--per-walk 6, CPU only). It is accepted if it loads, has a navmesh (none ship, so computed with experiment
12's agent), its largest navigable island is 40-250 m^2, its objects carry instance labels
(habitat_scene.scene_check) and it yields at least 60 trials; otherwise the next house is drawn, until 8
are accepted. They are named HSSD1..HSSD8 in draw order: walks in BIG/walks-roomHSSD<k> (data/walks-roomHSSD<k>
links there, as for the Replica walks), trials in results/export/dissimilar-roomHSSD<k>/. A skipped house's
render and trials move to BIG/hssd-skipped/<scene id>/. Every draw, and why it was accepted or skipped, goes
to results/hssd_selection.json, rewritten after each draw; a rerun resumes from it. Houses whose objects are
not downloaded yet (scripts/rtx4070/get_hssd.sh fetches the first 24 of the draw first) stop the run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import time

HSSD = os.path.expanduser("~/habitat-data/hssd-hab")
BIG = "/mnt/bigd/mechinterp"
HAB_PY = os.path.expanduser("~/miniforge3/envs/hab033/bin/python")
OUT = "results/hssd_selection.json"
REVISION = "4369cb9876214c7fbebcf552eb532380e4d287e4"      # the hssd-hab revision scripts/rtx4070/get_hssd.sh pins
HOUSES, MIN_TRIALS, DRAW_SEED, RENDER_SEED = 8, 60, 16, 1600
RENDER = ["--walks", "32", "--steps", "256", "--probes-per-walk", "128", "--margin", "0.15"]
TRIALS = ["--walks", "32", "--per-walk", "6"]


def draw_order(root=HSSD):
    ids = sorted(os.path.basename(f)[:-len(".scene_instance.json")] for f in glob.glob(f"{root}/scenes/*.scene_instance.json"))
    random.Random(DRAW_SEED).shuffle(ids)
    return ids


def downloaded(sid, root=HSSD):
    if os.path.exists(f"{root}/.download_done"):
        return True
    return os.path.exists(f"{root}/.priority_done") and sid in json.load(open(f"{root}/.priority_done"))


def last_json(text):
    lines = [l for l in text.splitlines() if l.startswith("{")]
    return json.loads(lines[-1]) if lines else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--houses", type=int, default=HOUSES)
    args = ap.parse_args()
    order = draw_order()
    rec = json.load(open(OUT)) if os.path.exists(OUT) else {}
    rec.update(about=__doc__.strip().split("\n\n")[1], dataset="hssd/hssd-hab", revision=REVISION, scene_ids=len(order),
               rules={"island_m2": [40, 250], "min_trials": MIN_TRIALS, "render": " ".join(RENDER), "trials": " ".join(TRIALS),
                      "draw_seed": DRAW_SEED, "render_seed": f"{RENDER_SEED} + draw position"},
               draw_order=order)
    draws = rec.setdefault("draws", [])
    save = lambda: json.dump(rec, open(OUT, "w"), indent=1)
    for pos, sid in enumerate(order):
        accepted = [d for d in draws if d["status"] == "accepted"]
        if len(accepted) >= args.houses:
            break
        if any(d["scene"] == sid for d in draws):
            continue
        if not downloaded(sid):
            print(f"stopping: house {pos} ({sid}) is not downloaded yet", flush=True)
            break
        name = f"HSSD{len(accepted) + 1}"
        walks, link, export = f"{BIG}/walks-room{name}", f"data/walks-room{name}", f"results/export/dissimilar-room{name}"
        for p in (walks, export):
            shutil.rmtree(p, ignore_errors=True)
        os.makedirs(walks)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(walks, link)
        d = {"position": pos, "scene": sid, "seed": RENDER_SEED + pos}
        t0 = time.time()
        r = subprocess.run([HAB_PY, "-m", "spatialcells.habitat_scene", "--dataset", f"{HSSD}/hssd-hab.scene_dataset_config.json",
                            "--scene", sid, "--hssd", "--walks-out", link, *RENDER, "--seed", str(d["seed"])],
                           capture_output=True, text=True)
        d["check"] = json.load(open(f"{walks}/scene.json")) if os.path.exists(f"{walks}/scene.json") else None
        if r.returncode == 3:
            d.update(status="skipped", reason=d["check"]["skip"])
        elif r.returncode != 0:
            d.update(status="skipped", reason=f"does not load or render (exit {r.returncode})", stderr_tail=r.stderr[-1500:])
        else:
            d["render"] = last_json(r.stdout)
            t = subprocess.run([".venv/bin/python", "-m", "spatialcells.dissimilar_trials", "--room", name, *TRIALS],
                               capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
            if t.returncode != 0:
                raise SystemExit(f"dissimilar_trials failed for {name} ({sid}):\n{t.stderr[-3000:]}")
            trials = json.load(open(f"{export}/trials.json"))
            d.update(trials=len(trials), novel=sum(not x["familiar"] for x in trials), trials_summary=t.stdout.strip())
            if len(trials) >= MIN_TRIALS:
                d.update(status="accepted", name=name, reason=f"{len(trials)} lure-free trials >= {MIN_TRIALS}")
            else:
                d.update(status="skipped", reason=f"trials: {len(trials)} lure-free, fewer than {MIN_TRIALS}")
        d["minutes"] = round((time.time() - t0) / 60, 1)
        if d["status"] == "skipped":
            dest = f"{BIG}/hssd-skipped/{sid}"
            shutil.rmtree(dest, ignore_errors=True)
            os.makedirs(dest)
            shutil.move(walks, f"{dest}/walks")
            if os.path.exists(export):
                shutil.move(export, f"{dest}/export")
            os.remove(link)
        draws.append(d)
        save()
        print(json.dumps({k: d.get(k) for k in ("position", "scene", "status", "name", "reason", "trials", "novel", "minutes")}), flush=True)
    accepted = [d for d in draws if d["status"] == "accepted"]
    rec["accepted"] = {d["name"]: d["scene"] for d in accepted}
    rec["skipped"] = {d["scene"]: d["reason"] for d in draws if d["status"] == "skipped"}
    save()
    print(f"{len(accepted)} accepted of {len(draws)} drawn -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
