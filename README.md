# Place and grid cells in vision-language models

**Do neurons in a vision-language model act like hippocampal place cells or entorhinal grid
cells when the model watches a first-person walk through a room? And can the model tell where a
new view was taken, when it faces a way it never faced from that spot?**

![One two-choice trial](results/example_trial.png)

*One trial. After a 256-frame walk through a 10 m room, the model sees a new view (left) and is
asked whether it was taken nearer Frame 130 or Frame 149. Frame 130 is 0.1 m away but faces
another way. Qwen3.8-27B picks Frame 149, 3.3 m away, which shows the same posters.*

This project applies rodent-style single-unit analysis (rate maps, spatial information,
gridness, heading invariance, all against shuffle nulls), linear probes, behavioural tests and
causal interventions (ablation, activation patching, steering) to Qwen3-VL 2B/8B/32B,
Qwen3.8-27B, Qwen3.6-35B-A3B, Gemma-4-31B, SenseNova-SI, DINOv2, the 3D vision model VGGT,
and Claude Haiku, Sonnet and Opus 5.5 (through the API). Everything runs in ray-cast
rendered rooms, so every frame's true position and heading are known.

## Key findings

1. **Units that pass rodent place-cell criteria are everywhere, and on their own they mean
   little.** 72% of raw pixels, 20-63% of DINOv2 units (trained or random) and 29-37% of
   Qwen3-VL-2B units pass. They are view cells: a unit's map facing north does not match its map
   facing south, and position does not decode across facing directions. Heading invariance is
   the test that separates place from view. ([Exp 1](docs/experiments.md#experiment-1-spatial-cells-pilot-spatialcells))
2. **Open VLMs recognise views, not places.** Qwen models from 2B to 32B cannot place a view
   seen from a new direction. In their activations, teleport probes decode at chance across
   headings (3.2-3.5 m; chance 3.6 m). In their answers, at most 3% of novel views are placed
   within 1 m, below random. Scale improves view matching only, and reasoning does not help.
   ([Exp 2-3](docs/experiments.md#experiment-2-teleport-probes-across-model-sizes))
3. **The failure is a landmark lure, and the walk makes it worse.** Open models pick the far
   frame when it shares a poster with the new view. On a lure-free trial set they reach 59-71%
   from the three images alone, but the walk does not help: answering directly they fall to
   chance, and with thinking Qwen3.8-27B does no better than without the walk.
   ([Exp 5](docs/experiments.md#experiment-5-why-do-open-models-fail-and-does-any-pass))
4. **Claude Opus 5.5 is the only off-the-shelf model that gains from the walk** (75% vs 61-66%
   without it on lure-free novel trials; paired sign test p = 0.008), and it resists the lure.
   Shuffling the walk costs nothing, so it uses the frames as a set of reference views, not as a
   path. Sonnet and Haiku 5.5 do not gain from the walk.
   ([Exp 5-6](docs/experiments.md#experiment-6-does-opus-follow-the-path-shuffled-walk))
5. **VGGT solves the task and has a place code, but its place neurons are not the mechanism.**
   With reference views it places 98% of novel views within 1 m. Its late blocks hold
   heading-invariant place units with broad fields and decode position across headings to
   0.97 m from a single frame. Activation patching finds a 10-dimensional residual-stream
   subspace that carries the position its camera head uses (85% transfer at block 21). Ablating
   or steering the place *neurons* barely moves the camera, and the code does not remap in a
   new room. ([Exp 4](docs/experiments.md#experiment-4-does-a-model-that-knows-where-it-is-have-place-cells-vggt),
   [Exp 8](docs/experiments.md#experiment-8-can-we-steer-vggts-place-neurons))
6. **A small model can learn the skill.** LoRA on 1,136 procedural rooms takes Qwen3-VL-8B from
   50% to 92% on the held-out room's novel trials, above Opus. Unlike Opus it needs the walk in
   order (shuffled: chance), and its place code changes only a little.
   ([Exp 9](docs/experiments.md#experiment-9-teaching-a-small-model-the-skill))
7. **The grid-cell scorer works, and no model has grid cells.** A path-integrating RNN trained
   following Sorscher et al. grows hexagonal grids, and the unchanged pipeline flags 93% of the
   strong ones. No grids with spacing of 2 m or more appear in pixels, DINOv2, Qwen or VGGT.
   ([Exp 7](docs/experiments.md#experiment-7-can-our-scorer-find-a-grid-cell-at-all-positive-control))

## Results at a glance

Two-choice test, lure-free novel-view trials (102 trials; chance 50%; walk = the 256-frame walk
in context; Claude models at high effort):

| | no walk | with walk |
|---|---|---|
| Qwen3-VL-8B | 59% (71% at 512 px) | 50% |
| Qwen3.8-27B, thinking | 61% (63% at 512 px) | 65% |
| Gemma-4-31B | 56% (68% at 1024 px) | 53% |
| Claude Haiku 5.5 | 67% | 56% |
| Claude Sonnet 5.5 | 64% | 70% |
| Claude Opus 5.5 | 63% | **75%** |
| Qwen3-VL-8B + LoRA on procedural rooms (64-frame walk) | 53% | **92%** |

On the original trials, which contain the landmark lures, every open model falls below chance
and VGGT gets all 104 right:

![Two-choice accuracy on the original trials](results/two_choice.png)

The full write-up, with methods, controls, caveats and per-experiment commands, is in
**[docs/experiments.md](docs/experiments.md)**.

## Repository layout

```
spatialcells/            the package (run modules with python -m spatialcells.<module>)
  room.py                ray-cast 10 m room, procedural layouts, foraging walks
  render_data.py         render walks -> data/walks
  render_probes.py       teleport probes (views from visited places, random headings)
  procedural_data.py     1,200 procedural rooms and trials for fine-tuning
  extract.py             record MLP neurons of DINOv2 / Qwen models along the walks
  analyze.py             place, grid and head-direction scoring, heading invariance, decoding
  analyze_probes.py      teleport-probe decoding across headings
  summarize.py           summary figures and tables
  behave.py, behave_reason.py, analyze_behave.py      "where am I?" behaviour tests
  dissimilar_trials.py   the lure-free trial set
  two_choice.py          two-choice test for any Hugging Face VLM (optionally a LoRA adapter)
  claude_test.py         the same test through the Anthropic API
  vggt_*.py              VGGT control, unit recording, ablation, patching, remapping, steering
  grid_rnn.py            grid-cell positive control (Sorscher-style RNN, synthetic grids)
  finetune.py, ft_seeds.py, adjacency_baseline.py     LoRA fine-tuning and its baselines
scripts/                 the exact job scripts behind each experiment (run_*.sh)
  vast1/                 helpers for running on a remote GPU container (docs/gpu_host.md)
results/                 summaries (JSON/CSV), per-trial answers (JSONL) and figures
  export/dissimilar/     the 192 lure-free trials (probes + trials.json)
docs/experiments.md      full write-up of experiments 1-9
writeup/probe_post/      figures for a post on probe vs causal directions in VGGT (make_figs.py)
```

## Reproducing

```bash
git clone https://github.com/molinzhao27/vlm-place-cells.git
cd vlm-place-cells
pip install -e .              # CPU: room, rendering, scoring, baselines
pip install -e ".[gpu]"       # + VLM extraction, behaviour tests, VGGT, LoRA (CUDA)
pip install -e ".[claude]"    # + the Claude API test
```

All paths are relative to the repository root. Generated data (rendered walks, activations,
checkpoints) goes to `data/`, which is not tracked.

```bash
python -m spatialcells.render_data                  # 48 walks x 256 frames -> data/walks (CPU)
python -m spatialcells.render_probes                # teleport probes
python -m spatialcells.analyze --tags pixels        # place/grid/HD scores for the raw-pixel baseline
python -m spatialcells.grid_rnn --synthetic         # gridness scorer on synthetic grid cells (CPU)
```

Each experiment in [docs/experiments.md](docs/experiments.md) ends with the commands that
produced it. The GPU experiments ran on one 96 GB RTX PRO 6000; `scripts/run_*.sh` are the
job scripts, and [docs/gpu_host.md](docs/gpu_host.md) describes the remote setup. The VGGT
experiments also need the [VGGT repository](https://github.com/facebookresearch/vggt) cloned.

Not tracked, because of size: the walk frames used by the Claude test
(`results/export/claude/`, 90 MB), raw generation records (`results/behave/qwen38-reason.npz`)
and per-layer rate-map figures (`results/figs/`). They will be archived with the preprint.

## Citation

A preprint is in preparation. Until then:

```bibtex
@misc{zhao2026placecells,
  title  = {Place and grid cells in vision-language models},
  author = {Zhao, Molin},
  year   = {2026},
  url    = {https://github.com/molinzhao27/vlm-place-cells}
}
```

## License

Code: [MIT](LICENSE). Figures and trial data in `results/`: CC BY 4.0.
