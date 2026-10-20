# Experiments: place and grid cells in vision-language models

The full write-up behind the [README](../README.md): every experiment in the order it was run,
with methods, tables, caveats and the commands that produced each result. Paths are relative
to the repository root.

Do neurons in a vision-language model act like hippocampal place cells or
entorhinal grid cells when the model watches a first-person walk through a room?

## Experiment 1: spatial cells pilot (`spatialcells/`)

1. **Room** (`room.py`): a 10 m x 10 m ray-cast room with four different walls,
   six posters, a door, a window and two pillars (asymmetric on purpose).
   Camera at 1 m, 90° field of view, 256 x 256 frames.
2. **Walks** (`render_data.py`): 48 rat-like foraging walks x 256 steps
   (12,288 frames): forward motion with drifting heading, on-the-spot head
   scans, turning away from walls and pillars.
3. **Activations** (`extract.py`), one set per condition:
   - `dinov2`, `dinov2-random`: DINOv2-base MLP neurons at the CLS token, 12 layers
     (repeats Luo, Mok & Love 2024 with pretrained vs random weights)
   - `qwen-single`: Qwen3-VL-2B language-model MLP neurons at `<|vision_end|>`,
     each frame seen alone
   - `qwen-walk`: the whole walk in one context, in order (the model can integrate)
   - `qwen-shuffled`: the same frames in one context, random order
   - `qwen-single-random`: random-weight Qwen3-VL, single frames
   - `pixels` (computed in `analyze.py`): 32 x 32 x 3 raw pixels as "units"
4. **Scores** (`analyze.py`): every neuron pole (relu(+a), relu(-a)) is scored with
   rodent criteria against circular-shift shuffles: place (spatial information,
   stability, 1-3 fields), grid (gridness), head direction (mean vector length),
   plus **heading invariance**, which separates place cells from view cells, and
   ridge decoding of (x, y) within and **across** facing directions.
5. **Summary** (`summarize.py`): `results/summary.png`, `results/summary.csv`,
   example rate maps in `results/figs/`.

### Results

- Units that pass the rodent place-cell criteria are everywhere: 72% of raw
  pixels, 20-63% of DINOv2 poles (pretrained or random), 29-37% of Qwen3-VL-2B
  poles on single frames, ~80% with random weights. They are view-bound: their
  maps from opposite facing halves barely agree (invariance / stability
  0.19-0.27), and position decoded from one facing direction does not transfer
  to the other (≈ chance, 3.7 m).
- With the whole walk in context, Qwen3-VL-2B units become heading-invariant
  (inv / stab up to 0.94) and cross-heading decoding drops to 1.5 m. The same
  frames shuffled stay at chance.
- That effect is memory of recent views, not a map. A causal 8-step moving average
  of single-frame activations, with no model in the loop, does better
  (inv ≈ 1.0, cross 0.5-0.6 m), and the same average of raw pixels gives 97%
  "place cells" with inv 0.98.
- No grid cells: high-gridness units were unstable between halves of the data.

## Experiment 2: teleport probes across model sizes

Smoothing explains the walk effect, so this test removes it. After each walk,
64 views from places the walk visited are appended in random order with
random headings (`render_probes.py`; `extract.py --mode walk|shuffled|probes-only`).
A probe has no recent history to smooth over, so it can be placed only if the
model links it to what it saw earlier. Decoding is
trained on even walks and tested on odd walks (`analyze_probes.py`), including a
train-heading x test-heading matrix (`results/probes_heading_matrix.json`).

Median error at each model's best layer (m), chance 3.62:

| | same facing direction | opposite facing direction |
|---|---|---|
| probe's own pixels | 1.35 | 4.14 |
| recall the most similar walk frame (8-step pixel average) | 1.49 | **1.68** |
| Qwen3-VL-2B: after walk / shuffled walk / no walk | 1.29 / 1.29 / 1.33 | 3.47 / 3.38 / 3.58 |
| Qwen3-VL-8B | 1.21 / 1.18 / 1.25 | 3.24 / 3.22 / 3.29 |
| Qwen3-VL-32B | 1.09 / 1.05 / 1.13 | 3.34 / 3.30 / 3.40 |
| Qwen3.8-27B (newer family) | 1.18 / 1.17 / 1.22 | 3.19 / 3.25 / 3.29 |

The Qwen3-VL sizes (2B, 8B, 32B) share a recipe and are the scaling series.
Qwen3.8-27B is reported separately: it is a newer model with different training
and a hybrid linear/full-attention architecture.

- Larger Qwen3-VL models are better view encoders: within-heading error falls
  from 1.29 m (2B) to 1.05 m (32B), better than pixels.
- None of them places a probe from the other side: cross-heading error stays
  near chance at every size and layer. The heading matrix is diagonal with
  opposite-facing cells worse than chance (4.4-4.6 m for the 27B), the same
  pattern as pixels.
- The walk does not help: after walk, after shuffled walk and alone differ by
  ≤ 0.15 m. The trivial recall baseline gets 1.68 m across headings, so the
  information is in the context. The models do not write it into the probe's
  representation.
- Caveat: this is a linear readout at one token. Experiment 3 asks the
  models directly.

Single units across sizes (`results/units_scaling.png`): Qwen3-VL-8B and 32B
repeat the 2B pattern, and so does Qwen3.8-27B.
- **One frame:** place-criteria units are view-bound (inv / stab 0.12-0.25) and
  cross-heading decoding is near chance (Qwen3-VL 3.45-3.79 m; Qwen3.8 3.41 m).
- **Walk in context:** cross-heading error drops to 1.23-1.48 m for Qwen3-VL,
  and to 0.85 m for Qwen3.8.
- **One frame + 8-step average:** does better than the walk for every model
  (Qwen3-VL 0.39-0.45 m, Qwen3.8 0.37 m; inv ≈ 1.0).
- No grid cells in any model.

## Experiment 3: asking the model where it is

`behave.py` shows the whole walk with each frame labelled "Frame k:". It then
shows one teleport probe and asks which numbered frame was taken from the
position closest to where the new view was taken. The model answers greedily
with a frame number.
- Each walk is prefilled once, and each probe continues from that KV cache,
  rewound between probes. The first two probes of each model are checked
  against a full recompute (same answer up to bf16 near-ties).
- `analyze_behave.py` splits the probes into two groups:
  - **familiar** (2690): some walk frame within 1 m faced within 45° of the
    probe.
  - **novel** (382): the place was visited but never seen facing this way.

Fraction of answers within 1 m of the true place (`results/behave.png`):

| | familiar views | novel views |
|---|---|---|
| pick the most similar-looking frame (pixels) | 0.67 | 0.03 |
| random frame | 0.12 | 0.06 |
| Qwen3-VL-2B | 0.20 | 0.03 |
| Qwen3-VL-8B | 0.34 | 0.01 |
| Qwen3-VL-32B | 0.43 | 0.01 |
| Qwen3.8-27B (newer family) | 0.58 | 0.01 |

- Within Qwen3-VL, familiar views get easier with size (20%, 34%, 43%).
  2B and 8B often fail the task format itself: 2B answers "256" 13% of the
  time. Qwen3.8-27B does best on familiar views (58%), close to pixel matching.
- Novel views fail in every model, below random. 69-80% of the larger models'
  picks face within 45° of the probe's heading, and for a novel probe every
  such frame is at least 1 m away. The models answer "where did I see
  something that looks like this?", not "where am I?".
- So, in their behaviour as well as their activations, these Qwen models
  recognise views, not places. Claude Opus 5.5 does not fail this way (3d). The test has limits: answers come without reasoning, from
  one prompt format, in one room. A 256-way frame index is also hard on its own.
  Answers on familiar views show the format is workable for the larger models.

### 3b: does reasoning, or an easier question, help? (Qwen3.8-27B)

`behave_reason.py` reruns the question on the 27B over 16 walks, with frames
labelled "Frame 001" so that questions batch together. It adds a two-choice
version: was the new view taken closer to where Frame a or Frame b was taken?
One option is the walk frame nearest the probe, the other a random frame more
than 3 m away, in random order (chance 50%). Each question is asked directly
on all probes, and with thinking on for the 104 novel probes. Thinking uses
medium effort and the model's sampling settings, closed at 3072 tokens
(9-17% hit the cap) (`results/behave_reason_qwen38.png`).

| | familiar, direct | novel, direct | novel, reasoning |
|---|---|---|---|
| nearest frame: answers within 1 m (random 12% / 6%) | 56% | 0% | 1% |
| two-choice: picked the nearer frame (chance 50%) | 69% | 38% | 38% |

- Reasoning changes nothing on novel views. The model takes 1300-1500 thinking
  tokens and still lands on the same answers.
- On novel views the two-choice score is below chance (40/104 and 39/104; exact
  binomial p = 0.024 and 0.014). The model picks
  whichever option faces more like the new view 71-74% of the time. For a novel
  view, that option is the farther frame 69% of the time, because the nearer
  frame was by definition taken facing another way.
- The reasoning traces say this outright: "the new view's content ... matches
  the region around Frame 149 much more than 130. So the new view is closer to
  Frame 149's location." The model sometimes tries to infer camera position
  from which wall it faces, but gets the layout wrong and falls back on matching
  content.

### 3c: a positive control and a spatial fine-tune

**VGGT can do the task** (`vggt_control.py`). VGGT (Wang et al., CVPR 2025) is a
3D vision model that predicts a camera pose for every image. Each trial gives it
every 4th walk frame, the two option frames and the probe, one probe at a time.
Its predicted cameras are put into room coordinates by a similarity fit on the
walk frames only.

On the 104 novel trials:
- 98% of probes are placed within 1 m (median error 0.30 m, heading error 1.6°).
- The nearest-frame pick is within 1 m 96% of the time.
- Two-choice: 104/104 correct.

So the 256 x 256 frames hold everything needed to place a novel view. The VLMs'
failure is not about resolution or information.

**Spatial fine-tuning doesn't help.** SenseNova-SI-1.1-Qwen3-VL-8B is
Qwen3-VL-8B fine-tuned on 8M spatial QA samples; it reports 68.7% on VSI-Bench
and 85.6% on MindCube. Against the base model on every test:

| | Qwen3-VL-8B | SenseNova-SI 8B |
|---|---|---|
| teleport probes, cross-heading (best layer, after walk) | 3.24 m | 3.21 m |
| teleport probes, within-heading | 1.21 m | 1.18 m |
| single frame: place-unit invariance / stability | 0.17-0.21 | 0.16-0.26 |
| walk in context vs 8-step average, cross-heading | 1.23 vs 0.40 m | 1.26 vs 0.41 m |
| nearest frame within 1 m, familiar / novel | 34% / 1% | 33% / 0% |
| two-choice, familiar / novel (chance 50%) | 62% / 35% | 59% / 38% |

`results/two_choice.png` puts all systems on the same novel trials.

### 3d: a frontier model through the API (Claude Opus 5.5)

`claude_test.py` asks the two-choice question through the Anthropic API. The
prompt is the VLM prompt: 256 labelled frames as 256 x 256 JPEGs, the probe,
and the question. It runs all 104 novel trials plus 3 familiar trials per walk.
- Thinking is adaptive (it cannot be turned off on Opus 5.5) at effort high.
- Each walk is cached across its questions, so the whole run cost $4.86.
- Behaviour only: the activations are not accessible.

| | familiar (48) | novel (104) |
|---|---|---|
| Claude Opus 5.5 | 90% | **75%** (78/104; 95% CI 66-82%) |
| Qwen3.8-27B with reasoning | n/a | 38% |
| VGGT | 100% | 100% |

- Opus 5.5 is well above chance on novel views (exact binomial p = 3e-7) and
  well above every Qwen model (z = 5.5 against Qwen3.8-27B with reasoning).
- It is not view matching:
  - It picks the option facing like the probe only 48% of the time; the Qwen
    models do so 71-74% of the time.
  - When pixels favour the farther frame it is still right 68% of the time;
    the Qwen models manage 23-34%.
- It is hardest when the nearer frame faces nearly the opposite way (39/59,
  66%) and easiest when the farther option is more than 7 m away (11/11).
- There is no answer-order bias (first-listed option picked 49%).
- Its thinking summaries build a room layout from wall materials and pillars
  ("cream is opposite brick, and blue is opposite wood, with the pillars ...
  helping orient each shot").
- On the 45 novel trials of the human-baseline page it gets 33/45.

**Does it need the walk?** The `--no-walk` control shows only the two option
frames (with their frame numbers), the new view and the same question.

- Opus 5.5 still gets 69/104 (66%; p = 0.001 against chance), against 75% with
  the walk.
- Paired by trial, the walk helped on 16 and hurt on 7. That is not a
  significant difference (McNemar chi2 = 2.8, p ≈ 0.1).
- The control cost $1.36.

Most of the skill is reasoning about the three views themselves (which walls,
posters and pillars are visible, and from how far). On the lure-free trials of
experiment 5, though, the walk adds about 10 points, and that gain is significant
(p = 0.008).

So at least one frontier model shows the behaviour the Qwen models lack:
recognising a place seen from a new direction. Whether it has a place-like code
inside cannot be tested here, because its activations are not available.
A human baseline on the same trials is collected at
https://claude.ai/artifact/BjEa1fLN8M8sFWhYE7ehHA.

`results/probes.png` has the figure for experiment 2. Run with `scripts/run_scaling.sh`
(extraction and probe analysis) and `scripts/run_scaling_units.sh` (single-unit
scores for the larger models). `scripts/run_behave.sh` runs experiment 3 and
`scripts/run_sensenova.sh` runs every test for SenseNova-SI. `vggt_control.py`
needs the VGGT repo cloned to `/workspace/vggt` in the container.

## Experiment 4: does a model that knows where it is have place cells? (VGGT)

VGGT places every frame correctly (3c), so it is the one model here whose units
are worth scoring as a positive case. `vggt_cells.py` records the MLP neurons
(fc2 input, 4096 per block) at each walk frame's camera token in 7 of VGGT's 24
global-attention blocks. Those units then go through the same place, grid and
head-direction scoring as experiment 1.

Each sequence starts with the same 8 anchor views (the room's centre facing
each wall, and four spots 3 m in from the corners facing the centre), so
VGGT's output frame is the same for every input. There are three conditions:

- **alone**: each walk frame by itself, with nothing to localise against.
- **single**: the 8 anchors plus one walk frame. VGGT can localise the frame
  but has no walk memory, so no smoothing over recent views is possible.
- **walk**: the 8 anchors plus 64 consecutive walk frames.

The camera head places frames with a median error of 0.42 m in single and
0.29 m in walk. Heading error is under 2° in both.

Best layer per condition (cross-heading decoding chance is 3.72 m):

| | cross-heading decoding | place units that are heading-invariant | place-unit invariance / stability |
|---|---|---|---|
| VGGT alone | 3.64 m (block 9) | 6% | 0.36 |
| **VGGT single** | **0.97 m (block 19)** | **23%** | **0.73** |
| VGGT walk | 1.08 m (block 19) | 30% (block 23) | 0.89 (block 19) |
| Qwen3-VL 2B / 8B / 32B, single frame | 3.79 / 3.55 / 3.45 m | 2% | 0.21-0.24 |
| Qwen3.8-27B, single frame | 3.41 m | 4% | 0.25 |
| SenseNova-SI 8B, single frame | 3.31 m | 3% | 0.26 |
| Qwen3-VL-8B, 8-step pixel average | 0.40 m | 42% | 0.98 |

- **The place-cell criteria still pass trivially.** 25-34% of units in VGGT
  alone count as place units and over 80% as head-direction units in most
  blocks. That is the same as pixels and random networks (experiment 1).
- **VGGT single is the clean case.** Given reference views, its late blocks
  (15-23) encode where the camera is in a way that transfers across facing
  directions. It does this from one frame, with no walk to smooth over. No Qwen
  model does this from a single frame. They only get there through smoothing,
  which VGGT single cannot do.
- **Most of block 19's place units are heading-invariant, and their fields are
  broad.** Across all place units in block 19:
  - 64% are heading-invariant in single and 80% in walk.
  - Invariant units carry more spatial information than heading-specific ones
    (median 0.06 vs 0.01 bits). They make up 77% of the top tenth by spatial
    information.
  - Fields are broad in both groups. The median field covers about a third of
    the room (0.32 above half-maximum). For Qwen3-VL-8B with the 8-step
    average it is 0.15-0.20.
  - `results/figs/vggt-single_L19.png` shows examples:
    - The 4 highest-information units are heading-specific, with small fields
      that fire in one corner only when facing north. They are the exception.
    - The invariant units shown (invariance 0.98-0.99) are smooth gradients,
      each rising toward one wall or corner.
  - This is a heading-invariant place code, but coarse: a few broad, overlapping
    fields rather than a tiling of small ones.
- **Late blocks mix position with heading.** In block 23, position decodes to
  0.09 m within a facing direction but only 2.12 m across directions. VGGT
  predicts camera extrinsics, whose translation t = -RC is in camera
  coordinates and so depends on heading. The final blocks seem to move toward
  that output form.
- **No grid cells** in any condition (≤ 0.1% of units).

**Are the units used? The test can't tell** (`vggt_ablate.py`). Each condition
replaces neurons at the walk frame's camera token by their mean and measures
VGGT's own camera-head position error (768 frames, anchors + 1 frame):

| neurons ablated | block 19 | blocks 15+19+23 |
|---|---|---|
| none | 0.42 m | 0.42 m |
| heading-invariant place neurons (532 / block) | 0.43 m | 0.45 m |
| heading-specific place neurons (same count) | 0.42 m | 0.43 m |
| random neurons (same count, 2 draws) | 0.42 m | 0.43 m |
| **all 4,096 neurons (positive control)** | 0.43 m | 0.48 m |

Removing every MLP neuron at that token in three blocks only adds 6 cm. The
position estimate is rebuilt from the residual stream and from attention to the
patch tokens and anchors. So this ablation cannot show whether the place units
matter. The units carry position, but here they look like a redundant readout.
A real causal test would need to remove the position subspace from the residual
stream across blocks, not single MLP neurons. The next test does that.

**Is the position code used? Yes, from block 19 on** (`vggt_patch.py`). This is
an interchange test on the residual stream at the walk frame's camera token,
right after a global block. It uses 128 target frames, each paired with a donor
frame more than 3 m away (median 5.3 m), in single mode in room A. Part of the
donor's token is written into the target's run, and the test measures how far
VGGT's predicted position moves toward the donor's: 0 means no effect, 1 means
the target is placed where the donor was.

The subspaces are found on 2,048 other frames:
- "probe" directions are the ones a ridge probe to (x, y) reads.
- "PCA" subspaces are the top principal directions of the mean token per 1 m
  position bin (position), 30° heading bin (heading), or 2 m x 90° pose bin (pose).

"Share" is how much of the target-donor difference (squared norm) falls in the
subspace.

| written into the target's token | share | block 15 | block 19 | block 21 |
|---|---|---|---|---|
| the whole token | 100% | 0.03 | 0.41 | **0.87** |
| position PCA, 10 of 1,024 dims | 93% | 0.03 | 0.40 | **0.85** |
| everything except pose PCA 30 | 2% | 0.00 | 0.00 | 0.00 |
| heading PCA, 6 dims | 58-69% | 0.00 | 0.07 | 0.32 |
| probe directions for (x, y), 2 dims | 0.1% | 0.00 | 0.00 | 0.00 |
| random 10 / 30 dims | 1% / 3% | 0.00 / 0.00 | 0.00 / 0.01 | 0.00 / 0.01 |
| erase pose PCA 30: position error (clean 0.43 m) | | 0.51 m | 1.76 m | **3.37 m** |

Across the earlier blocks, the whole token transfers nothing through block 12
(≤ 0.001) and 0.03 at block 15. At block 23 it transfers 0.46, with only the
global-attention half patched. Patching what the camera head reads (both halves
after block 23) transfers 1.00.

- **A 10-dimensional position code carries what the camera head uses.** At block
  21, writing just these 10 directions moves the prediction 85% of the way to
  the donor. Writing everything else does nothing. Erasing them raises the
  error from 0.43 m to 3.4 m.
- **It is used only from block 19 on.** Through block 15 the token carries
  position just as readably (probe R² 0.98), but patching it does nothing: later
  blocks rebuild position from the patch tokens and anchors. At block 19 about
  40% survives, at block 21 nearly all.
- **A good probe is not the causal direction.** The ridge probe decodes position
  with R² 0.99, but it reads directions that hold 0.1% of the difference between
  frames, and writing them changes nothing. The first run used only probe
  subspaces and found no effect at any block. That came from where the probe
  reads, not from the code being unused.

So the heading-invariant place units at block 19 sit in a position code that
the model starts to use there. At block 15 the same readable code is a
by-product that gets overwritten. The MLP ablation above missed this because
removing neurons at one token leaves the residual stream's position code
in place.

**Does the code remap in a new room? No** (`vggt_remap.py`). Hippocampal place
cells remap between different rooms: a cell's field moves somewhere unrelated, or
the cell falls silent. Room B (`render_data --room B`) is the same 10 m box with
different walls, floor, posters and pillar positions (`results/roomB_anchors.png`).
VGGT single mode was run there with anchors at the same 8 poses. Its camera head
is less accurate in room B (0.85 m median error, against 0.42 m in room A).

| | block 9 | block 15 | block 19 | block 23 |
|---|---|---|---|---|
| room-A place units: map in A vs map in B (median r) | 0.69 | 0.87 | **0.95** | 0.86 |
| same, within room A (even vs odd walks) | 0.93 | 0.93 | 0.95 | 0.84 |
| same, random pairs of place units across rooms | 0.54 | 0.18 | 0.17 | 0.28 |
| room-A place units that are place units in B (chance) | 61% (31%) | 73% (35%) | 87% (47%) | 80% (68%) |
| decoder trained in A, tested in B / within A (m) | 3.84 / 0.64 | 1.31 / 0.17 | 0.74 / 0.15 | 0.64 / 0.12 |
| population vector, same spot across rooms / within A | 0.69 / 0.82 | 0.77 / 0.78 | 0.82 / 0.82 | 0.67 / 0.67 |

- **From block 15 on, the units keep their fields in the new room.** At block 19,
  a unit's map in room B matches its map in room A as well as two halves of the
  room-A data match each other. Almost none go silent (0.4%), and mean rates are
  unchanged (B/A 0.93).
- A position decoder trained in room A works in room B (0.74 m against a chance
  of about 3.7 m). The population pattern at a spot is as similar across rooms as
  within one.
- **Block 9 is more room-specific.** There a decoder trained in one room fails in
  the other (3.8 m), though single-unit maps still partly agree.
- So VGGT's late place units are coordinates in the anchors' frame, not a memory
  of a particular room. That is what a camera-pose estimator should compute, with
  anchors at the same poses in a box of the same shape. It is the opposite of
  hippocampal global remapping.

So the one model that localises novel views also has heading-invariant place
units, from a single frame with no walk to smooth over, which no Qwen model
has. They appear only when VGGT has reference views to localise against (single
and walk, not alone), and only in blocks 15-23. Their fields are broad (about
a third of the room each), and grid cells are absent. The position code they
belong to is used from block 19 on. It does not remap in a new room: it is a
coordinate system in the anchors' frame, not a map of a particular place.

`results/vggt_units.png` plots all three measures by block. Run with
`scripts/run_vggt_cells.sh` (~70 min on the RTX PRO 6000; single mode is 12,288
runs of 9 images and takes most of it), then `python -m spatialcells.vggt_fields`
for the field-size comparison and `python -m spatialcells.vggt_ablate
[--all-neurons]` for the ablation (~30 min). `scripts/run_vggt_remap.sh` runs
room B and the patching test (~2.5 h). Room B's walks are rendered first, on CPU,
with `python -m spatialcells.render_data --room B --out data/walks-roomB`.
`python -m spatialcells.vggt_patch --blocks 15 19 21 --out
results/vggt_patch_subspaces.json` adds the PCA subspaces (~35 min).
In progress: `python -m spatialcells.vggt_patch --probes --blocks 15 19 21 --out
results/vggt_patch_probes.json` tests whether the probe result depends on how the probe was
regularised. It sweeps the ridge penalty up to 1e6 and also patches the Haufe pattern Cov(H) W
and the cross-covariance direction.

## Experiment 5: why do open models fail, and does any pass?

**The lure is a shared landmark.** `dissimilar_trials.py` ray-casts which posters
and pillars each view shows. In 50 of the 104 original novel trials, the far
option shows a landmark that the new view also shows, and the near option shows
none. The open models fall for it; Opus 5.5 mostly does not:

| original novel trials | far option shares a landmark (50) | neither option does (29) |
|---|---|---|
| Qwen3-VL-8B / SenseNova 8B / Qwen3.8-27B | 18% / 30% / 14% | 41% / 45% / 59% |
| Qwen3.6-35B-A3B / Gemma-4-31B | 8% / 18% | 59% / 52% |
| Claude Opus 5.5 (walk / no walk) | 60% / 56% | 86% / 76% |

**A lure-free trial set** (`results/export/dissimilar/trials.json`, 192 trials,
102 novel). In each trial:
- Both options face more than 90° away from the new view (median 140°).
- Neither option shows any of the new view's landmarks.
- Their pixel similarity to the new view is matched, and favours the near option
  in exactly half the trials.

So pure view matching scores 50%. The near option is within 1 m (median 0.6 m)
and the far one more than 3 m away (median 4.1 m).

One cue is left. The near option is usually closer in frame number to the walk
frame that looks most like the new view (81% of familiar and 66% of novel
trials). A model that remembers when it saw a view could use that. Scoring
reports accuracy on the trials where this cue points the wrong way.

**Results** (`two_choice.py`, a generic Hugging Face runner using the Claude
prompt; novel trials, greedy, no thinking; p is one-sided against 50%):

| novel trials | original, walk | lure-free, walk | lure-free, no walk, 256 px | lure-free, no walk, 512 / 1024 px |
|---|---|---|---|---|
| Qwen3-VL-8B | 38% | 50% (49% at 512 px) | 59% | **71% / 70%** (p < 1e-4) |
| Qwen3.8-27B | 38% | 58% (p = 0.07) | 56% | 64% (p = 0.004) / 55% |
| Qwen3.6-35B-A3B (new, MoE) | 33% | 50% (54% with 128 frames) | 61% | 59% / 59% |
| Gemma-4-31B (other family) | 38% | 53% | 56% | 64% / **68%** (p < 1e-3) |
| Qwen3-VL-8B, original trials, no walk | | | 40% | 38% (512 px) |

- **Without lures and with enough pixels, open models can partly do it.** Shown
  only the new view and the two options, Qwen3-VL-8B gets 71% at 256 tokens per
  frame. At 64 tokens per frame it gets 59%, so the token budget mattered.
  Gemma-4-31B reaches 68%, Qwen3.8-27B 64% (at 512 px) and Qwen3.6-35B-A3B 59-61%.
  Qwen3.8 answers in prose without thinking, so it was given 768 tokens to answer.
  The 8B and Gemma stay above chance (74% and 77% for the 8B) when
  the frame-order cue points the wrong way. The cue isn't usable without the walk
  anyway.
- **A lure overrides it.** On the original trials the same 8B, with no walk at
  512 px, gets 38%: it takes the landmark-sharing far frame.
- **The walk makes it worse.** With the 256-frame walk in context, every open
  model is at chance on the lure-free trials, even at 512 px (256 tokens per
  frame, ~70k-token prompt). Their familiar-trial accuracy with the walk
  (62-69%) fits the frame-order cue more than spatial reasoning.
- **Claude Opus 5.5 is the only model that resists the lure and gains from the
  walk** (75% with it, 66% without, on the original trials). It does the same on
  the lure-free trials, where Sonnet 5.5 and Haiku 5.5 do not (below). (A fine-tuned
  Qwen3-VL-8B later does both: experiment 9.)

**Does thinking help the open models? A little, but not with the walk.** Same
lure-free novel trials, with thinking on: medium effort and the model's sampling
settings, forced to answer after 4,096 thinking tokens (Qwen3.8) or 8,192
(Qwen3.6, which has no effort setting). The no-walk runs use `two_choice.py --think`.
The walk runs use `behave_reason.py --trials dissimilar`, which caches the walk once
per walk; its greedy answers matched `two_choice.py` on 181/192 trials.

| lure-free novel trials (102) | no walk, 256 px | no walk, 512 px | walk, 256 px |
|---|---|---|---|
| Qwen3.8-27B, direct | 56% | 64% | 58% |
| Qwen3.8-27B, thinking | 61% | 63% | **65%** (p = 0.002) |
| Qwen3.6-35B-A3B, direct | 61% | 59% | 50% |
| Qwen3.6-35B-A3B, thinking | 53% | 60% | 52% |

- **Qwen3.8 with thinking is the best open model with the walk (65%)**, and keeps
  63% where frame order points to the far option. But trial by trial, the walk
  is no better than its no-walk thinking run (22 trials better, 18 worse,
  p = 0.32). It does not use the walk the way Opus does (+10-14 points,
  p = 0.008).
- **Qwen3.6 thinks itself into the cap.** With the walk, 117 of 192 answers hit
  8,192 tokens and were forced, and 7% never gave a valid answer. It stays at
  chance.

**Claude models on the lure-free trials** (`claude_test.py --trials dissimilar`,
adaptive thinking at each effort). Opus 5.5 rejects requests with thinking
turned off. Haiku 5.5 accepts `disabled`. Sonnet 5.5 needs `between_tools`
instead, and then reasons in its visible reply (median 390 tokens) despite being
asked for a frame number only.

| lure-free novel trials (102) | thinking off | low | medium | high |
|---|---|---|---|---|
| Haiku 5.5, walk | 60% | 53% | 54% | 56% |
| Haiku 5.5, no walk | 63% | 59% | 62% | 67% |
| Sonnet 5.5, walk | 50% | 52% | 52% | **70%** (p < 1e-4) |
| Sonnet 5.5, no walk | 56% | 52% | 54% | 64% |
| Opus 5.5, walk | | **75%** (p < 1e-6) | **75%** | **75%** |
| Opus 5.5, no walk | | 61% | 66% | 63% |
| Opus 5.5, walk, where frame order points to the far option (35) | | 71% | 71% | 74% |

- **Only Opus gains from the walk.**
  - Opus: comparing walk and no walk trial by trial, averaged over the three
    efforts, the walk version is right more often on 38 novel trials and less
    often on 19 (sign test, p = 0.008). Each effort on its own gives p of 0.02
    to 0.05.
  - Sonnet at high effort: 70% with the walk against 64% without, but trial by
    trial that is 24 trials better and 18 worse (p = 0.22).
  - Haiku: the walk hurts slightly, as it does for the open models (30 trials
    better, 43 worse; at high effort p = 0.04 that it hurts).
- **Sonnet with thinking off, or at low and medium effort, barely does the
  task.** It picks the first-listed option 70-91% of the time. At high effort it
  thinks about twice as long (1,160 tokens) and the bias is gone.
- **Opus doesn't need much thinking.** All three efforts score 75%, and they
  give the same answer on 73% of novel trials. Low effort uses about 380 output
  tokens.
- **The frame-order cue fits the smaller models.** On familiar trials with the
  walk, Haiku and Sonnet get 29-41% where frame order points to the far option.
  So they follow it, like the open models. Opus gets 59-76% there, and 71-74% on
  novel trials.
- **Without the walk, the three Claude models are about equal, and no better
  than the open models.** Their best scores are 67% (Haiku), 64% (Sonnet) and
  66% (Opus), against 71% for Qwen3-VL-8B at 512 px. So guessing from what each
  frame shows of the room doesn't improve with model size. Using the walk does,
  and only at the top.
- Familiar trials with the walk: Haiku 54-62%, Sonnet 53-64%, Opus 79-84%.
- Cost: Opus $22.48, Sonnet $15.97, Haiku $0.99. Per-trial results are in
  `results/claude/<model>-dissimilar*.jsonl`, and the table in
  `results/claude/dissimilar_grid.json`.

Not tested:
- **Hy-Embodied-VLM-1.0** (Tencent, the strongest open model on multi-view
  benchmarks). Its remote code needs transformers 4.57. There it loads with layer
  0's dense MLP and the q/k norms unused and the experts randomly initialised, so
  its answers were noise and were discarded.
- **Qwen3.5-122B-A10B**, the largest Qwen that fits. Its GPTQ-Int4 checkpoint
  needs `gptqmodel` kernels built from source.

Run with `python -m spatialcells.dissimilar_trials`, then
`scripts/run_two_choice_new.sh`, `scripts/run_two_choice_qwen.sh` and
`scripts/run_two_choice_more.sh`, and `python -m spatialcells.two_choice --score`.
The Claude grids run locally with `bash scripts/run_claude_grid.sh <model> > results/claude/grid-<model>.log 2>&1`, and `python scripts/claude_grid_table.py` makes its table.

## Experiment 6: does Opus follow the path? (shuffled walk)

Opus 5.5 is the only model that gains from the walk on the lure-free trials (experiment 5).
Does it use the walk as a path (where it went, step by step) or as a set of reference
views? `claude_test.py` now has three controls, all at low effort on the 192 lure-free trials:
- `--shuffle-walk`: the 256 frames in a fixed random order (per walk), each still labelled
  with its real frame number, same prompt text.
- `--shuffle-walk --relabel`: the same shuffled order, but the frames are relabelled
  1, 2, 3, ... as shown, and the question names the two options by their new labels. No
  order information is left anywhere in the prompt.
- `--walk-frames 64`: only 64 of the 256 frames, in order (every option frame of that walk's
  trials plus frames spread evenly in time).

| lure-free trials, Opus 5.5, low effort | novel (102) | familiar (90) |
|---|---|---|
| walk in order (experiment 5) | 75% | 79% |
| walk shuffled, real frame numbers | **81%** | 79% |
| walk shuffled and relabelled | **75%** | 70% |
| 64-frame walk, in order | 77% | 83% |
| no walk | 61% | 63% |

Paired by trial on the novel trials (two-sided sign test):

| | vs walk in order | vs no walk |
|---|---|---|
| shuffled | 14 better, 7 worse (p = 0.19) | 31 better, 10 worse (p = 0.001) |
| shuffled and relabelled | 11 better, 10 worse (p = 1.0) | 27 better, 12 worse (p = 0.02) |
| 64 frames | 12 better, 9 worse (p = 0.66) | 29 better, 12 worse (p = 0.01) |

- **The walk's order doesn't matter.** Shuffling the frames costs nothing, and the gain over
  no walk stays (+14 to +20 points).
- **Not even the frame numbers' order.** With real frame numbers, the shuffled run's thinking
  often rebuilds the path from them ("frames 246-253 ... 132-139"; 66 of 192 summaries
  mention frame ranges or sequences, 23 for the ordered walk). The relabelled run can't do
  that (1 of 192 mention it), and still scores 75% on novel trials.
- **64 frames are enough.** A quarter of the walk gives the same score.
- So Opus uses the walk as a **set of reference views** of the room, not as a path it
  integrates. Its thinking describes the same thing: it works out which walls, posters and
  pillars surround each view and compares layouts.
- Caveats: one run per condition, at low effort only. The shuffled run's 81% is probably
  noise: two ordered runs at different efforts disagree on about as many trials (84/102 same
  answer) as shuffled vs ordered (81/102). Familiar trials drop to 70% when relabelled
  (9 better, 17 worse than ordered, p = 0.17). That fits the frame-order cue, which is gone
  there, but it is not significant.
- Cost: $4.71 (shuffled) + $4.95 (relabelled) + $2.30 (64 frames) = $11.96. Per-trial results
  are in `results/claude/claude-opus-5-5-dissimilar-{shuffled,shuffled-relabel,f64}-low.jsonl`,
  the comparison in `results/claude/shuffled_walk.json`.

## Experiment 7: can our scorer find a grid cell at all? (positive control)

Every model so far had "no grid cells", but the gridness scorer in `analyze.py` had never been
shown to find one. Two checks, both through `analyze.analyze_layer` unchanged (rate maps,
autocorrelogram, gridness, the layer's pooled 99th-percentile shuffle threshold, split-half
stability), along the same 48 walks every model was scored on (`grid_rnn.py`).

**Synthetic grid cells** (`--synthetic`). Ideal hexagonal grids (random orientation and
phase, field width 0.16 x spacing) read out along the walks, plus Gaussian noise (SD as a
fraction of the peak), mixed with as many place-like distractors:

| grid spacing | 1 m | 1.5 m | 2 m | 3 m | 4 m | 5 m |
|---|---|---|---|---|---|---|
| no noise | 0% | 2% | 100% | 100% | 100% | 100% |
| noise 0.5 x peak | 0% | 0% | 85% | 100% | 100% | 100% |
| noise 1 x peak | 0% | 0% | 0% | 92% | 100% | 100% |

Place-like distractors flagged as grid cells: 0.1%. So the scorer works for grids with spacing
of 2 m or more. Finer grids are invisible, because the rate maps use 0.5 m bins and smoothing.

**A trained grid network** (`--train`). A path-integrating RNN following Sorscher et al.
(ganguli-lab/grid-pattern-formation): 4,096 ReLU units, velocity input only, trained to
predict 512 difference-of-Gaussians place cells; their 2.2 m box scaled up to our 10 m room
(place cells 0.55 m wide); foraging walks from `room.foraging_walk`; 100k steps. Units are read
along data/walks with the state reset from the true position every ~20 network steps (as in
training; "anchored"). It took a few tries:

| RNN run | grid cells flagged (data/walks) | dense maps: units with gridness > 0.3 | position error |
|---|---|---|---|
| foraging walks, 0.22 m steps (our walks' step), TF32 | 0 of 4,040 | 0.2% | 0.17 m |
| **foraging walks, 0.093 m steps (Sorscher's, scaled), fp32** | **145 (3.6%)** | **15%** | 0.17 m |
| same, weight decay from step 1 instead of after 5k steps | 20 (0.6%) | 2.7% | 0.17 m |
| Sorscher's own trajectory generator, weight decay from step 1 | 20 (0.5%) | | 0.17 m |

"Dense maps" use 1,000 new walks (728 samples per bin instead of ~30) to see what the units
really are, independent of our sampling. The first run learned to path-integrate but formed
patchy, square-ish multi-field units, not hexagons (`results/grid_rnn_base_dense1000.png`).
With Sorscher-sized steps it formed clean hexagonal grids with ~3 m spacing
(`results/grid_rnn_s093_anchored.png`, `results/grid_rnn_s093_dense1000.png`). Weight decay
from the first step shrank the recurrent weights before the task signal grew (in the first run
the loss stuck at chance), so the main runs start it after 5,000 steps. The two runs that
started it at step 1 trained fine with small steps but formed far fewer grids.

**How sensitive is the scorer at our sampling?** For the 0.093 m run, compared with its dense
maps:

| units whose dense map has gridness | > 0.3 (613) | > 0.5 (314) | > 0.8 (81) |
|---|---|---|---|
| flagged along data/walks | 24% | 46% | 93% |

None of the 145 flagged units had a dense gridness below 0.3.

- **The scorer finds grid cells when they exist.** A clear grid cell with spacing of 2-3 m or
  more is flagged at our sampling (93% of strong ones), with almost no false positives.
- **It is conservative.** With ~30 samples per 0.5 m bin, the shuffle threshold is high
  (0.6-0.7), so weaker grid-like units are missed (about half of those with gridness 0.5-0.8).
- **So the earlier "no grid cells" results hold for coarse grids only.** In pixels, DINOv2,
  Qwen and VGGT, any grid cells with spacing of 2 m or more and a clean pattern would have been
  found. Grids finer than ~1.5 m could not have been seen.
- Free-running path integration (no resets over 255 steps) drifts (1.6 m error by the end) and
  halves the flagged count (74).
- Caveats: one seed per run, 100k steps. Getting grids at all depended on the trajectory
  statistics and the weight-decay schedule, as Schaeffer, Khona & Fiete (2022) warned.

Run with `scripts/run_grid_rnn.sh` (~1 h for the three retrains) and
`python -m spatialcells.grid_rnn --synthetic` (CPU, 30 s); results in `results/grid_rnn.json`.

## Experiment 8: can we steer VGGT's place neurons?

Experiment 4 found heading-invariant place neurons in VGGT's late blocks and a 10-d position
code in the residual stream that the camera head uses (interchange transfer 0.40 at block 19,
0.85 at block 21). Ablating the neurons did nothing. Does *writing* into them move VGGT's
predicted camera? `vggt_steer.py`, single mode (8 anchors + 1 walk frame), 96 test frames
from odd walks. Block 21 was added to the recorded activations (`vggt_cells.py --blocks 21`).

Place neurons, found with `analyze.analyze_layer` as before:

| | block 19 | block 21 |
|---|---|---|
| heading-invariant place neurons | 951 | 1,981 |
| heading-specific place neurons | 532 | 1,377 |
| invariant place units on the positive side of the neuron | 33 | 17 |

Most invariant place units are the *negative* side of a neuron (the small negative lobe of
GELU), so only a few can be driven "above their peak" in the usual sense.

**(a) Population steering.** Each test frame gets a target location L, the centre of a 1 m
square on average 5.7 m from the camera's predicted position. The neurons are set to their
average activity in that square (from the even walks, all headings). "Transfer" is how far the
prediction moves toward L: 0 = no effect, 1 = placed at L. Reference: the whole residual
token after the block set to its average in L's square.

| written at the walk frame's camera token | block 19 | block 21 | blocks 19 + 21 |
|---|---|---|---|
| **reference: whole residual token → its average at L** | **0.29** (moves 1.7 m) | **0.64** (moves 4.0 m) | |
| invariant place neurons → their average at L | −0.001 | −0.002 | −0.003 |
| heading-specific place neurons (same count) | −0.001 | 0.000 | −0.001 |
| random neurons (same count as invariant) | 0.000 | −0.001 | −0.001 |
| all 4,096 neurons | −0.001 | −0.003 | −0.004 |
| invariant, pushed 4x as far from their mean | −0.005 | −0.007 | −0.011 (moves 11 cm) |
| all 4,096, pushed 4x | −0.006 | −0.010 | −0.016 (moves 15 cm) |
| top 8 / 32 / 128 invariant by spatial information | ≤ 0.001 | ≤ 0.001 | ≤ 0.002 |

- **Writing a location into the neurons does not move the camera there.** Every neuron
  condition moves the prediction by 1-15 cm, not toward L. There is no dose-response: more
  neurons or a 4x push only make the small shift larger.
- **If anything, it moves slightly away from L**, consistently (−0.003 for the invariant
  neurons in both blocks, −0.011 at 4x, each many standard errors below 0).
- Writing the same location into the whole residual token does work (29% and 64% of the way).
  So the location is carried by the residual stream as a whole, not by these neurons' outputs
  at that token.

**(b) One neuron at a time.** Each invariant place unit on the positive side (33 in block 19,
17 in block 21) is set to 4x its largest activation on any walk frame, on 16 test frames.

| | block 19 | block 21 |
|---|---|---|
| prediction moves (median) | 3.0 cm | 6.6 cm |
| same for 50 random neurons with no place field | 0.3 cm | 0.2 cm |
| cosine between the move and the direction to the unit's field centre | −0.07 ± 0.06 | −0.18 ± 0.08 |
| units whose move points toward their field centre | 15 of 33 | 4 of 17 |
| same cosine with another unit's field centre (control) | +0.02 | +0.09 |

- A place unit moves the prediction about 10 times more than a random neuron does, so these
  units matter more than most. But the move doesn't point toward the unit's field. In block 21
  it tends to point away (13 of 17 units; two-sided sign test p ≈ 0.05).

**(c) Joystick.** Inside the 10-d position-PCA subspace of the residual token after the block
(as in `vggt_patch.py`), least squares on the fit frames gives the direction that changes as the
camera moves 1 m east, and the one for 1 m north. alpha times that direction is added to the
residual at the camera token, on 64 test frames.

| | block 19 | block 21 |
|---|---|---|
| predicted move per "metre" added, east / north | 0.16 / 0.15 m | 0.18 / 0.17 m |
| straight line? (R² of move vs alpha, alpha = −3 ... +3) | 1.00 / 1.00 | 1.00 / 1.00 |
| sideways move per metre (east → north, north → east) | 0.01 / −0.02 m | 0.01 / −0.04 m |
| heading change at alpha = ±3 | 1-2° | 3-4° |
| random direction, same size, alpha = 3 | 1 cm | 6 cm |

- **The position subspace works as a joystick.** Pushing "east" moves the predicted camera
  east and "north" moves it north, in a straight line, with little sideways drift and
  little heading change. A random direction of the same size does almost nothing.
- **But the gain is about 1/6.** Adding the token change that goes with 1 m of real movement
  moves the prediction only 16-18 cm. Only 11-21% of the variation inside the subspace is a
  straight-line function of (x, y) (the rest varies with heading and in curved ways), so one
  straight direction is only part of how the code changes with position.

**Summary.** VGGT's place neurons are not knobs for position. Writing a location into them
(as a group or one at a time, in block 19 or 21 or both) barely moves the camera, and slightly
in the wrong direction. The location lives in the residual stream: overwriting the token moves
the camera most of the way, and a straight push inside the 10-d position subspace steers it
smoothly, at about a sixth of the expected gain. This fits experiment 4: the neurons
*read out* position (they are place-tuned), but at this token their MLP output is not how the
camera head gets position. Caveats: steering only at the walk frame's camera token, in one or
two blocks. Neurons at the patch tokens, or in all blocks at once, were not tested. The neuron
targets are averages over all headings, which no real frame has.

`results/vggt_steer.json`, `results/vggt_steer.png`. Run with `scripts/run_steer.sh` (block-21
extraction ~55 min, then the steering ~45 min).

## Experiment 9: teaching a small model the skill

Can an open 8B model learn what Opus does, and do place units appear when it does?

**Training data** (`procedural_data.py`). `room.random_layout(seed)` makes procedural rooms: the
same 10 m box with random wall and floor patterns and colours, 1-3 posters per wall cut from
skimage's bundled pictures (sometimes a plain door-like panel), and 1-3 pillars at random places
(`results/preview/procedural.png`). Each room gets one 256-step foraging walk, of which the 64
frames a 64-image prompt shows are rendered (frame numbers 1, 5, 9, ... 256), plus 40 teleport
views. Two-choice trials follow the lure-free rules of experiment 5, with options from the shown
frames: near within 1 m, far more than 3 m away, both facing more than 90° away from the new view,
neither showing its posters or pillars. Pixel similarity and frame order (which option is nearer
in time to the most similar-looking frame) each favour the near option in exactly half the trials,
including each room's first question. 1,200 rooms gave 9,595 trials (8 per room, 42% novel;
near / far medians 0.59 / 4.07 m). Room A was never generated.

**Fine-tuning** (`finetune.py`). LoRA (rank 16) on every attention and MLP projection of
Qwen3-VL-8B's language model (44M parameters, 0.5%); vision tower frozen. Each sequence is one
room: the walk's 64 frames in the two-choice prompt, then its 8 questions as chat turns, with the
loss on the answers only. 1,136 training rooms, 64 held out, 2 epochs (284 updates of 8 rooms,
96 min, 29 GiB peak).

The loss sat at the coin-flip level (ln 2 per answer) for ~130 updates, with held-out accuracy
42-59%. Then it dropped, and held-out accuracy rose to 69% at the end of epoch 1 and 92% at the
end.

**Room A, lure-free trials** (`two_choice.py --adapter`; 64-frame walks unless noted):

| | novel (102) | familiar (90) |
|---|---|---|
| base Qwen3-VL-8B, 64-frame walk | 50% | 57% |
| base, 256-frame walk (experiment 5) | 50% | 66% |
| base, no walk | 59% | 47% |
| **fine-tuned, 64-frame walk** | **92%** (p = 5e-20) | **91%** |
| fine-tuned, full 256-frame walk | 79% | 86% |
| fine-tuned, no walk | 53% | 56% |
| fine-tuned, walk shuffled (real frame numbers) | 50% | 51% |
| fine-tuned, walk shuffled and relabelled | 50% | 54% |
| Opus 5.5, low effort, 256 / 64-frame walk (experiments 5-6) | 75% / 77% | 79% / 83% |

Paired on the novel trials: fine-tuned vs base 44 trials better, 1 worse (p = 3e-12); walk vs
no walk 43 better, 3 worse; shuffled vs ordered walk 4 better, 47 worse (p = 2e-10).

- **The skill transfers to the held-out room.** Trained only on procedural rooms, the 8B model
  gets 92% of room A's novel trials, more than Opus 5.5.
- **It is not a shortcut we balanced for.** It is right 91% of the time when frame order points to
  the far option, it picks the more similar-looking option 53% of the time (view matching would
  be 50%), and first-listed 47%.
- **It also resists the lures.** On the original trials, where the far option shares a landmark
  with the new view, it gets 91% (base 37%).
- **It needs the walk, in order.** Without the walk it is at chance. With the same frames
  shuffled it is at chance, even with their real frame numbers. That is the opposite of Opus,
  which uses the walk as a set of views and doesn't care about the order (experiment 6). The
  fine-tuned model uses the path.
- **It is more than "find a look-alike frame, then pick the option nearer in time".** That rule
  (`adjacency_baseline.py`), with pixels or DINOv2 features, gets 58-69% on the novel trials.
- It generalises partly to a 4x longer walk than it was trained on (79% with 256 frames).

**Did place units appear?** `extract.py --model qwen8b-ft` repeats experiment 2's recordings
(256-frame walks, teleport probes, single frames) with the adapter merged; `analyze.py` and
`analyze_probes.py` score them as before.

| | base Qwen3-VL-8B | fine-tuned |
|---|---|---|
| single frame: place units that are heading-invariant | 1.0-1.9% | 1.1-2.2% |
| single frame: cross-heading decoding (best layer) | 3.55 m | 3.45 m |
| walk in context, layers 23-35: place units that are heading-invariant | 1.2-3.8% | 7.5-11.1% |
| walk in context, layers 23-35: cross-heading decoding | 2.20-2.94 m | 1.73-2.18 m |
| walk in context, best layer (layer 3) | 1.23 m | 1.31 m |
| one frame + 8-step average (best layer) | 0.40 m | 0.39 m |
| teleport probes after the walk: cross-heading decoding (best layer) | 3.24 m | **2.78 m** (layer 35) |
| same, after a shuffled walk / with no walk | 3.22 / 3.29 m | 2.81 / 3.14 m |
| teleport probes: decoding from views facing the *opposite* way | 3.7-4.6 m | 3.1-3.9 m |
| grid units | ≤ 0.2% | ≤ 0.2% |

(Chance for decoding is 3.62 m.)

- **A little, in the late layers.** With the walk in context, the fine-tuned model's late layers
  keep more heading-invariant place units (7-11% vs 1-4%) and decode position across headings
  better. Teleport probes (the clean test: no smoothing possible) decode across headings at
  2.78 m instead of 3.24 m, and the walk now helps by 0.36 m (vs 0.05 m in the base).
- **But nothing like VGGT.** Position from opposite-facing views is still at about chance at the
  probe's token (VGGT single mode: 0.97 m across headings from one frame). Single frames are
  unchanged, and the 8-step pixel-average baseline still beats every layer. There are no grid
  units.
- So the model learned the behaviour much more than it learned a place code at the new view's
  token. The comparison probably happens when the question is asked, by attending back over
  the walk. These recordings read each image's last token with no question in the prompt.
- Caveats: one training run (one seed). Room A shares picture and pattern families with the
  procedural rooms, though not their layout. The 192 room-A trials are the same ones every model
  got. The unit analysis uses experiment 2's prompts (256 frames, no question), not the trained
  format.

Run with `python -m spatialcells.procedural_data --rooms 1200` (CPU, ~10 min), then
`scripts/run_finetune.sh` (~2.5 h: training 96 min, evaluation, recordings and analyses),
`scripts/run_ft_shuffle.sh` and `python -m spatialcells.adjacency_baseline`. Per-trial results
are in `results/two_choice/Qwen3-VL-8B-Instruct+qwen8b-proc-*.jsonl`, and the comparison is in
`results/two_choice/finetune_comparison.json`. The adapter is at `data/ft/qwen8b-proc` on the host.

## Experiment 9b (in progress): training seeds and a second room

Two more LoRA seeds with the same data and settings, each tested on room A's and room B's
lure-free trials (64-frame walk, no walk, shuffled walk). Run with `scripts/run_ft_seed.sh SEED`;
`python -m spatialcells.ft_seeds` summarises the three seeds. Results will be added here.

## Prior work this builds on

- Luo, Mok & Love 2024, *The inevitability and superfluousness of cell types in
  spatial cognition*: image networks (even untrained ones) have "place",
  "border" and "head-direction" units on single frames.
- Banino et al. 2018; Cueva & Wei 2018; Schaeffer, Khona & Fiete 2022: grid cells
  in navigation-trained RNNs, and how much they depend on design choices.
- Whittington, Warren & Behrens 2022: transformers with recurrent positional
  encodings produce grid and place cells.
- Gurnee & Tegmark 2024: linear space and time representations in Llama-2.
- Park et al. 2025: in-context random walks on a grid reorganise LLM token
  representations into the grid.
