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
Qwen3-VL-8B fine-tuned on 8M spatial QA samples; its model card reports 64.8% on VSI-Bench
and 73.8% on MindCube-Tiny for this 8B model (the card's headline 68.7% and 85.6% are for the
SenseNova-SI family). Against the base model on every test:

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
(two-sided p = 0.016).

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

**Was it how the probe was chosen? Partly** (`vggt_patch.py --probes`,
`results/vggt_patch_probes.png`). The probe above uses the ridge penalty λ with
the best held-out R². That choice can favour "filters" (Haufe et al. 2014):
weights that read low-variance directions to cancel noise, not the directions
along which position moves the token. So the test was repeated, with the same
targets, donors and fit frames, for 2-d subspaces for (x, y) chosen in other ways:
- the ridge probe at each λ from 0.1 to 10⁶;
- the activation pattern Cov(H) W of the best-R² probe (Haufe et al. 2014);
- the cross-covariance direction Hᵀ Y, the regression version of a mass-mean
  probe (Marks & Tegmark 2023);
- the top two position-PCA directions, a 2-d match for the 10-d subspace.

"R²" is the probe's own held-out R², and in brackets that of a least-squares
readout of (x, y) from the same two directions, which is not lowered by the
shrinkage of a heavily regularised probe. "Erase" sets the two directions to
their mean over the fit frames.

| 2 dims written into the target's token | R² at block 21: probe (readout) | share at block 21 | block 15 | block 19 | block 21 (± SEM) | erase at block 21: error (clean 0.43 m) |
|---|---|---|---|---|---|---|
| ridge, λ = 0.1 (best R² at blocks 15, 19) | 0.99 (0.99) | 0.02% | 0.00 | 0.00 | 0.00 ± 0.00 | 0.43 m |
| ridge, λ = 1 (best R² at block 21) | 0.99 (0.99) | 0.09% | 0.00 | 0.00 | 0.00 ± 0.00 | 0.43 m |
| ridge, λ = 10 | 0.99 (0.99) | 0.4% | 0.01 | 0.01 | 0.01 ± 0.00 | 0.43 m |
| ridge, λ = 100 | 0.99 (0.99) | 2.0% | 0.01 | 0.04 | 0.03 ± 0.00 | 0.44 m |
| ridge, λ = 10³ | 0.98 (0.98) | 7.6% | 0.02 | 0.13 | 0.11 ± 0.01 | 0.64 m |
| ridge, λ = 10⁴ | 0.95 (0.96) | 13% | 0.03 | 0.17 | 0.18 ± 0.01 | 0.84 m |
| ridge, λ = 10⁵ | 0.77 (0.93) | 13% | 0.02 | 0.18 | 0.19 ± 0.01 | 0.88 m |
| ridge, λ = 10⁶ | 0.22 (0.84) | 15% | 0.02 | 0.18 | **0.20 ± 0.01** | 1.00 m |
| Haufe pattern Cov(H) W of the best probe | (0.72) | 17% | 0.02 | 0.18 | **0.20 ± 0.01** | 1.10 m |
| cross-covariance Hᵀ Y | (0.72) | 17% | 0.02 | 0.18 | **0.20 ± 0.01** | 1.10 m |
| position PCA, 2 dims | (0.52) | 22% | 0.02 | 0.18 | 0.18 ± 0.02 | 1.10 m |
| position PCA, 10 dims | (0.95) | 93% | 0.03 | 0.40 | **0.85 ± 0.02** | 3.38 m |

- **Transfer rises as the probe is regularised more, and when it is switched to
  the pattern.** At block 21 it goes from 0.00 for the best-R² probe to 0.20 at
  λ = 10⁶ or with the pattern (block 19: 0.00 to 0.18). Erasing those
  directions raises the error from 0.43 m to 1.0-1.1 m. Most of the rise has
  come by λ = 10⁴, where the probe still has R² 0.95.
- **The best-R² probe reads filters.** Its directions hold 0.02-0.09% of the
  target-donor difference at block 21. Heavier penalties tilt it toward the
  high-variance directions the pattern points along (15% at λ = 10⁶).
- **The pattern and the cross-covariance direction are the same subspace.**
  Cov(H) W is proportional to Hᵀ Y − λW, which is almost Hᵀ Y at the small λ
  the best probe uses. The two give the same share, R² and transfer at every
  block.
- **No pair of directions comes close to the 10-d subspace.** The most
  regularised probe, the pattern and the top two position-PCA directions all
  transfer 0.18-0.20 at block 21, against 0.85 for position PCA 10. At block 19
  it is 0.18 against 0.40. At block 15 nothing transfers.

The way the probe was chosen explains why its own directions do nothing: a
probe picked for R² reads filters. It does not explain the rest of the gap.
Pointed the right way, two linear directions for (x, y) carry about a fifth of
the effect at block 21. The code the camera head uses has more dimensions than
the two coordinates it encodes.

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
`python -m spatialcells.vggt_patch --probes --blocks 15 19 21 --out
results/vggt_patch_probes.json` runs the probe-choice test and draws
`results/vggt_patch_probes.png` (3.6 h on an RTX 4070 with `--batch 2`; it reproduced
the clean error, 0.43 m, and block 21's position-PCA-10 transfer, 0.85).

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
  two-sided p = 0.64). It does not use the walk the way Opus does (+10-14 points,
  two-sided p = 0.016).
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
    often on 19 (two-sided sign test, p = 0.016). Each effort on its own gives
    two-sided p of 0.03 to 0.11.
  - Sonnet at high effort: 70% with the walk against 64% without, but trial by
    trial that is 24 trials better and 18 worse (two-sided p = 0.44).
  - Haiku: the walk hurts slightly, as it does for the open models (30 trials
    better, 43 worse; at high effort 11 better and 22 worse, two-sided p = 0.08).
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
  gets 92% of room A's novel trials, more than Opus 5.5. This run turned out to be the best of three
  training seeds (experiment 9b: 70-92%, mean 82%).
- **It is not a shortcut we balanced for.** It is right 91% of the time when frame order points to
  the far option, it picks the more similar-looking option 53% of the time (view matching would
  be 50%), and first-listed 47%.
- **It also resists the lures.** On the original novel trials it gets 91% (base 37%), and on the
  50 where the far option shares a landmark with the new view, 88% (base 14%).
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
- Caveats: one training run (experiment 9b adds two seeds and room B). Room A shares picture and pattern families with the
  procedural rooms, though not their layout. The 192 room-A trials are the same ones every model
  got. The unit analysis uses experiment 2's prompts (256 frames, no question), not the trained
  format.

Run with `python -m spatialcells.procedural_data --rooms 1200` (CPU, ~10 min), then
`scripts/run_finetune.sh` (~2.5 h: training 96 min, evaluation, recordings and analyses),
`scripts/run_ft_shuffle.sh` and `python -m spatialcells.adjacency_baseline`. Per-trial results
are in `results/two_choice/Qwen3-VL-8B-Instruct+qwen8b-proc-*.jsonl`, and the comparison is in
`results/two_choice/finetune_comparison.json`. The adapter is at `data/ft/qwen8b-proc` on the host.

## Experiment 9b: training seeds and a second room

Experiment 9 rests on one training run tested in one room. This repeats the fine-tune with two more
seeds and tests all three runs in room B as well as room A. `finetune.py --seed` changes the LoRA
initialisation, the dropout and the order of the training rooms, while the data, the held-out rooms
and every other setting stay the same (seed 0 is experiment 9's run). Room B is experiment 4's
second room (the same 10 m box with other walls, floor, posters and pillar positions). Its
lure-free trials come from `dissimilar_trials.py --room B` under room A's rules (near option within
1 m, far more than 3 m away, both facing more than 90° away from the new view, no shared poster or
pillar), with one addition: frame order is balanced as well as pixel similarity, as in the training
rooms. Each favours the near option in 50-51% of both the novel and the familiar trials, whereas in
room A frame order favours it in 66% of the novel trials. That gives 192 trials (106 novel, 86
familiar; near / far medians 0.62 / 4.14 m). Room B is a second layout rather than a different kind
of room: its wall patterns (tiles, stripes, plain, panels) and its posters (skimage's camera, retina,
coins, moon, logo and page) belong to families the procedural generator also draws from.

**Training.** The three runs differ mainly in how long they stay at the coin-flip loss before they
start to learn (`results/two_choice/ft_seed_logs/train_log_s*.json`; loss per answer token, where a
coin flip gives ln 2 / 4 ≈ 0.17):

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| update at which the loss first falls below 0.15 (of 284) | 135 | 175 | 215 |
| loss over the last 5 updates | 0.034 | 0.079 | 0.137 |
| held-out procedural rooms, first question, end of training | 92% | 84% | 69% |

**Results** (`python -m spatialcells.ft_seeds` -> `results/two_choice/ft_seeds.json`; 64-frame walks;
base Qwen3-VL-8B and seed 0 in room B ran on Apple MPS, see the platform note below):

| room A, lure-free trials | base | seed 0 | seed 1 | seed 2 | three seeds: mean (range) |
|---|---|---|---|---|---|
| novel (102), walk | 50% | 92% | 83% | 70% | **82%** (70-92%) |
| novel, no walk | 59% | 53% | 56% | 57% | 55% (53-57%) |
| novel, walk shuffled (real frame numbers) | | 50% | 58% | 62% | 57% (50-62%) |
| familiar (90), walk | 57% | 91% | 88% | 72% | 84% (72-91%) |
| familiar, no walk | 47% | 56% | 56% | 43% | 51% (43-56%) |
| familiar, walk shuffled | | 51% | 66% | 58% | 58% (51-66%) |

| room B, lure-free trials | base | seed 0 | seed 1 | seed 2 | three seeds: mean (range) |
|---|---|---|---|---|---|
| novel (106), walk | 54% | 95% | 80% | 62% | **79%** (62-95%) |
| novel, no walk | 46% | 60% | 52% | 48% | 53% (48-60%) |
| novel, walk shuffled (real frame numbers) | | 52% | 61% | 56% | 56% (52-61%) |
| familiar (86), walk | 50% | 95% | 83% | 72% | 83% (72-95%) |
| familiar, no walk | 51% | 52% | 51% | 42% | 48% (42-52%) |
| familiar, walk shuffled | | 55% | 70% | 57% | 60% (55-70%) |

Opus 5.5 at low effort with the 256-frame walk gets 75% of room A's novel trials (61% without the
walk) and 64% of room B's (63% without; `results/claude/claude-opus-5-5-roomB{,-nowalk}-low.jsonl`).

Paired by trial on the novel trials, with the walk (trials better / worse, two-sided sign test):

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| room A: vs base | 44 / 1 (p = 3e-12) | 39 / 5 (p = 1e-7) | 29 / 9 (p = 0.002) |
| room A: vs the same model without the walk | 43 / 3 (p = 5e-10) | 33 / 5 (p = 4e-6) | 28 / 15 (p = 0.07) |
| room A: vs the same model, walk shuffled | 47 / 4 (p = 2e-10) | 34 / 8 (p = 7e-5) | 21 / 13 (p = 0.23) |
| room A: vs Opus 5.5 | 23 / 5 (p = 9e-4) | 20 / 11 (p = 0.15) | 18 / 23 (p = 0.53) |
| room B: vs base | 45 / 1 (p = 1e-12) | 34 / 6 (p = 8e-6) | 22 / 13 (p = 0.18) |
| room B: vs the same model without the walk | 39 / 2 (p = 8e-10) | 39 / 9 (p = 2e-5) | 32 / 17 (p = 0.04) |
| room B: vs the same model, walk shuffled | 50 / 4 (p = 4e-11) | 28 / 8 (p = 0.001) | 19 / 12 (p = 0.28) |
| room B: vs Opus 5.5 | 36 / 3 (p = 4e-8) | 28 / 11 (p = 0.009) | 23 / 25 (p = 0.89) |

- **The training seed matters a lot, and experiment 9's run was the best of three.** Over the three
  seeds the fine-tuned model gets 82% of room A's novel trials (range 70-92%) and 79% of room B's
  (62-95%). Every seed beats the base model in room A, and seeds 0 and 1 do in room B, where seed 2's
  62% is not significantly above the base model's 54%. The claim that the fine-tuned 8B does better
  than Opus 5.5 holds on average and for seed 0, but seed 1's lead in room A is not significant and
  seed 2 does slightly worse than Opus in room A and about as well in room B.
- **The spread follows how long each run took to leave the coin-flip plateau.** All three runs first
  sit at the coin-flip loss, and the later a run leaves it (update 135, 175, 215), the less learning
  rate the cosine schedule has left and the higher its final loss. Held-out accuracy on the training
  generator's own rooms ranks the seeds in the same order (92%, 84%, 69%) as rooms A and B do. Seed 2
  therefore looks unfinished rather than broken, and the two-epoch schedule is too short to train
  this skill reliably. Whether longer training closes the gap was not tested.
- **Room B shows what room A shows, with frame order balanced as well.** In room B the trials where
  frame order points to the far option are half the novel trials (53). The seeds get 96%, 75% and
  58% of them; they pick the more similar-looking option 52%, 56% and 60% of the time and the
  first-listed one 50%, 52% and 43% of the time. So the strongest run does not rely on the cues the
  trials balance, and the weakest leans slightly towards the look-alike.
- **The dependence on the walk's order is clearest in the strongest run.** Seed 0 falls to chance
  when the walk is shuffled, in both rooms (50% and 52%). Seeds 1 and 2 keep 56-62% with a shuffled
  walk, and for seed 2 the ordered walk is not significantly better than the shuffled one in either
  room. The weaker runs therefore use something a shuffle leaves intact, which these tests do not
  identify.
- **Without the walk, every model mostly picks the second-listed option** (67-82% of its answers on
  novel trials). Averaging accuracy over the two listing orders gives 53-59% in room A and 47-53% in
  room B, close to chance, except for seed 0 in room B at 61% (48 of 52 right when the near option is
  listed second, 16 of 54 when it is listed first). With the walk the fine-tuned runs pick the
  first-listed option 43-52% of the time, so their walk-versus-no-walk gains are not a listing
  artefact.
- **Opus gains nothing from the walk in room B** (64% vs 63%), unlike room A (75% vs 61%), so the
  fine-tuned model's lead over Opus is larger there for seeds 0 and 1.
- Caveats: the three seeds share one recipe (data, held-out rooms, hyperparameters), so this
  measures variation between training runs, not between recipes. Seeds 1 and 2 also trained on a
  different GPU type from seed 0 (below), so their difference from seed 0 mixes seed and platform,
  though the plateau-escape account fits all three runs (resolved by the GPU control below: a second
  run of the seed-0 configuration on that GPU reaches the original's level). Rooms A and B both share picture and
  pattern families with the training rooms. Experiments 10 and 12 tested seed 0 only.

Run with `python -m spatialcells.render_probes --room B --data data/walks-roomB` and
`python -m spatialcells.dissimilar_trials --room B` (CPU; room B's walks from
`render_data --room B --out data/walks-roomB`), then `scripts/run_roomB.sh` for base and seed 0 in
room B and `scripts/run_ft_seed.sh SEED` (~2 h each: training, then the six tests in rooms A and B)
for seeds 1 and 2, and `python -m spatialcells.ft_seeds`. Per-trial results are in
`results/two_choice/Qwen3-VL-8B-Instruct{,+qwen8b-proc,+qwen8b-proc-s1,+qwen8b-proc-s2}-{dissimilar,roomB}-256px-*.jsonl`.
The seed-1 and seed-2 adapters are at `data/ft/qwen8b-proc-s{1,2}` on vast-3 (seed 0 on vast-1).

*Platform (2026-10-25):* room B's runs of the base model and seed 0 ran on Apple silicon
(`two_choice.py --device mps`, tags ending `-mps`), after the same command reproduced the CUDA
answers of seed 0 on room A trial for trial (192 of 192); room B's walks on the Mac are byte-identical
to vast-1's. Seeds 1 and 2 are trained and tested on vast-3's RTX 5090 (32 GB), not vast-1's
RTX PRO 6000 (seed 0). The training code (`run_ft_seed.sh`, `finetune.py`) is unchanged, the packages
are vast-1's exact versions (`requirements-lock.txt`), and the training data are seed 0's byte for byte (all 1,205 files of
`data/proc` checked by md5). Regenerating them would not do: `room.py` draws `skimage.data.binary_blobs()`
posters without a seed, so the 242 rooms with one differ on every run, and those rooms and `trials.json`
were copied from vast-1. Training also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which
changes memory layout, not the arithmetic (seed 0 peaked at 29 GiB). The tests use the current
`two_choice.py`; before the move, it reproduced on vast-3 vast-1's CUDA answers for the seed-0 model on
room A (48 of 48 trials, the same answers and text).

*Amendment to experiments 9b, 10, 12 and 16 (2026-10-25, after the room-A/B seed results, before any
seed-1 or seed-2 answer in these scenes): all three seeds everywhere, and a GPU control.* Seed 0, used for
every fine-tuned result so far, is the best of three (room A 92 / 83 / 70%). So:
1. *GPU control:* seed 0 is retrained on vast-3's RTX 5090 with the identical script, data and settings
   (seeds 1-2 trained there; the original seed 0 on vast-1's RTX PRO 6000), then scored on room A and room
   B. If its room-A novel accuracy is within the range of the other seeds' spread of the original (i.e.
   at or above 83%) and its loss leaves the plateau at a similar update, the spread is attributed to the
   seeds; otherwise the GPU is reported as a possible cause and the three-seed numbers are qualified.
2. *Seeds 1-2 in every scene:* the ten held-out rooms (experiment 10), the three apartments (experiment
   12, which registered "seeds 1-2 when trained") and the eight houses (experiment 16), each with the
   64-frame walk, no walk and the shuffled walk, as for seed 0. Analysis: each experiment's registered
   contrasts per seed, and the three-seed mean with the range; the per-seed results are reported
   whichever way they go.
Platforms: vast-3 (CUDA) for scenes whose renders it has or can re-render byte-identically; the Mac
(MPS) for the apartments, HSSD2-3 and P5005 (its blobs poster cannot be re-rendered). Every run records
its platform.
*vast-3 inputs (2026-10-25):* the nine re-rendered held-out rooms (P5000-P5009 without P5005, with
experiment 10's commands) gave frames and probes byte-identical to the Mac's. Their walk poses differ by
at most 6e-14 m (rounding differences between the Mac's and vast-3's floating-point maths, which add up
along a walk); `two_choice.py` reads only the walk and step columns. The Mac's pose files and
`trials.json` were copied over, so all 36 walk files match the Mac's by md5. The houses (HSSD1, HSSD4-8)
are md5-identical copies of the Mac's. The GPU control is `run_ft_seed.sh 0 data/ft/qwen8b-proc-s0-5090`
(the script now takes the adapter folder as an optional second argument; nothing else changed).

**GPU control result (2026-10-25, gq vast3:152, 2.2 h).** A second run of the seed-0 configuration on
the RTX 5090 reaches the original seed 0's level, so by the registered criterion the spread between
the three runs is attributed to run-to-run variation, not to the GPU (`python -m spatialcells.ft_seeds`,
entry "0-5090"; training log `results/two_choice/ft_seed_logs/train_log_s0-5090.json`).

*Correction (2026-10-25, found by the vast-3 session after this result):* the control is not a replay of
seed 0. `finetune.py` calls `torch.manual_seed` only for a non-zero seed; seed 0 keeps torch's default
generator, and in the torch used here (2.14) that generator is seeded differently in every process (two
fresh processes gave different `torch.initial_seed()` values on vast-3 and on the Mac). So every seed-0
run, the original included, starts from its own random LoRA initialisation and dropout stream, and only
the room order (`random.Random(epoch)`) is shared; two 5-update seed-0 runs on the same GPU have
uncorrelated `lora_A` matrices. Seeds 1 and 2 do set the generator. The registered wording ("identical
script, data and settings") holds, but the rows below compare two independently initialised runs of one
configuration, and the answer agreement is between independent runs, not a numerical replay. On the
RTX 5090 the three runs span 70-94% on room A's novel trials (seeds 2 and 1, and this run). The
original seed 0 cannot be retrained exactly; its results reproduce from its saved adapter.


| | seed 0 (RTX PRO 6000) | seed-0 configuration, second run (RTX 5090) | seeds 1 / 2 (RTX 5090) |
|---|---|---|---|
| update at which the loss first falls below 0.15 (of 284) | 135 | 135 | 175 / 215 |
| loss at the end (update 280) | 0.034 | 0.032 | 0.079 / 0.137 |
| held-out procedural rooms, first question, end of training | 92% | 92% | 84% / 69% |
| room A novel (102), walk | 92% | **94%** | 83% / 70% |
| room A novel, no walk / shuffled walk | 53% / 50% | 61% / 59% | seed 1: 56% / 58%; seed 2: 57% / 62% |
| room B novel (106), walk | 95% (MPS) | 94% | 80% / 62% |
| room B novel, no walk / shuffled walk | 60% / 52% (MPS) | 64% / 59% | seed 1: 52% / 61%; seed 2: 48% / 56% |

- The two seed-0 loss curves are nearly the same (correlation 0.998 over the 56 logged updates, median
  difference 0.003), although their initialisations differ; they separate most around the escape from
  the plateau (largest difference 0.028 at update 150), which the second run leaves slightly faster.
  Both escape at the same update, so with the same room order the escape time did not depend on the
  initialisation here.
- With the walk, the second run gives the same answer as the original on 98 of room A's 102 novel
  trials (3 better, 1 worse) and on 103 of room B's 106 (1 better, 2 worse). Without the walk and with
  the shuffled walk, where both models are near chance, they agree on fewer trials (room A 84 and 81 of
  102; room B 92 and 86 of 106), and over all 192 of room A's trials the second run is right more
  often (no walk 19 / 7 trials, p = 0.03; shuffled 22 / 9, p = 0.03).
- Before training, held-out accuracy (the base model, since LoRA starts as the identity) is 61% on all
  three RTX 5090 runs and 59% on vast-1: one of the 64 first questions is answered differently, a
  platform difference.
- One reading in experiment 9b changes. The original seed 0 fell to chance with a shuffled walk (50%
  and 52%), and that section took the 56-62% of seeds 1 and 2 to mean that weaker runs use something a
  shuffle leaves intact. The second seed-0 run, as strong as the original with the walk, keeps 59% in
  both rooms with a shuffled walk (14 / 5 and 14 / 6 trials against the original, p = 0.06 and 0.12),
  within the range of seeds 1 and 2. So accuracy with a shuffled walk varies between runs by about as
  much as the seeds differ there, and what separates the seeds is accuracy with the ordered walk. The
  second run still needs the order (ordered vs shuffled 39 / 3 trials in room A, 41 / 4 in room B).
- Experiment 9b's caveat that seeds 1 and 2 mix seed and platform no longer applies: a run on the
  RTX 5090 reaches seed 0's level. Its adapter stays on vast-3 (`data/ft/qwen8b-proc-s0-5090`); every
  other experiment keeps using the original seed 0.

**Seeds 1-2 in every scene: results (2026-10-25).** `.venv-claude/bin/python scripts/seed_scenes_analysis.py`
-> `results/seed_scenes_summary.json`, with each experiment's own units: experiment 10 on novel trials
with rooms as the unit, experiment 12 on all lure-free trials pooled (novel trials as the secondary
outcome), experiment 16 on all lure-free trials with houses as the unit. Seed 0 is the original run
(the numbers already reported in each experiment; the script reproduces them). Platforms as registered:
seeds 1-2 on vast-3 (CUDA) in HSSD1, HSSD4-8 and the nine re-rendered held-out rooms, on the Mac (MPS) in
the apartments, HSSD2-3 and P5005; the JSON names every file. Four of the 63 vast-3 files hold one answer
that names neither option (scored wrong, as before; seed 0's scene files have 51 such answers).
Contrasts are paired by trial: trials better / worse, two-sided sign test; for experiments 10 and 16
also the mean over rooms or houses with its 95% bootstrap CI and the count of rooms or houses where the
difference is positive / negative.

*Experiment 10, ten held-out procedural rooms* (1,299 novel trials; mean over rooms [95% CI over rooms],
rooms positive / negative, and trials better / worse pooled):

| | seed 0 | seed 1 | seed 2 | three seeds: mean (range) |
|---|---|---|---|---|
| novel trials, walk | 92.5% | 80.6% | 68.7% | **80.6%** (68.7-92.5%) |
| novel trials, no walk | 55.9% | 54.9% | 49.7% | 53.5% (49.7-55.9%) |
| novel trials, shuffled walk | 51.3% | 63.4% | 60.9% | 58.5% (51.3-63.4%) |
| walk vs base (base 51.8%) | +40.7 [38.2, 43.4], 10 / 0 | +28.7 [26.1, 31.4], 10 / 0 | +16.8 [13.2, 20.7], 10 / 0 | |
| walk vs no walk | +36.6 [33.9, 39.6], 10 / 0 | +25.7 [23.3, 28.1], 10 / 0 | +19.0 [14.3, 23.5], 10 / 0 | |
| walk vs shuffled walk | +41.2 [38.9, 43.8], 10 / 0 | +17.2 [15.1, 19.3], 10 / 0 | +7.8 [4.4, 11.6], 10 / 0 | |
| walk vs Opus 5.5 (low effort, 69.2%) | +23.3 [20.7, 26.2], 10 / 0 (343 / 46) | +11.3 [7.7, 15.3], 10 / 0 (284 / 142) | -0.6 [-5.0, 3.5], 4 / 6 rooms (room sign test p = 0.75); 263 / 276 trials (p = 0.61) | |

Every contrast against base, no walk and shuffled walk is positive in all ten rooms for every seed
(pooled p <= 4e-7).

*Experiment 12, three Replica apartments* (503 lure-free trials, 58 novel; pooled):

| | seed 0 | seed 1 | seed 2 | three seeds: mean (range) |
|---|---|---|---|---|
| all trials, walk | 86.7% | 81.1% | 70.4% | **79.4%** (70.4-86.7%) |
| all trials, no walk | 52.5% | 50.3% | 49.1% | 50.6% (49.1-52.5%) |
| all trials, shuffled walk | 53.5% | 64.2% | 59.0% | 58.9% (53.5-64.2%) |
| novel trials, walk | 79.3% | 77.6% | 72.4% | 76.4% (72.4-79.3%) |
| walk vs base (base 55.1%) | +31.6 (187 / 28) | +26.0 (158 / 27) | +15.3 (118 / 41) | |
| walk vs no walk | +34.2 (197 / 25) | +30.8 (185 / 30) | +21.3 (148 / 41) | |
| walk vs shuffled walk | +33.2 (202 / 35) | +16.9 (115 / 30) | +11.3 (91 / 34) | |
| walk vs Opus 5.5 (84.1%) | +2.6 (62 / 49, p = 0.25) | -3.0 (52 / 67, p = 0.20) | -13.7 (50 / 119, p = 1e-7) | |

Every contrast against base, no walk and shuffled walk has p < 1e-6 on all trials. On the 58 novel
trials every seed beats base and its own no-walk run (p <= 0.012), but only seed 0 beats its shuffled
walk (24.1 points, p = 0.016; seeds 1 and 2 both 5.2 points, 7 / 4 trials, p = 0.55).

*Experiment 16, eight HSSD houses* (1,450 lure-free trials, 366 novel; mean over houses [95% CI over
houses], houses positive / negative, and trials better / worse pooled):

| | seed 0 | seed 1 | seed 2 | three seeds: mean (range) |
|---|---|---|---|---|
| all trials, walk | 92.0% | 87.6% | 73.2% | **84.3%** (73.2-92.0%) |
| all trials, no walk | 51.0% | 52.6% | 48.0% | 50.5% (48.0-52.6%) |
| all trials, shuffled walk | 54.4% | 68.7% | 63.3% | 62.1% (54.4-68.7%) |
| novel trials, walk | 86.4% | 81.6% | 66.3% | 78.1% (66.3-86.4%) |
| walk vs base | +31.7 [28.4, 34.8], 8 / 0 | +27.3 [24.6, 30.3], 8 / 0 | +12.9 [9.9, 15.7], 8 / 0 | |
| walk vs no walk | +41.0 [38.4, 43.3], 8 / 0 | +35.0 [32.1, 37.9], 8 / 0 | +25.2 [19.9, 31.1], 8 / 0 | |
| walk vs shuffled walk | +37.6 [35.9, 39.5], 8 / 0 | +18.9 [15.4, 22.1], 8 / 0 | +9.9 [7.3, 12.3], 8 / 0 | |
| walk vs Opus 5.5 | +4.1 [2.5, 5.7], 8 / 0 (140 / 79) | -0.3 [-1.2, 0.8], 2 / 6 (125 / 129) | -14.7 [-16.8, -12.5], 0 / 8 (106 / 320) | |

On novel trials: vs base +30.2, +25.4, +10.1 (8 / 0, 8 / 0, 7 / 1 houses; p <= 0.003 pooled); vs no walk
+35.3, +29.5, +20.8 (8 / 0 each); vs shuffled walk +30.7, +14.6, +4.4 (seed 2: [0.3, 8.5], 5 / 2 houses,
54 / 39 trials, p = 0.15); vs Opus +2.2, -2.6, -17.9 (seed 2: 0 / 7 houses, p = 9e-10).

Reading:
- **Every run learns a skill that transfers to every kind of new scene.** All three seeds beat the base
  model and their own no-walk run in all ten held-out rooms, all eight houses and the apartments, and
  without the walk every seed is near chance (48-56%). The skill comes from the walk in every run, and
  the transfer from ray-cast training rooms to photoreal homes is a property of the recipe, not of the
  one run tested before.
- **How much each run learned keeps the order of rooms A and B.** Seed 0 > seed 1 > seed 2 in every
  experiment (held-out rooms 92.5 / 80.6 / 68.7%, apartments 86.7 / 81.1 / 70.4%, houses 92.0 / 87.6 /
  73.2%), following how late each run left the loss plateau (experiment 9b).
- **The comparison with Opus depends on the run.** In the procedural held-out rooms, where Opus gains
  little from the walk, seeds 0 and 1 beat Opus in all ten rooms and seed 2 ties it. In the photoreal
  scenes, where Opus gains much more, seed 0 matches it (apartments, p = 0.25) or beats it slightly
  (houses, 8 / 0), seed 1 ties it, and seed 2 is 14-15 points behind. Over the three runs the fine-tuned
  model is ahead of Opus in the procedural rooms (80.6% vs 69.2%) and a few points behind in the photoreal
  scenes (apartments 79.4% vs 84.1%; houses 84.3% vs 87.9%, both means over houses). So "the fine-tuned 8B beats
  Opus" holds for the best run everywhere, and for the average run only in rooms like those it was
  trained on.
- **Every run uses the walk's order, the weaker ones less.** Ordered beats shuffled in every room and
  house for every seed, but the margin shrinks from seed 0 to seed 2 (held-out rooms 41 / 17 / 8 points;
  houses 38 / 19 / 10), because the weaker seeds keep 59-69% with a shuffled walk where seed 0 is at
  chance. With the second seed-0 run's 59% (GPU control above), accuracy with a shuffled walk varies
  between runs, so the claim that holds across runs is "ordered beats shuffled", not "shuffled is at
  chance".
- The analyses inside the fine-tuned model (experiments 13a, 14a, 15, unit scoring) used seed 0 only.

## Experiment 9c (pre-registered 2026-10-25, before any training): a longer training schedule

**Why.** In experiment 9b the three two-epoch runs differ mainly in when they leave the coin-flip
loss plateau (updates 135, 175, 215 of 284). The cosine learning rate then decays to near zero, and
the late runs stall before they finish learning: over their last ~100 updates, accuracy on the 64
held-out training-style rooms stays at 84% (run 1) and 67-70% (run 2), with the loss still at 0.079
and 0.137 (run 0: 92%, 0.034). The paper states that the schedule may be too short and that a longer
one was not tested. The user asked for more runs that are properly trained. The schedule below is
set from the training logs and the held-out training-style rooms only; the test scenes play no part
in choosing it. The two-epoch runs stay in every analysis and in the paper as the first recipe.

**Recipe.** Identical to experiment 9 (`spatialcells/finetune.py`: the same `data/proc` rooms, 1,136
for training and the same 64 held out; LoRA rank 16, alpha 32, dropout 0.05 on the language model;
answer-only loss; AdamW lr 1e-4, 30 warm-up updates, cosine decay; 8 rooms per update; 64-frame walks
of 256-px frames), except `--epochs 4` (568 updates, the cosine schedule stretched over all of them).
Runs: seeds 3, 4, 5, 6 and 7, each with `torch.manual_seed` set (unlike seed 0), adapters
`data/ft/qwen8b-proc-e4-s<seed>`. Training on vast-3's RTX 5090 (as seeds 1-2), through gq, one after
another (`EPOCHS=4 scripts/run_ft_seed.sh <seed> data/ft/qwen8b-proc-e4-s<seed>`).

**Training checks** (reported for every run, not used to drop any): the update at which the loss
first falls below 0.15, the loss at the end, and the held-out accuracy curve. Every run that finishes
is tested and reported, whatever its curve; a run that crashes is restarted with the same seed.

**Tests.** The same as for seeds 1-2 everywhere, each with the 64-frame walk, no walk and the shuffled
walk: room A and room B (experiment 9b), the ten held-out procedural rooms (experiment 10), the three
apartments (experiment 12) and the eight houses (experiment 16). Platforms as for seeds 1-2: vast-3
(CUDA) for rooms A and B, HSSD1, HSSD4-8 and the nine re-rendered held-out rooms (`scripts/run_scene_tests.sh`);
the Mac (MPS) for the apartments, HSSD2-3 and P5005 (`scripts/mac/run_heldout.sh`).

**Analysis.** Per run, each experiment's registered contrasts (as in `scripts/seed_scenes_analysis.py`
and `spatialcells/ft_seeds.py`); the five-run mean and range; and the two recipes side by side (three
two-epoch runs, five four-epoch runs) for every experiment's primary measure.

**Decision rule, set now.** The longer schedule "makes the recipe reliable" if all five runs end with
at least 85% held-out accuracy on the training-style rooms, the five-run mean on room A's novel trials
is at least 85%, and no run is below 80% there. Otherwise the result is reported as the longer schedule
not closing the gap (or closing it partly, with the numbers). Either way all five runs are reported,
and the paper shows both recipes; nothing is selected after the tests. The analyses inside the model
(experiments 13a, 14a, 15, unit scoring) stay on the original seed 0 and are not repeated here.

### Exp 9c results (2026-10-27 06:00 PDT)

All five runs trained without a crash or restart: gq vast3:253-254 and 275-282, then mac:75, 80 and 86. Every
test file is complete, 63 per run. Summaries are in `results/two_choice/ft_seeds.json` (`e4-s<k>`, `runs4ep/<cond>`,
`training`, `decision_rule_9c`) and `results/seed_scenes_summary.json` (`four_epoch_runs`, next to `three_seeds`).

**The decision rule is met: the four-epoch schedule makes the recipe reliable.**

| run | loss < 0.15 at update | held-out end | room A novel | room B novel |
|---|---|---|---|---|
| e4-s3 | 235 | 95.3% | 98.0% (100/102) | 97.2% (103/106) |
| e4-s4 | 180 | 95.3% | 99.0% (101/102) | 99.1% (105/106) |
| e4-s5 | 150 | 98.4% | 98.0% (100/102) | 98.1% (104/106) |
| e4-s6 | 185 | 96.9% | 99.0% (101/102) | 97.2% (103/106) |
| e4-s7 | 140 | 96.9% | 98.0% (100/102) | 98.1% (104/106) |

- Every run ends with at least 85% held-out accuracy (95.3-98.4%).
- The five-run room A novel mean is 98.4%, against a required 85%.
- No run is below 80% (minimum 98.0%).

Every run leaves the plateau by update 235 of 568, so the slowest run has 333 updates of learning
left. The two-epoch runs left the plateau at updates 135-215 of 284.

**Both recipes side by side** (primary measures; Opus 5.5 with the walk for reference):

| scene and measure | 2 epochs, runs 0/1/2 (mean) | 4 epochs, runs 3-7 (mean) | Opus |
|---|---|---|---|
| held-out procedural rooms, novel, room mean | 92.5 / 80.6 / 68.7 (80.6) | 96.3-97.4 (97.1) | 69.2 |
| apartments, all lure-free trials, pooled | 86.7 / 81.1 / 70.4 (79.4) | 90.9-92.2 (91.5) | 84.1 |
| apartments, novel trials only (58), pooled | 79.3 / 77.6 / 72.4 (76.4) | 84.5-87.9 (85.5) | 89.7 |
| houses, all trials, house mean | 92.0 / 87.6 / 73.2 (84.3) | 93.9-95.0 (94.5) | 87.9 |
| houses, novel trials, house mean | 86.4 / 81.6 / 66.3 (78.1) | 88.7-90.5 (89.6) | 84.2 |

**Each four-epoch run, by scene:**
- *Held-out procedural rooms:* every run beats the base model (+44.4 to +45.6 points), no walk and the
  shuffled walk in 10/10 rooms, and beats Opus by +27.0 to +28.2 in 10/10 rooms.
- *Houses:* every run beats base, no walk and shuffled in 8/8 houses, and beats Opus in 8/8 on all trials
  (+6.0 to +7.1). On novel trials the margin over Opus is +4.5 to +6.3 (sign tests p 0.006-0.06).
- *Apartments:* every run beats base, no walk and shuffled on all trials and on novel trials. On all trials
  it beats Opus by +6.8 to +8.2 (p ≤ 0.0006). On the 58 novel trials it is 1.7-5.2 points below Opus, a
  difference of 3 to 6 trials (p 0.51-1.0).
- *Ordered vs shuffled walk:* the ordered walk beats the shuffled one in every run and scene: by +15.1 to +42.9
  in the held-out rooms, +23.3 to +36.4 in the apartments and +17.7 to +40.2 in the houses. Shuffled-walk
  accuracy varies across runs (held-out rooms 54.4-82.1%; room A 59-81%). It is higher than in the two-epoch
  runs (51-63%), so how much of the order a run needs varies between runs. Run 7 depends on it least.

**Reading.**
- The spread across the two-epoch runs (experiment 9b) came from a schedule too short for late plateau
  escapes. With four epochs, all five runs land within 1-1.5 points of each other on every primary measure,
  at or above the original run 0.
- The comparison with Opus now holds for the recipe, not only for the best run, in the held-out rooms, on
  all apartment trials and in the houses.
- The exception is the apartments' 58 novel trials, where Opus is level or slightly ahead (not
  significant). This is the same subset where the two-epoch runs trailed. The apartment caveat from
  experiment 12 still applies: the walk may help partly by grouping views by room.
- The internal analyses stay on run 0, as registered.

## Experiment 10 (pre-registered 2026-10-22, run 2026-10-23): many held-out rooms

**Deviation found after the results below (2026-10-23, by an external review of the preprint).**
In the six rooms rebuilt with 32 walks (P5001-P5003, P5005, P5006, P5008), the Opus runs cover only
walks 0-15: 96 of 192 trials, 60-79 novel. `claude_test.py` defaults to `--walks 16` and the launch
script did not override it. This was a bug, not a choice. Opus has 859 novel trials over the ten
rooms against LoRA's 1,299. Six rooms fall below the registered 100 novel trials. The walk-minus-
no-walk gain is still paired (both Opus conditions miss the same walks), but the LoRA-minus-Opus
row below compares different trial sets in those rooms. On matched novel trials, LoRA - Opus is
+24.9 points [22.0, 27.7], positive in all ten rooms (the review's audit,
`results/review/preprint-audit-2026-10-23.json`). The missing walks were run the same day (`--walks 32`; Exp 10 Opus cost
$73.48 in all). On the full set (1,299 matched novel trials; `heldout_analysis.py` now checks every
run against the manifests): Opus 69.2% with the walk vs 62.8% without, **gain +6.4 points [+4.1,
+8.2]**, positive in 9 of 10 rooms (p = 0.021; trials 282 better / 200 worse); LoRA minus Opus
**+23.3 [+20.8, +26.2]**, 10/10 rooms. The 16-walk numbers (+5.3 [3.1, 7.2]; +24.9 matched) are kept in
`results/heldout_summary.json` under `walks_0_15`. The table below is the first pass.

**Results** (`scripts/heldout_analysis.py`, `results/heldout_summary.json`; novel trials, 102-152 per
room; means over the 10 rooms with 95% bootstrap CIs over rooms):

| novel trials | walk | no walk | shuffled walk |
|---|---|---|---|
| base Qwen3-VL-8B | 51.8% [49.2, 54.2] | 50.0% [47.6, 52.4] | 56.4% [54.6, 58.5] |
| LoRA (seed 0) | **92.5%** [91.4, 93.6] | 55.9% [53.0, 58.7] | 51.3% [48.2, 54.3] |
| Opus 5.5, low effort | 67.4% [64.6, 69.8] | 62.1% [59.2, 64.5] | (not run; amendment 1) |

- **Primary question: Opus's walk gain holds across rooms, but it is small.** It is +5.3 points
  [+3.1, +7.2], positive in 9 of 10 rooms (room-level sign test p = 0.02; pooled trials: 185
  better, 138 worse). That is smaller than room A's +14 and larger than room B's ~0.
- **The LoRA model:**
  - walk minus no walk: +36.6 points [+33.9, +39.6];
  - in order minus shuffled: +41.2 [+38.8, +43.8];
  - minus base: +40.7 [+38.2, +43.5];
  - minus Opus: +25.1 [+22.7, +27.7];
  - all positive in 10 of 10 rooms.
- Scope: these rooms come from the LoRA model's training generator (new layouts, same kind of
  room), and the LoRA results are for one training seed. Opus cost $48.84 for the 20 runs.

The pre-registration follows.

Rooms A and B disagree on one result: Opus 5.5 gains from the walk in room A (75% vs 61%) but not
in room B (64% vs 63%). Two rooms cannot say which is typical. This experiment repeats the
two-choice test in ten new rooms, with rooms (not trials) as the unit of analysis.

**Rooms.** Ten procedural rooms from the experiment-9 generator, `room.random_layout(seed)` for
seeds 5000-5009 (`--room P5000` ... `P5009`). The LoRA model was trained and validated on seeds
0-1199, so it has seen none of these layouts. They come from the same generator, so the claim
they support is "new rooms of the kind it was trained on", not new kinds of rooms.

**Trials.** In each room, 16 foraging walks of 256 steps (`render_data --walks 16 --seed <room
seed>`) and 64 teleport probes per walk (`render_probes`), then the lure-free two-choice trials
of `dissimilar_trials.py`, built exactly as in room B:
- near option within 1 m, far option more than 3 m away;
- both options facing more than 90 deg away from the new view;
- no poster or pillar shared with the new view;
- pixel similarity and frame order each favour the near option in half the trials;
- up to 12 trials per walk, novel views first.

The target is at least 100 novel trials per room. If a room has fewer, its walks are re-rendered
with 32 walks and the trials rebuilt. That decision is made from the trial counts alone, before
any model sees the room.

**Models and conditions** (the same prompts as before):
- Base Qwen3-VL-8B: 64-frame walk, no walk, and the 64-frame walk shuffled (real frame numbers).
  Greedy, bf16, on the Mac (`--device mps`; it matched CUDA on 192/192 room-A answers).
- The LoRA model: the same three conditions. Seed 0 first; seeds 1 and 2 once trained
  (experiment 9b).
- Claude Opus 5.5, low effort (as in experiment 6 and room B): 256-frame walk, no walk, and the
  256-frame walk shuffled (real frame numbers).

**Analysis.**
- Primary outcome: accuracy on novel trials in each room. Unparsed answers count as wrong.
- Every summary is the mean over rooms, with a 95% CI from a bootstrap over rooms (10,000
  resamples).
- Primary question: Opus's walk gain, its walk accuracy minus its no-walk accuracy in each room.
  It is reported as holding up if the 95% CI of the mean gain excludes 0, and is reported either
  way, along with the number of rooms where it is positive.
- Secondary, analysed the same way:
  - LoRA walk minus LoRA no walk;
  - LoRA in order minus LoRA shuffled;
  - LoRA minus base, both with the walk;
  - LoRA minus Opus, both with the walk.
- Trial-level paired sign tests, pooled over rooms, are reported as a supplement only.
- Familiar trials are reported but not tested.

**Budget.** Opus costs about $11 per room at low effort ($4.9 walk, $4.9 shuffled, $1.3 no
walk, from rooms A and B), about $110 in all. The approved cap is $300. Spend is logged per run
in `results/api_costs.csv`.

**Amendment (2026-10-23, before any result from rooms P5001-P5009; only P5000's Opus walk run
had started).** The Claude API budget is now $138 until next month, so Opus runs only the
256-frame walk and no walk, about $6.2 per room and $62 in all. Its shuffled-walk condition is
dropped. That condition was secondary: room A already showed that walk order does not matter to
Opus (experiment 6). The primary question is unchanged. The LoRA and base models keep all three
conditions, since they cost nothing. Caching is already near its limit: each walk's prefix is
written once and read by the walk's other 11 trials. About half of an Opus walk run is those 16
cache writes, and a quarter is thinking tokens, which cannot be turned off.

**Amendment 2 (2026-10-23, from trial counts only; no model had seen any room but P5000).** At
16 walks, most procedural rooms have fewer than 100 novel trials: 77-102, against room A's 102.
Their posters are more numerous than room A's, so the no-shared-landmark rule leaves fewer pairs.
As registered, such rooms are re-rendered with 32 walks. To keep every room at 192 trials (the
same cost and run time as the others), their trials are built with up to 6 per walk instead of
12, novel first (`dissimilar_trials --walks 32 --per-walk 6`). The trial rules are otherwise
unchanged. The first 16 walks and their probes stay identical, because both are drawn
sequentially from the same seeds. Rooms that already reach 100 novel trials at 16 walks keep
those trials.

## Experiment 11 (pre-registered 2026-10-23, not yet run): other frontier and open models

Experiment 5 found Opus 5.5 to be the only off-the-shelf model whose accuracy rose with the walk.
That claim rests on Claude and open models up to 35B. This experiment tests it on other providers'
flagship models and on the largest open models served by an API.

**Models.** Candidates, each kept only if a preflight shows it accepts the full protocol:
- Kimi K3 (Moonshot; reasoning `low`; temperature fixed at 1.0 by the provider);
- GPT-6 Astra (reasoning `low`; temperature not settable);
- Gemini 4 Argon (reasoning `low`; temperature ignored);
- GPT-5.5 at effort `none` with temperature 0, as a near-greedy control;
- Qwen3.5-397B-A17B and MiniMax-M3 (DeepInfra; thinking off, temperature 0);
- DeepSeek V4.1 Flash.

The preflight (`api_test.py --preflight`) is one request with all 257 images (256 walk frames
plus the new view) at 256 × 256, as JPEGs, unresized and labelled as in the Claude prompt. A
model or provider that cannot take that is listed as untestable, with the reason, and the protocol
is not changed for it. Known in advance:
- untestable: Llama 4 Maverick (3-20 images per request, by provider), Mistral Large 3 (8) and
  InternVL3.5-241B (no reliable host);
- 64-frame walk only: GLM-5.3 (150 images), so it is not tested.

Each model's settings and any provider quantisation are recorded with its results.

**Trials and conditions.** Room A's 192 lure-free trials (102 novel), with exactly the prompt
and images Opus got (`api_test.py` builds them with `claude_test.py`'s own functions). Two
conditions: the 256-frame walk, and no walk. A model gets the shuffled walk only if its walk
gain is significant. Every model with a significant gain is also run in the experiment-10 rooms,
if the budget allows.

**Analysis.**
- Accuracy on novel trials; unparsed answers count as wrong.
- Walk gain: a two-sided paired sign test of walk vs no walk on the novel trials, as for Opus in
  experiment 5 (two-sided p = 0.016).
- The claim "Opus 5.5 is the only off-the-shelf model that gains from the walk" stands only if
  no tested model has p < 0.05 for a positive gain. It is reported either way.
- An open model with a significant gain triggers a proposal to the user for a self-hosted
  activation analysis, with a cost estimate. Nothing is rented without asking.

**Budget.** $150 in all, outside the Claude limit. Per-model estimates for room A, walk plus no
walk: GPT-6 Astra about $25-40, Gemini 4 Argon about $10-38 (its image token count is measured
first), GPT-5.5 about $9, Kimi K3 about $10-14, Qwen3.5-397B and MiniMax-M3 about $1-2 each.
Each trial's cost is logged in `results/api/*.jsonl` and summed in `results/api_costs.csv`.

**Amendment (2026-10-23, before any experiment-11 trial): a fast-decision arm.** Two
decision-model APIs return one choice from a fixed set, with probabilities, and no reasoning
step:
- OpenAI's Decisions API (`gpt-6-luna`, public beta);
- TypeSafe's Jev 1.14, whose image input the provider labels experimental (it is sent with
  `TypeSafe-Danger-Zone-Images: true`).

They ask whether the walk gain needs deliberation. They get the same content in the same order:
the labelled walk as shared input, then "New view:", the probe and the question. The question's
last sentence ("Answer with one of the two frame numbers only") is replaced by the two choices
"Frame a" and "Frame b", listed in the trial's order.
- Scoring uses the chosen option, as for every other model.
- The probability given to the near option is recorded and reported, but not tested.
- The same preflight and untestable rule apply.

Together they are expected to cost under $2 for room A.

**Amendment 2 (2026-10-23, scope; no results seen from any of the dropped models).** After
experiment 10, Opus's walk gain is small (+5.3 points over 10 rooms). The fine-tuned 8B's is large
(+37). So the paper's claim no longer rests on Opus being the only model with a gain, and this
experiment shrinks to one appendix table. The models are GPT-6 Astra, Gemini 4 Argon, Kimi K3 (done)
and DeepSeek V4.1 Flash, plus OpenAI's Decisions API only if it is cheap and clean. Jev (image input
experimental by the provider's own account), GPT-5.5, Qwen3.5-397B and MiniMax-M3 are dropped. The
analysis and the untestable rule are unchanged.

**Amendment 3 (2026-10-23, scope; no decision model has been run).** The fast-decision arm is
dropped: OpenAI's Decisions API (gpt-6-luna) is not run, and Jev was already dropped. The
experiment is GPT-6 Astra, Gemini 4 Argon, Kimi K3 and DeepSeek V4.1 Flash.

*Note (2026-10-25, before any GPT-6 Astra or Gemini 4 Argon trial; logistics only):* the user supplied
keys on this Mac, so the remaining models run here with the same code, trials and images as the lab
package (its checksums verified), following the package's runbook (preflight, then the walk and
no-walk runs). *Correction (2026-10-25):* the keys are the user's own, like Kimi's and DeepSeek's, so
the spend counts toward the $150 other-provider budget (`scripts/cost_log.py`), not the lab.

**Amendment 4 (2026-10-25, scope; no Gemini 4 Argon trial has been run).** The user dropped Gemini 4
Argon; GPT-6 Astra is experiment 11's last model. *Transport note (2026-10-25, during the Astra walk run,
21 trials in):* GPT-6 Astra reuses its prompt cache only for exact repeats of these image-heavy prompts,
never for a shared walk prefix with a different question (tested on Chat Completions with one or two
user messages and a text-only question, and on the Responses API; about $2.5 of test requests, not
recorded as trials). So from trial 22 on, requests go out on OpenAI's flex tier (`--service-tier flex`;
the same request at batch pricing). Prompts, images, settings and parsing are unchanged; each row
records its service tier, and costs are logged at the standard rate as an upper bound.

**Results so far (room A, 102 novel and 90 familiar trials; `results/api/`).**

| Model | Novel, walk | Novel, no walk | Better / worse with walk | p | Familiar, walk / no walk | Cost |
|---|---|---|---|---|---|---|
| Kimi K3 | 57.8% | 60.8% | 12 / 15 | 0.70 | 56.7% / 57.8% | $14.92 |
| DeepSeek V4.1 Flash | 56.9% | 58.8% | 24 / 26 | 0.89 | 68.9% / 54.4% (25 / 12, p 0.047) | $1.27 |
| GPT-6 Astra (low reasoning) | 71.6% | 51.0% | 35 / 14 | 0.0038 | 77.8% / 66.7% | $42.22 logged |

Neither model gains from the walk on novel trials, the pre-registered test. DeepSeek is better with
the walk on familiar trials, where a seen view can be matched directly (one walk answer unparsed,
counted wrong). DeepSeek ran with thinking on and off-peak pricing; its median reasoning was about
5,300 tokens with the walk and 1,850 without. *GPT-6 Astra (2026-10-25, run on this Mac with the user's
key; walk run trials 1-21 at the standard tier, the rest and the no-walk run on the flex tier):* it
gains from the walk, the pre-registered test: 71.6% vs 51.0% on novel trials, 35 better / 14 worse,
two-sided p = 0.0038 (`walk_gain.py`). So a walk gain is not specific to Claude. Its median output was
about 60 tokens. By the registration, a significant gain adds a shuffled-walk run in room A and the
ten experiment-10 rooms if the budget allows; both wait for the user's decision. Costs are logged at
the standard rate ($40.28 walk, $1.94 no walk) against the $150 other-provider budget; flex should
roughly halve the actual charge for all but the first 21 walk trials. Gemini 4 Argon was dropped
(amendment 4).
*Astra's registered shuffled-walk run (2026-10-25, room A, flex tier, $40.48 logged):* novel trials,
shuffled 77.5% vs ordered 71.6% vs no walk 51.0%. Ordered minus shuffled -5.9 points, 7 better / 13
worse (p 0.26); walk-clustered -5.9 [-15.4, +2.5] (`results/review_stats.json`). Shuffled minus no
walk +26.5 [+16.4, +38.0], 10 walks up and none down, trials 36/9 (p 7e-5). So, like Opus, Astra
does at least as well with the walk shuffled (real frame numbers kept) as in order; the fine-tuned
model drops to chance. The ten held-out rooms were not run (the budget does not allow it; the
registration runs them only if it does).

**Amendment 5 (2026-10-25, before any trial of these models): a screen of the largest open-weight VLMs.**
Experiment 11's question is whether models other than Claude gain from the walk; the two that do
(Opus, GPT-6 Astra) are closed, so none can be inspected. This screen looks for an open-weight model
that passes. Models (OpenRouter, the newest and largest open-weight models that take images, Oct
2026; Kimi and DeepSeek already tested), in the order run, cheapest input price first:
nvidia/nemotron-3.5-super-vl, qwen/qwen3.8-flash, minimax/minimax-m3, xiaomi/mimo-v2.6-pro,
qwen/qwen3.5-397b-a17b, stepfun/step-5-preview, thinkingmachines/inkling, meta/muse-spark-open.
- Protocol as above: room A, the same 192 trials, images and prompts, walk and no walk, reasoning
  at low effort where the model has it, temperature not sent; a model that rejects the 257-image
  request is untestable. Requests go through OpenRouter restricted to hosts serving bf16, fp16 or
  fp8 weights; each record keeps the host that served it and OpenRouter's reported cost.
- Test: the registered walk-gain sign test on novel trials, but because the screen tests up to eight
  models and stops at the first pass, a model passes only at two-sided p < 0.05 / 8 = 0.00625
  (Bonferroni). Nominal p values are reported for every model; a model with 0.00625 <= p < 0.05 is
  "nominal, not confirmed" and the screen continues.
- Stopping rule: one model at a time, preflight first; the screen stops at the first model that
  passes, which then gets the registered shuffled-walk run. Models after it are reported as not run.
- Budget: the OpenRouter key's $50 limit; a model whose preflight projection would pass what is left
  is skipped and reported.
*Note (2026-10-25, after the first two preflights, before any recorded trial):* OpenRouter lists each
host's declared precision. nvidia/nemotron-3.5-super-vl is untestable: it spends about 1,445 tokens
per 256-px frame (371,365 image tokens for the walk), more than the 262,144-token context of its only
bf16 host (its 1M-context hosts serve fp4). Some models have no host that declares bf16/fp16/fp8 (Qwen3.8-Flash:
Alibaba only; Inkling: Together only; Muse Spark Open: fp4 or undeclared). For those, hosts with
undeclared precision are allowed, declared fp4 never; the serving host is recorded with every answer.
API runs go one at a time: concurrent ~6 MB image uploads on this network cause TLS errors.
*Note (2026-10-25, first pass of the screen):* untestable as registered: Nemotron 3.5 Super VL and
Qwen3.5-397B (the walk needs about 389k tokens, over their 262k context on hosts allowed by precision),
Qwen3.8-Flash (Alibaba's API takes at most 250 images per request), Inkling (its only host rejects the
request as too large). Step-5's first host (DeepInfra) takes at most 8 images, so it is retried on
StepFun's own fp8 host before being called untestable. MiniMax-M3 and MiMo-V2.6-Pro stopped on
malformed replies from their hosts (not rejections); such replies are now retried, and both are rerun.
Muse Spark Open, tested: 64.7% with the walk vs 59.8% without, 21 better / 16 worse, p = 0.51, no gain
(hosts Fireworks, Modal and Together, precision undeclared; $4.92).

**Screen result (2026-10-25, second pass complete; `scripts/or_screen.py` -> `results/api/or_screen.json`).**
No open-weight model passes. Three of the eight could be tested, and none gains from the walk, even
nominally; the other five cannot take the 257-image request on any allowed host. Room A, novel trials
(102), walk vs no walk, trials better / worse with the walk, two-sided sign test:

| model | walk | no walk | better / worse | p | hosts (precision) | cost |
|---|---|---|---|---|---|---|
| MiniMax-M3 | 52.9% | 58.8% | 21 / 27 | 0.47 | DeepInfra, GMICloud, Minimax, ModelRun, Novita, Parasail, StreamLake, Venice | $2.72 |
| MiMo-V2.6-Pro | 53.9% | 56.9% | 20 / 23 | 0.76 | DeepInfra, GMICloud, Novita, Xiaomi | $2.54 |
| Muse Spark Open | 64.7% | 59.8% | 21 / 16 | 0.51 | Fireworks, Modal, Together (undeclared) | $4.92 |
| Nemotron 3.5 Super VL, Qwen3.5-397B | untestable | | | | walk needs ~389k tokens; 262k context on allowed hosts | |
| Qwen3.8-Flash | untestable | | | | Alibaba: at most 250 images per request | |
| Step-5 (preview) | untestable | | | | DeepInfra: at most 8 images; StepFun: "exceeds the model's limitation" | |
| Inkling | untestable | | | | Together: request too large | |

With the walk, the tested models stay within 4 points of their no-walk accuracy, where GPT-6 Astra
gains 21 points and Opus 5.5 14 points (low effort, as here) on the same trials. Since no model passed, none had
the shuffled-walk run. The OpenRouter key spent $10.30 of its $50 (including preflights and the first
pass's runs that stopped on malformed replies).

**Amendment 6 (2026-10-25, after the screen, before any trial of this model): the largest Qwen with a
shorter walk.** The user asked to try the largest open Qwen, Qwen3.5-397B-A17B, which was untestable
only because OpenRouter's own estimate of the request (about 1,445 tokens per image, 388,798 in all)
exceeds the 262,144-token context of the hosts allowed by precision. A shorter walk fits.
- Walk: 160 frames per walk (`api_test.py --walk-frames 160`) if the preflight is accepted, otherwise
  128; nothing else is tried. The frames always include every option frame of the walk's trials, and
  the rest are spread evenly (farthest-point fill), shown in walk order with their real frame numbers.
  Precedent: Opus 5.5 with a 64-frame walk chosen the same way gets 77.5% of room A's novel trials
  (256 frames: 74.5%; no walk 60.8%; 29 better / 12 worse than no walk), so a shorter walk keeps the
  gain for a model that uses it.
- Everything else as in amendment 5: room A, the same 192 trials and prompts, reasoning at low effort,
  temperature not sent, OpenRouter restricted to hosts declaring bf16, fp16 or fp8, the host recorded per
  answer. The no-walk run is the usual one (two option frames only).
- Test: the registered walk-gain sign test on novel trials at two-sided p < 0.05 (a single model,
  tested on its own after the screen); p is also reported against the screen's 0.00625. If it passes,
  the shuffled walk with the same frames follows. Budget: what is left on the OpenRouter key.
- The result does not change the screen's verdict, which used the full walk; it is reported as a
  follow-up with a shorter walk.

**Amendment 6 result (2026-10-25).** The 160-frame preflight was accepted (161 images; the model's own
count was 11,672 prompt tokens, about 72 per image, so the full 256-frame walk would be about 19k tokens
and only OpenRouter's estimate blocks it). Qwen3.5-397B-A17B does not gain from the walk: on room A's 102
novel trials 52.9% with the 160-frame walk vs 56.9% without, 17 better / 21 worse, two-sided p = 0.63;
familiar trials 51.1% vs 50.0% (17 / 16). No shuffled run followed. Its reasoning often runs past the
16,000-token output cap at low effort: 30 walk and 29 no-walk answers stopped at the cap with no answer
(scored wrong, as for every model); on the 74 novel trials both runs answered, 63.5% vs 64.9%. It picks
the first-listed option on 53% / 48% of the novel trials it answered (52% / 48% of all answered trials). Hosts: AtlasCloud and Parasail (walk), plus DeepInfra and
GMICloud (no walk), all serving fp8. Cost $14.13 (key: $24.49 used of $50). Per-trial answers in
`results/api/qwen_qwen3.5-397b-a17b-dissimilar-{f160,nowalk}-low.jsonl`.

*Logistics (2026-10-23, no protocol change).* GPT-6 Astra and Gemini 4 Argon are run on the user's
university lab PC, billed to the lab, from a self-contained
package (`lab_package/exp11_lab.zip`, gitignored: the room-A trials, probe and walk JPEGs with
SHA-256 checksums, `claude_test.py`, `api_test.py` and a runbook for the agent there). `api_test.py`
gained the two models (prices and settings from the providers' docs, Oct 2026), Gemini's dated
price change, OpenAI's long-context rate, a `prompt_cache_key` per walk for OpenAI, thinking tokens
billed from `total_tokens` when `completion_tokens` leaves them out, the served model ID and UTC
time in each record, and a projected full-run cost in `--preflight`.

## Experiment 12 (pre-registered 2026-10-23, run 2026-10-23): photoreal apartments

Do the behavioural results hold in real scanned interiors? This is the first pass agreed in
docs/photoreal_plan.md: behaviour tests plus VGGT localisation. The place-cell analyses are left for
a later version.

**Scenes.** Three Replica apartments (Meta; Replica Research Terms): apartment_0, apartment_1 and
apartment_2, named ReplicaApt0-2. They are rendered with habitat-sim 0.3.3 from the
vertex-coloured mesh; the PTex textures do not render on the host's driver. The stage loads Z-up
while its navmesh is Y-up, and `habitat_scene.py --zup` handles this. The single-room scenes were
dropped from counts alone, before any model saw them: their walks revisit every place from every
direction, so almost no probe is novel. These are apartments, so "the same room" in the prompt is
really the same apartment.

**Walks and probes** (`scripts/rtx4070/render_replica.sh`):
- the same camera: 256 x 256, 90 deg horizontal FOV, 1 m high;
- the same foraging walk: 0.22 m steps, heading drift, head scans;
- 32 walks of 256 steps on each scene's largest navigable island, with 0.15 m clearance from
  obstacles (0.35 m confined the walks to single rooms);
- 128 teleport probes per walk, near visited places at random headings.

**Trials** (`dissimilar_trials --room ReplicaAptN --walks 32 --per-walk 6`):
- the lure-free rules of experiments 5 and 10: near option within 1 m, far more than 3 m away,
  both facing more than 90 deg from the new view, pixel similarity and frame order balanced;
- "no shared landmark" means no labelled object (other than walls, floor, ceiling, windows,
  doors and similar) covering 2% or more of both views;
- no view with more than 5% of pixels showing no geometry is used.

Counts: ReplicaApt0 176 trials (42 novel), ReplicaApt1 168 (6 novel), ReplicaApt2 159 (10 novel).

**Primary outcome: accuracy on all lure-free trials** in each scene, with an exact binomial CI,
and pooled over the three scenes. Novel views are rare here, so novel-trial accuracy is reported
pooled (58 trials) as a secondary outcome only. All trials are still hard by construction. In
room A, familiar lure-free trials separated the models about as well as novel ones did (base
57-66%, LoRA 91%, Opus about 80%).

**Models and conditions** (prompts as before):
- base Qwen3-VL-8B and the LoRA model (seed 0; seeds 1-2 when trained): 64-frame walk, no walk,
  shuffled 64-frame walk, on the Mac (MPS);
- Opus 5.5 at low effort: 256-frame walk and no walk, about $5 per scene from the Claude budget;
- VGGT positive control, as in experiment 3c: each trial gives it every 4th walk frame, the two
  options and the probe. Its cameras are fitted to the walk frames' true poses by a similarity
  transform (the probe never enters the fit). It reports the median probe position error, the
  share of probes placed within 1 m, and two-choice accuracy, for novel and familiar trials. (A
  first draft of this entry said "8 anchor views"; that was experiment 4's method, corrected
  before any VGGT run.)
  *Amendment, before any VGGT result:* every **8th** walk frame, so 32 walk frames plus the two
  options and the probe. 67 images do not fit the RTX 4070's 12 GB, even one trial at a time and
  with VGGT's unused heads removed; experiment 3c used every 4th frame on a 96 GB GPU.

**Questions.**
1. Does the LoRA model, trained only on ray-cast procedural rooms, transfer to real scans?
   Paired sign tests: LoRA minus base, and LoRA walk minus LoRA no walk, pooled over scenes.
2. Is the base model again near chance, with Opus in between?
3. Does VGGT still place probes?

Reported either way. Scope: three scenes, vertex-coloured renders, apartments rather than rooms.

*Deviation (2026-10-23, found after the first Opus pass, before any accuracy was tabulated).* The
first Opus pass used `claude_test.py`'s default of the first 16 walks with trials, a bug rather
than a choice (the same one as in experiment 10): 270 of the 503 trials (27 of 58 novel). Its progress log, which prints
running counts of right answers, had been seen. The remaining walks are run with `--walks 32` (about
$10 more), so Opus covers every trial as the primary outcome requires. Both the 16-walk numbers
and the full numbers are reported.

**Results (2026-10-23; `scripts/photoreal_analysis.py` -> `results/photoreal_summary.json`; all 503
lure-free trials, 58 novel; every run checked against the manifests).** Pooled accuracy, all trials
[exact 95% CI] (per scene Apt0 / Apt1 / Apt2; novel pooled; first-listed rate):

| | all trials | per scene | novel (58) | first-listed |
|---|---|---|---|---|
| base, 64-frame walk | 55.1% [50.6, 59.5] | 47 / 54 / 66% | 52% | 55% |
| base, no walk | 49.3% [44.9, 53.8] | 47 / 50 / 52% | 45% | 61% |
| base, shuffled | 51.9% [47.4, 56.3] | 45 / 52 / 59% | 66% | 55% |
| **LoRA, 64-frame walk** | **86.7% [83.4, 89.5]** | 88 / 84 / 88% | 79% | 47% |
| LoRA, no walk | 52.5% [48.0, 56.9] | 52 / 52 / 53% | 52% | 21% |
| LoRA, shuffled | 53.5% [49.0, 57.9] | 50 / 52 / 58% | 55% | 43% |
| **Opus, 256-frame walk** | **84.1% [80.6, 87.2]** | 87 / 76 / 89% | 90% | 55% |
| Opus, no walk | 55.7% [51.2, 60.1] | 51 / 50 / 67% | 48% | 56% |
| VGGT (every 8th frame) | 82.3% | 91 / 77 / 79% | 72% | |

Paired two-sided sign tests, pooled (trial level; trials share walks, 32 per scene):
- LoRA minus base, with the walk: +31.6 points (187 better / 28 worse, p = 5e-30); novel +27.6 (18/2).
- LoRA walk minus no walk: +34.2 (197/25, p = 2e-34); walk minus shuffled: +33.2 (202/35).
- Opus walk minus no walk: **+28.4** (174/31, p = 2e-25); novel +41.4 (24/0, p = 1e-7).
- LoRA minus Opus, with the walk: +2.6 (62/49, p = 0.25); novel -10.3 (3/9, p = 0.15).
- Base walk minus no walk: +5.8 (99/70, p = 0.03).
Opus's first pass (the first 16 walks with trials, 270 trials): 85.2% with the walk, 56.3% without
(novel 96% / 56%, n = 27), close to the full set. Cues are balanced (pixel similarity favours near on
51%, frame order on 50%); with the walk, Opus and LoRA stay at 83% on trials where frame order points
to the far option. VGGT: median probe error 0.34-0.85 m on familiar views; novel 0.87 m in Apt0
(n = 42) and 2.2-2.7 m in Apt1-2 (n = 6 and 10).

Answers. (1) **Yes, the LoRA model transfers** to real scans it never trained on: 86.7%, far above
the base model, and it still needs the walk in order. (2) The base model is again near chance, but
**Opus is not in between: with the walk it matches the LoRA model**, and its walk gain (+28 points)
is about four times its gain in the procedural rooms (+6.4). (3) VGGT still places views, but at
82% it is not the ceiling here (every 8th frame, vertex-coloured renders). Caveat for
interpretation: these are apartments. The far option (> 3 m) is often in another room, so the walk
can help by grouping views into rooms; room-level recognition, not metric localisation within a
room, may carry much of the gain. Most trials are familiar (445 of 503).

## Experiment 13 (pre-registered 2026-10-23, run 2026-10-23/24): what the walk is used for

External reviews of the preprint (GPT in `docs/reviews/`; Grok 4.7 and Muse Wonder in the repository root) asked what the
order effects and the walk's cost actually show. This experiment tests the cheapest ways to tell
the candidate explanations apart. No model has seen any of these presentations.

**13a. The LoRA model's use of order** (seed 0; the ten held-out rooms P5000-P5009, all 192 trials
each; the experiment-10 64-frame setting, which shows 61-63 frames). Three new presentations of
the same shown frames, each labelled 1, 2, ... in the order shown:
- *reversed*: the frames in reverse time order;
- *window-shuffled*: the frames cut into consecutive blocks of 8 (about 7 m of walking), with the
  block order shuffled by a seed taken from the walk and the order kept inside each block;
- *renumbered*: time order kept, with labels replaced by sorted, distinct random integers from
  1-999 (seeded by the walk), so that label differences no longer track steps.

They are compared with experiment 10's ordered walk (92.5%), shuffled walk (51.3%) and no walk
(55.9%). What each explanation predicts:
- distances read off label differences: renumbered well below ordered;
- only local time-neighbours used (find a look-alike frame, pick the option near it in time):
  window-shuffled stays well above shuffled, and reversed is about equal to ordered;
- long-range order needed (integration along the whole path, or any other long-range use):
  window-shuffled close to shuffled;
- dependence on the direction of travel: reversed well below ordered.
The hand-built adjacency rule (`adjacency_baseline.py`, pixels and DINOv2) is also scored under
each presentation, as the pattern that a look-alike-plus-time-neighbour strategy produces.
*Note (2026-10-23, from the rule's scoring, before any LoRA run):* "time" can be read from the shown
labels or from each frame's position in the prompt; both are reported (`results/adjacency_presentations.json`).
In the held-out rooms the rule is near chance under every presentation (ordered 54.6% pixels, 59.2%
DINOv2; window-shuffled 49.5% / 54.2%), because their trials balance frame order by construction. So it
gives no informative pattern to compare the LoRA model's against; the explanations are told apart by
the model's own contrasts above.

Analysis: novel trials, rooms as the unit, means and paired differences with 95% bootstrap CIs over
rooms, room-level sign tests. The three primary contrasts are window-shuffled minus shuffled,
ordered minus reversed, and ordered minus renumbered. Reported either way. The paper calls the
model "order-dependent" unless these results separate the explanations.

*13a result (2026-10-24, Mac MPS, all 30 runs complete; `scripts/exp13_analysis.py` ->
`results/exp13_summary.json`; 1,299 novel trials, means over rooms with 95% bootstrap CIs):*

| presentation | novel accuracy | first-listed |
|---|---|---|
| ordered (exp. 10) | 92.5% [91.4, 93.6] | 53% |
| renumbered | 91.9% [90.2, 93.8] | 54% |
| window-shuffled (blocks of 8) | 81.9% [79.4, 84.3] | 50% |
| reversed | 67.8% [65.5, 70.1] | 49% |
| no walk (exp. 10) | 55.9% [53.1, 58.7] | 23% |
| shuffled (exp. 10) | 51.3% [48.1, 54.3] | 43% |

Primary contrasts (rooms up/down, room-level sign test; trial-level pooled sign test):
- window-shuffled - shuffled: +30.6 [+27.3, +33.8], 10/0 rooms (p 0.002), 513/115 trials;
- ordered - reversed: +24.7 [+22.4, +26.7], 10/0 rooms (p 0.002), 360/37 trials;
- ordered - renumbered: +0.6 [-0.8, +1.8], 6/3 rooms (p 0.51), 28/21 trials (p 0.39).
Also: window-shuffled - ordered -10.6 [-12.7, -8.8] (0/10 rooms); reversed - shuffled +16.5
[+12.8, +20.3] (10/0).
Against the registered predictions: renumbering costs nothing, so the model does not read distances
off label differences. Window-shuffling keeps 74% of the ordered-minus-shuffled gap (30.6 of 41.2
points), so most of what the model uses lies within blocks of 8 shown frames (about 7 m of walking);
the remaining 10.6 points need order across blocks. Reversing costs 24.7 points while staying 16.5
above shuffled, so the model depends on the direction of travel, which a symmetric "look-alike frame,
then its time-neighbours" strategy would not. The explanations are separated: local, direction-
dependent use of the sequence, plus some longer-range order; not label arithmetic.

**13b. Why the base model drops to chance with the walk** (room A, 192 lure-free trials, base
Qwen3-VL-8B, 256 px). Existing runs on novel trials: no walk 58.8%, 64-frame walk 50.0%, 256-frame
walk 50.0%. A 64-frame walk is enough to cause the drop, so prompt length beyond 64 frames does not
explain it. New condition: an *irrelevant walk*, the 64-frame presentation with every frame except
the two options replaced by the frame at the same position of room B's walk with the same index,
labels unchanged. Same length and format, no information about room A beyond the options.
- If any extra images cost accuracy, the irrelevant walk scores about 50%.
- If room-A content specifically misleads the model, it scores about as well as no walk.
Analysis: paired sign tests, irrelevant minus no walk and irrelevant minus the 64-frame walk, with
95% intervals from a bootstrap over walks; first-listed answer rates are reported alongside.
*Amendment (2026-10-23, before any 13b run):* the no-walk and 64-frame references were run on CUDA
and the base model's CUDA-MPS agreement has not been checked, so both are rerun on MPS next to the
irrelevant walk and the comparisons use the MPS runs (the CUDA runs are reported too).

*13b result (2026-10-24, Mac MPS; `results/exp13_summary.json`; 102 novel trials, 95% intervals from
a bootstrap over walks):* irrelevant walk 52.9%, no walk 58.8%, 64-frame walk 49.0% (CUDA: no walk
58.8%, 64-frame 50.0%, 256-frame 50.0%).
- irrelevant - no walk: -5.9 [-15.2, +3.7], 11/17 (p 0.34);
- irrelevant - 64-frame walk: +3.9 [-10.5, +18.8], 24/20 (p 0.65).
Inconclusive: the irrelevant walk lands between the two references and differs from neither. The
drop it was meant to explain is itself not significant: 64-frame walk - no walk is -9.8 [-22.8, +2.0],
15/25 (p 0.15) on MPS; on CUDA 15/24 (p 0.20) with 64 frames and 19/28 (p 0.24) with 256. So the
supported statement is that the walk does not help the base model, not that it hurts it. First-listed
rates: irrelevant 27%, no walk 31%, room-A 64-frame walk 55%; the irrelevant walk leaves the no-walk
lean to the second-listed option in place, while room A's own walk removes it.

**13c. VGGT controls** (the experiment-4 patching setup: same model, test set, targets and donors;
blocks 19 and 21):
- the exact orthogonal complement of position PCA-10, patched (expected transfer near 0);
- a random 10-d erasure, as the matched control for the position-PCA-10 erasure already measured
  (block 21: 3.38 m against 0.43 m clean);
- fixed alignment: the patched run scored with the clean target run's similarity transform rather
  than one refitted after patching, for the whole token and position PCA-10, with the median
  movement of the anchor cameras reported;
- heading-matched donors: donors restricted to the target's heading quadrant, for the whole token
  and position PCA-10, with the change in predicted heading reported.
Expected: complement transfer below 0.05; fixed-alignment and heading-matched transfers within 0.1
of the originals. Departures are reported and narrow the claims in the paper.

*13c result (2026-10-23, `results/vggt_patch_controls.json`; 128 targets, clean error 0.43 m).* All
four registered expectations hold. Block 19 / block 21:
- exact complement of position PCA-10: transfer 0.00 / -0.01 (expected < 0.05);
- erasure: position PCA-10 raises the error to 1.65 / 3.38 m, a random 10-d subspace leaves it at
  0.43 / 0.43 m;
- fixed alignment (the clean run's similarity transform): whole token 0.41 / 0.87, position PCA-10
  0.40 / 0.85, identical to the refitted scores; anchors move 4 mm;
- heading-matched donors (heading gap 28 degrees instead of 91): position PCA-10 0.47 / 0.80, whole
  token 0.49 / 0.82 (originals 0.40 / 0.85 and 0.41 / 0.87), all within 0.1.
Also measured: with the original donors, patching position PCA-10 turns the predicted heading by a
median 25 / 77 degrees (23 degrees with heading-matched donors), so the 10-d subspace carries heading
as well as position, consistent with heading PCA-6 transferring 0.32.

**13d. Appearance cue beyond pixels** (no model behaviour). For every lure-free trial in rooms A,
B and P5000-P5009: the DINOv2 (ViT-B/14, CLS token) cosine similarity of the new view to each
option, the share of trials where it favours the near option (novel and familiar), and each
model's accuracy split by which option it favours.

**13e. Opposite headings, strictly** (added 2026-10-23 from the Gemini 4 review, before any
recomputation). The heading split behind "heading invariance" and "cross-heading decoding"
(`analyze.py`, `halves`) uses 180-degree halves (N = sin(theta) > 0, and so on). Two frames facing 1
and 359 degrees therefore count as opposite, although with a 90-degree field of view they show
almost the same scene; about half of each half lies within 45 degrees of a boundary. Both measures
are recomputed with four 90-degree sectors centred on E, N, W and S, comparing only opposite sectors
(N vs S, E vs W; at least 90 degrees apart, so no shared view): invariance as the correlation of
rate maps built from opposite sectors (and its ratio to split-half stability), and decoding trained
on one sector and tested on the opposite one (the off-diagonal cells of `heading_matrix`). Every
condition in `results/summary.json` is recomputed from the stored activations, on CPU, into a new
file; the original numbers are kept. The teleport-probe decoding (`analyze_probes.py`, `results/probes.json`) uses the same
180-degree halves and is recomputed the same way, at every layer, into `results/probes_quadrant.json`
(scope extended 2026-10-23; only the existing single-layer heading matrices in
`results/probes_heading_matrix.json` had been seen). The paper then reports the strict version, with the 180-degree
version noted. Expected: lower invariance and higher cross-heading error everywhere. The tests
that matter are whether the moving average still matches the walk-in-context model, whether
VGGT-single keeps a clear majority of heading-invariant place units, and whether random-weight
Qwen3-VL-2B's 2.79 m survives.

**13f. A positive control for the strict test** (added 2026-10-23 at the user's request, after the
13e results above; nothing about the RNN under the strict split had been computed). Under 13e no
model passes heading invariance with a linear readout, so the test needs a model that has a
position code by construction. The path-integrating RNN of experiment 7 (velocity input only,
trained to output place-cell activity) is scored with the same strict measures along the same 48
walks (`grid_rnn.py --score --quadrant` -> `results/grid_rnn_quadrant.json`): the `s093` run
(0.093 m training steps, `--sub 3`) and the `base` run (0.22 m steps), each anchored (state reset
every ~20 network steps, as in training) and free. Prediction: a clear majority of its place units
heading-invariant on opposite sectors and strict cross-heading error well below chance, near its
within-sector error. If it passes, the strict test is shown to be passable by a real position code;
if it fails, the strict test (a linear decoder trained on a quarter of the frames) is too harsh and
the paper says so. Its only heading information is the recent direction of motion.

**Re-analyses of existing data** (sensitivity, not new tests): room-A and room-B contrasts
re-reported with 95% intervals from a bootstrap over walks (10,000 resamples); first-listed answer
rates for every model and condition; experiment 10 recomputed on complete runs, with completeness
checked against the trial manifests. The ten-room results stay the primary evidence.

Cost: none (Mac and RTX 4070). Mac time about 8 h; 4070 about 2 h, behind the shared lock.

**Results so far (2026-10-23).**
- *13d* (`scripts/appearance_cue.py` -> `results/appearance_cue.json`). DINOv2 favours the near
  option on 52.0% of novel lure-free trials over the 12 rooms (n = 1,507; room A 52.0%, room B
  55.7%, P rooms 44-60%) and 55.0% of familiar ones; it agrees with the pixel cue on only about half
  of trials, so it is a separate cue and is roughly balanced. Accuracy split by its direction (near-
  favoured minus far-favoured, novel): the base 8B depends on it strongly (P rooms: walk +15.5
  points, no walk +16.9, shuffled +11.3; room B +26-28), as does the LoRA model without the walk
  (P rooms +14.3, room B +24.4). The LoRA model with the walk (+0.6 to +6.8, P rooms p = 0.21) and
  Opus with the walk (P rooms +1.2, p = 0.67; complete runs) do not; Opus without the walk
  depends on it modestly (+6.9, p = 0.011). Single room-A runs of other models show large
  splits (Qwen3.6 with walk +25, Gemma no walk 1024 px +20, Sonnet high with walk +20, DeepSeek no
  walk +27), uncorrected over about 160 runs.
- *Walk-clustered re-analysis* (`scripts/review_stats.py` -> `results/review_stats.json`; agrees with
  the GPT review's audit where they overlap). Room A, Opus walk minus no walk: low +13.7 [-1.3,
  +28.2], sign-flip p = 0.13; medium +8.8 [+1.0, +17.8], p = 0.094; high +11.8 [-1.1, +22.7],
  p = 0.13; mean of the three +11.4 [+2.3, +20.6], p = 0.046. Ordered minus shuffled -6.9 [-15.8,
  +2.0]; ordered minus relabelled -1.0 [-11.8, +10.6]. Room B, Opus +0.9 [-11.0, +12.5]. Every LoRA
  contrast in rooms A and B stays large with walks as the unit (e.g. room A walk minus no walk +39.2
  [+28.4, +52.0], p = 1e-4; ordered minus shuffled +42.2). No open model or other provider has an
  interval excluding 0.
- *First-listed rates* (same file): above 70% for Sonnet at none/low/medium effort (74-91%),
  Haiku thinking-off with the walk (72%), Gemma-4-31B with the walk (76%); second-listed above 70%
  for every LoRA no-walk run (74-84%) and several base no-walk runs (71-76%). Unparsed at most 1%
  (Qwen3.6 thinking 9/192).
- *13e* (`analyze.py --quadrant` -> `results/summary_quadrant.json`, 14 priority conditions so far,
  the other 21 running; `analyze_probes.py --quadrant` -> `results/probes_quadrant.json`, complete).
  Opposite 90-degree sectors; chance (predicting the training sector's mean) 3.70 m for units, 3.62 m
  for probes. The CPU recompute reproduces summary.json to 0.001 except qwen-shuffled (stored
  activations differ slightly; use the file's own 180-degree values for it).
  Original 180-degree -> strict, at the layer the paper reports (cross-heading error; invariant place
  units as a share of place units):
  pixels 3.87 -> 5.11 m, 15% -> 1%; pixels + moving average 0.79 -> 0.96 m, 100% -> 97%;
  Qwen3-VL-2B single L10 3.79 -> 5.68 m, 4% -> 0%; with the walk L2 1.48 -> 2.38 m, 88% -> 47%;
  shuffled 3.52 -> 4.71 m; single + moving average 0.45 -> 0.60 m, 98% -> 88%;
  random-weight Qwen3-VL-2B 2.79 -> 4.80 m (best strict layer 4.20 m), 10% -> 1%;
  DINOv2 3.06 -> 5.30 m; random DINOv2 3.06 -> 6.91 m; VGGT alone L9 3.64 -> 5.44 m;
  **VGGT single (8 anchors) block 19: 0.97 -> 3.41 m, 64% -> 45%**; VGGT walk block 19: 1.08 ->
  2.18 m, 80% -> 63%; Qwen3-VL-8B base and fine-tuned with the walk L3: 1.23 / 1.31 -> 2.41 m.
  VGGT single at every block: strict cross-heading error 3.41-8.92 m (best block 19), while within a
  sector it is 0.08-0.13 m at blocks 15-23; at blocks 21 and 23 opposite-sector predictions miss by
  7.7-8.9 m, about twice chance, i.e. mirrored.
  Teleport probes, best layer: every model 4.16-4.55 m (was 3.19-3.58), the fine-tuned 8B 2.78 ->
  3.63 m; walk, shuffled and no walk still alike; recall baseline 1.68 -> 1.95 m.
  Answers to the three questions: (1) the moving average still matches or beats the walk-in-context
  model, by a wider margin; (2) **no**: VGGT-single keeps 45% heading-invariant place units at block
  19 and does not decode position linearly across opposite headings; its population code decodes
  precisely within a sector but not across (VGGT walk does better, 2.18 m and 63%); see 13f for why the
  decoding half of this is not diagnostic; (3) **no**: random-weight
  Qwen3-VL-2B's 2.79 m was the 180-degree boundary; strictly it is worse than chance.
- *13e, complete (2026-10-24 02:11 UTC; all 35 conditions in `results/summary_quadrant.json`).* Strictly
  heading-invariant share of place units (best layer): single frames 0-6% in every model (pixels 1%,
  DINOv2 4%, random DINOv2 2%, Qwen3-VL-2B 0%, random 2B 1%, 8B 0%, fine-tuned 8B 1%, Qwen3.8-27B 0%,
  32B 0%, SenseNova 0%, VGGT alone 6%); walk in context 7-63% (8B 7%, fine-tuned 8B 14%, SenseNova
  17%, 27B 29%, 2B 47%, 32B 51%, VGGT 63%; VGGT single with anchors 45%); the 8-step moving average of
  single frames 90-94% for every Qwen-family model and 97% for pixels (2-step 22-34%, 32-step 100%).
  Strict cross-sector decoding (13f's caveat applies): single frames 4.2-6.6 m, walk in context
  1.57-2.41 m, moving average 0.49-0.96 m. So in every model the moving average of single frames
  gives at least the walk's invariance. The fine-tuned 8B's late layers (14%) exceed the base's (7%)
  but stay far below the moving average.
- *13f* (`results/grid_rnn_quadrant.json`). Path-integrating RNN, strict opposite sectors (invariant
  place units as a share of place units; decoding within a sector / across opposite sectors; chance
  3.70 m):
  `s093` anchored: 100% (inv/stab 0.99); 0.18 / **0.31 m**. Free: 100%; 1.50 / 1.64 m.
  `base` anchored: 99% (inv/stab 0.94); 0.23 / **7.57 m**. Free: 98%; 0.29 / 4.22 m.
  So the unit-level strict test is passed by both networks (98-100% of place units invariant, against
  45% for VGGT-single block 19, 63% VGGT-walk, 47% Qwen3-VL-2B with the walk, 88% its moving
  average, 0-1% single-frame pixels, DINOv2, Qwen and VGGT-alone at the reported layers, at most 6% at
  any layer). The population test is not
  reliable: the base RNN, whose units are all heading-invariant, fails strict cross-sector decoding
  with mirrored errors (7.57 m, twice chance) just as VGGT does (7.7-8.9 m at blocks 21-23). A
  population that also carries heading (the RNN's input is the direction of motion) can defeat a
  decoder trained on one heading sector, whatever its position code. So: (1) the strict *unit*
  measure is the criterion the paper uses; (2) strict cross-sector decoding is reported only with
  this caveat and is not evidence that a code is heading-dependent; (3) the earlier reading of
  VGGT's mirrored errors as a camera-relative code (13e above) is withdrawn as unsupported.
- *Experiment 10 completeness*: `heldout_analysis.py` now checks runs against the manifests and
  writes `results/heldout_summary_incomplete.json` until every run is complete. These files are
  rerun once the Opus walks 16-31 finish.

## Experiment 14 (pre-registered 2026-10-23, run 2026-10-23/24): the deferred controls

The reviews' remaining requests that the user chose to run (all but repeated API runs). No model has
seen any of these inputs or conditions.

**14a. The fine-tuned model, recorded while doing the task.** Experiment 9 recorded the LoRA model
under a walk prompt without labels or the question; reviewers asked for the task itself.
- *Models and inputs:* the LoRA model (seed 0) and the base Qwen3-VL-8B, on the exact `two_choice.py`
  prompts (64-frame walk as shown, 61-63 labelled frames, the new view and the question), for all
  1,920 trials of P5000-P5009. bf16 on the Mac, one forward pass per trial, no generation.
- *Sites:* the residual stream after every decoder layer at (i) the new view's image-end token,
  (ii) the mean over the new view's visual tokens, (iii) the question's last token and (iv) the
  last prompt token (where the answer is generated).
- *Targets:* the new view's position (x, y); its distance to option a minus its distance to
  option b; and which option is nearer (a or b).
- *Readouts:* ridge, and a one-hidden-layer MLP (256 units, early stopping on a validation split),
  trained on five rooms and tested on the other five, both ways, averaged (no room in both).
- *Questions:* (1) Is the nearer option decodable at (iv) in the LoRA model at about its behavioural
  accuracy (a pipeline check), and not in the base? (2) Is the distance difference decodable at
  (iii)/(iv), and from which layer, in the LoRA model compared with the base? (3) Is the new view's
  (x, y) decodable at any site in the LoRA model, and at (i) under the task prompt compared with
  experiment 9's recording prompt? For (x, y), cross-sector decoding (train on one 90-degree heading
  sector of the new view, test on the opposite) is reported with experiment 13f's caveat.
  Reported either way; the paper's representation claim is then stated from these results.

*14a result (2026-10-24, Mac MPS, both models complete; `data/acts_task/`, `spatialcells/task_probe.py` ->
`results/task_probe.json`; readouts trained on five rooms and tested on the other five, both ways).*
Behaviour on these exact prompts: LoRA 92.8% (novel 92.5%), base 52.8%. Chance: nearer option 54.0%,
(x, y) 3.50 m.
- (1) Pipeline check passes: at the last token the LoRA model's nearer option reads out at 92.5% (ridge,
  layer 27), its behavioural accuracy; the base model's at 55-57%, near chance.
- (2) Distance difference (d_a - d_b) at the question and last tokens: LoRA R^2 0.74-0.76 (ridge; MLP
  0.72), appearing between layers 18 and 20 (last token R^2 0.14 at layer 18, 0.54 at 20) together
  with the nearer option (66% -> 82%); base about 0 at every layer. The decision forms in layers
  18-20.
- (3) The new view's (x, y) across rooms: not decodable at any site in either model (best R^2 0.03;
  lowest median error anywhere 3.38 m, against 3.50 m chance). Within one heading sector across rooms
  the image sites reach 2.15-2.27 m at their best layers (ridge; LoRA 2.15/2.16, base 2.22/2.27) in both
  models alike; opposite sectors 4.3-4.9 m, worse than chance (13f's caveat). Experiment 9's
  room-A decoding (about 1.4 m) was within one room, so it is not comparable with this cross-room test.
*Exploratory (2026-10-24, not registered; from the same recordings):*
- (2) is mostly the answer itself. A binary predictor from the LoRA model's own answer already gives
  R^2 0.69 for d_a - d_b (the true sign gives 0.87), and a ridge readout of what is left after the
  answer reaches R^2 0.03-0.06 across rooms (last token, layers 20-30). So the readout shows the
  decision, with at most a small graded distance component.
- Within each room (ridge, five folds grouped by walk), the new view's position reads out at 1.43 m
  from the mean of its visual tokens (layer 6) in both models alike (base 1.44 m), matching experiment
  9's room-A figure under the walk prompt; at the last token, LoRA 2.10 m vs base 2.86 m (chance 3.48 m).
  So the task prompt and the fine-tuning leave the image-level position information as it was, and add
  the decision at the answer site, not a room-general position code.

**14b. Opus with the shuffled walk** (experiment 10's registered arm, dropped there for budget, and
the same arm in the apartments). Claude Opus 5.5, low effort, the 256-frame walk shuffled with its
real frame numbers (`claude_test.py --shuffle-walk`), as registered in experiment 10. Rooms: decided
from the Claude budget before any run (see below). Analysis: ordered minus shuffled on novel trials,
rooms as the unit with a bootstrap CI (procedural rooms); paired sign test pooled (apartments, all
trials and novel). "Opus does not use the walk's order" is claimed only if the 95% CI of ordered
minus shuffled lies within +/-3 points.

*14b result (2026-10-24; `results/claude/claude-opus-5-5-roomReplicaApt*-shuffled-low.jsonl`,
`scripts/photoreal_analysis.py`; $17.30).* Opus with the shuffled walk: **88.1%** of all 503 trials
[84.9, 90.8] (Apt0-2: 88 / 85 / 92%), against 84.1% in order and 55.7% with no walk; novel 88% (58).
Ordered minus shuffled: all trials **-4.0 points** (18 better / 38 worse, sign test p = 0.01;
bootstrap over 90 walks [-7.0, -1.0], whole-walk sign-flip p = 0.017); novel +1.7 (3/2, p = 1).
Shuffled minus no walk: +32.4 (186/23). So in the apartments Opus's large walk gain does not depend
on the walk's order, and the interval excludes any advantage for the ordered walk (it slightly
favours the shuffled one). The same direction as room A (shuffled 81% vs ordered 75%, p = 0.19) and
the base 8B in the procedural rooms (56.4% vs 51.8%); the opposite of the LoRA model (53.5% shuffled vs
86.7% ordered here).

**14c. The landmark lure, edited.** Room A's 50 original novel trials where only the far option
shares a landmark with the new view (`results/landmark_shares.json`).
- *Edit:* the shared landmark is removed from the room and the whole walk, the new view and both
  options are re-rendered at the same poses. A poster is replaced by the wall's own texture.
  Pillars cannot be removed without changing geometry and occlusion, so only trials whose shared
  landmarks are all posters are used (their number is reported).
- *Control edit:* in a second variant, one poster that is visible in none of the trial's three
  views (the lowest-numbered such poster) is removed instead, with the same re-rendering.
- *Models:* base Qwen3-VL-8B with the 256-frame walk and without the walk, as in figure 2b, on the
  original, edited and control renders (all on the Mac); Opus (walk and no walk) if the budget allows.
- *Analysis:* paired sign tests on the same trials, edited minus original and edited minus control.
*Clarifications (2026-10-23, after rendering, before any model run):* (1) "posters" means every flat
wall landmark `room.py` renders as a poster, including the door and window: 38 of the 50 lure trials
qualify (12 share a pillar). The six picture posters alone give 23 trials, reported as a subset. (2)
Three trials have no control (all eight wall landmarks appear in their views), so the control
comparison has 35. In one control trial (w11p0705) the removed window shows as a ~95-pixel sliver in
both options, below the landmark threshold; it is kept. (3) The base model runs through `two_choice.py`
on MPS for original, edited and control alike (figure 2b's 18% came from another runner; the
`two_choice.py` CUDA run scores 21% with the walk on these 38). (4) Opus runs at high effort, figure
2b's setting, reusing its existing original-trial runs; about $10.
  If the shared poster causes the errors, accuracy rises on the edited trials (toward the 41% the
  base model scores on no-landmark trials) and not on the control.

*14c result (2026-10-24; base model on the Mac, MPS; Opus at high effort; `scripts/lure_edit_analysis.py` ->
`results/lure_edit_summary.json`).* 38 qualifying trials (23 with picture posters only), 35 with a
control. MPS reproduces the CUDA answers on the original trials (38/38 with and without the walk).

| run | original | edited | control | edited - original | edited - control |
|---|---|---|---|---|---|
| base 8B, walk | 21% (8/38) | 26% (10/38) | 14% (5/35) | +5 (3/1, p 0.63) | +11 (5/1, p 0.22) |
| base 8B, no walk | 16% (6/38) | 24% (9/38) | 17% (6/35) | +8 (4/1, p 0.38) | +9 (4/1, p 0.38) |
| Opus high, walk | 66% (25/38) | 53% (20/38) | 74% (26/35) | -13 (1/6, p 0.13) | -17 (3/9, p 0.15) |
| Opus high, no walk | 61% (23/38) | 74% (28/38) | 77% (27/35) | +13 (7/2, p 0.18) | +0 (3/3, p 1) |

Picture posters only: the same pattern (base with walk 22% -> 26%). Removing the shared poster does not
rescue the base model: only 4 of its 38 answers change with the walk, it still picks the far option on
about three quarters of the trials, and accuracy stays far below the 41% of no-landmark trials. So the
shared poster is not what causes the lure errors. Opus's changes are not significant either way; its
no-walk gain on edited trials is matched by the control (+14 over the original), so it is not specific
to the shared poster.
*Exploratory (2026-10-24, not registered):* neither listing order nor raw pixels explain the base
model's pull to the far option on these trials (far option listed first in 19 of 38; pixels favour the
near option in 19 of 38, before and after the edit; the model picks far 30/38 with the walk and 32/38
without). DINOv2 (the 13d features) favours the far option in 26 of 38 original trials (68%), 25 of 38
edited (66%) and 23 of 35 controls (66%), with the median margin unchanged (+0.045 original, +0.055
edited). The lure trials are trials where the far option looks more similar overall; the shared poster
marks that similarity but is not its source.

**14d. VGGT, cross-room patching.** The experiment-4 setup at blocks 19 and 21 in room B (anchors at
the same poses, targets and donors built the same way). Subspaces (position PCA-10, PCA-2 and the
lambda = 1e4 ridge) fit on room A's fit frames are compared with the same subspaces fit on room B's.
If the causal subspace is a room-general position code, room-A-fit transfer in room B is within 0.1
of room-B-fit transfer.

*14d result (2026-10-24, `results/vggt_patch_crossroom.json`; 128 targets in room B; clean error
0.85 m in room B, 0.42 m on room A's fit frames).* Transfer, subspace fit in room A / fit in room B:
block 21 position PCA-10 0.87 / 0.87, PCA-2 0.19 / 0.22, ridge 1e4 0.15 / 0.16 (whole token 0.90);
block 19 PCA-10 0.45 / 0.42, PCA-2 0.17 / 0.18, ridge 0.16 / 0.17 (whole token 0.48). Paired A-fit
minus B-fit differences -0.04 to +0.03, all within the registered 0.1; the A- and B-fit subspaces
overlap 0.70-0.92. **The causal subspace carries over to the other room.**

**14e. VGGT remapping with a changed reference.** Block-19 camera-token rate maps in room A with
(i) the standard 8 anchors, (ii) the same anchors in another order (a different first anchor), and
(iii) all anchor poses rotated 90 degrees about the room centre. Map correlation with (i), unit by
unit, and with (i)'s maps rotated 90 degrees. If the units code position in the first anchor's or
the anchors' frame, (ii)/(iii) move the maps (lower correlation with (i); (iii) correlating with the
rotated maps); if they code the room's own frame, the maps stay. Reported either way; the remapping
paragraph is framed by it.
*Amendment (2026-10-23, before any 14e recording):* the 8 standard anchors sit at poses that map onto
each other under a 90-degree turn about the room centre, so (iii) as written renders the same 8 images
in another order and cannot separate the anchors' frame from the room's. (iii) is replaced by all
anchor poses rotated **45 degrees** about the room centre (new images, a moved anchor set), compared
with (i)'s maps rotated 45 degrees. (ii) stays (a corner view, (2, 2) facing the centre, first). To
share the RTX 4070 fairly, each anchor set runs as its own locked job.

*14e result (2026-10-24, RTX 4070; `results/vggt_remap_reference.json`, `spatialcells/vggt_reference.py`).*
The camera head works with every set (median error 0.42 m standard, 0.47 m reordered, 0.42 m rotated).
Block-19 place units of (i) (1,505; split-half stability median 0.95; a 45-degree turn of (i)'s own maps
correlates 0.66 with them, so the test can tell a turned map from a kept one). Medians over units:

| set | vs (i) | vs (i) turned +45 | vs (i) turned -45 | closer to turned +45 | also a place unit (chance) |
|---|---|---|---|---|---|
| (ii) reordered | 0.68 | 0.85 | 0.03 | 71% | 43% (22%) |
| (iii) rotated 45 | 0.68 | 0.95 | -0.01 | 96% | 72% (31%) |

The strictly heading-invariant place units (957) give the same picture (rotated: 0.97 vs turned +45,
99% closer to it). Position decoders trained on one set fail on another (2.7-3.8 m; within a set
0.13-0.15 m). So the maps move with the anchors: rotating the anchor poses turns the maps by the same
45 degrees, and reordering alone, which changes only the first anchor (from the centre facing east to
(2, 2) facing the centre, a 45-degree change of heading), turns them too. VGGT's place units code
position in the frame the anchors define (consistent with the first camera's frame, which VGGT's
outputs are expressed in), not in the room's own frame. This also explains why they keep their maps
across rooms A and B, where the anchors sit at the same poses. Turned +45 fits (ii) less well than
(iii) (0.85 vs 0.95), as expected if the first camera's position, which moves in (ii), also matters;
only rotation was registered, so this is not tested.

**Cost.** 14a, 14c (base model), 14d and 14e: none (Mac and RTX 4070). 14b and Opus in 14c come from
the Claude budget ($45 left of $138 until next month): 14b about $7 per procedural room and $6.5 per
apartment; Opus in 14c about $13. The rooms in 14b are fixed by the user's budget decision, before
any run.
*Budget decision (2026-10-23, before any 14b or Opus run):* stay within the $45 left. 14b runs in the
three apartments only (ReplicaApt0-2, all trials), and Opus runs in 14c. The ten procedural rooms
are not run, so the +/-3-point equivalence claim cannot be made for them and the paper keeps its
narrowed wording about Opus and order in those rooms. 14b's analysis is the pooled paired sign test
in the apartments (all trials and novel).
*Budget decision revised (2026-10-24, by the user, before any procedural-room 14b run):* a second
Claude API key with $200 pays for the arm that was left out. 14b now also runs in the ten held-out
procedural rooms P5000-P5009 exactly as registered above (Opus 5.5, low effort, 256-frame walk shuffled
with its real frame numbers, all trials), and its procedural-room analysis is the registered one:
ordered minus shuffled on novel trials, rooms as the unit, bootstrap CI over rooms, and the +/-3-point
equivalence claim only if that CI lies inside +/-3 points. The ordered and no-walk runs are experiment
10's. Cost about $5-7 a room; runs on the second key stop before its spend would pass $200
(`scripts/run_heldout_opus_shuffled.sh`; `scripts/cost_log.py` reports that key's spend separately).
*14b result, procedural rooms (2026-10-24/25; `results/claude/claude-opus-5-5-roomP50xx-shuffled-low.jsonl`,
all 192 trials in each of the ten rooms; `scripts/heldout_analysis.py` -> `results/heldout_summary.json`,
key `exp14b_procedural`, with its own rng so experiment 10's numbers are unchanged; $60.37 on the second
key, one connection error resumed).* Novel trials, means over rooms with 95% bootstrap CIs:
Opus with the shuffled walk 77.0% [75.5, 78.8], against 69.2% with the walk in order and 62.8% without.
- ordered - shuffled: -7.8 points [-11.6, -4.8]; ordered better in 0 rooms, worse in 10 (p 0.002);
  trials 114 better / 209 worse (p 1e-7);
- shuffled - no walk: +14.2 [+10.6, +18.2], 10/0 rooms; trials 330/153.
The registered equivalence test fails, but in the other direction: the CI lies outside +/-3 points
because the shuffled walk is better, not worse. So "Opus does not use the walk's order" cannot be
claimed as registered; what the data show is that presenting the walk in order does not help Opus and
presenting it shuffled (with the real frame numbers, so the order can still be read from the labels)
helps more, as in the apartments (shuffled 4.0 points above ordered). The option frames sit at the same
average position in both presentations (relative position 0.49-0.51; near after far in 51% vs 49% of
trials), so the shuffle carries no position cue. The fine-tuned model shows the opposite dependence
(shuffled 51.3%, at chance). Why shuffling helps Opus is not tested here.
*Exploratory (2026-10-25, not registered; same trials in both presentations, novel, pooled over rooms):*
the shuffled walk's advantage depends on where the near option sits in the ordered walk. By fifths of
the walk (first to last), ordered 79 / 68 / 61 / 68 / 70%, shuffled 80 / 79 / 74 / 78 / 75%: almost no
difference when the near frame comes early, 13 points when it sits in the middle (part of the middle dip,
74% vs 80%, is trial difficulty, since it shows in the shuffled runs too). By how far apart the two
options are in the ordered walk (quarters, close to far), ordered 64 / 71 / 77 / 78%, shuffled 73 / 79 /
80 / 82%. This fits a weakness for frames in the middle of a long, ordered image sequence, which
shuffling spreads out, but it is correlational. A direct test would rotate the ordered walk cyclically
(order kept, start moved) so the near option lands at the start or end.

## Experiment 15 (pre-registered 2026-10-24, run 2026-10-24/25): which walk frames the fine-tuned model reads

Experiment 13a showed that the LoRA model uses short stretches of the walk in the order walked, and
14a that its decision becomes readable at the answer site between layers 18 and 20, with no
room-general position code. This experiment intervenes inside the model to find which shown frames
the decision depends on. No model has seen these conditions.

- *Model and trials:* the LoRA model (seed 0), bf16 on the Mac (MPS), experiment 10's exact 64-frame
  prompts (61-63 labelled walk frames, the new view, the question; built as in `task_record.py`, with
  its token-count check against experiment 10), all 1,299 novel trials of P5000-P5009. Answers by
  greedy generation and parsing exactly as `two_choice.py` does.
- *Intervention (attention cut):* for a set S of shown walk frames, every token outside S is stopped
  from attending to the visual tokens of the frames in S (from `<|vision_start|>` to `<|vision_end|>`
  inclusive; the frames' text labels stay visible), by an additive -inf attention mask, in every
  decoder layer and in every generation step. Nothing else changes: same tokens, positions and
  images.
- *Frame sets* (8 frames each, never including the two option frames):
  - *near neighbours*: the 4 shown frames before and the 4 after the near option in the order shown
    (at a walk end, the missing ones are taken from the other side);
  - *far neighbours*: the same around the far option;
  - *look-alikes*: the 8 shown frames whose DINOv2-base CLS features (as in experiment 13d) are most
    similar (cosine) to the new view;
  - *random*: 8 shown frames drawn with a seed from the trial id (the control for cutting 8 frames).
  Sets may overlap; the overlaps (near neighbours with look-alikes, near with far neighbours) are
  reported, and the primary contrasts are repeated on the trials without that overlap.
- *Conditions:* clean (no cut), and each set cut in all layers (primary). Secondary, on rooms
  P5000-P5002 only (about 390 novel trials): near neighbours and look-alikes cut only in layers 1-18 or
  only in layers 19-36 (before and after the decision forms, per 14a).
- *Check:* the clean run reproduces experiment 10's answers on at least 99% of trials; otherwise the
  run stops and the mismatch is reported before any cut condition is run.
- *Analysis:* as 13a. Novel-trial accuracy per room, means over rooms with 95% bootstrap CIs (10,000
  resamples), paired per-room differences with room-level sign tests, and trial-level pooled sign tests.
  Primary contrasts: near neighbours minus random, look-alikes minus random, near neighbours minus far
  neighbours. The secondary windows are compared with clean and with the all-layer cut.
- *What each account predicts:*
  - the decision reads the walk around the near option (local sequence where the query was taken):
    cutting near neighbours costs well more than random, and more than far neighbours;
  - the decision starts from frames that look like the query: cutting look-alikes costs well more
    than random;
  - no 8-frame set is special (the information is spread along the walk): every set costs about as
    much as random.
  The windows then say whether the frames that matter are read before layer 19 (feeding the decision
  as it forms) or after it. Reported either way; the paper's statement about what the fine-tuned model
  reads is made from these results only, and without a claim about how it computes the answer.
- *Cost:* none (Mac). About 6.5 h for the primary conditions and 1.5 h for the windows; one job at a
  time on the Mac.
*Note (2026-10-24, before any run; implementation, no design change):* `spatialcells/frame_cut.py`,
`scripts/mac/run_frame_cut.sh`, `scripts/frame_cut_analysis.py` -> `results/frame_cut_summary.json`.
Checks before the run: token counts match experiment 10 on 20 prompts; every cut set covers 8 frames'
528 tokens; with the hooks on and nothing cut the logits are bit-identical to no hooks; in eager
attention, tokens outside a cut set give it exactly zero attention in every hooked layer (prefill and
decoding), and only in layers 19-36 with that window; the clean run reproduced experiment 10's answers
on 12 of 12 trials. Choices the registration left open: the tokens of the cut frames still attend to
each other; the "without overlap" analyses use the 503 trials where near neighbours and look-alikes
share no frame (for near and look-alikes minus random) and the 1,028 where near and far neighbours
share none (for near minus far). Timing on the Mac is 4.9-5.6 s per trial, so the run takes about
12.5 h rather than 8.

*Result (2026-10-25; `results/frame_cut/`, `scripts/frame_cut_analysis.py` -> `results/frame_cut_summary.json`).*
The check passed: the clean run gave experiment 10's answer on 1,299 of 1,299 trials (92.5%). Novel
trials, means over rooms with 95% bootstrap CIs:

| frames cut (all layers) | accuracy | minus clean |
|---|---|---|
| none (clean) | 92.5% [91.4, 93.6] | |
| 8 random | 91.6% [90.0, 93.1] | -1.0 [-2.0, -0.0] |
| 8 look-alikes of the new view | 90.3% [88.7, 92.3] | -2.2 [-3.4, -0.9] |
| 8 around the far option | 89.1% [86.3, 91.5] | -3.5 [-5.6, -1.5] |
| 8 around the near option | 83.0% [80.9, 84.7] | -9.5 [-10.8, -8.3] |

Primary contrasts (rooms better/worse, room-level sign test; trial-level pooled sign test):
- near neighbours - random: -8.6 [-10.0, -7.3], 0/10 rooms (p 0.002), trials 30/143 (p 8e-19); on the
  503 trials where near neighbours and look-alikes share no frame, -6.4 [-9.6, -3.3], 1/8 rooms (p 0.04);
- look-alikes - random: -1.2 [-2.7, +0.2], 3/6 rooms (p 0.51), trials 39/56 (p 0.10); without overlap,
  +0.4 [-1.9, +2.8];
- near neighbours - far neighbours: -6.1 [-8.2, -3.8], 0/9 rooms (p 0.004); on the 1,028 trials
  without overlap, -7.1 [-9.4, -4.7], 0/10 rooms (p 0.002).
Against the registered predictions: the decision depends on the walk around the near option (cutting
those 8 frames costs well more than random and more than the far option's neighbours), not on the
frames that look like the new view (no more than random), and the information is not spread evenly
along the walk. The far option's neighbourhood matters a little (-3.5 vs clean).
Secondary windows (P5000-P5002, 399 novel trials; clean 91.9%, near cut in all layers 80.4%): near
neighbours cut only in layers 1-18 give 86.5% (-5.5 vs clean), only in layers 19-36 88.8% (-3.1);
each window alone costs less than the cut in all layers (+6.1 and +8.4 above it). Look-alikes: 89.3%
(1-18) and 91.4% (19-36), against 87.7% in all layers. So the near option's neighbourhood is read
both before and after layer 19, with the larger share before; the window comparison is on three rooms
and was not tested formally.

*Reproducibility note (2026-10-25, found while preparing the LoRA seeds on vast-3).* `room.py` draws
the "blobs" poster texture with `skimage.data.binary_blobs()` without a seed, so any room that uses it
renders differently each time: 242 of the 1,200 procedural training rooms and held-out room P5005. Room A
and room B (fixed posters) re-render byte-identically. Every model saw a single render of each room (the
fine-tuning data rendered once on vast-1; P5005 rendered once on the Mac and used by all experiment-10
models), so the results are internally consistent, but those rooms cannot be re-rendered exactly. The
existing renders are the data of record and are to be archived with the preprint; seeding the call
would change those rooms relative to the record, so the code is left as it is.

## Experiment 16 (pre-registered 2026-10-25, run 2026-10-25): more out-of-distribution houses (HSSD)

Experiment 12 tested three Replica apartments. This adds more and larger interiors from a different
source, to see whether its results generalise beyond three scenes. No model has seen any of these
scenes, and none has been rendered.

**Scenes.** HSSD (Habitat Synthetic Scenes Dataset, `hssd/hssd-hab` on Hugging Face, CC BY-NC 4.0):
211 artist-built houses based on real floor plans, rendered with habitat-sim 0.3.3. Eight houses are
chosen by a fixed draw: the scene ids are sorted, shuffled with `random.Random(16)`, and taken in that
order. A drawn house is used if it loads, has a navmesh (the shipped one, or one computed with the
agent settings of experiment 12), its largest navigable island is 40-250 m^2, its objects carry
instance labels for the landmark rule, and it yields at least 60 lure-free trials. Otherwise the next
house in the order is drawn. The skipped houses and the reason for each are reported. These are
houses, so, as with the apartments, "the same room" in the prompt means the same house, and the walk
may help by grouping views into rooms.

**Walks, probes and trials.** Exactly as in experiment 12:
- the same camera, foraging walk, 32 walks of 256 steps on the largest island with 0.15 m clearance,
  and 128 probes per walk;
- the lure-free rules (near option within 1 m, far more than 3 m, both facing more than 90 deg from
  the new view, pixel similarity and frame order balanced), "no shared landmark" by instance labels
  (no labelled object other than walls, floor, ceiling, windows, doors and similar covering 2% or
  more of both views), no view with more than 5% of pixels showing no geometry, 6 trials per walk.

**Models and conditions** (prompts as before):
- base Qwen3-VL-8B and the LoRA model (seed 0): 64-frame walk, no walk, shuffled 64-frame walk (Mac);
- Opus 5.5 at low effort: 256-frame walk, no walk, and the shuffled walk with real frame numbers (the
  second Claude key; about $11.5 per house, about $92 in all; the runner stops before that key's spend
  would pass $200);
- VGGT as in experiment 12 (every 8th walk frame, the two options and the probe; RTX 4070).

**Outcomes.** As in experiment 12, the primary outcome is accuracy on all lure-free trials, per house
with an exact binomial CI and pooled over houses; novel-trial accuracy pooled is secondary. With eight
houses, the contrasts are also reported with houses as the unit: means over houses with 95% bootstrap
CIs (10,000 resamples, seed 0) and house-level sign tests, beside the pooled trial-level sign tests.
Questions:
1. Does the LoRA model transfer: LoRA minus base, and LoRA walk minus no walk and minus shuffled?
2. Does Opus gain from the walk, and is it, as in experiments 12 and 14b, at least as good with the
   walk shuffled as in order?
3. Is the base model near chance?
4. Does VGGT still place probes?
Reported either way, and pooled with experiment 12's apartments only as a separate, labelled summary.

*Implementation note (2026-10-25, after rendering, before any model saw a house):* `spatialcells/hssd_select.py`,
`habitat_scene.py --hssd`, `scripts/rtx4070/render_hssd.sh`; selection record `results/hssd_selection.json`.
- The Hugging Face release holds 168 houses, not the 211 of the HSSD paper; the draw ran over those 168
  as registered (sorted ids, `random.Random(16)`).
- 13 houses were drawn and 8 accepted (HSSD1-8 in draw order). Skipped: three whose largest navigable
  island is a yard or a whole lot (327-866 m^2), one at 35.6 m^2, and one whose largest island is
  outdoors and gave 10 trials (88% of views more than 5% holes).
- HSSD ships no navmesh, so it is computed with experiment 12's agent (1.5 m tall, 0.1 m radius),
  furniture counted as obstacles. HSSD's semantic ids are per category, so the instance image uses each
  object's own id; the stage (walls, floor, ceiling) is structure, and doors, windows, carpets (as rugs),
  stairs, beams and wall panels are mapped onto experiment 12's non-landmark list. Walk seeds are 1600
  plus the draw position.
- Accepted houses: largest islands 57-128 m^2, no view more than 5% holes. Trials (all / novel): HSSD1
  188/68, HSSD2 192/53, HSSD3 180/38, HSSD4 192/38, HSSD5 156/36, HSSD6 170/40, HSSD7 180/55, HSSD8
  192/38; 1,450 in all, 366 novel.

*Result (2026-10-25; `scripts/hssd_analysis.py` -> `results/hssd_summary.json`).* Qwen ran on the Mac for
HSSD1-3 and on vast-3 for HSSD4-8 (platform note above); Opus on the second Claude key ($103 for the eight
houses); VGGT on the RTX 4070. Pooled over the eight houses, all 1,450 lure-free trials [exact 95% CI];
novel trials (366):

| | all trials | novel | per house (all) |
|---|---|---|---|
| base, 64-frame walk | 60.3% [57.7, 62.8] | 55.5% | 55-65% |
| base, no walk | 51.5% | 50.0% | |
| base, shuffled | 59.8% | 59.0% | |
| **LoRA, 64-frame walk** | **91.9% [90.4, 93.3]** | **85.0%** | 87-95% |
| LoRA, no walk | 51.0% | 51.4% | |
| LoRA, shuffled | 54.2% | 54.9% | |
| **Opus, 256-frame walk** | **87.7% [85.9, 89.4]** | **83.3%** | 82-94% |
| Opus, no walk | 61.7% | 61.7% | |
| Opus, shuffled | 89.2% [87.5, 90.8] | 84.7% | |
| VGGT (every 8th frame) | 82.7% | 74.9% | |

Contrasts, pooled trial-level sign tests and houses as the unit (mean [95% CI], houses up/down):
1. LoRA transfers: LoRA minus base with the walk +31.7 (498/39), houses +31.7 [+28.4, +34.8], 8/0; novel
   +29.5, houses 8/0. It needs the walk in order: walk minus no walk +41.0 (8/0), walk minus shuffled
   +37.7 (8/0).
2. Opus gains from the walk: +26.1 (453/75), houses +26.2 [+24.3, +28.4], 8/0; novel +21.6 (8/0). As in
   experiments 12 and 14b it does at least as well shuffled: shuffled minus ordered +1.5 (94/72, p 0.10),
   houses +1.4 [-0.7, +3.3], 5/3; novel +1.4 (p 0.52).
3. The base model: with the walk +8.8 over no walk (325/198, p 3e-8; 8/0 houses), mostly on familiar
   trials; novel +5.5 (68/48, p 0.08). Unlike in the procedural rooms, the walk helps it somewhat here.
4. VGGT places probes as in the apartments (two-choice 82.7%; median probe error 0.5-1.8 m on novel
   trials, 0.26-0.95 m on familiar ones, by house).
LoRA minus Opus: +4.2 on all trials (8/0 houses), +1.6 on novel (p 0.56). The house caveat applies: the
walk may help by grouping views into rooms.

*Platform amendment (2026-10-25, after the Mac had run HSSD1 and part of HSSD2):* to finish sooner, the
Qwen runs are split by house. HSSD1-3 ran on the Mac as registered (MPS, files tagged `-mps`) and HSSD4-8
on vast-3's RTX 5090 (CUDA, untagged files; gq jobs vast3:83-86 and 88). All six conditions of a house
ran on one machine, in `scripts/mac/run_heldout.sh`'s order, from md5-identical copies of the walks and
trials. Before the split, vast-3 reproduced the Mac's HSSD1 answers on walks 0-3 for the base model with
the walk and the LoRA model with and without it (72 of 72 trials, the same answers and text).

## Follow-up pilot WP (pre-registered 2026-10-26, before any of these models saw our walks): navigation policies and a streaming 3D model

`docs/followup_world_models.md` plans a follow-up paper that asks which models trained on egocentric
video and self-motion form a place code that passes our tests. This pilot runs the plan's ranks 2 and 4
on the Mac and the RTX 4070: the Berkeley navigation policies ViNT and NoMaD (GNM as a check), which
were trained on real robot driving and have a head that predicts how many steps away a goal image is,
and CUT3R, a streaming 3D reconstruction model that carries a learned state across the whole walk,
together with TTT3R's update rule on the same weights. NWM (rank 1) waits for the user to accept its
gated license. When this section was written, the weights had been downloaded and the code read, but
no model had seen any frame of ours. The section fixes the plan's W0, W1, W2, W4 and W5 for these
models (as WP0-WP5). Room A is the selection scene, so the pilot's labels are provisional; the follow-up
registers its own confirmation scenes.

**Models, code and weights** (official pages, licenses and weights re-checked on 2026-10-26):
- ViNT, NoMaD and GNM: `robodhruv/visualnav-transformer` (MIT, commit dca7981), with the weights
  `vint.pth`, `nomad.pth` and `gnm.pth` from the Google Drive folder that its README links, and the
  configs `train/config/{vint,nomad,gnm}.yaml`. ViNT and GNM see the current frame and the 5 before it
  at 85 × 64 px; NoMaD sees the current frame and the 3 before it at 96 × 96 px. Their distance heads
  were trained on goals 0-20 steps ahead.
- CUT3R: `CUT3R/CUT3R` (CC BY-NC-SA 4.0, commit 8bc15dc), the final checkpoint
  `cut3r_512_dpt_4_64.pth` from the README's Google Drive link. TTT3R: `Inception3D/TTT3R` (CC BY-NC-SA
  4.0 for the CUT3R parts and MIT for its own, commit edd6d8c), the same checkpoint, with its
  `ttt3r` state-update rule.
- License handling (plan Section 7): the repositories and weights live outside this repository
  (`~/wm_models/`), and each model family has its own virtual environment. Our adapters
  (`spatialcells/wm_vint.py`, `spatialcells/wm_cut3r.py`, MIT) put a checkout on `sys.path` and import
  from it; no code is copied. Checkpoints are unpickled only with `weights_only=True` and an allowlist
  of the classes they contain, and models are built from our own reading of the stored configuration.

**Inputs.** The existing renders (256 px, 90° horizontal field of view, camera 1 m above the floor),
each passed through the model's own preprocessing:
- ViNT family: its training preprocessing, a center crop to 4:3 (256 × 192) resized to the model's
  input size, then ImageNet normalization. One walk step (0.22 m × U(0.7, 1.3)) counts as one model
  step: the training data are spaced 0.25 m (RECON) and 0.255 m (SACSoN), and RECON's camera is
  0.95 m high.
- CUT3R and TTT3R: the demo's loader for a square image at size 512, which upsamples to 512 × 512 and
  center-crops to 512 × 384.
- "Anchored" sequences start with the 8 anchor views of experiment 4 (`vggt_cells.anchors`), so every
  sequence's output frame is the first anchor's, as it was for VGGT. The state is never reset.
Scenes: room A for every analysis, Replica apartment 0 (`walks-roomReplicaApt0`) also for the
competence gates, and the registered lure-free trials for WP5. The only new render is a set of 256
teleport probes per walk in room A (WP1).

**WP0. Competence gates** (the plan's thresholds, fixed before any run). A model that fails its gate in
room A but passes in the apartment is reported as competent only on photoreal scenes; its WP1, WP2 and
WP4 analyses, which need the 10 m arena of `analyze.py`, are left to the follow-up. A model that fails
in both scenes is reported as not competent on our renders and left out of everything below.
- ViNT, NoMaD and GNM: Spearman ρ ≥ 0.5 between the predicted temporal distance and the true step
  distance k, pooled over pairs. The observation is the frame at step t = 5, 15, ..., 235 of every
  walk, with its own past walk frames as context; the goal is the frame k = 0, 1, ..., 20 steps later
  (NoMaD with the goal unmasked). 95% CI from a bootstrap over walks (10,000 resamples, seed 0).
- CUT3R and TTT3R: median camera-position error ≤ 0.5 m after a similarity alignment. The first 16 walks
  of each scene are each streamed from a fresh state without anchors, and each walk's predicted camera
  centers are aligned to the true ones (x, y, 1 m) by one similarity transform (Umeyama, all 256
  frames). Also reported: the error over the first and the last 64 frames (forgetting) and, in room A,
  the anchored error with the alignment fit on the 8 anchors only (VGGT single: 0.42 m).

**WP1. Strict heading invariance of place units.** `analyze.py --quadrant`, unchanged: rodent place
criteria against circular-shift shuffles, and invariance as the correlation between rate maps built
from opposite 90° heading sectors (N vs S, E vs W). The measure is the share of place units that are
strictly invariant (correlation > 0.5), `frac_place_heading_invariant_quadrant / frac_place`. On this
measure the path-integrating RNNs score 98-100%, VGGT single 45% (block 19), VGGT with the walk 63%, and
single-frame models 0-6% (experiments 13e-f).
- Units recorded (float16, `data/acts/<tag>/L<k>.npy`, rows in `frames.npy` order):
  - ViNT: L0 the current frame's EfficientNet-B0 feature (1,280); L1-L4 the MLP hidden units (2,048)
    of transformer layers 1-4 at the current frame's token; L5-L9 the five layers after the transformer
    (512, 256, 128, 64 and 32 units, after ReLU), which mix all tokens.
  - NoMaD: L0 as for ViNT; L1-L4 the MLP hidden units (1,024) at the current frame's token, with the
    goal masked (its exploration mode); L5 the pooled observation encoding (256) that its distance head
    and diffusion policy read.
  - GNM: L0 the pooled features of its observation encoder, which stacks the context frames as
    channels (1,280); L1 the compressed observation encoding (1,024); L2-L3 its joint layers (256 and
    32), with the goal equal to the current frame.
  - CUT3R and TTT3R: L3, L6, L9 and L12 the MLP hidden units (3,072) of image-decoder blocks 3, 6, 9
    and 12 at the pose token, the counterpart of VGGT's camera token; L13 the state after this frame's
    update, averaged over its 768 tokens (768); L0 the encoder's mean patch token (1,024), a
    single-frame reference.
- Conditions (room A, 48 walks of 256 steps):
  - ViNT family: *walk* (the model's past walk frames as context; ViNT and GNM take the current frame
    as the goal, NoMaD masks the goal), *single* (the context is the current frame repeated), and
    *shuffled* (the context frames are drawn at random from the same walk, seeded by walk and step).
  - CUT3R: *walk* (anchors, then the walk with the state carried), *single* (anchors, then the frame
    alone), *alone* (the frame as the first view of a fresh sequence), and *shuffled* (anchors, then
    the walk's frames in an order shuffled with the walk as seed; rows are put back in walk order
    before scoring). TTT3R: *walk* and *single*.
  - For every model, the 8-step moving average of *single* (`<tag>-single+ema8`), the control that
    reaches 90-97% in every model tested so far.
- Each condition is reported at its layer with the highest share, with every layer in the file. The
  provisional labels are "place-like" at ≥ 45% (VGGT single's share), "view-like" at ≤ 10%, and
  "intermediate" in between. Because the moving average alone reaches 90-97%, a high share along the
  walk counts as evidence of integration only when the walk condition beats the model's own
  *single* + EMA-8. The test that removes smoothing is the one on teleport probes, next.
- *Teleport probes* (the plan's W1 measure): 256 probes per walk in room A (12,288; `render_probes
  --per-walk 256 --seed 17` into `data/walks-p256/`, which links room A's walks), about 30 per 0.5 m
  bin. Units are scored on the probe activations with the same criteria: `analyze.py` runs on a probe
  arena in which a probe's walk is its walk and its step is its index, and since probes come in random
  order, the circular shifts act as permutations. Probe conditions: for CUT3R and TTT3R, *probes-walk*
  (each probe read against the state at the end of its anchored walk, with the state's write disabled,
  `update = False`) and *probes-single* (read against the state after the anchors); for the ViNT family,
  *probes-single* (the context is the probe repeated: these models keep at most 5 frames, so after a
  teleport they have nothing else). References on the same probes: pixels, and VGGT single at blocks
  19 and 21 (`vggt_cells` on the 4070). Same labels and layer rule.

**WP2. Teleport probes across headings.**
- *Primary (the plan's W2):* the novelty cost. A ridge decoder (λ chosen as in `analyze.decode`) is
  trained on the even walks' walk-frame activations at all headings, in the condition that matches the
  probe condition (walk for probes-walk, single for probes-single), and tested on the odd walks'
  256-per-walk probes, split into familiar probes (some walk frame within 1 m faced within 45°,
  `analyze_behave.geometry`) and novel ones. The novelty cost is the median error on novel probes minus
  that on familiar ones. Baselines under the same protocol: pixels, and the recall baseline (the 8-step
  moving average of pixels at the walk frame that looks most like the probe). Decision rule: "beyond
  view recognition" requires a novel-probe error below the recall baseline's and a novelty cost below
  the pixels', each difference with a 95% CI that excludes zero (paired bootstrap over the 24 odd
  walks). Layer: the one WP1 selects on probes.
- *Model-native (CUT3R, TTT3R):* the pose head's position for each probe, read with `update = False`
  after the anchored walk and mapped to room coordinates by the similarity fit on that sequence's
  anchors: median error on novel and familiar probes, and its novelty cost.
- *Strict cross-sector decoding:* a ridge decoder trained on one 90° sector of the even walks' probes
  and tested on the opposite sector of the odd walks' (as `analyze_probes --quadrant`), reported with
  experiment 13f's caveat and used for no claim.

**WP4. CUT3R pose-head patching** (experiments 4, 13c and 14d; the plan's W4). Single mode (anchors,
then one walk frame), so that the anchors' poses, and with them the alignment to the room, are the same
in every run. The 128 targets, their donors (more than 3 m away) and the 2,048 fit frames are drawn as
in `vggt_patch.py` (rng seed 0 and the same metadata, hence VGGT's frames). Sites: the pose token's
residual stream after each of the 12 image-decoder blocks. Written from the donor: the whole token;
position PCA-10 and PCA-2 (`vggt_patch.mean_subspace`, the mean token per 1 m bin); the ridge probe for
(x, y) at the λ with the best held-out R² (`ridge_subspace`); a random 10-d subspace (seeded by block);
and the exact complement of PCA-10. Erased (set to the fit frames' mean): PCA-10 and random-10, scored
by position error. Readout: the pose head's camera position for the walk frame in room coordinates.
Transfer is how far that position moves toward the donor's clean prediction, as a fraction of the
distance between them (`vggt_patch`'s definition). Decision rule (plan): a "causal position code"
requires PCA-10 transfer ≥ 0.5 at some site, complement and random-10 transfer ≤ 0.05, and an erasure
error at least twice the clean error with the random erasure within 10% of clean (VGGT: 0.85, 0.00 and
0.00; 3.38 m vs 0.43 m). Secondary, in walk mode: the state and the pose memory entering a target walk
frame are replaced by those of a donor walk at the donor frame, and transfer is scored the same way;
this asks whether the remembered map, rather than the view, sets the pose. If the primary rule passes,
experiment 14d's cross-room test follows (subspaces fit in room A and written in room B; the test is
whether room-A-fit minus room-B-fit transfer lies within 0.1).

**WP5. The temporal-distance head on the registered two-choice trials** (the plan's W5; ViNT, NoMaD,
GNM). For each option X (a walk frame), the observation is X with the walk frames before it as context
(5 for ViNT and GNM, 3 for NoMaD; at the start of a walk the earliest frame is repeated), and the goal
is the new view. The model picks the option with the smaller predicted temporal distance; a tie counts
as half. No-walk control: the context is X repeated. Secondary: the reverse direction, with the new view
(repeated as context) as the observation and each option as the goal. Trials: every registered
lure-free trial, i.e. rooms A and B (192 each), P5000-P5009 (1,920, of which 1,299 novel), the three
apartments (503) and the eight houses (1,450). Outcomes as in experiments 10, 12 and 16: novel-trial
accuracy in P5000-P5009 with rooms as the unit (mean over rooms, 95% bootstrap CI, room-level sign
test); all-trial accuracy in the apartments and houses, pooled with exact binomial CIs and with houses
as the unit; rooms A and B with exact binomial CIs. "Above chance" requires a CI that excludes 50%;
"uses the walk" requires walk minus no walk above zero with a 95% CI over rooms that excludes zero.
References: base Qwen3-VL-8B near 50%, the LoRA model 92.5%, Opus 69.2% (novel trials, P rooms) and
VGGT 82-83% (photoreal).

**Predictions** (from the plan; every result is reported either way): ViNT, NoMaD and GNM are view-like
in WP1 and WP2 and near 50% on novel trials in WP5; CUT3R is place-like in WP1 and WP4 if any learned
state is.

**Multiple comparisons.** Holm across the five models for each primary decision (the WP1 probe label,
WP2, WP4 and WP5).

**Compute and cost.** No API and no paid service. Jobs go through gq on the Mac (`--priority 150`, below
experiment 9c's) and, if its system disk allows, the PC, each split to about an hour so that 9c's Mac
tests never wait longer. Outputs: `results/wm_pilot/` (gates, probes, patching, two-choice) and
`results/wm_pilot/summary_quadrant.json` (unit scores; the main `results/summary_quadrant.json` is
not touched).

*Implementation notes (2026-10-26, no change to any test).* The unit scores are written per model
(`results/wm_pilot/units/<model>/` and `results/wm_pilot/probes/<model>/`, so that two scoring jobs
never write one file); `scripts/wm_pilot/*.sh` run the steps; `spatialcells/wm_pilot.py` builds the
probe arena and computes WP2, WP5 and the summary (`results/wm_pilot/summary.json`). The probes rendered
on the PC are byte-identical to the Mac's (MD5). CUT3R's final checkpoint was over its Google Drive
quota, so it came from the authors' Hugging Face release (`zhangify/CUT3R_release`, which a CUT3R
author links in the official issue tracker); its SHA-256 matches the release's LFS hash. The stored
model string matches our configuration apart from training-only fields, and every weight loads (no
missing or unexpected keys). On the Mac, CUT3R's PyTorch fallback for rotary embeddings cannot take
the pose token's position −1, so the adapter replaces it with the closed form that the CUDA kernel
computes (identical for positions ≥ 0). Check (`scripts/wm_pilot/cut3r_check.py`): our preprocessing
equals CUT3R's `load_images` exactly, and our step loop reproduces CUT3R's own forward pass (the demo's
path) bit for bit over 12 frames; over 24 frames batched encoding drifts by up to 3.5 cm.

*Results (2026-10-26).*
- **WP0.** ViNT, NoMaD and GNM pass in room A and fail in the apartment (Spearman ρ, 95% CI over walks):
  ViNT 0.570 [0.549, 0.590] and 0.407 [0.381, 0.434]; NoMaD 0.512 [0.494, 0.531] and 0.382 [0.352, 0.413];
  GNM 0.536 [0.516, 0.556] and 0.377 [0.353, 0.402]. In room A the mean predicted distance rises over all
  20 steps (ViNT 1.5 at k = 0, 4.1 at k = 1, 11.8 at k = 10, 13.7 at k = 20); in the apartment it is flat
  after about 6 steps. **CUT3R fails in both scenes**: median error after alignment 0.99 m in room A
  (first 64 frames 1.10 m, last 64 1.15 m) and 0.85 m in the apartment (0.71 and 1.05 m); CUT3R's own
  forward pass gives 1.04 m on walk 0. As registered, CUT3R is reported as not competent on our renders,
  and its WP1, WP2 and WP4 are not run. Exploratory: on walk 0, 54 of 255 steps turn more than 30°
  (median 59°); CUT3R's per-step displacement error is 0.53 m after those steps and 0.19 m after the
  others (true step 0.22 m). **TTT3R fails in room A and passes in the apartment**: 0.59 m (first and
  last 64 frames 0.48 and 0.65 m) and 0.40 m (0.39 and 0.43 m). It is therefore reported as competent
  only on photoreal scenes, and its WP1 and WP2, which need the 10 m arena, are left to the follow-up.
  TTT3R ran on the RTX 4070 (CUDA, 2 walks per batch) and CUT3R on the Mac; CUT3R's gate rerun on the
  4070 gives 1.03 and 0.80 m (`results/wm_pilot/pc/`), so the gap is the update rule's, not the
  platform's.
- **WP1** (room A; share of place units that are strictly heading-invariant, at each condition's best
  layer; references: path-integrating RNNs 98-100%, VGGT single 45%, VGGT with the walk 63%):
  ViNT walk 1.3%, single 2.0%, shuffled 2.8%, its 8-step moving average 97.0%, teleport probes 3.0%;
  NoMaD walk 5.2% (77 place units), single 0.3%, shuffled 0.1%, moving average 56.9%, probes 0.3%;
  GNM walk 40.2% (layer L0, 117 place units), shuffled 29.2%, single 0.4%, moving average 69.7% at the
  same layer (86.7% at its best layer, 15 units), probes 0.4%; pixels on the probes 0.9%; VGGT single on
  the probes 36.7% (block 19; 33.0% at block 21), against its 45% along the walk, with probe positions
  from its camera head at 0.41 m median (novel 0.39 m, familiar 0.41 m). All three policies are
  view-like on the probes, and the probe measure gives a model that localizes an intermediate share. GNM's walk share comes from its input layer, which stacks the 6 context frames
  as channels, and stays below its own moving average there, so it counts as smoothing. ViNT barely uses
  its context: the walk and single contexts change its units after the transformer by 0.1-0.2% and its
  predicted distance by 0.004 steps on average.
- **WP2.** No model is "beyond view recognition". Novel-probe error at the probe-selected layer: ViNT
  2.49 m, NoMaD 2.08 m, GNM 2.36 m, against the recall baseline's 2.28 m and the pixels' 1.14 m. The
  novelty cost is negative for every single-frame representation (pixels −0.16 m, the models −0.3 to
  −0.6 m), so the cost criterion is met by ViNT and GNM only because of a flaw in the measure:
  "familiar" is defined against the probe's own walk, which a decoder trained on other walks never saw,
  so the split tracks where the probe is rather than what the decoder knows. The follow-up should define
  familiarity against the decoder's training frames, or train on the probe's own walk. Strict
  cross-sector probe decoding is worse than chance for all three (4.5-5.3 m; chance 3.68 m).
- **WP5** (walk; P5000-P5009 novel trials, mean over rooms [95% CI], rooms above/below 50%): ViNT 51.5%
  [49.1, 54.2], 4/4; NoMaD 47.3% [44.7, 49.7], 4/6; GNM 53.5% [51.6, 55.8], 8/1 (sign test p = 0.039,
  0.12 after Holm over the three). Walk minus no walk: 0.0, −1.6 and −0.4 points, all with CIs that
  include zero, so none uses the walk. All trials in the apartments 53.3%, 53.7%, 50.1% and in the houses
  51.7%, 53.0%, 51.2% (NoMaD's houses pooled [50.4, 55.6], houses as the unit [49.3, 56.5]). Room A novel
  60.8% [50.6, 70.3], 55.9%, 57.8%. Reverse direction (secondary), P rooms novel: 54.3% [51.8, 56.4],
  50.7%, 55.6% [51.7, 58.8]. The base 8B is near 50%, the LoRA model 92.5% and Opus 69.2% on the same
  trials.

*Amendment 1 (2026-10-26 11:26 PDT, approved by the user; written before any sub-stepped frame was rendered
or shown to a model): CUT3R and TTT3R gated again on walks whose turns are split into sub-steps.* CUT3R
failed its gate in both scenes and TTT3R in room A. The exploratory per-step errors above point to the
walks' large turns between consecutive frames, which are rare in these models' training videos (plan
Section 3.2). This amendment gates both models again on the same walks, rendered with extra frames inside
every large turn. It adds a test and changes nothing above: the first gates stand as registered and are
reported beside the new ones.
- *Re-render rule.* Between original steps t and t + 1 of a walk, let Δθ be the heading change wrapped to
  (−180°, 180°]. If |Δθ| > 15°, n − 1 sub-frames are inserted, with n = ceil(|Δθ| / 15°): sub-frame j
  (j = 1, ..., n − 1) sits at step t's position (x_t, y_t) with heading θ_t + j Δθ / n. The camera thus
  turns on the spot and then steps, as the walk generator moves (`room.foraging_walk` turns first and
  then steps along the new heading), and the last transition turns Δθ / n ≤ 15° and takes the original
  step. Every sub-frame is rendered at its exact pose, with the scene's renderer and settings unchanged
  (256 px, 90° field of view, camera 1 m above the floor; in the apartment the floor height is snapped at
  (x_t, y_t), as for step t). The original frames keep their poses, read from each scene's `meta.npz`,
  and their re-renders must be byte-identical to the existing frames; if they are not, the difference is
  reported here and no model sees the new frames until it is resolved. On the first 16 walks this gives
  510-574 frames per walk in room A (8,557 in all; 44% of transitions turn more than 15°) and 612-941 in
  the apartment (11,700; 53%).
- *Scenes and files.* Room A (ray cast with `room.py` on the Mac, and again on the RTX 4070 PC, where the
  gates run; the two renders must match by MD5) and Replica apartment 0 (`habitat_scene.py` with
  experiment 12's settings, on the RTX 4070), the first 16 walks of each, in the
  new folders `data/walks-sub15` and `data/walks-roomReplicaApt0-sub15` (`data/walks` and
  `data/walks-roomReplicaApt0` are not touched). Their `meta.npz` adds, per frame, `orig` (true for an
  original step) and `orig_step` (the original step it belongs to, or that it follows for a sub-frame).
- *Gate.* As registered in WP0, with one change of input: each model streams every frame of a sub-stepped
  walk, original and inserted, from a fresh state without anchors. Each walk's similarity alignment
  (Umeyama) is fit on its 256 original frames only, and only those frames are scored: the median
  camera-position error, and the error over the first and the last 64 original frames. The threshold
  stays at a median of 0.5 m, with the same consequences as registered (fails in room A but passes in the
  apartment: competent only on photoreal scenes; fails in both: not competent on our renders). CUT3R and
  TTT3R both run on the RTX 4070 (CUDA), where CUT3R's first gate gave 1.03 and 0.80 m and TTT3R's 0.59
  and 0.40 m. Exploratory: the per-step displacement error between consecutive original steps, after
  steps that turn more than 30° and after the others, as in the first gate.
- *If a model now passes in room A,* its registered WP1, WP2 and WP4 analyses follow, and the conditions
  that stream a walk (WP1 *walk*; WP2 *probes-walk* and the model-native probe positions, read against
  the state at the end of the walk; WP4's secondary walk mode) take the sub-stepped walks as input: all
  48 room-A walks, rendered by the same rule, with units, poses and swaps recorded at the original steps
  only. The conditions without a walk sequence (*single*, *alone*, *probes-single*, WP4's single mode) and
  the *shuffled* control (the original frames in shuffled order) stay as registered. These runs start
  only after the gate results are reported here.

*Amendment 1 results (2026-10-26 14:00 PDT).* The renders followed the rule: no transition turns more
than 15° (largest 15.0°). Room A has 8,557 frames, ray cast on the Mac and on the PC with the same MD5.
Apartment 0 has 11,700 frames. In both scenes all 4,096 re-rendered original frames are byte-identical
to the existing ones (`results/wm_pilot/substep_<scene>-sub15.json`; the apartment's frames stay on the
PC). The gates ran on the
RTX 4070 (gq pc:567 and pc:568; `results/wm_pilot/gate_<rule>_<scene>-sub15.json`). **Neither model
passes in room A, so no WP1, WP2 or WP4 run follows.** Median camera-position error over the original
steps after alignment, with the first gates on the same GPU in brackets:
- CUT3R: room A 1.44 m (1.03 m), first and last 64 original frames 1.68 and 1.61 m, per walk 0.76-2.11 m;
  apartment 0.75 m (0.80 m). It fails in both scenes again and stays "not competent on our renders".
- TTT3R: room A 0.59 m (0.59 m), first and last 64 frames 0.44 and 0.65 m, per walk 0.45-0.97 m;
  apartment 0.40 m (0.40 m), passing as before. It stays "competent only on photoreal scenes".
- Exploratory, per-step displacement error between consecutive original steps (median), after steps that
  turn more than 30° and after the others (22% of the room-A steps and 32% of the apartment's turn that
  far): CUT3R 0.33 vs 0.17 m in room A and 0.25 vs 0.14 m in the apartment; TTT3R 0.47 vs 0.13 m and
  0.35 vs 0.10 m. The first gate's figure (0.53 vs 0.19 m) came from walk 0 only, so it is not strictly
  comparable. Still, a large turn costs both models two to four times the error of an ordinary step,
  even when it reaches them in steps of 15° or less.

Splitting the turns does not make either model competent in room A. The large turns still carry most of
the per-step error when they come in small pieces, so the size of a single frame-to-frame rotation was
not the cause. The inserted frames are pure rotations on the spot, which carry no parallax, and that
may matter more. CUT3R does worse on the longer sequences (about 535 frames per walk instead of 256),
already over the first 64 original frames; this fits the forgetting reported for CUT3R's state, but we
did not test it. TTT3R is unchanged in both scenes. Its unit and probe tests stay with the photoreal
scenes (Amendment 2).

*Amendment 2 (2026-10-26 11:49 PDT; written before TTT3R or VGGT saw any frame of the new probes or anchors, before any
TTT3R unit was recorded in an apartment, and before any unit was scored there; only the apartments' positions,
headings and probe counts had been looked at): TTT3R's WP1 and WP2 in the Replica apartments, and a corrected WP2
novelty cost.* TTT3R passed WP0 only in apartment 0 (0.40 m; room A 0.59 m), so as registered it counts as competent
only on photoreal scenes, and its WP1 and WP2 were left until `analyze.py` could score units outside room A's 10 m
box. This amendment adds that option, replaces WP2's decoder (the results above show that its novelty cost is
confounded), and runs both analyses in the apartments. It changes no earlier result: the registered WP2 numbers
stay above, and the corrected ones are reported beside them. CUT3R stays out (it failed in both scenes), WP4 is not
part of this amendment, and amendment 1 (sub-stepped walks) is unaffected; if TTT3R passes room A there, amendment
1's room-A runs follow as it says.
- *Scenes.* Replica apartment 0 is the primary scene, since TTT3R passed its gate there. Apartments 1 and 2 first
  take the same WP0 gate (the first 16 walks, each from a fresh state without anchors, per-walk similarity alignment
  over all 256 frames, median ≤ 0.5 m), and WP1 and WP2 run in each one that passes. Results are reported per
  apartment. The label that counts is apartment 0's, and it is called consistent across apartments if every
  apartment that passes gets the same label. Apartments 1 and 2 have rendering holes (29% and 45% of frames have
  more than 5% of their pixels with no geometry; apartment 0 has none), which is a reason to expect worse gates
  there.
- *Arena* (`analyze.py --arena`; without the flag nothing changes). The grid is a square of 0.5 m bins laid over the
  bounding box of the positions in the scored folder's `meta.npz`, with its corner on the 0.5 m lattice through
  (0, 0), so a scene's walk and probe arenas share bin edges. Bins with fewer than 3 frames are masked, as in room A,
  so walls, unvisited rooms and the square's padding drop out of every score; the field-size criterion (≤ 40% of
  the arena) counts only those visited bins. Everything else in `analyze.py --quadrant` is unchanged: the place
  criteria, 100 circular-shift shuffles, the opposite 90° sectors, the 1-bin (0.5 m) smoothing, and the share of
  strictly invariant place units. The grids: apartment 0 26 × 26 bins with 218 visited (≥ 3 walk frames),
  apartment 1 21 × 21 with 214, apartment 2 20 × 20 with 170 (room A 20 × 20 with 384). The smoothing can blend
  two bins on either side of a thin wall; this is a known limitation and is not corrected. Check on room A, where
  the bounding box gives the default grid exactly (corner (0, 0), 20 bins): the HEAD version of `analyze.py`, the new one and the new one with `--arena` give bit-identical output for pixels and pixels + EMA-8 along the walk, GNM's walk layer L0 and pixels on the 256-per-walk probes, and match `results/summary_quadrant.json` and the pilot's own files in every unit score (the ridge decoding errors in `results/summary_quadrant.json`, computed on vast-1, differ by at most 0.6 mm for the HEAD version too).
- *Probes.* The apartments were rendered with 128 probes per walk (about 20 per visited bin, so many opposite-sector
  maps fall below 3 probes per bin). To match room A's probe test (256 per walk, about 30 per visited bin), 256 new
  probes per walk are rendered on the PC through gq (`habitat_scene.py --probes-for`, drawn as the existing probes
  are: the position of a random frame of the walk plus N(0, 0.15 m) noise in x and y, redrawn until navigable and
  0.15 m from obstacles, heading uniform; probe seed 17, the pathfinder seeded as the walks were), into
  `data/walks-roomReplicaApt<k>-p256` (8,192 per apartment, 38-48 per visited bin), with a probe arena as for room
  A (`wm_pilot arena`). The existing 128-per-walk probes are not used. The renderer is checked first: the first 16
  frames of walk 0 rendered again must equal the stored frames byte for byte, or nothing runs.
- *Anchors.* Each apartment gets 8 anchor views in room A's pattern (the centre facing E, N, W and S, then four
  spots 3 m from it in x and in y, SW, SE, NE and NW, each facing the centre), placed on walk positions so that each
  is navigable: the centre is the walk position nearest the median of all walk positions, and each other spot is
  the walk position nearest centre + (±3 m, ±3 m). They are rendered with the walks' settings into
  `data/walks-roomReplicaApt<k>/anchors.npz`. As registered, the first anchor sets the output frame, and a
  similarity fit on the anchors' predicted cameras maps the pose head to apartment coordinates (used only by the
  pose-head readout).
- *TTT3R conditions and units* (as registered for TTT3R in WP1): *walk* (anchors, then the walk with the state
  carried), *single* (anchors, then the frame alone) and *single* + EMA-8, on all 32 walks; on the probes,
  *probes-walk* (each probe read against the state at the end of its own anchored walk, with the state's write
  disabled) and *probes-single* (read against the state after the anchors). Units L0, L3, L6, L9, L12 and L13 as
  registered. Each condition is reported at its best layer, with the registered labels (place-like ≥ 45%,
  view-like ≤ 10%), and a high share along the walk counts as integration only if *walk* beats *single* + EMA-8.
  The labels' thresholds come from room A, so scene-matched references are scored the same way: pixels and pixels
  + EMA-8 on the walk frames and pixels on the probes in every apartment, and in apartment 0 VGGT single at blocks
  19 and 21 (the registered probe reference) on the walk frames and the probes, with the same anchors.
- *WP2, corrected decoder.* In the registered WP2, the decoder was trained on the even walks and tested on the odd
  walks' probes, while familiarity was defined against the probe's own walk, which that decoder never saw. The
  split therefore tracked where and which way the probe looked rather than what the decoder knew, and the cost came
  out negative even for pixels (−0.16 m). Defining familiarity against the decoder's training frames does not work
  here: the even walks cover room A so densely that only 6 of 6,144 odd-walk probes are novel. Instead, every
  probe is decoded by a ridge trained on its own walk's frames (`wm_pilot novelty --own-walk`): the decoder has
  then seen exactly the walk that familiarity refers to, which is also the walk that a model's state carries in
  *probes-walk*. Familiarity is unchanged (a frame of the probe's own walk within 1 m, facing within 45°). λ is
  chosen once per representation from the registered grid, with each walk's first half of steps predicting its
  second half and vice versa (median error over all walks). The novelty cost is the median error on novel probes
  minus that on familiar ones, pooled over all walks' probes. 95% CIs come from a paired bootstrap over walks
  (with repetition, 10,000 resamples). The recall baseline uses the same protocol. The decision rule is unchanged:
  "beyond view recognition" needs a novel-probe error below the recall baseline's and a novelty cost below the
  pixels', each with a CI that excludes zero, at the layer that WP1 selects on the probes. The model-native readout
  (the pose head's probe positions after the anchored walk) and strict cross-sector probe decoding are reported as
  registered. Check on room A's existing data (`results/wm_pilot/novelty_ownwalk.json`): pixels novel 2.51 m,
  familiar 1.21 m, cost +1.30 m [1.11, 1.52]; recall novel 2.12 m, cost +1.40 m; VGGT single (block 19) novel
  0.37 m, familiar 0.19 m, cost +0.18 m [0.15, 0.21] (block 21 +0.19 m); VGGT's pose head 0.39 m novel, 0.41 m
  familiar. A view code thus pays a large cost, and a model that localizes from the anchors pays a small one. The
  navigation policies, rescored with the corrected decoder: at the layer each selects on the probes, ViNT (L4) novel 2.81 m, familiar 2.08 m, cost +0.73 m [0.56, 0.92]; NoMaD (L4) 2.82 m, 2.24 m, +0.58 m [0.37, 0.77]; GNM (L0) 2.67 m, 2.09 m, +0.58 m [0.38, 0.82]. Their costs are below the pixels' only because they decode badly everywhere: their novel-probe errors exceed the recall baseline's by 0.55-0.71 m (CIs exclude zero), so none is beyond view recognition, as before. VGGT single (block 19) passes both parts (novel minus recall −1.75 m [−1.92, −1.60], cost minus pixels −1.12 m [−1.32, −0.95]).
- *Compute.* On the RTX 4070 PC through gq (`scripts/wm_pilot/pc_apartments.sh`): renders, TTT3R's gates and
  activations at priority 150, then the VGGT reference at 140, and the scoring as CPU jobs there. TTT3R runs with
  CUDA and 2 walks per batch, as in its gate. The order is fixed: renders and gates, then activations only in
  apartments that pass, then scoring. If the PC stays unreachable, TTT3R's activations may move to the Mac (MPS,
  gq, priority 150), but the renders must wait for the PC. Holm across the five models applies as registered.
- *Scene-matched pixel references (2026-10-26 11:58 PDT; scored before any TTT3R unit was recorded in an
  apartment; `results/wm_pilot/apartments/ReplicaApt<k>/units/pixels/`).* The share of place units that are
  strictly heading-invariant is much higher for pixels along the walk in the apartments than in room A (1%):
  18% in apartment 0, 38% in apartment 1 and 72% in apartment 2 (place units 63%, 89% and 42% of poles).
  Pixels + EMA-8 reach 100% in all three (room A 95%). In these apartments, then, a pure view code clears the
  room-A threshold for "intermediate" in apartments 0 and 1 and for "place-like" in apartment 2. A likely cause
  is that the apartments' rooms differ in appearance, so any view from inside a room carries the room's identity
  whichever way it faces; room A is one room. The registered labels are therefore not interpretable in the
  apartments on their own, and TTT3R's shares are read only against the same apartment's references (pixels on
  the same frames and probes, and VGGT single in apartment 0). The rule for that comparison is set in a further
  dated note before any TTT3R unit is recorded.
- *Comparison rule in the apartments (2026-10-26 12:00 PDT; no TTT3R unit has been recorded in an apartment, and
  no probe or anchor has been rendered).* This replaces the registered labels in the apartments.
  - *Primary test (apartment 0, teleport probes).* Take the layer at which TTT3R's *probes-walk* share is highest,
    and read *probes-single* at that same layer. A condition's own best layer is not used, since choosing the best
    layer separately for each condition would bias the difference in favor of a gap. The finding "TTT3R's state
    adds a place code" requires the *probes-walk* share at that layer to exceed the *probes-single* share there by
    at least 10 percentage points, and the pixels' share on the same 256-per-walk probes by at least 10 points.
    Otherwise the result reads "no evidence that the state carries a place code beyond the scene's view floor."
    Every layer of both conditions is reported, with each condition's share of place units and their number, so
    that a margin resting on a handful of units is visible.
  - *Positive reference.* VGGT single (the same 8 anchors as TTT3R, then the probe), blocks 19 and 21, on the same
    apartment-0 probes. It shows whether a model that localizes clears the pixel floor there by the same 10-point
    margin. If VGGT cannot be run, or does not clear the floor, a TTT3R null in the apartment is uninterpretable,
    and it is reported that way.
  - *Secondary.* The along-walk shares (*walk*, *single*, *single* + EMA-8, against pixels and pixels + EMA-8) are
    reported, but EMA-8 already reaches 100% there, so they carry no decision. Apartments 1 and 2, if they pass
    their gates, are descriptive only. WP2 (own-walk decoder) is unchanged, since its baselines already come from
    the same scene. No new invariance measure (for example, one computed within rooms) is built for this
    amendment.

*Amendment 2 results (2026-10-26 15:10 PDT; VGGT's apartment reference added 20:05 PDT).*
`results/wm_pilot/apartments/ReplicaApt0/`, `results/wm_pilot/summary.json` ("apartments (amendment 2)"); jobs on the
RTX 4070 through gq (pc:554-582).
- *Setup.* Apartment 0's renders passed the check: the first 16 frames of walk 0 rendered again were byte-identical.
  It has 8,192 new probes (none with holes) and the 8 anchors. TTT3R's gate failed in apartments 1 and 2, with a
  median error of 0.57 m in both (first and last 64 frames 0.63 and 0.54 m; 0.53 and 0.65 m). So, as registered,
  WP1 and WP2 ran only in apartment 0, and apartments 1 and 2 have only their pixel references. The first TTT3R
  activation run crashed on a bug in `wm_cut3r.py`: the recording hook returned `False` when recording was off,
  and PyTorch passes a pre-hook's non-None return value on as the layer's input. It was fixed, checked on the PC
  and rerun; no unit had been recorded before the fix.
- *The anchored stream is not competent.* TTT3R places the 8 anchors badly (the similarity fit's RMS is 2.1 m;
  four of the views turn 90° on the spot), so the pose-head readout in apartment coordinates is unusable: median
  4.2 m for walk frames and 4.3 m for probes (chance about 3.4 m). Exploratory: WP0's gate measure on the anchored
  stream (the 8 anchors, then each of the first 16 walks, aligned per walk in 3D as in the gate) gives 0.77 m
  (first and last 64 frames 0.93 and 0.79 m), against 0.40 m for the same walks from a fresh state. The
  registered anchored conditions therefore put TTT3R below its own competence threshold.
- *WP1, primary test (teleport probes, the comparison rule).* The share of place units that are strictly
  heading-invariant, per layer (place units as a share of active poles in parentheses), is:

  | layer | *probes-walk* | *probes-single* |
  |---|---|---|
  | L0 | 0.6% (45.3%) | 0.6% (45.5%) |
  | L3 | **39.2%** (12.7%, 492 units) | 0.3% (27.4%) |
  | L6 | 16.3% (14.9%) | 0.8% (33.6%) |
  | L9 | 23.8% (16.4%) | 1.4% (39.6%) |
  | L12 | 27.7% (18.6%) | 1.7% (36.6%) |
  | L13 | (not recorded; the write is disabled) | 0.4% (29.2%) |

  Pixels on the same probes score 19.3%. At L3, the layer where *probes-walk* is highest, the margins are 38.9 points
  over *probes-single* and 19.9 points over pixels, so **the registered rule is met**. Three observations limit
  what that means. First, all 256 probes of a walk are read against one state, the one at the end of that walk.
  Second, the walk's probes lie along its path, so a component that marks which walk a probe belongs to also marks
  roughly where the probe is, whichever way it faces. Third, the state makes the probe code worse across walks:
  at L3, ridge decoding of probe positions trained on even walks and tested on odd ones gives 2.49 m with the
  state and 0.74 m without (chance 3.40 m), and the place units' split-half stability falls from 0.92 to 0.67.
  - Exploratory, not registered: removing each walk's mean probe activation, per unit, before scoring leaves
    *probes-walk* at 0.1-6.4% (L3-L12), *probes-single* at 0.0-0.7% and pixels at 0.0%. The same transform on room
    A's probes, however, cuts VGGT single from 36.7% to 9.7% (block 19) and 13.6% (block 21) and raises ViNT (L4)
    from 3.0% to 20.3%, because a different offset per walk breaks the cross-walk consistency that the rate maps
    need. The control thus fails its own validation, and it cannot separate a walk signature from a code for the
    probe's position. It is reported and used for nothing.
- *WP1 along the walk (secondary).* *Walk* 94.6% at L13 (the state itself, 315 place units) and 13.5-30.5% at
  L3-L12; *single* 0.4-1.2%; *single* + EMA-8 95.6-99.1% at every layer; pixels 18.1%, pixels + EMA-8 100%. The
  walk does not beat its own EMA-8 control at any layer, so along the walk the registered reading is smoothing.
- *WP2 (own-walk decoder).* The apartment's walks cover headings so densely that only 108 of the 8,192 probes are
  novel, so every CI is wide. Pixels: novel 2.20 m, familiar 0.77 m, cost +1.43 m [1.02, 2.21]. Recall: novel
  2.31 m, cost +1.90 m. At L3, the probe-selected layer, *probes-walk* has novel 2.34 m, familiar 1.22 m, cost
  +1.12 m; novel minus recall is +0.03 m [−1.07, 1.18] and cost minus pixels −0.31 m [−1.77, 0.56], so it is not
  beyond view recognition. No layer of either condition passes both parts; the closest is *probes-walk* L12
  (novel minus recall −0.98 m [−1.61, −0.27], but cost minus pixels −0.56 m [−1.40, 0.05]). The pose head's
  readout (median 4.3 m) is uninterpretable, for the anchor reason above. Strict cross-sector probe decoding
  (L3-L12) is 3.7-3.9 m with the state and 4.0-4.2 m without (chance 3.42 m).
- *Positive reference: VGGT single in apartment 0* (the same 8 anchors, then the frame; pc:569, 5 h). Its pose
  head localizes walk frames and probes to 1.06 m median (room A 0.43 and 0.41 m). On the probes, VGGT's share is
  25.7% at block 19 (1,280 place units, split-half stability 0.97) and 15.1% at block 21. That is only 6.4 points
  above the pixels' 19.3%, so **VGGT does not clear the floor by the registered 10-point margin**. Along the walk it
  scores 25.5% and 13.0% (pixels 18.1%). Its units still decode probe positions across walks to 0.55 and 0.50 m
  (TTT3R with the state: 2.15-2.52 m). In WP2, VGGT is beyond view recognition at both blocks: at block 19, novel
  1.07 m, familiar 0.31 m, cost +0.75 m [0.63, 1.00], novel minus recall −1.25 m [−1.66, −0.58] and cost minus
  pixels −0.68 m [−1.49, −0.12]; at block 21, novel 0.75 m, cost +0.47 m, −1.57 m [−2.02, −1.00] and −0.96 m
  [−1.79, −0.49]. The exploratory walk-centring leaves VGGT at 0.1-0.2%, which confirms that the centring control
  is unusable.
- *Reading.* By the registered rule, TTT3R's state adds heading-invariant place-like units on the apartment
  probes. In this scene, however, the WP1 probe test has not shown that it can detect a localizer: VGGT, which
  localizes and decodes well across walks, gains only 6.4 points over pixels. TTT3R's state-carried units score
  higher than VGGT's (39.2% against 25.7%), yet decode position four to five times worse across walks, and they come
  from an anchored stream that fails the competence threshold (0.77 m). That combination fits a state that carries
  a signature of the walk and its region more than a code for the probe's position. WP2 separates the two models
  cleanly: VGGT is beyond view recognition at both blocks, and TTT3R is not at any layer. The pilot therefore makes
  no claim that TTT3R has a place code. A swap control (each walk's probes read against another walk's final
  state) or anchors that TTT3R can track (for example, sub-stepped turns as in amendment 1) would test the walk
  signature directly; neither is registered.

## Experiment 17 (pre-registered 2026-10-26, before any model saw benchmark data): standard spatial benchmarks

**Why.** Every test so far uses our own two-choice task. Two questions about the fine-tune are still
open. Does the walk skill carry over to spatial questions it was never trained on? And does the training
cost the model spatial skills it already had? Here the base model and every LoRA run take two published
spatial benchmarks, each with its authors' prompts, frame sampling and metrics, so the scores can be set
beside published ones. Nothing is tuned on either benchmark. When this section was written the data had
been downloaded and counted, and only a timing run on synthetic frames had touched the model.

**Benchmarks and versions.** The protocol details below were checked against the official pages and code
on 2026-10-26 (notes with sources: the session notes `std-bench/vsibench_spec.md` and `mindcube_spec.md`).
- *VSI-Bench* (Yang et al., CVPR 2025; Hugging Face `nyu-visionx/VSI-Bench` at revision 3a6a1cf9).
  It has 5,130 questions on 288 videos of real indoor scenes (ARKitScenes 150, ScanNet 88, ScanNet++ 50;
  all 640x480). There are eight tasks. Four ask for a number: object count, absolute distance, object
  size and room size. Four are multiple choice: relative distance, relative direction (easy, medium and
  hard), route planning and appearance order. The authors' VSI-Bench-Debiased subset (the 2,362 questions
  not listed in `pruned_ids.txt`) is scored as well, as the dataset card recommends.
- *MindCube* (Yin et al., ICLR 2026; `MLL-Lab/MindCube` at revision 9c941b46; code `mll-lab-nu/MindCube`
  at b8b7062). We use MindCube-Tiny (`MindCube_tinybench.jsonl`), the authors' official test set: 1,050
  multiple-choice questions on 2-4 views of a scene, in three settings (among 600, around 250,
  rotation 200) and 124 image groups.
- No subset is drawn: both sets are run in full for every model.

**Models.** Base Qwen3-VL-8B-Instruct. The two-epoch runs 0, 1 and 2 (`data/ft/qwen8b-proc`,
`-s1`, `-s2`; experiments 9 and 9b). The four-epoch runs with seeds 3-7 (`data/ft/qwen8b-proc-e4-s<seed>`,
experiment 9c), every run that finishes training, whether or not it meets 9c's decision rule. Each adapter
is merged into the base weights, as in every other test. Run 0's second copy (the GPU control) is not
included.

**Protocol, VSI-Bench** (lmms-eval's `vsibench` task as the authors wrote it, with lmms-eval's default
settings for Qwen3-VL):
- the system message "You are a helpful assistant.", then one user turn with the video and then the text;
- the text is "These are frames of a video.", the question, and for multiple choice "Options:" with the
  options one per line and "Answer with the option's letter from the given choices directly.", or for
  numbers "Please answer the question using a single word or phrase.";
- 32 frames at `round(linspace(0, N - 1, 32))` of the video's N decoded frames (qwen-vl-utils' rule),
  at their native 640x480 (300 visual tokens per pair of frames), with the frame times taken from the
  video's frame rate (Qwen3-VL writes a timestamp before each pair of frames);
- greedy decoding with at most 16 new tokens;
- scoring as the authors' `utils.py`. The first space-separated token of the reply, with a final "."
  removed, must equal the letter. For numbers it is read as a float and scored by mean relative accuracy
  over the thresholds 0.50, 0.55, ..., 0.95. A reply that cannot be read scores 0. The overall score is
  the mean of the eight task scores, with relative direction scored as the mean of its three levels.
  The more lenient number parser that lmms-eval adopted in July 2026 is logged as a secondary score.

**Protocol, MindCube-Tiny** (the repo's raw-QA setting, the baseline used for off-the-shelf models):
- no system message; one user turn with the views first, in their listed order, and then the raw-QA text
  (the "[Task]" and "[Answer Instruction]" block, then "[Question]" and the question with its options);
- each view capped at 480 x 480 = 230,400 pixels, as the repo's processor does (Qwen3-VL's own minimum);
- greedy decoding with at most 4,096 new tokens; whitespace collapsed;
- the repo's `extract_answer`, unchanged. A question is correct when the extracted letter equals the
  answer; a reply with no letter is wrong. The overall score is accuracy over all 1,050 questions, and
  there is one score per setting.

**Platforms.** VSI-Bench runs on CUDA: vast-3's RTX 5090, bf16, after experiment 9c's jobs there
finish. If vast-1 is used as well, the videos are split between the machines, and every model answers a
given video's questions on the same machine, so paired comparisons never cross machines. MindCube-Tiny
runs on the Mac (MPS, bf16) for every model. Every job goes through gq at priority 150, below 9c's, and
each Mac job stops after about 20 minutes (the next one resumes), so 9c's Mac tests never wait for long.

**Hypotheses.** For each benchmark, the walk fine-tune either (H1) improves the score, (H2) lowers it,
or (H0) leaves it unchanged. We have no strong prior. The walk skill relates views taken from different
places, which some VSI-Bench and MindCube questions need. But the training covered one answer format
(a frame number) in one kind of scene, so a small loss is also plausible.

**Primary measures and contrasts.** The VSI-Bench overall score (full set) and MindCube-Tiny overall
accuracy, in points. On each benchmark, three contrasts, each a model minus the base:
- C1: run 0 (the model analyzed in the paper);
- C2: the mean of the two-epoch runs 0-2;
- C3: the mean of the four-epoch runs 3-7.
A recipe mean averages its runs' scores. The range over runs is reported beside it.

**Statistics.** The comparisons are paired by question: both models answer the same questions. The 95% CI
of a contrast comes from a bootstrap that resamples scenes with replacement (VSI-Bench: the 288 videos;
MindCube: the 124 image groups). Every question of a drawn scene is kept and both models' official scores
are recomputed (10,000 resamples, seed 0). The runs of a recipe are held fixed. A bootstrap over
questions, stratified by task, is reported beside it. Code: `scripts/std_bench_analysis.py` ->
`results/std_bench_summary.json`.

**Decision rule, set now.** Each contrast is labeled "helps" if its scene-bootstrap CI lies above 0,
"hurts" if it lies below 0, "no change" if it lies within plus or minus a margin, and "inconclusive"
otherwise. The margin is 2 points for VSI-Bench and 5 for MindCube-Tiny. Two published evaluations of
the base model on VSI-Bench differ by 1.5 points (57.9 and 59.4). MindCube-Tiny's 1,050 questions in 124
groups give a CI of about plus or minus 3 points. The verdict for each benchmark is:
- "transfers" if C2 or C3 helps and none of C1-C3 hurts;
- "hurts" if any of C1-C3 hurts and none helps;
- "no change" if all three are "no change";
- otherwise "mixed", with the numbers.
No correction is applied for the six primary labels, so a single label on its own is weak evidence.
Every model's scores are reported, whatever the verdict.

**Secondary analyses** (all reported, none with a decision rule):
1. Per-task scores and per-task contrasts with scene-bootstrap CIs.
2. Where a transfer would show. On VSI-Bench, the tasks that relate views taken at different places or
   times (relative direction, relative distance, route planning, appearance order) against the metric
   estimates (counting, object size, room size, absolute distance), as a difference of differences with
   a CI, for C1-C3. If the fine-tune transfers, we expect it to transfer more to the first group.
   MindCube is reported per setting, with no prediction.
3. Answer format: the share of replies each official parser can read, per model and task; the score on
   readable replies; and VSI-Bench with lmms-eval's lenient number parser. A loss that comes only from
   unreadable replies means a format effect rather than a loss of skill.
4. VSI-Bench-Debiased: every model and contrast, labeled separately (margin 3 points, as it has fewer
   questions).
5. A language-prior control: the base model and run 0 with no video and no images, on both benchmarks
   (the same text). If C2 or C3 is labeled "helps" or "hurts" on a benchmark, that recipe's other runs
   are run without the visual input there too.
6. MindCube's yes/no shortcut ("Yes" is correct on 243 of Tiny's 244 yes/no questions): accuracy on the
   other questions.
7. The four-epoch mean minus the two-epoch mean.
8. Exploratory, across the eight runs: the rank correlation between each run's two-choice accuracy
   (room A novel trials) and its benchmark change from the base model.

**Checks before any LoRA run is scored.** Every prompt is printed in a dry run first. Then the base model
answers 20 questions of each benchmark on the Mac as a smoke test (`data/smoke/`; these are not counted).
The base model's full VSI-Bench score is compared with the published 57.9 (32 frames) and 59.4 (Qwen's
report). If it is more than 8 points below 57.9, or fewer than 95% of its replies can be read, the
pipeline is checked for bugs before any LoRA run is scored. The settings stay as registered unless a bug
is found, and any change is recorded here as an amendment first. The published MindCube-Tiny score of the
base model (29.4, on 1,040 of the questions, through another toolkit) is only reported.

**Compute and cost.** No API. VSI-Bench takes about 1.5 hours per model on the 5090 (estimated from
synthetic frames), about 15 hours for all nine models and the two runs without video, starting when 9c's
vast-3 jobs end (about Tue 05:30 PDT). MindCube-Tiny takes about 30 minutes per model on the Mac: the base
model and runs 0-2 first, then each four-epoch run once 9c has copied its adapter to the Mac. Outputs:
`results/std_bench/<benchmark>/<model>[-blind].jsonl`, one line per question, and
`results/std_bench_summary.json`.

*Execution note (2026-10-27, before the analysis).* Because 9c's vast-3 chain failed for a few hours on
Monday (its jobs had been submitted with a malformed option), VSI-Bench started on vast-3 about a day
early. 9c was resubmitted and went first again; the one VSI-Bench job it interrupted resumed where it
stopped. Two VSI-Bench runs, run 2 and the four-epoch seed 3, ran back to back on vast-3 (05:00-07:42
PDT) and answered every question with "!!!!!!!!!!!!!!!!". That is token 0 at every step, which greedy
decoding gives when the logits are not finite. It was a fault on the machine, not a property of the two
models:
- the runs just before and after were normal;
- both adapters are identical on the two machines and contain only finite values;
- the same adapters answered MindCube normally on the Mac;
- 20-question reruns of both on vast-3 were normal.
Both runs were redone in full on vast-3. The failed files are kept in `results/std_bench/failed_nan/`.
The runner now stops with an error instead of saving such a reply. No other file has one.

*Result (2026-10-27; `scripts/std_bench_analysis.py` -> `results/std_bench_summary.json`).*
- Platforms were as registered: VSI-Bench on vast-3 (CUDA) and MindCube-Tiny on the Mac (MPS).
- Every model answered every question, and the official parsers could read 100% of every model's
  replies on both benchmarks. So none of the differences below comes from the answer format. lmms-eval's
  lenient number parser gives the same VSI-Bench scores to 0.1 point.
- The base model passed the registered check: 55.4 on VSI-Bench, against the published 57.9 (32 frames)
  and 59.4 (Qwen's report). On MindCube-Tiny it scores 30.1 (published 29.4), where chance (one over the
  number of options, averaged over questions) is 33.0.

Scores in points; "no input" means the same question without the video or images.

| | VSI-Bench | VSI-Bench, no input | VSI-Bench-Debiased | MindCube-Tiny | MindCube-Tiny, no input |
|---|---|---|---|---|---|
| base | 55.4 | 35.0 | 48.9 | 30.1 | 31.6 |
| run 0 | 53.6 | 36.5 | 45.9 | 30.0 | 29.1 |
| run 1 | 54.7 | 36.5 | 47.3 | 30.2 | |
| run 2 | 54.5 | 36.6 | 47.1 | 30.8 | |
| 4 epochs, seed 3 | 53.3 | 37.6 | 46.5 | 36.6 | 31.7 |
| seed 4 | 54.0 | 37.0 | 46.4 | 33.9 | 30.3 |
| seed 5 | 53.3 | 37.4 | 46.1 | 30.5 | 30.8 |
| seed 6 | 52.5 | 36.8 | 45.5 | 32.4 | 34.8 |
| seed 7 | 53.6 | 36.9 | 46.0 | 37.7 | 31.8 |

Primary contrasts (model minus base, points, with the 95% scene-bootstrap CI, the range over the runs and
the registered label):

| | VSI-Bench | MindCube-Tiny |
|---|---|---|
| C1 run 0 | -1.83 [-2.90, -0.78], hurts | -0.10 [-3.13, +2.93], no change |
| C2 two-epoch mean | -1.19 [-2.05, -0.34], hurts (runs -1.83 to -0.79) | +0.22 [-2.54, +2.87], no change (-0.10 to +0.67) |
| C3 four-epoch mean | -2.12 [-3.14, -1.13], hurts (runs -2.96 to -1.40) | +4.11 [+1.02, +7.25], helps (+0.38 to +7.62) |
| verdict | **hurts** | **transfers** (by the registered rule; see below) |

The question bootstrap gives nearly the same CIs. Run by run, seven of the eight runs are labeled "hurts"
on VSI-Bench. The exception is run 1: -0.79 [-1.75, +0.18], within the 2-point margin. On MindCube-Tiny,
the four-epoch seeds 3, 4 and 7 are labeled "helps" (+6.5, +3.8, +7.6), seed 5 "no change" (+0.4) and
seed 6 "inconclusive" (+2.3). Runs 0-2 are all "no change".

Secondary analyses (as registered):
1. *Per task.* The VSI-Bench loss sits in two metric tasks:
   - object counting: run 0 -5.7, two-epoch mean -4.0, four-epoch mean -5.4;
   - absolute distance: -6.3, -3.3 and -6.4 (every CI below 0);
   - appearance order: -4.1 [-7.5, -0.8] for the four-epoch mean; the two-epoch mean (-1.9) and run 0
     (-2.4) have CIs that include 0.
   Object size, room size, relative distance, relative direction and route planning change by 1.6 points
   or less, with CIs that include 0. On MindCube-Tiny the four-epoch gain is in the "around" setting:
   +14.9 [+8.0, +21.9]. The "among" setting moves +1.7 [-1.2, +4.8] and rotation -2.2 [-11.8, +5.8].
2. *View-relating against metric tasks (VSI-Bench).* The tasks that relate views taken at different places
   or times are unchanged: run 0 -0.6 [-2.6, +1.4], two-epoch mean -0.3, four-epoch mean -0.9, every CI
   including 0. The metric estimates lose: -3.0 [-3.9, -2.2], -2.1 [-2.8, -1.4] and -3.4 [-4.2, -2.5]. The
   difference of differences is +2.4 [+0.1, +4.6], +1.9 [+0.0, +3.6] and +2.5 [+0.4, +4.5]. The registered
   prediction (a transfer would be larger on view-relating tasks) has no transfer to test. What the
   contrast does show is that the loss falls on the metric estimates.
3. *Answer format.* Every reply was readable (above), so no part of the loss on either benchmark is a
   format effect.
4. *VSI-Bench-Debiased* (margin 3 points). C1 -2.97 [-4.66, -1.30], C2 -2.13 [-3.41, -0.83] and C3 -2.81
   [-4.37, -1.24] are all labeled "hurts", and so is every run (-1.6 to -3.4), so the verdict is "hurts".
5. *Without the visual input.* Run on every model, because both recipe means were labeled on both
   benchmarks.
   - VSI-Bench: without the video, the fine-tuned models score slightly above the base model. The
     two-epoch mean is +1.5 [+0.5, +2.5] and the four-epoch mean +2.1 [+1.0, +3.2]. So the loss with the
     video is not a worse language prior: the video helps them less. The base model gains 20.4 points from
     the video. That gain shrinks by 3.3 [1.9, 4.8] for run 0, by 2.7 [1.4, 4.0] for the two-epoch mean
     and by 4.2 [2.8, 5.7] for the four-epoch mean.
   - MindCube-Tiny: the base model gains nothing from the images (with minus without: -1.5 [-4.2, +1.2]).
     Without images, the four-epoch mean matches the base model (+0.3 [-2.2, +2.6]; "around" -2.5
     [-6.0, +1.0]). So its gain needs the images: it gains 3.9 [+1.0, +6.9] points more from them than
     the base model does.
6. *MindCube without its yes/no questions* (244 questions, where "Yes" is correct on 243). The base model
   scores 23.9; chance on the remaining questions is 27.9. Against it, the four-epoch mean is +5.1 [+1.6,
   +8.5], the two-epoch mean +2.4 [-0.8, +5.4] and run 0 +2.2 [-1.2, +5.5].
7. *The four-epoch mean minus the two-epoch mean*: VSI-Bench -0.93 [-1.60, -0.29], MindCube-Tiny +3.89
   [+2.00, +5.95]. The longer recipe costs slightly more on VSI-Bench and gains more on MindCube-Tiny.
8. *Exploratory, across the eight runs.* The rank correlation between room-A novel two-choice accuracy and
   the change from the base model is -0.67 on VSI-Bench and +0.52 on MindCube-Tiny. With eight runs, and
   all five four-epoch runs at 98-99% in room A, this mostly separates the two recipes.

*Exploratory, chosen after seeing the results (not registered).*
- MindCube-Tiny. Every model scores at or below chance. The four-epoch mean (34.2) moves from below
  chance to about chance.
  - On the 87 yes/no "around" questions, where "Yes" is always correct, the four-epoch runs answer "Yes"
    62-69 times against the base model's 33. That accounts for most of their "around" gain. On the other
    "around" questions the four-epoch runs score 28.2-41.7 (base 30.7, chance 28.1).
  - On the "among" questions without yes/no, the base model scores 17.8, well below chance (25.3), and
    every LoRA run scores 22-29.
  - So the gain is best read as the fine-tuned models losing some of the base model's below-chance
    answer biases, mostly on questions they answer from the images. No model shows MindCube skill
    clearly above chance.
- VSI-Bench. The fine-tuned models more often give a default answer:
  - "2" for counts: 417 of 565 counting answers for run 0 and 434 for seed 6, against 385 for the base
    model;
  - shorter, rounder distances: a median of 1.3 m (run 0) and 1.2 m (seed 6) against 1.7 m for the base
    model, with 1.8 m correct.
  This fits the smaller gain from the video: the fine-tune makes the model rely more on typical values
  when it has to estimate from the video.

**In short.** The walk fine-tune does not carry over to VSI-Bench. The VSI-Bench questions that relate
views taken at different places or times are unchanged, so the walk skill shows no transfer there. The
training costs 1.2-2.1 points of VSI-Bench's overall score (2.1-2.8 on the Debiased subset), all of it
on metric estimates, and the model draws less on the video for them. On MindCube-Tiny, the two-epoch
runs, including the paper's run 0, change nothing. The four-epoch recipe's +4.1 meets the registered
"transfers" rule, and it depends on the images. But every model, base and fine-tuned, is at or below
chance there, so the gain shows the model losing some of its biases, not learning to reason about space.

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
