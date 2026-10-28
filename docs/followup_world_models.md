# Follow-up paper: places in world models, navigation policies and JEPA video models (scoping plan)

**Update (2026-10-26).** Ranks 2 and 4 (ViNT/NoMaD/GNM and CUT3R/TTT3R) have been piloted; see Section 9 and
the pilot's registration in `docs/experiments.md`. The rest of this plan is unchanged.

**Status (2026-10-25).** This is a scoping plan for a follow-up to the preprint ("Recognising views, not
places"). Nothing in it has been run: no GPU job, no API call, no download, and no model listed here has
seen our walks. The preprint is not changed by it. The model facts below come from official pages
(GitHub READMEs, Hugging Face cards and file trees, arXiv) read on 2026-10-25 by four research passes, and
the entries that decide the shortlist (NWM, V-JEPA 2 ViT-g, InJePA, Cosmos3-Edge, ViNT/NoMaD, CUT3R,
Zing-0.5) were fetched a second time. The shared web-search budget ran out near the end of the survey, so
releases from September and October 2026 may be missing, and every entry should be checked again before it
is run. The references in Section 1 come from the same passes, and each is checked against arXiv before it
goes into a paper, as was done for the preprint. Entries marked † are inferences (mostly memory needs and
run times worked out from parameter counts), not statements read on a page.

## 1. The question, and what would be new

The preprint shows that open vision-language models recognise views rather than places. Units pass the
rodent place-cell criteria everywhere (72% of raw pixels), so the criterion that matters is strict heading
invariance: rate maps built from opposite 90° heading sectors must agree. Path-integrating RNNs pass it with
98-100% of their place units (Exp 13f), single-frame VLM units with 0-6% (Exp 13e), and the rise to 7-51%
with the walk in context is matched by an 8-step moving average of single-frame activations (90-94%).
Teleport probes decode at chance across headings, no open model gains from the walk, and the one model that
localises novel views, VGGT, has a 10-dimensional position subspace that its camera head uses (patching
transfer 0.85 at block 21, the same subspace working in a second room). A LoRA fine-tune learns the
two-choice behaviour without learning a room-general position code (Exp 14a). None of these models was ever
trained to predict what it would see after moving. The follow-up asks whether models that were trained on
egocentric video together with self-motion form a place code that passes the same tests. That covers
navigation world models, interactive video world models, navigation policies and robot VLAs, and JEPA video
models with and without an action-conditioned predictor.

What is known already points both ways. Small purpose-built networks trained to predict egocentric views
from earlier views, most of them also given self-motion, grow place-like units (Uria et al. 2020, bioRxiv 2020.11.11.378141;
Recanatesi et al. 2021, Nat Commun; Gornet & Thomson 2024, Nat Mach Intell; Levenstein et al. 2024, bioRxiv
2024.04.28.591528; Wang et al. 2026, arXiv 2605.21356), while untrained image networks pass the same
criteria (Luo, Mok & Love, eLife reviewed preprint RP99047). Heading has been controlled unit by unit only
in small DQN agents (Vijayabaskaran & Cheng 2022, PLoS Comput Biol; Kappel & Cheng 2025, Front Comput
Neurosci), with per-heading field centres rather than opposite sectors and without a path-integrating
positive control. In large pretrained models the nearest work reads camera pose and 3D structure out of
video diffusion models and V-JEPA (AC3D, arXiv 2411.18673; Huang et al., CVPR 2026, arXiv 2512.19949; Musa
et al., arXiv 2609.01551), linear and steerable state out of VLAs (Häon et al., CoRL 2025, arXiv
2509.00328; Buurmeijer et al., arXiv 2603.05487), and maps out of the memory of blind Habitat agents
(Wijmans et al., ICLR 2023). Two papers supply control designs we should adopt. Kim et al. (arXiv
2609.34684) show that probe accuracy on natural trajectories can hide a weak response when targets are
crossed with contexts. Tiwari et al. (arXiv 2609.12090) show that WorldMem keeps 94% of its memory benefit
when it is handed the wrong memory from the same trajectory. The NWM paper itself suggests that egocentric
next-frame prediction may produce allocentric representations and leaves that untested. The survey found no
paper that tests for place codes, with or without a heading control, inside a pretrained action-conditioned
world model, navigation policy, VLA or V-JEPA 2 model. The literature search also ran out of budget before
a full 2025-26 sweep, so a targeted arXiv search is needed before the paper claims novelty, as was done for
the preprint.

The survey also gives the plan its organising principle, which is to ask who integrates the path. Many
"spatially consistent" world models are handed the camera pose: as extrinsics or Plücker rays per frame, as
the key that retrieves memory frames by field-of-view overlap, or as the frame in which a 3D cache is built.
A position readout then succeeds by reading the conditioning back, so these models cannot show that a place
code emerges. Only models conditioned on per-step motion (relative pose changes, key presses and mouse
deltas) or on no motion at all leave the network to work out where it is. The second axis is memory. A
place code that survives a new heading needs a state or context that spans the walk, and most motion-only
models see between 4 and about 30 frames. The candidates therefore fall into a two-by-two design, crossing actions
(none, or per-step motion) with memory (a few frames, or the whole walk). Positive controls that are known to
localise sit alongside it.

## 2. Open-weight candidates (October 2026)

Our machines are the RTX 4070 (12 GB, the PC with Habitat), the RTX 5090 on vast-3 (32 GB; its container
is capped at 24 GB of host RAM, which matters when large checkpoints are loaded), the RTX PRO 6000 on vast-1
(96 GB, busy with another project, so only for models that need it), and the Mac (Apple M5 Pro, 48 GB of
unified memory, MPS, one GPU job at a time). Stacks that need flash-attn, apex, TransformerEngine or JAX on
GPU do not run on the Mac. In the fit columns, ✓ means inference in bf16 with activation hooks fits, ~ means
it needs quantisation, offloading, a shorter window or code edits, and ✗ means it does not fit or the stack
does not run.

### 2.1 Action-conditioned navigation world models

| Model | Links | Licence (code / weights) | Size | Inputs and actions | Memory | Fit 4070 / 5090 / 6000 / Mac |
|---|---|---|---|---|---|---|
| **NWM** (Meta; Bar et al., CVPR 2025) | [arXiv 2412.03572](https://arxiv.org/abs/2412.03572), [code](https://github.com/facebookresearch/nwm), [weights](https://huggingface.co/facebook/nwm) (gated) | CC BY-NC 4.0 / CC BY-NC 4.0 (the card text; the HF metadata says cc-by-4.0) | CDiT S 50M, B 200M, L 700M, XL 1B; XL-Ego4D 1B | 4 context frames at 4 fps, 224 × 224 through the SD VAE (196 tokens per frame); action = target (x, y, yaw) in the current ego frame, x and y divided by the dataset's `metric_waypoint_spacing` (RECON 0.25 m, SACSoN 0.255, SCAND 0.38, GoStanford 0.12, TartanDrive 0.72) and normalised with `action_stats`, yaw in radians, plus a time shift (frame offset / 128, up to ±16 s) | none beyond 4 frames | ✓ / ✓ / ✓ / ~ (scripts hard-code CUDA autocast) |
| RAE-NWM (Tsinghua, ECCV 2026) | [arXiv 2603.09241](https://arxiv.org/abs/2603.09241), [code](https://github.com/20robo/raenwm), [weights](https://huggingface.co/zmkun20/raenwm) | MIT / MIT | ~350M | NWM's action format, but predicts frozen DINOv2 patch tokens instead of VAE latents; also trained on 1,000 Habitat/Matterport3D trajectories | 4 frames | ✓ / ✓ / ✓ / ✓† |
| CompACT (CVPR 2026) | [arXiv 2603.05438](https://arxiv.org/abs/2603.05438), [weights](https://huggingface.co/kdwon/CompACT) | not stated | CDiT-B/L on a 16-token tokenizer | NWM-style | 4 frames† | ✓† |

NWM was trained on RECON, SCAND, TartanDrive and HuRoN/SACSoN (an indoor robot at Berkeley), and the
XL-Ego4D checkpoint adds about 900 h of Ego4D navigation clips (time shift only). The released checkpoints
were retrained after face blurring, so they may differ from the paper. Its CDiT blocks self-attend over the
target frame's tokens, cross-attend to the context tokens and take the action, time shift and diffusion step
through AdaLN. Activations are read at the target tokens with the true frame noised to a chosen level, the
probing protocol of AC3D and Huang et al. Hugging Face's pickle scan flags `cdit_s_100000.pth.tar`, so the
checkpoints are loaded with `weights_only=True` in an isolated environment. Three related models released no
weights (checked): NavWAM (Cosmos-Predict2-2B trained on HM3D), UniNav and DR-NWM. PEVA's code is listed as
"coming soon". MWM is a LoRA on NWM-XL whose release is unclear.

### 2.2 Interactive and camera-motion world models conditioned on motion only

| Model | Links | Licence | Size | Actions | Memory over a 256-frame walk | Fit 4070 / 5090 / 6000 / Mac |
|---|---|---|---|---|---|---|
| **Cosmos 3 Edge / Nano** (NVIDIA; Nano 31 May 2026, Edge 20 Jul 2026) | [arXiv 2606.02800](https://arxiv.org/abs/2606.02800), [Edge](https://huggingface.co/nvidia/Cosmos3-Edge), [Nano](https://huggingface.co/nvidia/Cosmos3-Nano), [framework](https://github.com/NVIDIA/cosmos-framework) | OpenMDW-1.1, ungated, commercial use allowed | Edge 4B, Nano 16B | "general camera motion" domain: 9-D relative pose change per transition (3 translation + 6-D rotation, OpenCV axes), T = 16-400 steps; normalisation constants not documented | the whole clip (Edge 50-150 frames at 256p/480p; Nano up to ~300-400), bidirectional attention | Edge ✗ / ~† / ✓ / ✗; Nano ✗ / ✗ / ✓ / ✗ (Linux, CUDA, BF16 only) |
| Infinite-World (Meituan; ICML 2026) | [arXiv 2602.02393](https://arxiv.org/abs/2602.02393), [weights](https://huggingface.co/MeiGen-AI/Infinite-World) | MIT / MIT | 1.3B (Wan2.1) | binary move and turn keys, labelled from VGGT pose changes with unpublished thresholds | **pose-free**: up to 320 frames compressed into 20 memory tokens, trained for revisits | ✗ / ✗ / ✓ (~45 GB) / ✗ (flash-attn pinned) |
| Zing-0.5 (seedleap, Aug 2026) | [arXiv 2609.17909](https://arxiv.org/abs/2609.17909), [weights](https://huggingface.co/seedleap/zing-0.5), [code](https://github.com/seedleap/zing-world-model) | Apache-2.0 | 5B (Wan2.2-TI2V) | continuous W/A/S/D/I/J/K/L intensities, 24 fps | causal KV window: 97 latents (~388 frames) needs ≥ 80 GB, 33 latents (~132 frames) on smaller cards | ✗ / ~ / ✓ / ✗ |
| Waypoint-1.5 (Overworld, Apr 2026) | [1B](https://huggingface.co/Overworld/Waypoint-1.5-1B), [360p](https://huggingface.co/Overworld/Waypoint-1.5-1B-360P), [runtime](https://github.com/Wayfarer-Labs/world_engine) | weights Apache-2.0; runtime GPL-3.0 (a Diffusers pipeline is the alternative) | 1.2-2B | per-frame keys + mouse (dx, dy), uncalibrated gain | 512-frame context | ~ / ✓ / ✓ / ~ (360p model) |

Cosmos 3 pairs an autoregressive "reasoner" tower with a diffusion "generator" tower that share attention
in every layer. Nano's towers start from Qwen3-VL-8B, the preprint's own model; Edge starts from an internal
model. Video goes through the Wan2.2 VAE, which compresses time 4× and space 16×. The camera-motion training
data are about 4.6k h of web video whose poses were estimated in metric scale (ViPE with Depth Anything 3);
another 41k h is egocentric manipulation. Several models with motion-only actions but short memory can serve
as short-memory controls: Matrix-Game 2.0 (MIT, 1.8B, about 6 latent frames, needs apex and flash-attn and
≥ 24 GB), Oasis 500M (MIT, 32 frames, Minecraft), DIAMOND CS:GO (MIT, 4 frames) and Micro-World 2 (AMD,
OpenRAIL, mouse in degrees per frame, built for ROCm). Yume 1.5 (Apache-2.0) keeps a compressed whole-history
memory but takes its actions as text.

### 2.3 World models that are handed the pose (upper bounds, not emergence tests)

SANA-WM (NVlabs, Apache-2.0 tag) has a true recurrent state (Gated DeltaNet), but its key-press mode is
turned into a camera-to-world trajectory before the network sees it, and it needs 47 GB in bf16.
Matrix-Game 3.0 and 3.5 (Apache-2.0) retrieve memory by pose and field of view, and 3.5 takes extrinsics
directly. HY-WorldPlay (Tencent's licence excludes the EU, the UK and South Korea; 72 GB) retrieves context
by field-of-view overlap. LingBot-World v1/v2 (Apache-2.0 / CC BY-NC-SA) and WorldMem (S-Lab, non-commercial,
memory retrieved by ground-truth pose) are also handed pose. Echo-Memory (CC BY 4.0) trains one Wan2.1-1.3B
backbone with raw, compressed and state-space memories but gives the pose relative to each segment's start.
VMem, Voyager, GEN3C, SPMem, Uni3C, Aether, ReCamMaster, CameraCtrl and the Wan-Fun camera models take
extrinsics or a pose-indexed 3D cache. If any of these is run, it is to show what our probes find when the
pose is given, or, for SANA-WM and Echo-Memory, to probe the state with the pose input held fixed. None is in
the first round.

### 2.4 Other Cosmos models

Cosmos-Predict2.5 (2B/14B, October 2025, archived; gated; NVIDIA Open Model License) has a robot
action-conditioned variant (256p, 4 fps) but no ego-motion variant. Cosmos-Predict2's action-conditioned
sample is a Bridge robot arm (7-D actions). A camera-conditioned Predict1 checkpoint is described in the
paper, but its release could not be confirmed. Cosmos-Reason2-8B (gated, Open Model License) is Qwen3-VL-8B
post-trained for physical reasoning, partly on egocentric data, with no actions. It is a cheap
same-architecture control for the preprint's Qwen3-VL-8B result, and `extract.py` and `two_choice.py` should
run it with a new model id†. Cosmos Policy is non-commercial and for manipulation only, and the Cosmos
tokenizers are plain autoencoders. The NVIDIA Open Model License ends automatically for anyone who bypasses
a safety guardrail without a substitute, so its models are run with the guardrail on. OpenMDW (Cosmos 3) has
no such clause.

### 2.5 JEPA and latent world models

| Model | Links | Licence | Size | Inputs | Memory | Fit 4070 / 5090 / 6000 / Mac |
|---|---|---|---|---|---|---|
| **V-JEPA 2** (Meta, Jun 2025) | [arXiv 2506.09985](https://arxiv.org/abs/2506.09985), [ViT-L](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256), [ViT-g](https://huggingface.co/facebook/vjepa2-vitg-fpc64-256) | ViT-L/H MIT, ViT-g Apache-2.0; ungated | ViT-L 300M (24 blocks), H 600M (32), g 1B (40, width 1408) | clips of up to 64 frames in training (256 or 384 px), tubelets of 2 frames × 16 × 16 px, 3D RoPE, no frame cap in the HF code (a 256-frame clip is 32,768 tokens at 256 px); no actions | the clip, bidirectional | ✓ / ✓ / ✓ / ~ (64 frames at 256 px; decord does not run on macOS) |
| V-JEPA 2-AC | same repo (`vjepa2-ac-vitg.pt`, 11.8 GB) | MIT† | frozen ViT-g + ~300M block-causal predictor | 7-D end-effector state and 7-D action deltas at 4 fps, from ~62 h of DROID filmed by a fixed exterior camera | a few frames | ✓ |
| V-JEPA 2.1 (Mar 2026) | [arXiv 2603.14482](https://arxiv.org/abs/2603.14482); files on dl.fbaipublicfiles.com, no official HF release | MIT | ViT-B 80M and L 300M (distilled from G), g 1B, G 2B; 384 px | as V-JEPA 2, with a loss on every token | the clip | B/L ✓ everywhere |
| **InJePA** (28 Sep 2026) | [arXiv 2610.23896](https://arxiv.org/abs/2610.23896), [code + checkpoint](https://github.com/zhuxun132/InJepa) | code MIT; checkpoint non-commercial academic use under StreamVLN and MP3D terms | frozen V-JEPA 2.1 ViT-B + ~135M action-conditioned JEPA modules | discrete VLN-CE actions (forward 0.25 m, turn 15°, stop); trained on 8,004 StreamVLN trajectories in 50 MP3D buildings (Habitat 0.2.4, 640 × 480, 79° HFOV, camera at 1.25 m) | 4 context frames | ✓ (Habitat evaluation Linux/CUDA) |

V-JEPA 2 was pretrained on VideoMix22M (SSv2, Kinetics, HowTo100M, YT-1B, ImageNet), with no navigation or
robot data in it. The HF code returns every block's per-patch tokens and the mask predictor's output.
V-JEPA 2-AC cannot take our walks: its actions move an arm in front of a static camera, so a moving camera
has no expression in its action space. The V-JEPA 2.1 paper's TartanDrive navigation world model is not
released. DINO-WM, Meta's JEPA-WMs (CC BY-NC), LeWorldModel and PLDM work on top-down or third-person 2-D
tasks or arm scenes, VL-JEPA (gated, FAIR non-commercial) and LeJEPA have no actions, the TD-MPC2
checkpoints are not egocentric, 1X GENIE has egocentric robot video but no actions, and PEVA has no weights.
Published probes of V-JEPA read camera pose and camera motion (Huang et al.: pose AUC@30 0.619 vs DINOv2's
0.508 on CO3Dv2; Musa et al.: camera-motion ROC-AUC above 90%, peaking at 60-70% of depth), and none probes
position over a walk.

### 2.6 Navigation policies

| Model | Links | Licence (code / weights) | Size | History | Odometry or past actions as input | Output | Fit |
|---|---|---|---|---|---|---|---|
| **ViNT** (Berkeley, CoRL 2023) | [arXiv 2306.14846](https://arxiv.org/abs/2306.14846), [code + weights](https://github.com/robodhruv/visualnav-transformer) (Google Drive) | MIT | 31M | current + 5 past frames, 85 × 64 | no | 5 waypoints (x, y, cos, sin) + temporal distance to a goal image | ✓ everywhere |
| **NoMaD** (Berkeley, 2023) | [arXiv 2310.07896](https://arxiv.org/abs/2310.07896), same repo | MIT | 19M | 3 past frames (config), 96 × 96 | no | 8 waypoints by diffusion + temporal distance; goal masking | ✓ everywhere |
| GNM (Berkeley, ICRA 2023) | [arXiv 2210.03370](https://arxiv.org/abs/2210.03370), same repo | MIT | small | 5 frames, 85 × 64 | no | as ViNT | ✓ |
| CityWalker (NYU, CVPR 2025) | [arXiv 2411.17820](https://arxiv.org/abs/2411.17820), [code](https://github.com/ai4ce/CityWalker) | Apache-2.0 | 214M | 5 frames through frozen DINOv2-B | **yes: 5 past positions from visual odometry** | 5 waypoints + arrival | ✓ |
| DD-PPO PointNav (Habitat) | [README](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-baselines/habitat_baselines/rl/ddppo/README.md) | MIT† / Matterport3D terms | ResNet/SE-ResNeXt + LSTM 512 | whole episode | **yes: previous action and GPS + compass goal vector** | discrete actions | ✓ |
| PIRLNav ObjectNav (CVPR 2023) | [code](https://github.com/Ram81/pirlnav) | not stated | CNN + GRU 2048 | whole episode | GPS + compass† | discrete | ✓ |
| StreamVLN (Jul 2025) | [arXiv 2507.05240](https://arxiv.org/abs/2507.05240), [code](https://github.com/InternRobotics/StreamVLN), [weights](https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3) | CC BY-NC-SA 4.0 | LLaVA-Video Qwen2-7B, 16 GB | sliding window of 8 dialogue turns + voxel-pruned memory built with depth | **yes: o1 a1 o2 a2 ... in the window** | VLN-CE actions (0.25 m, 15°) | ✗ / ✓ / ✓ / ~ |
| NaVILA (RSS 2025) | [code](https://github.com/AnjieCheng/NaVILA), [weights](https://huggingface.co/a8cheng/navila-llama3-8b-8f) | Apache-2.0 / not stated | 8B | current + 7 sampled frames | no | language actions ("move forward 75 cm") | ✗ / ✓ / ✓ / ~ |
| Uni-NaVid (RSS 2025) | [code](https://github.com/jzhzhang/Uni-NaVid), [weights](https://huggingface.co/Jzzhang/Uni-NaVid) | MIT / not stated | Vicuna-7B + EVA-CLIP | 64-frame buffer, then one merged token per frame | no | next 4 actions (25 cm, 30°) | ~ / ✓ / ✓ / ~ |
| LoGoPlanner (Dec 2025) | [arXiv 2512.19629](https://arxiv.org/abs/2512.19629), [weights](https://huggingface.co/InternRobotics/LoGoPlanner) (gated) | CC BY-NC-SA 4.0 | VGGT backbone (~1B†) | causal RGB-D sequence | no; has an internal camera-pose head | (Δx, Δy, Δθ) chunks | ~ / ✓ / ✓ / ? |

The Berkeley models were trained on more than 100 h of real robot driving (RECON outdoors; SCAND,
GoStanford, SACSoN/HuRoN indoors and on campus) and normalise waypoints by the dataset's step length, as NWM
does. Their temporal-distance head, which predicts how many steps separate the current view from a goal
image, is a ready-made model-native version of our two-choice question. Several other models are open but
fit our question less well: NaVid (CC BY-NC 4.0), InternVLA-N1/DualVLN (8B, CC BY-NC-SA), NavDP and X-NavDP
(RGB-D), MiniCPM-RobotTrack (0.9B, Apache-2.0, target tracking), OmniNav (ModelScope) and MTU3D (3D point
queries). NavFoM, Qwen-RobotNav and Qwen-VLA have no released weights. Wijmans et al. released no agent
checkpoints. Every Habitat agent above receives GPS and compass, so position is part of its input.

### 2.7 Manipulation VLAs

Almost all of them condition on a single timestep, and the rest on two: OpenVLA (7.5B; MIT, with the weights
carrying the Llama-2 licence), OpenVLA-OFT, openpi's π0, π0-FAST and π0.5 (3.3B; Apache-2.0 plus the Gemma
licence; JAX, with a PyTorch port for π0 and π0.5), SmolVLA (450M, Apache-2.0), CogACT, SpatialVLA,
X-VLA, MolmoAct and MolmoAct2 (Apache-2.0), Octo (2-frame window, JAX) and RDT-1B (2 frames). GR00T N1.x
(Apache-2.0 code, NVIDIA Open Model License weights; N1.7's backbone is gated; CUDA, flash-attn and TensorRT)
and LingBot-VLA 2.0 (Apache-2.0) add large amounts of egocentric human video. MiniCPM-RobotManip (1.5B,
Apache-2.0) streams up to a minute of history. They see tabletops from fixed or wrist cameras and never move
through a room. π0.6 and later are not open. These models could at most serve as negative controls for
"action training without self-motion", and they are left out of the first round.

### 2.8 Streaming 3D reconstruction models with a state (positive controls)

| Model | Links | Licence | Size | State | Pose output | Fit 4070 / 5090 / Mac |
|---|---|---|---|---|---|---|
| **CUT3R** (CVPR 2025) | [arXiv 2501.12387](https://arxiv.org/abs/2501.12387), [code](https://github.com/CUT3R/CUT3R) (weights on Google Drive) | CC BY-NC-SA 4.0 | 793M | fixed 768 tokens × 768, updated every frame and returned per frame; `inference_step(view, state_args)` takes the state as an argument, and per-view `update`/`reset` masks gate the write | metric camera-to-world in the first frame's coordinates | ✓ (~6-8 GB) / ✓ / plausible† |
| TTT3R (ICLR 2026) | [arXiv 2509.26645](https://arxiv.org/abs/2509.26645), [code](https://github.com/Inception3D/TTT3R) | CC BY-NC-SA + MIT | CUT3R's weights | same state, a different update rule | same | ✓ (6 GB) |
| StreamVGGT (ICLR 2026) | [arXiv 2507.11539](https://arxiv.org/abs/2507.11539), [weights](https://huggingface.co/lch01/StreamVGGT) | CC BY-NC-SA / CC BY-NC | 1.26B (VGGT made causal) | growing KV cache (~25 GB at 200 frames) | VGGT's 9-D | ✗ / ~ / ✗ |
| Point3R (NeurIPS 2025) | [arXiv 2507.02863](https://arxiv.org/abs/2507.02863), [code](https://github.com/YkiWu/Point3R) | CC BY-NC-SA | ~0.8B† | explicit pointer memory with 3D positions | metric, first-frame coordinates | ? / ✓ / ? |
| HorizonStream (2026) | [arXiv 2605.23889](https://arxiv.org/abs/2605.23889), [weights](https://huggingface.co/NicolasCC/HorizonStream) | CC BY-NC-SA (VGGT lineage) | ViT-L | 1024 × 1024 linear-attention states at 4 layers, ~8.5 GB flat | world-to-camera, metric | ✓ / ✓ / ? |
| LingBot-Map (2026) | [arXiv 2604.14141](https://arxiv.org/abs/2604.14141), [weights](https://huggingface.co/robbyant/lingbot-map) | Apache-2.0 | ~1.2B† | first 8 + last 64 frames in full, 6 tokens per older frame | camera-to-world | ~ / ✓ / ✗ |

MUSt3R, Spann3R, Mem3R, ZipMap and STream3R also keep a state. Pi3, MapAnything, VGGT-Long, Fast3R and
VGGT-Ω have none, so they add nothing to our existing VGGT control. CUT3R's state forgets: one report has its
error growing 3.8-fold from 50 to 250 frames (arXiv 2601.22615), and 250 frames is our walk length.

## 3. From our walks to each model's inputs

### 3.1 Poses to actions

Every walk stores (x, y, θ) per frame in `meta.npz`: x east and y north in metres, θ counterclockwise from
+x. The camera is 1 m above the floor with a 90° horizontal field of view at 256 × 256, looking along
f = (cos θ, sin θ, 0), with its right-hand axis r = (sin θ, −cos θ, 0). `room.foraging_walk` turns first
and then steps forward along the new heading (0.22 m × U(0.7, 1.3)). On 15% of steps it instead scans on the
spot by 30-90°. Because the poses are exact, every action format below is computed, not estimated. The
relative motion from frame t to frame t + k, in frame t's body frame (ROS convention: x forward, y left,
yaw counterclockwise), is

    dxw, dyw = x[t+k] - x[t], y[t+k] - y[t]
    fwd  =  cos(θ[t]) * dxw + sin(θ[t]) * dyw
    left = -sin(θ[t]) * dxw + cos(θ[t]) * dyw
    dyaw = wrap(θ[t+k] - θ[t])          # in (-π, π]

and the camera-to-world pose in OpenCV axes (x right, y down, z forward) is R_c2w = [r | (0, 0, −1) | f]
with centre C = (x, y, 1.0), with intrinsics fx = fy = cx = cy = W / 2 at a 90° field of view. The formats
differ as follows.

- **Relative planar displacement** (NWM, RAE-NWM, CompACT, ViNT/GNM waypoints, CityWalker): (fwd, left,
  dyaw) to the target frame, with fwd and left divided by the training dataset's `metric_waypoint_spacing`.
  The main setting uses SACSoN/HuRoN's 0.255 m (indoors, and close to our 0.22 m step), and RECON's 0.25 m
  serves as a check. NWM's time shift is k / 128 for k frames at 4 fps. The sign convention of
  `to_local_coords` is checked on one walk before any run by plotting the actions against the frames.
- **9-D relative pose change** (Cosmos 3 camera motion): ΔT = T[t−1]⁻¹ T[t] in OpenCV camera axes. The
  translation is (−left, 0, fwd) and the rotation is R_y(−dyaw) about the camera's down axis; the 6-D
  rotation is its first two columns. The normalisation is undocumented, so it is calibrated by commanding
  known turns and steps and measuring the generated camera motion with VGGT (Section 4.3).
- **Discrete VLN-CE steps** (InJePA, StreamVLN, Uni-NaVid, DD-PPO): rather than quantising the existing
  walks, new walks are generated directly in the action set (forward 0.25 m, turn 15° or 30°; 10° for
  DD-PPO†) with the
  foraging walk's statistics. A 60° scan then becomes four 15° turns, each rendered at its exact pose.
  NaVILA's language actions ("move forward 75 cm") are sums of the same steps.
- **Keys and mouse** (Infinite-World, Zing-0.5, Waypoint-1.5, Matrix-Game 2.0): W is forward translation,
  and a turn key or mouse dx is yaw. Our walks never strafe. The gains (metres per key-frame, degrees per
  mouse unit) and Infinite-World's key thresholds are unknown, so they are calibrated per model the same way
  as Cosmos 3's.
- **Extrinsics** (pose-informed models, and evaluation of CUT3R and VGGT): R_c2w and C as above.
- **Arm actions** (V-JEPA 2-AC, OpenVLA, π0, Octo, GR00T): there is no faithful mapping from a moving camera
  to an arm moving in front of a fixed camera, so these action spaces are not used.

### 3.2 Matching the camera and the motion

Our frames differ from every model's training data. The camera height (1 m) sits between robot cameras
(about 0.3-0.6 m on the RECON and HuRoN robots†), Habitat agents (1.25 m for InJePA) and head-mounted
video (about 1.6 m†). The 90° field of view and square 256 px frames differ from 85 × 64 (ViNT), 224 (NWM),
640 × 480 at 79° (InJePA) and 16:9 at 480p (Cosmos 3). The motion has one frame per 0.22 m step and scans of
up to 90° within a single frame, whereas turns of more than about 30° between frames are rare in all of the
training sets†. Both renderers take any pose, so each model gets its own render of the same walks:
- the model's camera height, field of view and aspect ratio. `room.py` fixes `IMG`, `HFOV` and `CAM_H` as
  module constants and `habitat_scene.py` fixes `RES`, `HFOV` and `CAM_H`, so both need small parameter
  changes;
- scans split into sub-steps of at most 15°;
- for models with a 4× temporal VAE (Cosmos 3, Zing-0.5, Infinite-World), four sub-frames per 0.22 m step,
  so that one latent frame covers one original step rather than mixing four poses.

Every sub-frame has an exact pose. The analyses use either the frames that fall on the original steps or all
frames with their own poses, and the choice is fixed per model in the pre-registration. The original 256 px
walks remain the primary input for V-JEPA 2 and CUT3R, whose training cameras are generic.

### 3.3 Scenes

The scenes are the ray-cast rooms A and B, the ten held-out procedural rooms P5000-P5009 (Exp 10), the three
Replica apartments (Exp 12) and the eight HSSD houses (Exp 16), and all of their walks, probes and lure-free
trials exist already. For models trained on real video the photoreal scenes are the primary test. For unit
scoring in Habitat scenes, denser coverage is needed (64 walks per scene instead of 32, rendered on the
4070). `analyze.py` also assumes the 10 m box (`L = 10`, 20 bins of 0.5 m), so it needs each scene's
bounding box and a mask for unvisited bins. The Habitat-trained models (InJePA, StreamVLN, DD-PPO) were
trained on MP3D, HM3D or Gibson, none of which includes Replica or HSSD†; this is checked against each
model's data list before it is run.

## 4. Which analyses apply, and the code they reuse

The existing pipeline needs only an extractor per model. `analyze.py` reads
`data/acts/<tag>/L<layer>.npy` (frames × units, float16, rows aligned with the walk's `meta.npz`) and
scores every unit; `analyze_probes.py` reads `<model>-probes-<mode>`; the `+ema<h>` tag suffix builds the
moving-average control from any stored activations; and the lure-free trial files
(`results/export/dissimilar*/trials.json`) define the behavioural tests. Each new model therefore gets one
module (`spatialcells/wm_<model>.py`) that renders or loads its walk variant, feeds the model and writes
activations in that format. The scoring, statistics and figures then run unchanged.

| Analysis (preprint) | Existing code | NWM, RAE-NWM | ViNT, NoMaD | V-JEPA 2 / 2.1, InJePA | CUT3R, TTT3R | Cosmos 3, Waypoint, Zing | StreamVLN |
|---|---|---|---|---|---|---|---|
| Rodent unit scoring, strict opposite-sector invariance, moving-average control (Exp 1, 13e) | `analyze.py --quadrant`, tags `<base>+ema8` | ✓ target tokens, per block and noise level | ✓ per-frame tokens, transformer layers, pooled context | ✓ encoder blocks, last tubelet; InJePA predictor | ✓ state, pose token, decoder tokens | ✓ both towers, per latent frame | ✓ LLM layers at frame tokens |
| Positive-control reference (Exp 7, 13f) | `grid_rnn.py --score --quadrant` | numbers reused; optional RNN trained on the same action encoding | same | same | same | same | same |
| Teleport probes (Exp 2) | `render_probes.py`, `analyze_probes.py --quadrant` | probe as target after 4 walk frames | probe as current frame | probe as last tubelet after the walk window | probe read with `update=False` after the walk | probe as the last latent frame | probe after the walk, as in `extract.py` |
| Patching a position subspace, controls, cross-room, reference frame (Exp 4, 13c, 14d, 14e) | `vggt_patch.py` (subspaces, transfer, erasure, `--controls`, `--crossroom`), `vggt_remap.py`, `vggt_reference.py` | readout: predicted frame localised by VGGT | readout: distance head, waypoints | readout only through InJePA's planner | readout: pose head (closest to VGGT) | readout: generated frame localised by VGGT | readout: two-choice answer |
| Two-choice behaviour (Exp 9-12, 16) | trial files; `two_choice.py` scoring; `heldout_analysis.py`, `photoreal_analysis.py`, `hssd_analysis.py` | turn-in-place 2AFC | distance-head 2AFC on the same trials | InJePA: turn-in-place; encoders: 14a-style readout | localise the view, as VGGT | turn-in-place 2AFC | the two-choice prompt itself |
| New: action manipulations and dark stretches | template: `causal_ema`, `grid_rnn` free vs anchored | actions zeroed, shuffled, mirrored | (no action input) | InJePA only | (no action input) | ✓ | actions in context |

### 4.1 Unit scoring and strict heading invariance

`analyze.py --quadrant` runs unchanged on each model's activations: rodent place, grid and head-direction
criteria against circular-shift shuffles, strict invariance on opposite 90° sectors, and the decoding suite.
Three conditions carry over from the preprint for every model with temporal context: single frames (the
context replaced by the current frame alone, or by a static clip of it), the walk in the model's native
context, and the walk shuffled. The moving-average control applies the 8-step EMA to the model's own
single-frame activations. Because that control already reaches 90-97% strictly invariant units, invariance
along the walk cannot separate a place code from smoothing in any model. The unit measure that can is
computed on teleport probes (Section 4.2), where no recent history matches the view. That needs more probes
than the preprint used: 256 per walk instead of 64, which gives about 30 per 0.5 m bin in room A, the
density of the walk frames the preprint scored. `analyze.py` needs an option to score probe activations
against `probes_meta.npz`; since probes come in random order, its circular shuffles act as permutations
there. Diffusion models (NWM, Cosmos 3) are read at two noise levels chosen on room A before any other room is scored, since
3D information peaks at early but not first denoising steps in video diffusion models (Huang et al.). The
V-JEPA encoders attend in both directions within a clip, so frame t is read from a causal window that ends
at t, with frame t duplicated into the last tubelet so that the tubelet does not mix two poses. Cosmos 3
attends in both directions too, so its clip is truncated at t.

### 4.2 Teleport probes

Probes are views at random headings from places the walk visited, appended after the walk (Exp 2). The
preprint decoded them across opposite heading sectors. Exp 13f showed that this population test can fail
for a real position code: the base RNN, whose units are all heading-invariant, misses by 7.57 m with
mirrored errors. The primary probe measure here therefore changes. A ridge decoder is trained on walk frames
of the even walks at all headings and tested on probes of the odd walks, separately for familiar probes
(some walk frame within 1 m faced within 45°; `analyze_behave.geometry`) and novel ones. The **novelty
cost** is novel minus familiar error. A view code pays a large novelty cost and a place code pays little,
and the decoder's training on all headings removes the heading-specific shortcut behind the mirrored
failure. The pixel baseline and the recall baseline (the most similar walk frame's 8-step moving average,
1.68 m at 180° halves and 1.95 m strict) are recomputed under the same protocol, and the strict cross-sector
version is reported with 13f's caveat as before.

How a probe enters each model:
- **CUT3R:** read with the state's write disabled (`update=False`), so later probes do not see earlier ones.
- **V-JEPA:** as the last tubelet after the whole walk as one 256-frame clip (four times the training
  length), and as a second condition after only the walk's last 62 frames.
- **NWM:** as the target frame after the last four walk frames, with the action from the last frame to the
  probe's pose. In this case the decoder also gets the displacement through the input, so NWM's probes are
  reported with and without the action.
- **ViNT:** as the current frame, with the five frames before it rendered along a straight approach to the
  probe's pose.

### 4.3 Patching a position subspace

`vggt_patch.py` holds the whole recipe: subspaces fit on held-out frames, the transfer score, erasure, and
the 13c and 14d controls. The subspaces are position PCA-k from the mean token per 1 m bin, the ridge probe
at each λ, the Haufe pattern, the cross-covariance direction, random subspaces and the exact complement.
Transfer is the fraction of the target-to-donor distance that the model's own position output moves. Its
VGGT-specific parts are the model call and the readout, so the plan is to move the subspace and scoring
code into a model-agnostic module and give each model a readout:
- **CUT3R:** its pose head, the closest analogue of VGGT's camera head. Patching sites are the current
  frame's tokens after each decoder block and the state itself. Patching the state between a target walk
  and a donor walk asks whether the remembered map, rather than the current view, sets the pose.
- **NWM, Cosmos 3, Waypoint:** the predicted frame (the one-step x̂₀ estimate at the chosen noise level, or a
  short sample), localised by a **pose meter**. The meter is VGGT in single mode with the 8 standard anchors,
  whose error in room A is known (0.42 m median); in the photoreal scenes every 8th walk frame serves as
  the reference, as in Exp 12. A nearest-neighbour search over DINOv2 features of a dense
  render bank (0.25 m × 15°, about 38,000 views per room, CPU) serves as a second meter. Both are checked on
  ground-truth frames before use. Transfer is how far the localised pose moves toward the donor's.
- **ViNT, NoMaD:** the temporal distance to a fixed goal image and the predicted waypoints, patched at the
  transformer tokens or the pooled 32-d context.
- **InJePA:** the planner's action and its cost to a goal image. The plain V-JEPA encoders have no position
  output, and a probe trained on clean activations is not an independent readout, so they are not patched.
- **StreamVLN:** the two-choice answer, as in Exp 15's attention cuts, or a position subspace at the new
  view's tokens.

Exp 14d's cross-room test (subspace fit in room A, transfer measured in room B) applies to every model
patched. Exp 14e's reference-frame test needs a reference. VGGT and CUT3R take the 8 anchor views as a
prefix. For models without anchors the reference is the start of the walk, so the test prefixes every walk
with the same short scan at a fixed pose and then repeats it with the scan rotated 45° about the room
centre. A place code tied to the room keeps its maps. One tied to the start rotates them, as VGGT's did.

### 4.4 Behavioural analogues of the two-choice task

All variants use the registered lure-free trials: room A, the ten held-out rooms (1,920 trials, 1,299
novel), the apartments (503) and the houses (1,450). In every trial the near option is within 1 m, the far
option more than 3 m away, both face more than 90° away from the new view, neither shows its landmarks, and
pixel similarity and frame order are balanced. Each model's numbers therefore go into the same tables as
Qwen, the LoRA model, Opus and VGGT.
- **Distance-head 2AFC (ViNT, NoMaD, GNM).** The observation is an option frame with the walk frames before
  it (at the model's spacing) and the goal is the new view. The model picks the option with the smaller
  predicted temporal distance. The reverse direction (the new view with a rendered approach as the
  observation, each option as the goal) is secondary.
- **Turn-in-place 2AFC (NWM, RAE-NWM, Cosmos 3, Waypoint, InJePA).** For each option X, the model is given
  its context ending at X and the action "turn on the spot from X's heading to the new view's heading". It
  then scores the new view under that hypothesis. Diffusion models score it by the conditional denoising
  loss over a fixed set of noise levels and seeds, the diffusion-classifier method (Li et al. 2023). JEPA
  models score it by the distance between the predicted embedding and the new view's embedding. The lower
  score is the choice.
  - Because the new view faces more than 90° away from both options and shares no landmark with them, the
    frames at X do not contain it. A model can beat 50% only if its context holds what the room looks like
    behind X. This is the same demand the two-choice question makes.
  - Controls: no walk (X alone), the walk shuffled, another room's walk, and, as a sanity check, the true
    action from X's pose to the new view's pose, under which both options should score alike.
- **Localising the view (CUT3R; StreamVGGT if run).** The new view is placed against the walk state with
  the write disabled, and the model picks the option nearer the predicted position, as VGGT did in Exps 12
  and 16 (82-83%).
- **The prompt itself (StreamVLN).** StreamVLN was co-trained on video QA, so `two_choice.py`'s 64-frame
  prompt can be asked as it is†, with the preprint's parsing and statistics. That makes it directly
  comparable to base Qwen3-VL-8B and the LoRA model.
- **Encoders without actions (V-JEPA 2/2.1).** These have no behaviour to score. Exp 14a's readout
  (the nearer option decoded from the new view's tokens, trained on five held-out rooms and tested on the
  other five) is reported instead and labelled as a readout, not behaviour.
- **Generation quality (secondary).** NWM's generated revisit views are scored against the ground-truth
  render (DreamSim and LPIPS).

### 4.5 New for action-conditioned models: action manipulations and dark stretches

Two manipulations are possible only with an action input. First, the actions are corrupted at test time:
- zeroed;
- shuffled in time within the walk, which keeps their distribution;
- mirrored (left and dyaw negated);
- scaled by 0.5 and 2.

A code that depends on path integration should degrade under shuffling, mirror under mirroring and rescale
under scaling. A code built from views alone should be unaffected while frames are visible. Second, **dark
stretches** of k frames (k = 8, 32) are replaced by a uniform grey frame, or by the model's own predictions
in a generative rollout, while the true actions continue. Position is then decoded at the end of the
stretch with a decoder trained on lit frames. The path-integrating RNN is the reference: 0.17 m anchored and
1.5-1.6 m after 255 free-running steps (Exp 7, 13f). This test separates path integration from view
recognition, and it needs a model with both an action input and a memory longer than k: Cosmos 3,
Waypoint-1.5, Zing-0.5 and Infinite-World, and StreamVLN with its actions in context. NWM's four-frame
context limits it to k ≤ 4 unless its own rollouts carry the information forward, which is itself a result.

## 5. Shortlist: what to run first

The ranking weighs four things: which cell of the actions-by-memory design a model fills, whether the
network itself must integrate motion, whether it runs on the machines that are free (the 4070, the 5090 and
the Mac), and how directly it can be compared with the preprint. GPU hours are estimates for the
shortlisted conditions in room A, room B, three Replica apartments and two HSSD houses, plus the behavioural
tests on all registered trials. There is no API spend and no new data, and everything runs through gq. The
weights take about 40 GB of disk and the activations about 50-100 GB, which go to `/mnt/bigd` on the PC and
the Docker volume on vast-3.

| Rank | Model | Cell | Why first | Compute (estimate) | Cost and access |
|---|---|---|---|---|---|
| 1 | **NWM** (CDiT-XL and XL-Ego4D; S, B, L as a size series) with RAE-NWM | motion-only actions, 4-frame memory | The canonical navigation world model, whose paper leaves this exact question open. Its action format is ours, its indoor robot data are the nearest real domain, it fits every GPU, and the sizes give a scaling series like the Qwen one. RAE-NWM puts the same model in DINOv2's feature space, which is already one of our baselines. | ~50k forward passes per scene and condition (walk frames and probes at two noise levels), about 20-25 min on the 4070†; five conditions in seven scenes ~12 h, the S/B/L sizes ~4 h, turn-in-place 2AFC ~3 h on the 5090†; ~20 GPU-h in all | GPU time only. The user must accept the gated licence on Hugging Face. CC BY-NC: research use, no redistribution of derived weights. |
| 2 | **ViNT and NoMaD** (GNM as a check) | no action input, ~1 s of frames; trained with action labels | The cheapest and fastest result. The temporal-distance head answers the registered two-choice trials directly, so the numbers sit beside Qwen, Opus, the LoRA model and VGGT. The training data are real egocentric robot navigation. | < 3 h on the Mac or CPU† | MIT; weights on Google Drive. |
| 3 | **V-JEPA 2** (ViT-L, ViT-g at 256 px) and **V-JEPA 2.1 ViT-B with InJePA** | no actions with walk-length clips; motion actions with 4 frames on the same encoder | The JEPA arm. InJePA trains an action-conditioned predictor in Habitat on top of the same frozen ViT-B, which gives a with-actions versus without-actions contrast on one encoder. The 64-frame clips give the memory-without-actions cell. Ungated. | one 64-frame causal window per walk frame and per probe: ViT-g ~1.5 h per scene and condition on the 5090† (four scenes), ViT-L and ViT-B on the 4070 or Mac (seven scenes); InJePA small; ~25-30 GPU-h | V-JEPA MIT/Apache; InJePA checkpoint non-commercial academic. |
| 4 | **CUT3R** (+ the TTT3R update rule on the same weights) | positive control: a learned state over the whole walk | A model that localises from a persistent state, between VGGT (45% strict, no state) and the RNN (98-100%). Its state can be read and swapped, and its pose head makes patching as direct as VGGT's. If it fails the strict tests, the tests are too strict for learned states, which must be known before any world-model null is read. | streaming at ~5 frames/s on the 4070†, probes batched against each walk's final state: ~0.8 h per scene and condition; with patching ~20 GPU-h | CC BY-NC-SA: keep its code outside our MIT repository (import it, do not vendor it). |
| 5 | **Cosmos3-Edge** (Nano on the RTX PRO 6000 if it becomes free) | motion-only 9-D relative actions; the whole walk in one clip | The cell nothing else fills on our machines: per-step motion, so the network integrates the path, and memory that spans the walk. If Nano can be run, its towers start from Qwen3-VL-8B, so Qwen3-VL-8B, Cosmos-Reason2-8B and Nano's two towers form a same-initialisation series. | a one-day feasibility check on the 5090, then ~15-20 GPU-h†. With 4× sub-stepped renders, one latent frame = one walk step. | OpenMDW-1.1, ungated. Linux/CUDA only. |

**Alternates, in order.**
- **StreamVLN**, the closest sequel to the preprint: a video-LLM fine-tuned for navigation with past actions
  in context, about 8 GPU-h on the 5090. Its slow memory prunes image patches by 3D voxel after
  back-projecting them with depth (one variant also uses odometry), which can carry position into the
  memory†, so it is run with that memory off.
- **Waypoint-1.5**, which takes Cosmos3-Edge's place if Edge fails its feasibility check: Apache-2.0
  weights, a 512-frame context, fits the 4070. It has no paper and undisclosed training data.
- **Infinite-World and Cosmos3-Nano**, the strongest emergence tests, which both need the RTX PRO 6000:
  Infinite-World's 20-token pose-free memory is the one place a place code would have to live.
- **DD-PPO RGB and CityWalker**, as controls whose inputs carry position or odometry.

Engineering time is about 2-3 days per model family: the extractor, the walk variant and the readout. The
shared parts are generalising the patching module, the pose meter, the behavioural runners and the arena
generalisation in `analyze.py`, about a week. With queue sharing, the five shortlisted models take about
3-4 weeks of calendar time and roughly 90 GPU-hours in all. If vast-3 is billed by the hour, about 40 of
those hours fall on it.

## 6. Draft pre-registration

Written in the style of `docs/experiments.md`. It becomes registered only when the user approves it and it
is copied there with a date, before any model in it sees our walks. Amendments are dated and made before
results, as for Exps 10-16.

**Experiment W0: competence gates** (run first, on room A and one Replica apartment; nothing else is
analysed until a model passes or fails its gate). A null result from a model that cannot handle our renders
says nothing, so each model must first show it works on them:
- NWM: one-step predictions beat copying the last frame (median DreamSim lower, 95% CI over walks);
- ViNT/NoMaD: the predicted temporal distance correlates with the true step distance within 20 steps along
  our walks (Spearman ρ ≥ 0.5);
- V-JEPA 2: within-sector position decoding (`decode_within_quadrant`) is no worse than DINOv2's;
- CUT3R: the median camera-position error after a similarity alignment is ≤ 0.5 m;
- Cosmos 3: once the action scale has been calibrated on one set of commands, a second set of 15° turns and
  0.25 m steps produces camera motion within 25% of the command in VGGT's estimate;
- InJePA: its planner's first action points toward a goal 2 m ahead in an open part of the scene more often
  than chance.

A model that fails in the ray-cast rooms is tested only in the photoreal scenes, and one that fails there
too is reported as not competent on our renders and left out of the tests below. The thresholds are fixed
here, before any run.

**Experiment W1: heading-invariant place units on teleport probes.**
- *Models:* the shortlist, their random-weight versions, and the existing references (pixels, DINOv2,
  Qwen3-VL-8B, VGGT single and walk, and the path-integrating RNN on walk frames only, since a teleport
  gives it no input).
- *Inputs:* the model's walk variant (Section 3.2) and 256 probes per walk in room A (selection), then room B,
  the apartments and two houses (confirmation).
- *Measure:* the share of place units, by the unchanged `analyze.py` criteria, whose opposite-sector maps on
  the probes correlate above 0.5.
- *Layer and noise level:* chosen on room A as the best layer by that share, then fixed for every other
  scene.
- *Conditions:* walk in the model's native context, shuffled walk, no walk; for action models also true,
  zeroed and shuffled actions.
- *Analysis:* means with 95% bootstrap CIs over walks; between-condition differences paired by walk.
- *Decision rule:* a share ≥ 45% (VGGT single's) in the confirmation scenes is called "place-like", ≤ 10%
  (the single-frame range is 0-6%) is called "view-like", and anything between is reported as intermediate.
- Unit invariance along the walk and the model's own EMA-8 control (Exp 13e) are reported as secondary,
  since smoothing alone reaches 90-97%.

**Experiment W2: novel-probe localisation.**
- *Decoder:* ridge, trained on even walks' frames at all headings and tested on odd walks' probes.
- *Outcome:* the novelty cost (median error on novel probes minus familiar ones) at W1's fixed layer, with
  the pixel and recall baselines under the same protocol.
- *Decision rule:* "beyond view recognition" requires a novel-probe error below the recall baseline's and a
  novelty cost below the pixels', both with 95% CIs over walks excluding zero.
- Strict cross-sector decoding is reported with Exp 13f's caveat and is not used for any claim.

**Experiment W3: path integration in action-conditioned models with memory** (Cosmos 3, alternates
Waypoint-1.5 and StreamVLN; NWM with k ≤ 4).
- *Conditions:* dark stretches of k = 8 and 32 frames with true actions, the same with shuffled actions, and
  lit frames with mirrored actions.
- *Outcome:* position decoded at the end of the stretch by W2's decoder.
- *Decision rule:* "path integration" requires the dark-stretch error below chance and below the
  shuffled-action error (95% CIs excluding zero) at k = 8. Under mirrored actions the decoded displacement
  along the walk should correlate better with the mirrored path than with the true one.
- *Reference:* the RNN's free-running drift.

**Experiment W4: use of the position code.**
- *Models:* the shortlist models with a readout (Section 4.3).
- *Setup:* Exp 4's patching design with 128 targets and donors more than 3 m away, subspaces fit on 2,048
  other frames, at three sites chosen on room A.
- *Conditions:* position PCA-10, PCA-2, the ridge λ-sweep, the Haufe pattern, random 10-d, the exact
  complement of PCA-10, and erasure of PCA-10 and of random 10-d.
- *Cross-room and reference frame:* Exp 14d (fit in A, test in B) and Exp 14e (start scan or anchors
  rotated 45°).
- *Decision rule:* "causal position code" requires PCA-10 transfer ≥ 0.5 at some site, complement and random
  transfer ≤ 0.05, and an erasure error at least twice the clean error, with the random erasure within 10% of
  clean. This mirrors VGGT's 0.85 / 0.00 / 0.00 and its 3.38 m vs 0.43 m.

**Experiment W5: behaviour on the registered two-choice trials.**
- *Variants:* those in Section 4.4, per model.
- *Primary:* novel-trial accuracy in P5000-P5009 with rooms as the unit (means over rooms with 95% bootstrap
  CIs, room-level sign tests) and walk minus no walk; all-trial accuracy in the apartments and houses with
  exact binomial CIs and houses as the unit.
- *Turn-in-place controls:* shuffled and other-room context.
- *Decision rule:* "uses the walk" requires walk minus no walk and walk minus other-room context both above
  zero with 95% CIs over rooms.

**Experiment W6: what action training adds** (between-model contrasts, secondary to W1-W5).
- *Contrasts:* InJePA versus V-JEPA 2.1 ViT-B (same frozen encoder; W1, W2); RAE-NWM's predictor versus
  DINOv2, the frozen features it predicts (W1, W2); and Qwen3-VL-8B, Cosmos-Reason2-8B and Cosmos3-Nano's
  towers (W1, W2, W5), if Nano can run.
- The contrasts are stated as differences with CIs over walks. They show what the action-conditioned
  training adds to the same starting weights, not why.

**Predictions** (stated so they can be wrong; every result is reported either way):
- The short-context models (NWM, ViNT/NoMaD, InJePA) stay view-like in W1 and W2 and near 50% on novel
  trials in W5, whatever their action training.
- V-JEPA 2 resembles the VLMs.
- CUT3R is place-like in W1 and W4 if any learned state is.
- A long-memory motion-only world model is the one candidate for path integration in W3.

**Multiple comparisons.** Holm correction across the shortlisted models for each primary decision. Layers
and noise levels are chosen on room A only.

**Cost.** None beyond GPU time on the existing machines (Section 5) and the user's acceptance of the NWM
licence.

## 7. Risks

**Domain gap.** The ray-cast rooms are textured boxes with posters. NWM and ViNT were trained on real robot
footage, mostly outdoors or in offices, with other cameras and smooth motion. V-JEPA 2 and Cosmos 3 were
trained on web video. The competence gates (W0) and the per-model renders (Section 3.2) address this, and
the photoreal Replica and HSSD scenes are the primary test for models trained on real video. A remaining
risk is that a model passes the gate but uses texture shortcuts in the ray-cast rooms. The procedural rooms'
random textures and the photoreal scenes are there to show that. Habitat-trained models (InJePA, StreamVLN)
are at home in the simulator's look, which helps them relative to the real-video models. Model comparisons
are therefore made within a scene type, never across.

**Action-space mismatch.** NWM and the planar formats match exactly. Cosmos 3's normalisation, the game
models' key and mouse gains and Infinite-World's thresholds are undocumented and must be calibrated, and a
wrong scale would look like failed path integration. The calibration is part of the gate, and W3's scaled
and mirrored actions test for this directly. Arm action spaces (V-JEPA 2-AC, manipulation VLAs) cannot
express a moving camera and are not used. Discrete VLN-CE steps are generated natively rather than
quantised.

**Position leaking through the input.** Pose-informed world models, DD-PPO's GPS and compass, CityWalker's
odometry and, probably, StreamVLN's voxel-pruned memory† give position away, so a probe would read back the
input. These models are excluded from the emergence tests or run with that input removed. For NWM's probes,
the displacement in the action is reported separately (Section 4.2).

**Seeing the future.** V-JEPA's encoder and Cosmos 3 attend in both directions within a clip, so the
representation of frame t sees later frames. Causal windows ending at t (V-JEPA) and clips truncated at t
(Cosmos 3) remove this. A check that the walk's last frames do not change earlier frames' activations is
added to W0.

**Readouts that cannot tell views from places.** Natural-trajectory probes can look good while the model
does not respond when targets are crossed with contexts (Kim et al.). Memory can help without being
place-specific (Tiwari et al.: WorldMem keeps 94% of its memory benefit with a wrong memory). The novelty
cost, the other-room and shuffled contexts, and patching with complement and random controls are the
designs that answer this. The pose meter adds VGGT's own error (0.42 m) to every generated-frame readout,
so transfer is computed relative to the meter's error on ground-truth frames.

**Engineering.** Several stacks are CUDA-only and pin flash-attn or apex (Cosmos 3, Infinite-World,
Matrix-Game 2.0, StreamVLN). The Mac runs only the small PyTorch models. vast-1 is busy, which puts
Cosmos3-Nano and Infinite-World out of reach for now. vast-3's container has 24 GB of host RAM, so a
large checkpoint must be loaded to the GPU directly from safetensors. NWM's checkpoints are pickles flagged
by Hugging Face's scanner and are loaded with `weights_only=True` in an isolated environment.

**Licences.**
- *Non-commercial:* NWM (gated; the user accepts the terms), InJePA's checkpoint (also under StreamVLN and
  MP3D terms), StreamVLN, CUT3R, TTT3R and NaVid. All allow a research paper.
- *ShareAlike code:* CUT3R, TTT3R and StreamVLN are CC BY-NC-SA, so their code must not be copied into our
  MIT repository. Adapters import it.
- *Gated:* NWM, LoGoPlanner and the Cosmos-Predict2.5/Reason2 models.
- *Guardrail clause:* the NVIDIA Open Model License ends if a safety guardrail is bypassed, so those models
  run with it on. Cosmos 3's OpenMDW has no such clause.
- *Territorial limits:* Tencent's community licences exclude the EU, the UK and South Korea, which matters
  for collaborators.
- *Other terms:* Habitat agents' weights fall under the Matterport3D terms of use, OpenVLA's under Llama 2's
  and openpi's under Gemma's. HSSD (CC BY-NC) and Replica are already in use.
- *Derived weights:* fine-tuned weights from non-commercial models stay non-commercial. Results can be
  published; whether stored activations can be released with the paper is checked per licence.

**Novelty and pace.** The field moves monthly: several models above were released in August-October 2026,
and the literature search ran out of budget. An arXiv search for place codes, cognitive maps and position
probes in world models, VLAs and JEPAs is repeated before writing and before submission.

**A null across the board.** If every pretrained model is view-like, the paper's claim rests on the tests
being passable. The RNN (98-100%), VGGT (45%; transfer 0.85) and CUT3R are what show that, which is why
CUT3R is on the shortlist and why W0's gates are fixed before any result.

## 8. Decisions for the user before anything runs

- Whether to accept NWM's gated licence on Hugging Face (only the account holder can).
- Whether the RTX PRO 6000 can be booked later for Cosmos3-Nano and Infinite-World.
- Whether the first round is the five shortlisted models, or a smaller pilot of ranks 1, 2 and 4
  (NWM, ViNT/NoMaD and CUT3R; about 45 GPU-hours, mostly on the 4070 and the Mac) to test the pipeline first.
- Whether to add a training arm later. It would train a small action-conditioned predictor on frozen V-JEPA
  2 or DINOv2 features of the 1,136 procedural training rooms, with and without actions and with short and
  long memory, then run W1-W5 on it. It is the controlled version of Gornet & Thomson's and Uria et al.'s
  models under our strict tests, at about 10-20 GPU-hours on the 5090†.

## 9. Pilot of ranks 2 and 4 (2026-10-26)

Registered as "Follow-up pilot WP" in `docs/experiments.md` before any of these models saw our walks; the full
numbers, caveats and implementation notes are there, and `results/wm_pilot/summary.json` collects them. The pilot
ran ViNT, NoMaD and GNM (rank 2) and CUT3R with TTT3R's update rule (rank 4) on the Mac and the RTX 4070 through
gq, with each family's code and weights outside this repository (`~/wm_models/` on the Mac, `/mnt/bigd/wm_models/`
on the PC) and imported by `spatialcells/wm_vint.py` and `spatialcells/wm_cut3r.py`. NWM still waits for its
license. Room A is the selection scene, so every label below is provisional.

**Competence gates (W0).**

| Model | Room A | Replica apartment 0 | Verdict |
|---|---|---|---|
| ViNT | ρ 0.570 [0.549, 0.590] | ρ 0.407 [0.381, 0.434] | competent in room A only |
| NoMaD | ρ 0.512 [0.494, 0.531] | ρ 0.382 [0.352, 0.413] | competent in room A only (narrowly) |
| GNM | ρ 0.536 [0.516, 0.556] | ρ 0.377 [0.353, 0.402] | competent in room A only |
| CUT3R | 0.99 m (gate ≤ 0.5 m) | 0.85 m | not competent on our renders |
| TTT3R (same weights, its update rule) | 0.59 m | 0.40 m | competent in the apartment only |

The navigation policies' distance heads grade distance along our room-A walks over all 20 steps, but flatten
after about 6 steps in the photoreal apartment. CUT3R misses its gate from the first frames on (1.10 m over
the first 64 frames in room A). Our adapter reproduces CUT3R's own forward pass bit for bit, so the failure is
the model's, and the same gate on the RTX 4070 gives 1.03 and 0.80 m. Exploratory: its per-step error is
0.53 m after the walks' large in-place turns (more than 30°; 21% of steps on walk 0) and 0.19 m otherwise,
against a 0.22 m step. TTT3R's update rule on the same weights cuts the error to 0.59 m in room A and 0.40 m in
the apartment and passes in the apartment, so its unit and probe tests belong in the photoreal scenes, which
needs `analyze.py`'s arena generalized (Section 3.3).

*Sub-stepped renders (amendment 1, 2026-10-26).* We then gated both models on Section 3.2's sub-stepped walks.
Every turn of more than 15° between frames is split into on-the-spot turns of at most 15°, at exact poses,
and only the 256 original steps are aligned and scored. This takes room A to about 535 frames per walk and
the apartment to about 730. The re-rendered original frames are byte-identical to the old ones. Neither
model passes in room A:

| Model (RTX 4070) | Room A, original walks | Room A, sub-stepped | Apartment 0, original | Apartment 0, sub-stepped |
|---|---|---|---|---|
| CUT3R | 1.03 m | 1.44 m | 0.80 m | 0.75 m |
| TTT3R | 0.59 m | 0.59 m | 0.40 m | 0.40 m |

Even in small pieces, a large turn still costs two to four times the per-step error of an ordinary step
(TTT3R in room A 0.47 vs 0.13 m; CUT3R 0.33 vs 0.17 m). So the size of a single rotation between frames
was not the cause. The inserted frames are pure rotations without parallax, which may be the harder part.
CUT3R does worse on the longer sequences, which fits its state's reported forgetting. The verdicts stand:
CUT3R is not competent on our renders, and TTT3R is competent only on photoreal scenes. The streaming-3D
positive control therefore cannot be tested in the 10 m arena with these weights. A follow-up that needs it
there could try camera paths that keep moving forward while they turn (so every frame has parallax), or
another streaming model, under its own registration.

**Strict heading invariance (W1), room A.** The share of place units whose rate maps from opposite 90° sectors
correlate above 0.5, at each condition's best layer:

| | along the walk | own 8-step moving average | teleport probes (256 per walk) |
|---|---|---|---|
| Path-integrating RNNs (exp. 13f) | 98-100% | | (no input on a teleport) |
| VGGT single, 8 anchors (exp. 13e) | 45% (63% with the walk) | | 36.7% (probe positions 0.41 m) |
| ViNT | 1.3% | 97.0% | 3.0% |
| NoMaD | 5.2% | 56.9% | 0.3% |
| GNM | 40.2% (frame-stacking input layer) | 69.7% at that layer | 0.4% |
| Pixels | 1% (exp. 13e) | 97% (exp. 13e) | 0.9% |

All three navigation policies are view-like, with no more place-like units than pixels. VGGT, which
localizes the same probes to 0.41 m, keeps an intermediate share on them, so the probe measure separates the
two kinds of model. GNM's 40% along the
walk comes from its input layer, which stacks the six context frames as channels, and stays below its own
moving average at that layer. ViNT hardly reads its context at all (0.1-0.2% change in its post-transformer
units and 0.004 steps in its predicted distance).

**TTT3R in Replica apartment 0 (W1 and W2; amendment 2).** TTT3R passed its gate only in apartment 0 (0.40 m;
apartments 1 and 2 0.57 m), so these tests ran there alone, on a 0.5 m grid over the apartment
(`analyze.py --arena`) with 256 new probes per walk. The room-A labels do not carry over: in the apartments,
pixels are already 18-72% strictly invariant along the walk (room A: 1%), probably because the rooms differ in
appearance from every heading. The registered rule therefore compares conditions within the apartment:

| apartment 0 | along the walk | own 8-step moving average | teleport probes |
|---|---|---|---|
| TTT3R, state carried (walk; probes-walk) | 94.6% at the state (L13), 13-31% at L3-L12 | | 39.2% (L3, 492 units) |
| TTT3R, anchors only (single; probes-single) | 1.2% | 96-99% | 1.7% (0.3% at L3) |
| VGGT single, same 8 anchors (positive reference) | 25.5% (block 19) | | 25.7% (block 19; probe positions 1.06 m) |
| Pixels | 18.1% | 100% | 19.3% |

The rule is met (L3: 38.9 points over probes-single and 19.9 over pixels), but in this scene the probe test has
not shown that it can detect a localizer: VGGT gains only 6.4 points over pixels, short of the 10-point margin,
even though its units decode probe positions across walks to 0.50-0.55 m. TTT3R's state-carried units score
higher than VGGT's, yet decode four to five times worse across walks (2.15-2.52 m). They also come from an
anchored stream that tracks the walks at only 0.77 m (0.40 m without the anchors). This fits a signature of the
walk and its region better than a code for where the probe is. W2 with the corrected decoder separates the two
models: VGGT is beyond view recognition at both blocks, and TTT3R is not at any layer (only 108 of 8,192 probes
are novel). So the pilot makes no claim that TTT3R has a place code. A walk-centred control was tried and failed
its own validation (it also removes VGGT's share), so it settles nothing.

**Behavior (W5).** On the registered lure-free trials, the distance heads choose the nearer option no better
than base Qwen3-VL-8B: novel trials in the ten held-out rooms 51.5% (ViNT), 47.3% (NoMaD) and 53.5% (GNM;
0.12 after Holm), against the LoRA model's 92.5% and Opus's 69.2%. The walk adds nothing (walk minus no walk
0.0, −1.6 and −0.4 points).

**What the pilot changes in the plan.**
- The W2 novelty cost, as defined in Section 4.2, is confounded: familiarity is taken against the probe's own
  walk, which a decoder trained on other walks never saw. It came out negative for every single-frame
  representation, pixels included. Define familiarity against the decoder's training frames, or decode from
  the probe's own walk, before it is used for any claim. *Fixed by amendment 2 (2026-10-26):* the first option
  fails in room A, where the training walks cover the room so densely that only 6 of 6,144 test probes are novel.
  The second works: each walk's probes are decoded by a ridge trained on that walk's own frames
  (`wm_pilot novelty --own-walk`). On room A, pixels pay a novelty cost of +1.30 m [1.11, 1.52], the recall
  baseline +1.40 m, and VGGT single (block 19) +0.18 m [0.15, 0.21], with novel probes at 0.37 m. ViNT, NoMaD and
  GNM pay +0.58 to +0.73 m, but only because they decode badly everywhere: their novel-probe errors (2.7-2.8 m)
  are worse than the recall baseline's (2.12 m), so none is beyond view recognition, and VGGT is.
  W2 now uses this decoder.
- The competence gates did their job: they kept a model that cannot track our walks (CUT3R) out of the tests
  whose nulls would otherwise have been read as evidence.
- The 256-per-walk probe arena works with `analyze.py` unchanged (pixels on the probes: 77% place units, 0.9%
  strictly invariant), so the W1 probe measure can be used for every model.
- Still to do: NWM (waits for its license). CUT3R and TTT3R on sub-stepped renders: done (amendment 1); neither
  passes in room A. TTT3R in the apartments: done (amendment 2, above). Separating a walk signature from a
  probe-position code would take a swap control (probes read against another walk's final state) or anchors that
  TTT3R can track; neither is registered yet.
