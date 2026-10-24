# Plan: photoreal rooms (item 4)

Written 2026-10-23, before any building. The goal is to repeat the behavioural test, VGGT
localisation and, if time allows, the place-cell analyses, in 2-3 real scanned rooms instead of
the ray-cast box.

## Choices

- **Dataset: Replica** (Meta, 18 scanned rooms, offices and apartments). It is the only photoreal
  scan set that has all three things we need: per-object instance labels (for the no-shared-landmark
  rule), a navmesh (for walks), and no approval step. Its licence (Replica Research Terms, in the
  repository's `LICENSE`) allows non-commercial research use and publishing results and figures.
  **You need to read and accept it**; there is no form. Using the download script binds you to it.
- **Pilot: ReplicaCAD** (CC BY 4.0, under 1 GB, nothing to accept). It is an artist-made replica of
  one Replica apartment, used to debug the pipeline before the real scans arrive.
- **Not now: HM3D.** It is larger and has house-scale scenes, but Matterport has stopped free
  accounts from requesting access since 8 Oct 2026, expected back in November (a user report on the
  dataset's GitHub). It can be added later if it reopens. MP3D, Gibson and ScanNet++ need forms or
  applications and look worse.
- **Renderer: Habitat-Sim** on the RTX 4070, installed per user with Miniforge (no sudo, nothing
  system-wide). The current release (0.3.3) only renders Replica from its vertex-coloured mesh,
  which is visibly blurrier. Full-quality Replica textures (PTex) need the older 0.2.4, so there are
  two environments: 0.2.4 for Replica and 0.3.3 for ReplicaCAD. If 0.2.4 will not run on the
  4070's new driver, the fallback is the vertex-coloured mesh, a resolution caveat in the paper.

## Status (2026-10-23)

- Done: steps 1-4.
  - Miniforge with `hab024` (0.2.4) and `hab033` (0.3.3) on the 4070, plus `libopengl` (the host
    lacks `libOpenGL.so.0`).
  - ReplicaCAD and 9 Replica room scenes are in `~/habitat-data` (12 GB).
  - Headless EGL rendering works: 1,400-3,600 fps at 256 x 256 for RGB, depth and instance labels.
- **No PTex.** habitat-sim 0.2.4's PTex shader fails to link on the 4070's driver (595), so
  Replica is rendered from its vertex-coloured mesh in 0.3.3. Views are somewhat soft but
  recognisable. This needs a caveat.
- **Frame fix.** In 0.3.3 the Replica stage loads in its own Z-up frame while its shipped navmesh
  is Y-up: navmesh (X, Y, Z) is scene (X, -Z, Y). That frame is our room's (x east, y north,
  z up), so `habitat_scene.py --zup` poses the camera there. Holes went from 11-54% of pixels
  (cameras misplaced) to 0%.
- **Scenes** (`results/preview/replica_survey_zup/survey.json`, 300 random views each): room_0
  (17.1 m^2, 7.8 x 4.7 m, 62 landmark objects), office_4 (19.1 m^2, 6.5 x 6.5 m, 41) and office_3
  (17.8 m^2, 8.6 x 9.2 m, 78; 2.4% hole pixels).
- Next: step 5, walks and probes on the navmesh.
- **Scope (agreed with the user, 2026-10-23):** this version covers steps 1-8 only, the behaviour
  tests and VGGT localisation in 2-3 scenes. In the paper that is one figure and one paragraph.
  Step 9 (the place-cell analyses in photoreal scenes) is left for a later version.

## Steps

| Step | Where | Time |
|---|---|---|
| 1. Miniforge, the two Habitat environments, a test render (headless EGL) | 4070, CPU and disk | ~0.5 h |
| 2. ReplicaCAD pilot: pose-driven rendering at 256 × 256, 90 deg HFOV, camera at 1 m; RGB, depth and instance labels; Habitat's Y-up frame converted to our (x, y, theta) | 4070 | ~1 h |
| 3. Download Replica, extracting only the chosen scenes from the stream (the full archive is 34 GB; the 4070's disk is shared) | 4070 | 0.5-2 h, mostly waiting |
| 4. Choose 2-3 scenes by single-room navigable area (candidates room_0, room_2, office_3) and render anchors | 4070 | ~0.5 h |
| 5. Port the walk and probe generators to the navmesh: same step length, heading drift and head scans, steps restricted to the largest navigable island; 16-32 walks plus 64 probes per walk, as in experiment 10 | Mac (code), 4070 (rendering, minutes) | 3-5 h |
| 6. Lure-free trials from instance labels: a landmark is any object instance (not wall, floor or ceiling) covering at least 2% of the image, the equivalent of 8 of 64 x 64 rays. The near/far, heading and cue-balancing rules are as before. **Pre-registered before any model sees a scene**; NEAR and FAR may need scaling to the room size, decided from the geometry alone. | Mac | 2-3 h |
| 7. Behaviour: base Qwen3-VL-8B and the LoRA model (walk, no walk, shuffled) on the Mac, about 70 min a scene; Opus (walk, no walk) through the API, about $6 a scene | Mac, API | ~4 h of runs |
| 8. VGGT localisation per scene (the positive control) | 4070 | ~1-2 h |
| 9. Only if time allows: the place-cell, heading-invariance and teleport-probe analyses (Qwen activations on the Mac, VGGT on the 4070) | both | 1-2 days |

**Total to first-pass results (steps 1-8): about 3-4 working days, plus a day if the 0.2.4 PTex
build fails.** This is well inside the late-January target. GPU steps on the 4070 queue on its shared
GPU lock with other projects' jobs.

## Costs

No compute rental. Opus is about $6 per scene, $12-18 for 2-3 scenes, from the Claude budget
($138 until next month, about $62 of it allotted to experiment 10). The Qwen and VGGT runs cost
nothing.

## Risks

- **Scale.** Replica rooms are smaller than our 10 m box (several metres across). The 1 m / 3 m
  near/far rule may leave too few trials and may need scaling, which is decided before any model
  sees a scene.
- **Distribution.** The LoRA model was trained only on procedural ray-cast rooms, so photoreal
  scans are a real test of transfer. A drop there is a finding, not a bug.
- **Lighting and texture.** Real scans have reflections and holes. Trials whose views are mostly
  holes are dropped by a rule fixed before any model sees them.
- **Shared 4070.** Rendering is short. Disk is kept small by extracting only the chosen scenes.
