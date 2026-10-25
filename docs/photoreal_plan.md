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

## HSSD houses (experiment 16, prepared 2026-10-25)

- **Data.** `scripts/rtx4070/get_hssd.sh` downloads `hssd/hssd-hab` at revision `4369cb98` into
  `/mnt/bigd/mechinterp/hssd-hab` (linked as `~/habitat-data/hssd-hab`), file by file, because
  anonymous Hugging Face requests may resolve only 3,000 files per 5 minutes (about 1.5 h for the
  ~57k files). It leaves out `urdf/`, `scenes-articulated/`, `scenes-uncluttered/` and
  `scene_filter_files/`, which the default scenes never use, and fetches the objects of the first 24
  houses of the draw first. The default scene set has 168 houses, not the 211 of the HSSD paper, and
  the draw is over those 168.
- **Frame and navmesh.** HSSD loads Y-up like its navmesh, so no `--zup`. No navmeshes ship, so
  `habitat_scene.compute_navmesh` builds one with experiment 12's agent (habitat-sim's defaults: 1.5 m
  tall, radius 0.1 m, climb 0.2 m, slope 45 deg, which Replica's shipped navmeshes carry) and the
  scene's static objects as obstacles, since HSSD's furniture are separate objects rather than part
  of a scanned mesh. It takes about 4 s a house.
- **Instance labels.** HSSD's semantic ids are per category (from each object's config), so they
  cannot separate two chairs. With `--hssd`, the instance sensor renders each object's own id
  instead (`CameraSensorSpec.semantic_target = OBJECT_ID`). Objects get ids 1..N in a house, and the
  stage (walls, floors and ceilings) gets 0. `hssd_classes()` names each object from
  `metadata/fpmodels-with-decomposed.csv` (its `main_category`, or its WordNet synsets when that is
  empty). Doors, windows, carpets, stairs, beams and wall panels are mapped onto experiment 12's
  excluded classes (door, window, rug, stair, beam, panel), and the stage counts as structure.
  Every other object is a possible landmark, including those with neither a category nor a synset
  (7% of the objects in the eight accepted houses).
- **Acceptance checks** (`habitat_scene.scene_check`, run before rendering, written to each walk
  folder's `scene.json`). A house *loads* if every object instance in its scene file is created. It
  *has instance labels* if, over 64 random views on the largest island, at least one possible
  landmark covers 2% or more of a view and the only id that is not an object is the stage's 0. The
  draw itself (`spatialcells/hssd_select.py`, run by `scripts/rtx4070/render_hssd.sh` under the GPU
  lock) renders each house as in experiment 12 (seed 1600 plus its place in the draw), builds its
  trials and counts them. Skipped renders are kept in `/mnt/bigd/mechinterp/hssd-skipped/`.
- **Preview** (`results/preview/hssd_106366371_174226743*.png`, the first house drawn). That
  house's largest island is its yard (327 m^2, sky = holes); its second is the interior (317 m^2, 0%
  holes, ceilings present, plain but sensible lighting); its third is a flat roof. Yards and roofs
  are screened out by the island-size rule or by the 5% hole rule (one drawn house kept only 10
  trials for that reason). Rendering runs at about 230 fps for RGB, depth and instance labels, and a
  whole house (12,288 views plus its trials) takes about 1 minute.
- **Result** (`results/hssd_selection.json`; 13 drawn, 8 accepted, 2026-10-25): HSSD1-8 =
  104862639_172226823, 106879080_174887211, 107734176_176000019, 102816786, 106878960_174887073,
  103997424_171030444, 104862513_172226580, 105515505_173104584. Five were skipped: three had a
  largest island outside 40-250 m^2 (327, 866 and 723 m^2), one had an island of only 35.6 m^2, and one
  (108294846_176710506, whose largest island is outdoors) gave only 10 trials. The trials were built
  on the 4070. Rebuilt on the Mac, they come out the same, except that `sim_best_frame` differs in the
  4th decimal in one trial each of HSSD4 and HSSD5.
  Trials (all / novel): HSSD1 188/68, HSSD2 192/53, HSSD3 180/38, HSSD4 192/38, HSSD5 156/36,
  HSSD6 170/40, HSSD7 180/55, HSSD8 192/38; 1,450 in all, 366 of them novel. Largest islands are
  57-128 m^2 and the walks span 11-20 m. The walks (`data/walks-roomHSSD<k>/`) and trial exports
  (`results/export/dissimilar-roomHSSD<k>/`) were copied to the Mac, and all 1,507 files match the
  4070's SHA-256 sums. The Claude walk-frame JPEGs (`results/export/claude-roomHSSD<k>/`) were made
  on the Mac from the copied frames.

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
