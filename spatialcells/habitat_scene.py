"""Render first-person views of a Habitat-Sim scene at given poses, for the photoreal rooms
(docs/photoreal_plan.md). Runs in the 4070 host's Habitat environments (Python 3.9), not the
project venv; GPU use goes through scripts/rtx4070/run.sh, which holds the shared lock.

    ~/miniforge3/envs/hab033/bin/python -m spatialcells.habitat_scene --pilot \
        --dataset ~/habitat-data/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json \
        --scene Baked_sc0_staging_00 --out results/preview/habitat_pilot

The camera matches the ray-cast room's: 256 x 256, 90 deg horizontal FOV, 1 m above the floor.
Poses are (x, y, theta) on the floor, as in room.py: x east, y north, theta counterclockwise from
+x. Habitat is Y-up, with the camera looking down -Z, so x = X and y = -Z, and the camera is turned
by theta - 90 deg about +Y. --pilot reports the navmesh, renders RGB, depth and instance labels at
four headings from a few navigable points (a PNG grid), and times 200 renders.

HSSD (experiment 16; --hssd, with --dataset .../hssd-hab/hssd-hab.scene_dataset_config.json and a
scene id) is Y-up like the navmesh, so no --zup. It ships no navmesh, so one is computed with
experiment 12's agent (compute_navmesh). Its semantic ids are per category, not per object, so the
instance sensor renders each object's own id instead (OBJECT_ID target), and hssd_classes() names
the objects from HSSD's metadata; the stage (walls, floors, ceilings) is not an object.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

RES, HFOV, CAM_H = 256, 90.0, 1.0


def make_sim(dataset, scene, semantic=True, navmesh=None, zup=False, object_ids=False, compute=False):
    """dataset: a scene_dataset_config.json, or None to load `scene` as a file (e.g. Replica's
    textured mesh.ply, which only habitat-sim <= 0.2.4 renders); navmesh: a .navmesh to load.
    object_ids: the instance sensor shows each object's id (HSSD) rather than its semantic id.
    compute: with no navmesh shipped or given, compute one (compute_navmesh). sim.navmesh_source
    says which it is (file, shipped, computed or none)."""
    import habitat_sim
    import magnum as mn
    cfg = habitat_sim.SimulatorConfiguration()
    if dataset:
        cfg.scene_dataset_config_file = os.path.expanduser(dataset)
    cfg.scene_id = os.path.expanduser(scene)
    cfg.gpu_device_id = 0
    cfg.enable_physics = False
    specs = []
    kinds = [("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)]
    if semantic:
        kinds.append(("instance", habitat_sim.SensorType.SEMANTIC))
    for uuid, kind in kinds:
        s = habitat_sim.CameraSensorSpec()
        s.uuid, s.sensor_type, s.resolution = uuid, kind, [RES, RES]
        s.hfov = mn.Deg(HFOV)
        s.position = [0.0, CAM_H, 0.0]
        if uuid == "instance" and object_ids:
            s.semantic_target = type(s.semantic_target).OBJECT_ID
        specs.append(s)
    agent = habitat_sim.agent.AgentConfiguration(sensor_specifications=specs)
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
    if navmesh:
        sim.pathfinder.load_nav_mesh(os.path.expanduser(navmesh))
    sim.navmesh_source = "file" if navmesh else "shipped"
    if not sim.pathfinder.is_loaded:
        sim.navmesh_source = "computed" if compute and compute_navmesh(sim) else "none"
    sim.zup = zup
    return sim


def compute_navmesh(sim):
    """A navmesh for a scene that ships none (HSSD), with experiment 12's agent: Replica's shipped
    navmeshes carry habitat-sim's default settings (agent 1.5 m tall, radius 0.1 m, climb 0.2 m,
    slope 45 deg), as in survey(). HSSD's furniture are separate static objects, not part of the
    stage as in a scan, so they are included as obstacles."""
    import habitat_sim
    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    ns.agent_height, ns.agent_radius = 1.5, 0.1
    ns.include_static_objects = True
    return sim.recompute_navmesh(sim.pathfinder, ns) and sim.pathfinder.is_loaded


def to_habitat(x, y, theta, floor, zup=False):
    """Our floor pose -> Habitat agent position (on the floor at height `floor`) and rotation.
    zup: the scene was loaded in its own Z-up frame (Replica in habitat-sim 0.3.3, whose shipped
    navmesh is nevertheless Y-up). That is our own frame (x east, y north, z up), so the agent sits at
    (x, y, floor) and is turned so that its local +Y (where the camera's 1 m offset points) is world +Z
    and the camera looks along (cos theta, sin theta, 0)."""
    from habitat_sim.utils.common import quat_from_angle_axis
    if zup:
        level = quat_from_angle_axis(np.pi / 2, np.array([1.0, 0.0, 0.0]))
        return (np.array([x, y, floor], dtype=np.float32),
                quat_from_angle_axis(theta - np.pi / 2, np.array([0.0, 0.0, 1.0])) * level)
    return np.array([x, floor, -y], dtype=np.float32), quat_from_angle_axis(theta - np.pi / 2, np.array([0.0, 1.0, 0.0]))


def from_habitat(p):
    """Habitat point (X, Y, Z) -> our (x, y) and the floor height Y."""
    return float(p[0]), float(-p[2]), float(p[1])


def render(sim, x, y, theta, floor):
    import habitat_sim
    st = habitat_sim.AgentState()
    st.position, st.rotation = to_habitat(x, y, theta, floor, getattr(sim, "zup", False))
    sim.get_agent(0).set_state(st, reset_sensors=True)
    return sim.get_sensor_observations()


def pilot(args):
    from PIL import Image
    import habitat_sim
    sim = make_sim(args.dataset, args.scene, semantic=args.semantic, navmesh=args.navmesh, zup=args.zup,
                   object_ids=args.hssd, compute=args.hssd)
    pf = sim.pathfinder
    islands = (sorted(((pf.island_area(i), i) for i in range(pf.num_islands)), reverse=True)
               if hasattr(pf, "num_islands") and hasattr(pf, "island_area") else [])
    lo, hi = pf.get_bounds()
    info = {"scene": args.scene, "habitat_sim": habitat_sim.__version__, "navmesh": sim.navmesh_source,
            "navigable_area_m2": round(pf.navigable_area, 2),
            "islands_m2": [round(a, 2) for a, _ in islands[:5]],
            "bounds_x_m": [round(float(lo[0]), 2), round(float(hi[0]), 2)], "bounds_y_m": [round(float(-hi[2]), 2), round(float(-lo[2]), 2)]}
    pf.seed(0)
    big = islands[0][1] if islands else None
    rows, ids = [], set()
    for _ in range(3):
        p = pf.get_random_navigable_point(island_index=big) if big is not None else pf.get_random_navigable_point()
        x, y, floor = from_habitat(p)
        row = []
        for k in range(4):
            obs = render(sim, x, y, k * np.pi / 2, floor)
            rgb = obs["rgb"][..., :3]
            d = obs["depth"]
            info.setdefault("hole_share", []).append(round(float((d <= 0).mean()), 3))
            depth = (255 * np.clip(d / 6.0, 0, 1)).astype(np.uint8)[..., None].repeat(3, -1)
            panels = [rgb, depth]
            if args.semantic:
                inst = obs["instance"].astype(np.int64)
                ids |= set(np.unique(inst).tolist())
                panels.append(np.stack([inst * 37 % 255, inst * 91 % 255, inst * 53 % 255], -1).astype(np.uint8))
            row.append(np.concatenate(panels, 0))
        rows.append(np.concatenate(row, 1))
        info.setdefault("points", []).append([round(x, 2), round(y, 2), round(floor, 2)])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Image.fromarray(np.concatenate(rows, 0)).save(args.out + ".png")
    t0 = time.time()
    for i in range(200):
        render(sim, x, y, i * 0.03, floor)
    info.update(fps_rgb_depth_instance=round(200 / (time.time() - t0), 1), instance_ids_seen=len(ids),
                rgb_mean=float(np.mean(rows[0][:RES])))
    json.dump(info, open(args.out + ".json", "w"), indent=1)
    print(json.dumps(info))


STRUCTURE = {"wall", "floor", "ceiling", "undefined", "window", "door", "beam", "stair", "rug", "panel", "base-cabinet"}


def survey(args):
    """Choose scenes (docs/photoreal_plan.md step 4). For each scene: walkable area and bounds, and
    over N random views (largest navigable island, random heading) the share of pixels with no
    geometry (depth 0: holes and open sides of the scan). Also the number of labelled objects that
    could act as landmarks (not walls, floors and the like). Writes <out>/<scene>.png (16 views)
    and <out>/survey.json."""
    import habitat_sim
    from PIL import Image
    os.makedirs(args.out, exist_ok=True)
    rows = {}
    for scene in args.scenes:
        sim = make_sim(args.dataset, scene, semantic=True, zup=args.zup)
        pf = sim.pathfinder
        nav_lo, nav_hi = pf.get_bounds()
        bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
        if args.recompute_navmesh:       # rebuild the navmesh from the scene as loaded, instead of the shipped file
            ns = habitat_sim.NavMeshSettings()
            ns.set_defaults()
            ns.agent_height, ns.agent_radius = 1.5, 0.1
            sim.recompute_navmesh(pf, ns)
        islands = sorted(((pf.island_area(i), i) for i in range(pf.num_islands)), reverse=True)
        pf.seed(0)
        rng = np.random.default_rng(0)
        holes, ids, views = [], set(), []
        for k in range(args.n):
            x, y, floor = from_habitat(pf.get_random_navigable_point(island_index=islands[0][1]))
            obs = render(sim, x, y, rng.uniform(0, 2 * np.pi), floor)
            holes.append(float((obs["depth"] <= 0).mean()))
            ids |= set(np.unique(obs["instance"]).tolist())
            if k < 16:
                views.append(obs["rgb"][..., :3])
        Image.fromarray(np.concatenate([np.concatenate(views[i:i + 4], 1) for i in range(0, 16, 4)], 0)).save(f"{args.out}/{scene}.png")
        info = f"{os.path.dirname(os.path.expanduser(args.dataset))}/{scene}/habitat/info_semantic.json"
        objs = json.load(open(info)).get("objects", []) if os.path.exists(info) else []
        landmarks = [o for o in objs if o.get("class_name", "").lower() not in STRUCTURE]
        lo, hi = pf.get_bounds()
        h = np.array(holes)
        rows[scene] = {"navigable_m2": round(pf.navigable_area, 1), "largest_island_m2": round(islands[0][0], 1),
                       "extent_m": [round(float(hi[0] - lo[0]), 1), round(float(hi[2] - lo[2]), 1)],
                       "hole_share_mean": round(float(h.mean()), 3), "views_over_5pct_holes": round(float((h > 0.05).mean()), 3),
                       "instances_seen": len(ids), "landmark_objects": len(landmarks), "habitat_sim": habitat_sim.__version__,
                       "navmesh_recomputed": bool(args.recompute_navmesh),
                       "shipped_navmesh_bounds": [[round(float(v), 2) for v in nav_lo], [round(float(v), 2) for v in nav_hi]],
                       "scene_bounds": [[round(float(v), 2) for v in bb.min], [round(float(v), 2) for v in bb.max]]}
        print(scene, json.dumps(rows[scene]), flush=True)
        sim.close()
    json.dump(rows, open(f"{args.out}/survey.json", "w"), indent=1)


MARGIN, LANDMARK_SHARE = 0.35, 0.02     # as room.py's walks; a landmark covers >= 2% of a view (8 of 64 x 64 rays)


def nav(x, y, floor):
    """Our (x, y) on the floor -> a point in the navmesh's Y-up frame."""
    return np.array([x, floor, -y], dtype=np.float32)


def walkable(pf, x, y, floor, island, margin=MARGIN):
    p = nav(x, y, floor)
    return (pf.is_navigable(p, 0.5) and pf.get_island(p) == island
            and pf.distance_to_closest_obstacle(p, 2 * margin) >= margin)


def nav_walk(pf, rng, n, island, floor, margin=MARGIN, step=0.22, turn_sd=np.deg2rad(15), p_scan=0.15):
    """room.foraging_walk on a navmesh: the same step, heading drift, head scans and turning away
    from obstacles; positions stay on the largest island, MARGIN from obstacles."""
    while True:
        x, y, _ = from_habitat(pf.get_random_navigable_point(island_index=island))
        if walkable(pf, x, y, floor, island, margin):
            break
    th = rng.uniform(0, 2 * np.pi)
    out = np.zeros((n, 3))
    for i in range(n):
        out[i] = x, y, th % (2 * np.pi)
        if rng.random() < p_scan:
            th += rng.choice([-1, 1]) * rng.uniform(np.deg2rad(30), np.deg2rad(90))
            continue
        th += rng.normal(0, turn_sd)
        for _ in range(20):
            s = step * rng.uniform(0.7, 1.3)
            nx, ny = x + s * np.cos(th), y + s * np.sin(th)
            if walkable(pf, nx, ny, floor, island, margin):
                x, y = nx, ny
                break
            th += rng.choice([-1, 1]) * rng.uniform(np.deg2rad(60), np.deg2rad(180))
    return out


# HSSD's names for the kinds of structure STRUCTURE excludes (doors, windows, rugs, stairs, beams, wall
# panels): an object's main_category in HSSD's metadata, or, without one, its WordNet synsets. HSSD has no
# built-in base cabinets. Walls, floors and ceilings are the stage, which is not an object.
HSSD_STRUCTURE = {"door": "door", "window": "window", "carpet": "rug", "stairway.n.01": "stair", "stairs.n.01": "stair",
                  "beam.n.02": "beam", "wall_panel.n.01": "panel"}
ISLAND_M2 = (40.0, 250.0)       # experiment 16: the largest navigable island's area, m^2


def hssd_classes(sim, root):
    """HSSD, instance sensor on object ids: object id -> class, from metadata/fpmodels-with-decomposed.csv
    (by the object's template). A kind of structure gets STRUCTURE's name (HSSD_STRUCTURE); any other
    object, labelled or not, is a possible landmark, named by its main_category, synsets or "object". Ids
    that are not objects (the stage) are absent, so they count as structure ("undefined")."""
    import csv
    meta = {r["id"]: r for r in csv.DictReader(open(f"{root}/metadata/fpmodels-with-decomposed.csv"))}
    out = {}
    for obj in sim.get_rigid_object_manager().get_objects_by_handle_substring().values():
        r = meta.get(os.path.basename(obj.creation_attributes.handle).split(".")[0], {})
        names = [r["main_category"]] if r.get("main_category") else [n for n in r.get("wnsynsetkey", "").split(",") if n]
        out[int(obj.object_id)] = next((HSSD_STRUCTURE[n] for n in names if n in HSSD_STRUCTURE), names[0] if names else "object")
    return out


def scene_check(sim, classes, root, args, n=64):
    """Experiment 16's acceptance checks for an HSSD house, before anything is rendered. `skip` names the
    first that fails, else None:
    - loads: every object instance in the scene file was created;
    - navmesh: shipped or computed (compute_navmesh), with at least one island;
    - island: the largest navigable island covers ISLAND_M2 (--island-range);
    - labels: over n random views on that island (seed 0), the instance image shows object ids, at least
      one possible landmark (an object not in STRUCTURE) covers >= 2% of a view, and at most one id that
      is not an object appears (the stage's)."""
    pf = sim.pathfinder
    want = len(json.load(open(f"{root}/scenes/{args.scene}.scene_instance.json")).get("object_instances", []))
    islands = sorted(((pf.island_area(i), i) for i in range(pf.num_islands)), reverse=True) if pf.is_loaded else []
    out = {"scene": args.scene, "seed": args.seed, "objects_in_scene_file": want, "objects_created": len(classes),
           "navmesh": sim.navmesh_source, "islands_m2": [round(a, 1) for a, _ in islands[:5]],
           "largest_island_m2": round(islands[0][0], 1) if islands else 0.0, "island_range_m2": list(args.island_range)}
    seen, marks, other, share = set(), set(), set(), 0.0
    if islands:
        pf.seed(0)
        rng = np.random.default_rng(0)
        for _ in range(n):
            x, y, floor = from_habitat(pf.get_random_navigable_point(island_index=islands[0][1]))
            inst = render(sim, x, y, rng.uniform(0, 2 * np.pi), floor)["instance"]
            for i, c in zip(*(a.tolist() for a in np.unique(inst, return_counts=True))):
                if i in classes:
                    seen.add(i)
                    share += c / inst.size / n
                    if c >= LANDMARK_SHARE * inst.size and classes[i] not in STRUCTURE:
                        marks.add(i)
                else:
                    other.add(i)
    kinds = {}
    for c in classes.values():
        kinds[c] = kinds.get(c, 0) + 1
    out.update(label_views=n if islands else 0, objects_seen=len(seen), object_pixel_share=round(share, 3),
               landmark_objects_seen=len(marks), non_object_ids_seen=sorted(other),
               structure_objects={k: kinds[k] for k in sorted(kinds) if k in STRUCTURE},
               unnamed_objects=kinds.get("object", 0))
    lo, hi = args.island_range
    out["skip"] = (f"loads: {len(classes)} of {want} objects created" if len(classes) < want else
                   "navmesh: none" if not islands else
                   f"island: largest {islands[0][0]:.1f} m^2, not {lo:g}-{hi:g}" if not lo <= islands[0][0] <= hi else
                   f"labels: {len(marks)} possible landmarks seen, non-object ids {sorted(other)}"
                   if not marks or len(other) > 1 else None)
    return out


def walks(args):
    """Walks and teleport probes in a Habitat scene, in the files render_data.py and
    render_probes.py write (frames.npy, meta.npz, probes.npy, probes_meta.npz), so that
    dissimilar_trials.py, two_choice.py and claude_test.py read them unchanged. Also:
    landmarks.json, the labelled objects (not walls, floors and the like) covering >= 2% of each
    view, for the no-shared-landmark rule in place of ray casting; and each view's share of pixels
    with no geometry (`holes`)."""
    sim = make_sim(args.dataset, args.scene, semantic=True, zup=args.zup, object_ids=args.hssd, compute=args.hssd)
    pf = sim.pathfinder
    root = os.path.dirname(os.path.expanduser(args.dataset))
    if args.hssd:
        classes = hssd_classes(sim, root)
        check = scene_check(sim, classes, root, args)
        os.makedirs(args.out, exist_ok=True)
        json.dump(check, open(f"{args.out}/scene.json", "w"), indent=1)
        if check["skip"]:
            print(json.dumps(check), flush=True)
            sys.exit(3)
    else:
        info = f"{root}/{args.scene}/habitat/info_semantic.json"
        classes = {o["id"]: o.get("class_name", "").lower() for o in json.load(open(info)).get("objects", [])} if os.path.exists(info) else {}
    pf.seed(args.seed)
    island = max(range(pf.num_islands), key=pf.island_area)
    floor = from_habitat(pf.get_random_navigable_point(island_index=island))[2]
    rng = np.random.default_rng(args.seed)

    def shoot(poses):
        imgs, marks, holes = [], [], []
        for x, y, th in poses:
            f = from_habitat(pf.snap_point(nav(x, y, floor)))[2]
            obs = render(sim, x, y, th, f)
            imgs.append(obs["rgb"][..., :3].copy())
            ids, counts = np.unique(obs["instance"], return_counts=True)
            marks.append(sorted(int(i) for i, c in zip(ids, counts)
                                if c >= LANDMARK_SHARE * obs["instance"].size and classes.get(int(i), "undefined") not in STRUCTURE))
            holes.append(float((obs["depth"] <= 0).mean()))
        return np.stack(imgs), marks, np.array(holes)

    poses = np.stack([nav_walk(pf, rng, args.steps, island, floor, args.margin) for _ in range(args.walks)])
    flat = poses.reshape(-1, 3)
    frames, marks, holes = shoot(flat)
    os.makedirs(args.out, exist_ok=True)
    walk, step = np.meshgrid(np.arange(args.walks), np.arange(args.steps), indexing="ij")
    np.save(f"{args.out}/frames.npy", frames)
    np.savez(f"{args.out}/meta.npz", walk=walk.ravel(), step=step.ravel(), x=flat[:, 0], y=flat[:, 1], theta=flat[:, 2], holes=holes)
    rows = []
    for w in range(args.walks):                     # probes as render_probes.py: near visited places, random heading
        for k in range(args.probes_per_walk):
            while True:
                x, y = poses[w, rng.integers(args.steps), :2] + rng.normal(0, 0.15, 2)
                if walkable(pf, x, y, floor, island, args.margin):
                    break
            rows.append((w, k, x, y, rng.uniform(0, 2 * np.pi)))
    rows = np.array(rows)
    probes, pmarks, pholes = shoot(rows[:, 2:])
    np.save(f"{args.out}/probes.npy", probes)
    np.savez(f"{args.out}/probes_meta.npz", walk=rows[:, 0].astype(int), idx=rows[:, 1].astype(int),
             x=rows[:, 2], y=rows[:, 3], theta=rows[:, 4], holes=pholes)
    json.dump({"frames": marks, "probes": pmarks, "landmark_share": LANDMARK_SHARE, "scene": args.scene,
               "excluded_classes": sorted(STRUCTURE),
               **({"hssd_structure": HSSD_STRUCTURE, "classes": {str(k): v for k, v in sorted(classes.items())}} if args.hssd else {})},
              open(f"{args.out}/landmarks.json", "w"))
    print(json.dumps({"scene": args.scene, "frames": len(frames), "probes": len(probes),
                      "extent_m": [round(float(np.ptp(flat[:, 0])), 2), round(float(np.ptp(flat[:, 1])), 2)],
                      "frame_holes_over_5pct": round(float((holes > 0.05).mean()), 3),
                      "probe_holes_over_5pct": round(float((pholes > 0.05).mean()), 3),
                      "landmarks_per_view": round(float(np.mean([len(m) for m in marks])), 2)}), flush=True)


def anchor_poses(xy):
    """Eight anchor poses for a scene, by room A's pattern (vggt_cells.ANCHORS: the centre facing E, N, W, S, then
    spots 3 m in x and y from it facing it, SW, SE, NE, NW), on walk positions so that each is navigable: the centre is
    the walk position nearest the median of all of them, each other spot the walk position nearest centre + (+-3, +-3)."""
    c = xy[np.argmin(np.linalg.norm(xy - np.median(xy, 0), axis=1))]
    poses = [(c[0], c[1], th) for th in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)]
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        p = xy[np.argmin(np.linalg.norm(xy - (c + 3.0 * np.array([sx, sy])), axis=1))]
        poses.append((p[0], p[1], float(np.arctan2(c[1] - p[1], c[0] - p[0]) % (2 * np.pi))))
    return np.array(poses)


def probes_for(args):
    """Follow-up pilot WP, amendment 2: for walks rendered earlier by walks() (--probes-for SRC), new teleport probes
    (--probes-per-walk per walk, drawn as walks() draws them, with their own seed --probe-seed) into --probes-out, which
    links SRC's frames.npy and meta.npz as data/walks-p256 does room A's, and the scene's 8 anchor views (anchor_poses)
    into SRC/anchors.npz, linked from --probes-out. The pathfinder is seeded with the walks' --seed, so the floor height
    is walks()'s. The first 16 frames of walk 0 are rendered again and must equal SRC's byte for byte (exit 4 if not)."""
    src, out = args.probes_for, args.probes_out
    meta = dict(np.load(f"{src}/meta.npz"))
    sim = make_sim(args.dataset, args.scene, semantic=True, zup=args.zup)      # walks()'s sensors
    pf = sim.pathfinder
    pf.seed(args.seed)
    island = max(range(pf.num_islands), key=pf.island_area)
    floor = from_habitat(pf.get_random_navigable_point(island_index=island))[2]

    def shoot(poses):
        imgs, holes = [], []
        for x, y, th in poses:
            obs = render(sim, x, y, th, from_habitat(pf.snap_point(nav(x, y, floor)))[2])
            imgs.append(obs["rgb"][..., :3].copy())
            holes.append(float((obs["depth"] <= 0).mean()))
        return np.stack(imgs), np.array(holes)

    rows0 = np.flatnonzero(meta["walk"] == 0)[:16]
    again, _ = shoot(np.stack([meta["x"][rows0], meta["y"][rows0], meta["theta"][rows0]], 1))
    same = bool((again == np.load(f"{src}/frames.npy", mmap_mode="r")[rows0]).all())
    if not same:
        print(json.dumps({"scene": args.scene, "rerender_identical": False}), flush=True)
        sys.exit(4)
    os.makedirs(out, exist_ok=True)
    A = anchor_poses(np.stack([meta["x"], meta["y"]], 1))
    aimg, ahol = shoot(A)
    np.savez(f"{src}/anchors.npz", images=aimg, poses=A, holes=ahol)
    rel = os.path.relpath(src, out)
    for f in ("frames.npy", "meta.npz", "anchors.npz"):
        if not os.path.lexists(f"{out}/{f}"):
            os.symlink(f"{rel}/{f}", f"{out}/{f}")
    rng = np.random.default_rng(args.probe_seed)
    walks = np.unique(meta["walk"])
    poses = {w: np.stack([meta[k][meta["walk"] == w] for k in ("x", "y")], 1) for w in walks}
    rows = []
    for w in walks:                     # as walks(): near a visited place of the walk, random heading
        for k in range(args.probes_per_walk):
            while True:
                x, y = poses[w][rng.integers(len(poses[w]))] + rng.normal(0, 0.15, 2)
                if walkable(pf, x, y, floor, island, args.margin):
                    break
            rows.append((w, k, x, y, rng.uniform(0, 2 * np.pi)))
    rows = np.array(rows)
    probes, pholes = shoot(rows[:, 2:])
    np.save(f"{out}/probes.npy", probes)
    np.savez(f"{out}/probes_meta.npz", walk=rows[:, 0].astype(int), idx=rows[:, 1].astype(int),
             x=rows[:, 2], y=rows[:, 3], theta=rows[:, 4], holes=pholes)
    print(json.dumps({"scene": args.scene, "rerender_identical": same, "probes": len(probes),
                      "probe_holes_over_5pct": round(float((pholes > 0.05).mean()), 3),
                      "anchors": [[round(float(v), 3) for v in a] for a in A],
                      "anchor_holes": [round(float(h), 3) for h in ahol]}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="scene_dataset_config.json (omit to load --scene as a file)")
    ap.add_argument("--navmesh", help=".navmesh to load (when the dataset does not provide one)")
    ap.add_argument("--no-semantic", dest="semantic", action="store_false", help="RGB and depth only")
    ap.add_argument("--zup", action="store_true", help="the stage loads Z-up while its navmesh is Y-up (Replica, 0.3.3): pose the camera in the stage's frame")
    ap.add_argument("--scene", help="--pilot: the scene (a name in --dataset, or a file)")
    ap.add_argument("--out", default="results/preview/habitat_pilot")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--survey", nargs="+", dest="scenes", help="survey these scenes of --dataset (writes into --out)")
    ap.add_argument("--n", type=int, default=300, help="--survey: random views per scene")
    ap.add_argument("--recompute-navmesh", action="store_true", help="--survey: rebuild each navmesh from the loaded scene")
    ap.add_argument("--walks-out", dest="walk_out", help="render walks and probes of --scene into this folder")
    ap.add_argument("--walks", type=int, default=32)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--probes-per-walk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=MARGIN, help="walks keep this far from obstacles (m)")
    ap.add_argument("--hssd", action="store_true", help="an HSSD scene: instance labels from object ids, navmesh computed "
                    "if none ships, and (with --walks-out) experiment 16's checks first (scene_check; exit 3 if one fails)")
    ap.add_argument("--island-range", type=float, nargs=2, default=ISLAND_M2, metavar=("LO", "HI"),
                    help="--hssd: accepted area of the largest navigable island (m^2)")
    ap.add_argument("--probes-for", help="amendment 2: new probes and the anchors for the walks in this folder (probes_for)")
    ap.add_argument("--probes-out", help="--probes-for: the output folder")
    ap.add_argument("--probe-seed", type=int, default=17, help="--probes-for: seed of the probe draws (--seed is the walks')")
    args = ap.parse_args()
    if args.probes_for:
        probes_for(args)
    elif args.pilot:
        pilot(args)
    elif args.scenes:
        survey(args)
    elif args.walk_out:
        args.out = args.walk_out
        walks(args)


if __name__ == "__main__":
    main()
