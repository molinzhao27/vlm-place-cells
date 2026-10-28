"""CUT3R and TTT3R on our walks: a streaming 3D model with a learned state (follow-up pilot WP, docs/experiments.md).

    python -m spatialcells.wm_cut3r gate [--rule ttt3r] [--data data/walks-roomReplicaApt0 --scene ReplicaApt0]
    python -m spatialcells.wm_cut3r acts --cond walk [--probes data/walks-p256] [--rule ttt3r]
    python -m spatialcells.wm_cut3r acts --cond single|alone|shuffled [--rule ttt3r]
    python -m spatialcells.wm_cut3r acts --cond single --frames probes --data data/walks-p256
    python -m spatialcells.wm_cut3r patch

CUT3R (Wang et al., CVPR 2025; CC BY-NC-SA 4.0) and TTT3R (CC BY-NC-SA 4.0 / MIT) stay outside this repository:
--repo is a checkout (CUT3R/CUT3R for --rule cut3r, Inception3D/TTT3R for --rule ttt3r; its src/ is imported) and
--ckpt the final checkpoint cut3r_512_dpt_4_64.pth. The model is built from the configuration of the repository's
config/dpt_512_vary_4_64.yaml (checked against the string stored in the checkpoint, which is read but never
evaluated) and the weights loaded strictly. Frames go through the demo's loader for a square image at size 512
(upsampled to 512 x 512, center crop 512 x 384, normalized to [-1, 1]).

Each step runs the model's own pieces (encoder, state/image decoders, pose retriever, pose head; the dense heads are
skipped), so a batch of walks streams in parallel and the state's write can be disabled per step (update = False,
used for probes). --rule ttt3r applies TTT3R's update: the new state enters with per-token weight sigmoid(mean over
layers, heads and image tokens of the state-to-image cross-attention logits), as in TTT3R's code, per walk.
Anchored sequences start with the 8 anchor views of experiment 4 (vggt_cells.anchors), whose predicted cameras
give a similarity transform to room coordinates (the anchors come first from a fresh state, so it is the same for
every walk). The state is never reset.

  gate   WP0: the first 16 walks, each from a fresh state without anchors; per-walk similarity alignment of the
         predicted camera centers to the true ones (x, y, 1 m) over all 256 frames; median error, and over the
         first and last 64 frames -> results/wm_pilot/gate_<rule>_<scene>.json
  acts   WP1 units -> data/acts/<rule>-<cond>/L<k>.npy and pose.npz (room coordinates via the anchors):
         L0 the encoder's mean patch token (1024); L3, L6, L9, L12 the MLP hidden units (3072) of image-decoder
         blocks 3, 6, 9, 12 at the pose token; L13 the state after this step's update, mean over its 768 tokens
         (not recorded where the write is disabled, i.e. for probes-walk).
         walk: anchors, then the walk; single: anchors, then the frame alone; alone: the frame as the first view
         of a fresh sequence; shuffled: anchors, then the walk in an order shuffled with the walk as seed (rows
         returned to walk order). --probes DIR (walk only): after each walk, its probes (DIR/probes.npy) are read
         against the final state with update = False -> <rule>-probes-walk. --frames probes: run the condition
         on DIR/probes.npy instead -> <rule>-probes-<cond>. Outside room A (--scene) every tag ends in -<scene>,
         and the anchors must already be in --data (habitat_scene.py --probes-for; amendment 2).
  patch  WP4: single mode; the 128 targets, donors (> 3 m away) and 2,048 fit frames drawn as in vggt_patch.py.
         At the pose token after each of the 12 image-decoder blocks: full token, position PCA 10 and 2, the
         best-R^2 ridge probe for (x, y), random 10, the exact complement of PCA 10; erasure of PCA 10 and
         random 10. Transfer of the pose head's room position toward the donor's. Walk mode (secondary): the
         state and/or pose memory entering a target walk frame replaced by the donor walk's at the donor frame.
         -> results/wm_pilot/patch_<rule>.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from .extract import save
from .two_choice import walk_order
from .vggt_cells import anchors
from .vggt_control import umeyama
from .wm_common import WM_ROOT, add_path, device, safe_load

INF = float("inf")
CONFIG = dict(state_size=768, state_pe="2d", pos_embed="RoPE100", rgb_head=True, pose_head=True,
              patch_embed_cls="PatchEmbedDust3R", img_size=(512, 512), head_type="dpt", output_mode="pts3d+pose",
              depth_mode=("exp", -INF, INF), conf_mode=("exp", 1, INF), pose_mode=("exp", -INF, INF),
              enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12,
              dec_num_heads=12, landscape_only=False)
UNIT_BLOCKS = (3, 6, 9, 12)                      # 1-based image-decoder blocks recorded as units
STUB = ("omegaconf.", "typing.")                 # the training run's config, kept as inert stand-ins
ALLOW = ("builtins.dict", "collections.defaultdict", "collections.OrderedDict")


def stored_model_string(obj, seen=None):
    """The 'ARCroco3DStereo(...)' string inside the checkpoint's stand-in config objects (read, never evaluated)."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return None
    seen.add(id(obj))
    if isinstance(obj, str):
        return obj if obj.startswith("ARCroco3DStereo(") else None
    items = (obj.values() if isinstance(obj, dict) else obj if isinstance(obj, (list, tuple))
             else vars(obj).values() if hasattr(obj, "__dict__") else ())
    for v in items:
        found = stored_model_string(v, seen)
        if found:
            return found
    return None


def _rope1d(self, tokens, pos1d, cos, sin):
    """1-D rotary embedding with the angle computed from the position, as the CUDA kernel does (pos * base^(-i/Q)),
    so that negative positions work; the same values as the fallback's table for positions >= 0."""
    D = cos.shape[-1]
    inv = 1.0 / (self.base ** (torch.arange(0, D, 2, device=tokens.device).float() / D))
    f = pos1d.float()[..., None] * inv
    f = torch.cat((f, f), dim=-1).to(tokens.dtype)[:, None]
    return tokens * f.cos() + self.rotate_half(tokens) * f.sin()


class Stream:
    def __init__(self, repo, ckpt, rule, dev, record=False):
        add_path(os.path.join(repo, "src"))
        from dust3r.heads.postprocess import postprocess_pose
        from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig
        from dust3r.utils.camera import pose_encoding_to_camera
        from dust3r.utils.image import ImgNorm, _resize_pil_image
        self._post, self._cam, self._norm, self._resize = postprocess_pose, pose_encoding_to_camera, ImgNorm, _resize_pil_image
        import models.pos_embed as pe                   # croco, put on sys.path by dust3r
        if hasattr(pe.RoPE2D, "apply_rope1d"):          # no compiled cuRoPE2D (the Mac): its PyTorch fallback
            pe.RoPE2D.apply_rope1d = _rope1d            # looks angles up by position, so the pose token's -1 fails
        ck = safe_load(ckpt, ALLOW, STUB)
        self.stored = stored_model_string(ck.get("args"))
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in ck["model"].items()}
        if not any(k.startswith("dec_blocks_state") for k in sd):  # as ARCroco3DStereo.load_state_dict
            sd.update({k.replace("dec_blocks", "dec_blocks_state"): v for k, v in list(sd.items()) if k.startswith("dec_blocks")})
        model = ARCroco3DStereo(ARCroco3DStereoConfig(**CONFIG))
        missing, unexpected = torch.nn.Module.load_state_dict(model, sd, strict=False)
        assert not missing, f"missing weights: {missing[:5]}"
        self.unexpected = unexpected
        self.model = model.to(dev).eval()
        self.rule, self.dev = rule, dev
        self.rec, self.recording, self.patch = {}, False, {}
        blocks = self.model.dec_blocks
        for g, blk in enumerate(blocks, 1):
            blk.register_forward_hook(self._block_hook(g))
            if record and g in UNIT_BLOCKS:
                blk.mlp.fc2.register_forward_pre_hook(self._unit_hook(g))

    def _unit_hook(self, g):
        def f(_, inp):                                  # returns None: a pre-hook's return value replaces the input
            if self.recording:
                self.rec[f"L{g}"] = inp[0][:, 0].float()
        return f

    def _block_hook(self, g):
        def f(_, __, out):
            x, rest = out[0], out[1:]                   # TTT3R's blocks also return attention maps
            if self.recording:
                self.rec[f"tok{g}"] = x[:, 0].float()
            if g in self.patch:
                x = x.clone()
                x[:, 0] = self.patch[g].to(x.device, x.dtype)
                return (x, *rest)
        return f

    def prep(self, frames):
        """uint8 (N, 256, 256, 3) -> (N, 3, 384, 512): the demo's load_images at size 512 for a square image."""
        out = []
        for f in frames:
            im = self._resize(Image.fromarray(np.asarray(f)), 512)
            W, H = im.size
            cx, cy = W // 2, H // 2
            halfw = ((2 * cx) // 16) * 8
            halfh = 3 * halfw / 4 if W == H else ((2 * cy) // 16) * 8
            out.append(self._norm(im.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))))
        return torch.stack(out).to(self.dev)

    def fresh(self, B):
        """Initial state, its positions and the pose memory for a batch of B sequences."""
        m = self.model
        dummy = torch.zeros(B, 1, 2, device=self.dev, dtype=torch.long)
        state, pos = m._init_state(torch.zeros(B, 1, m.enc_embed_dim, device=self.dev), dummy)
        mem = m.pose_retriever.mem.expand(B, -1, -1)
        return {"state": state, "pos": pos, "mem": mem, "init_state": state.clone(), "init_mem": mem.clone(), "first": True}

    @torch.no_grad()
    def step(self, img, S, update=True):
        """One frame per sequence: img (B, 3, H, W) normalized; S from fresh() or an earlier step. Returns the camera
        (center (B, 3), optical axis (B, 3)) in the first view's frame and the next S (S itself if not update)."""
        m, B = self.model, len(img)
        self.rec.clear()                                # no unit outlives its step (L13 exists only if update)
        shape = torch.tensor(img.shape[-2:], device=self.dev)[None].repeat(B, 1)
        [feat], pos, _ = m._encode_image(img, shape)
        if self.recording:
            self.rec["L0"] = feat.mean(1).float()
        g_feat = m._get_img_level_feat(feat)
        pose_feat = (m.pose_token.expand(B, -1, -1) if S["first"] else m.pose_retriever.inquire(g_feat, S["mem"]))
        pose_pos = -torch.ones(B, 1, 2, device=self.dev, dtype=pos.dtype)
        mask = torch.ones(B, dtype=torch.bool, device=self.dev)
        args = (S["state"], S["pos"], feat, pos, pose_feat, pose_pos, S["init_state"])
        kw = dict(img_mask=mask, reset_mask=~mask, update=mask)
        if self.rule == "ttt3r" and not S["first"] and update:
            new_state, dec, _, cross, _, _ = m._recurrent_rollout(*args, **kw, return_attn=True)
            w = torch.sigmoid(torch.stack(cross).mean(dim=(0, 2, 4)))[..., None]    # (layers, B, heads, state, img)
        else:
            out = m._recurrent_rollout(*args, **kw)
            new_state, dec = out[0], out[1]
            w = 1.0
        pose = self._post(m.downstream_head.pose_head(dec[-1][:, 0].float()), m.downstream_head.pose_mode)
        c2w = self._cam(pose.float())
        cam = (c2w[:, :3, 3].cpu().numpy().astype(np.float64), c2w[:, :3, 2].cpu().numpy().astype(np.float64))
        if not update:
            return cam, S
        new_mem = m.pose_retriever.update_mem(S["mem"], g_feat, dec[-1][:, 0:1])
        state = new_state * w + S["state"] * (1 - w) if self.rule == "ttt3r" and not S["first"] else new_state
        if self.recording:
            self.rec["L13"] = state.mean(1).float()
        return cam, {**S, "state": state, "mem": new_mem, "first": False}


def expand(S, B):
    return {k: (v.expand(B, *v.shape[1:]).clone() if torch.is_tensor(v) else v) for k, v in S.items()}


def take(S, idx):
    return {k: (v[idx] if torch.is_tensor(v) else v) for k, v in S.items()}


def anchored(st, data, room="A"):
    """State after the 8 anchors (batch 1) and the similarity transform from the anchors' predicted cameras."""
    if room != "A":     # a photoreal scene's anchors come from habitat_scene.py --probes-for (amendment 2), never room.py
        assert os.path.exists(f"{data}/anchors.npz"), f"{data}/anchors.npz is missing"
    A = anchors(data, room)
    S = st.fresh(1)
    C = []
    for im in A["images"]:
        (c, _), S = st.step(st.prep([im]), S)
        C.append(c[0])
    gt = np.c_[A["poses"][:, :2], np.ones(len(A["poses"]))]
    T = umeyama(np.array(C), gt)
    rms = float(np.sqrt(((to_room_c(np.array(C), T) - gt) ** 2).sum(1).mean()))
    return S, T, rms


def to_room_c(C, T):
    s, R, t = T
    return (s * (R @ C.T)).T + t


def to_room(cam, T):
    """(x, y) and heading in room coordinates."""
    C, f = cam
    s, R, t = T
    Cr, fr = to_room_c(C, T), (R @ f.T).T
    return Cr[:, :2], np.arctan2(fr[:, 1], fr[:, 0])


def load_scene(data, frames="frames"):
    X = np.load(f"{data}/{'probes' if frames == 'probes' else 'frames'}.npy", mmap_mode="r")
    meta = dict(np.load(f"{data}/{'probes_meta' if frames == 'probes' else 'meta'}.npz"))
    return X, meta


def gate(st, args):
    """Sub-stepped walks (substep.py; meta has `orig`) are streamed whole, and only their original steps are aligned
    and scored (amendment 1); walks may then differ in length, and a batch drops each walk as it ends."""
    frames, meta = load_scene(args.data)
    order = walk_order(meta)
    walks = sorted(order)[:16]
    orig = meta["orig"].astype(bool) if "orig" in meta else np.ones(len(frames), bool)
    t0 = time.time()
    C = {w: np.zeros((len(order[w]), 3)) for w in walks}
    for b0 in range(0, len(walks), args.batch):
        wb = walks[b0:b0 + args.batch]
        S = st.fresh(len(wb))
        live = list(wb)
        for s in range(max(len(order[w]) for w in wb)):
            keep = [i for i, w in enumerate(live) if s < len(order[w])]
            if len(keep) < len(live):
                S, live = take(S, keep), [live[i] for i in keep]
            (c, _), S = st.step(st.prep([frames[order[w][s]] for w in live]), S)
            for i, w in enumerate(live):
                C[w][s] = c[i]
            if s % 64 == 0:
                print(f"walks {wb[0]}-{wb[-1]}, step {s}, {time.time() - t0:.0f}s", flush=True)
    err, step_err, turned = [], [], []
    for w in walks:
        fi = order[w][orig[order[w]]]
        gt = np.stack([meta["x"][fi], meta["y"][fi], np.ones(len(fi))], 1)
        P = to_room_c(C[w][orig[order[w]]], umeyama(C[w][orig[order[w]]], gt))
        err.append(np.linalg.norm(P - gt, axis=1))
        step_err.append(np.linalg.norm(np.diff(P, axis=0) - np.diff(gt, axis=0), axis=1))
        turned.append(np.abs((np.diff(meta["theta"][fi]) + np.pi) % (2 * np.pi) - np.pi) > np.deg2rad(30))
    err, step_err, turned = np.array(err), np.concatenate(step_err), np.concatenate(turned)
    res = {"rule": st.rule, "scene": args.scene, "data": args.data, "device": str(st.dev), "n_walks": len(walks),
           "frames_streamed": int(sum(len(order[w]) for w in walks)), "frames_scored": int(err.size),
           "median_error_m": float(np.median(err)),
           "median_error_first64_m": float(np.median(err[:, :64])), "median_error_last64_m": float(np.median(err[:, -64:])),
           "per_walk_median_m": [float(np.median(e)) for e in err], "pass": bool(np.median(err) <= 0.5),
           "step_error_after_turn_over30_m": float(np.median(step_err[turned])),
           "step_error_other_m": float(np.median(step_err[~turned])), "share_steps_turn_over30": float(turned.mean()),
           "seconds": time.time() - t0}
    os.makedirs("results/wm_pilot", exist_ok=True)
    json.dump(res, open(f"results/wm_pilot/gate_{st.rule}_{args.scene}.json", "w"), indent=1)
    print(f"gate {st.rule} {args.scene}: median error {res['median_error_m']:.2f} m (first 64 frames "
          f"{res['median_error_first64_m']:.2f}, last 64 {res['median_error_last64_m']:.2f}); {res['seconds']:.0f}s", flush=True)


def acts(st, args):
    frames, meta = load_scene(args.data, args.frames)
    n = len(frames)
    S0, T, rms = anchored(st, args.data, args.scene)
    print(f"anchors: fit rms {rms:.3f} m", flush=True)
    rec, xy, th = {}, np.full((n, 2), np.nan), np.full(n, np.nan)
    prec, pxy, pth = {}, None, None
    t0 = time.time()

    def keep(store, rows):
        for k, v in st.rec.items():
            if k.startswith("tok"):
                continue
            store.setdefault(k, np.zeros((n if store is rec else len(pframes), v.shape[1]), np.float16))[rows] = \
                v.cpu().numpy().astype(np.float16)

    part = tuple(map(int, args.part.split("/")))
    done, pdone = [], []
    st.recording = True
    if args.cond in ("single", "alone"):
        mine = np.array_split(np.arange(n), part[1])[part[0] - 1]
        for b0 in range(mine[0], mine[-1] + 1, args.batch):
            rows = np.arange(b0, min(mine[-1] + 1, b0 + args.batch))
            done.append(rows)
            S = expand(S0, len(rows)) if args.cond == "single" else st.fresh(len(rows))
            cam, _ = st.step(st.prep(frames[rows]), S)
            keep(rec, rows)
            if args.cond == "single":
                xy[rows], th[rows] = to_room(cam, T)
            if b0 % (args.batch * 20) == 0:
                print(f"{b0 + len(rows)}/{n} frames, {time.time() - t0:.0f}s", flush=True)
    else:
        order = walk_order(meta)
        if args.probes:
            pframes, pmeta = load_scene(args.probes, "probes")
            pxy, pth = np.full((len(pframes), 2), np.nan), np.full(len(pframes), np.nan)
        ws = list(np.array_split(np.array(sorted(order)), part[1])[part[0] - 1])
        for b0 in range(0, len(ws), args.batch):
            wb = ws[b0:b0 + args.batch]
            seq = []
            for w in wb:
                fi = order[w]
                if args.cond == "shuffled":
                    fi = fi[np.random.default_rng(int(w)).permutation(len(fi))]
                seq.append(fi)
            seq = np.array(seq)                      # (walks, steps) frame rows in the order shown
            done.append(seq.ravel())
            S = expand(S0, len(wb))
            for s in range(seq.shape[1]):
                cam, S = st.step(st.prep(frames[seq[:, s]]), S)
                keep(rec, seq[:, s])
                xy[seq[:, s]], th[seq[:, s]] = to_room(cam, T)
            if args.probes:
                for i, w in enumerate(wb):
                    pr = np.flatnonzero(pmeta["walk"] == w)
                    pdone.append(pr)
                    for p0 in range(0, len(pr), args.batch * 4):
                        rows = pr[p0:p0 + args.batch * 4]
                        cam, _ = st.step(st.prep(pframes[rows]), expand(take(S, [i]), len(rows)), update=False)
                        keep(prec, rows)
                        pxy[rows], pth[rows] = to_room(cam, T)
            print(f"walks {wb[0]}-{wb[-1]} done, {time.time() - t0:.0f}s; median position error so far "
                  f"{np.nanmedian(np.hypot(xy[:, 0] - meta['x'], xy[:, 1] - meta['y'])):.2f} m", flush=True)
    st.recording = False
    sfx = "" if args.scene == "A" else f"-{args.scene}"
    tag = f"{st.rule}-" + ("probes-" if args.frames == "probes" else "") + args.cond + sfx
    if np.isfinite(xy[:, 0]).any():
        err = np.hypot(xy[:, 0] - meta["x"], xy[:, 1] - meta["y"])
        print(f"{tag}: position error median {np.nanmedian(err):.2f} m (anchored; this part)", flush=True)
    save_part(f"data/acts/{tag}", part, n, rec, np.concatenate(done), dict(x=xy[:, 0], y=xy[:, 1], theta=th, anchor_fit_rms=rms))
    if args.probes:
        perr = np.hypot(pxy[:, 0] - pmeta["x"], pxy[:, 1] - pmeta["y"])
        print(f"{st.rule}-probes-walk{sfx}: probe position error median {np.nanmedian(perr):.2f} m (this part)", flush=True)
        save_part(f"data/acts/{st.rule}-probes-walk{sfx}", part, len(pframes), prec, np.concatenate(pdone),
                  dict(x=pxy[:, 0], y=pxy[:, 1], theta=pth, anchor_fit_rms=rms))
    print(f"{tag}: part {args.part} done, {time.time() - t0:.0f}s", flush=True)


def save_part(d, part, n, rec, rows, pose):
    """Write this part's rows of every layer and of pose (arrays over all n rows; scalars kept); once every part of
    d is on disk, merge them into L<k>.npy and pose.npz (extract.save) and remove the parts."""
    i, N = part
    os.makedirs(d, exist_ok=True)
    np.savez(f"{d}/part{i}of{N}.npz", rows=rows, **{k: v[rows] for k, v in rec.items()},
             **{f"pose_{k}": (v[rows] if np.ndim(v) else v) for k, v in pose.items()})
    paths = [f"{d}/part{j}of{N}.npz" for j in range(1, N + 1)]
    if not all(os.path.exists(p) for p in paths):
        print(f"{d}: part {i}/{N} saved", flush=True)
        return
    full, pz = {}, {}
    for p in paths:
        z = np.load(p)
        r = z["rows"]
        for k in z.files:
            if k == "rows":
                continue
            if k.startswith("pose_"):
                if z[k].ndim == 0:
                    pz[k[5:]] = z[k]
                else:
                    pz.setdefault(k[5:], np.full(n, np.nan))[r] = z[k]
            else:
                full.setdefault(k, np.zeros((n, z[k].shape[1]), np.float16))[r] = z[k]
    filled = np.zeros(n, bool)
    for p in paths:
        filled[np.load(p)["rows"]] = True
    assert filled.all(), f"{d}: {(~filled).sum()} rows missing after merging"
    save(d, full)
    np.savez(f"{d}/pose.npz", **pz)
    for p in paths:
        os.remove(p)
    print(f"{d}: merged {N} parts", flush=True)


def patch(st, args):
    from .vggt_patch import mean_subspace, ridge_subspace
    rng = np.random.default_rng(0)
    frames, meta = load_scene(args.data)
    xy_true = np.stack([meta["x"], meta["y"]], 1)
    even = np.flatnonzero(meta["walk"] % 2 == 0)
    odd = np.flatnonzero(meta["walk"] % 2 == 1)
    fit = np.sort(rng.choice(even, args.n_fit, replace=False))          # vggt_patch.py's draws, in its order
    targets = rng.choice(odd, args.n_test, replace=False)
    donors = np.array([rng.choice(odd[np.hypot(*(xy_true[odd] - xy_true[t]).T) > 3]) for t in targets])
    S0, T, rms = anchored(st, args.data)
    t0 = time.time()

    def single(idx, repl=None):
        """Room (x, y) of each frame in single mode, and its pose token after every block (clean runs)."""
        out, toks = [], {}
        st.recording = repl is None
        for b0 in range(0, len(idx), args.batch):
            rows = idx[b0:b0 + args.batch]
            st.patch = {} if repl is None else {g: v[b0:b0 + args.batch] for g, v in repl.items()}
            cam, _ = st.step(st.prep(frames[rows]), expand(S0, len(rows)))
            out.append(to_room(cam, T)[0])
            for k, v in st.rec.items():
                if k.startswith("tok"):
                    toks.setdefault(int(k[3:]), []).append(v.cpu())
        st.patch, st.recording = {}, False
        return np.concatenate(out), {g: torch.cat(v) for g, v in toks.items()}

    _, H_fit = single(fit) if not args.walk_mode else (None, None)
    pred_t, H_t = single(targets)
    pred_d, H_d = single(donors)
    gap = pred_d - pred_t
    path = f"results/wm_pilot/patch_{st.rule}.json"
    res = json.load(open(path)) if os.path.exists(path) else {"blocks": {}}
    res.update({"rule": st.rule, "anchor_fit_rms_m": rms, "n_test": len(targets),
                "clean_error_m": float(np.median(np.hypot(*(pred_t - xy_true[targets]).T))),
                "target_donor_distance_m": float(np.median(np.linalg.norm(gap, axis=1)))})
    print(f"clean passes {time.time() - t0:.0f}s: target error {res['clean_error_m']:.2f} m, "
          f"target-donor {res['target_donor_distance_m']:.2f} m", flush=True)

    def summarize(pred, erase=False, pred_d=pred_d):
        moved = pred - pred_t
        r = {"moved_m_median": float(np.median(np.linalg.norm(moved, axis=1))),
             "error_m_median": float(np.median(np.hypot(*(pred - xy_true[targets]).T)))}
        if not erase:
            g = pred_d - pred_t
            tr = (moved * g).sum(1) / (g ** 2).sum(1)
            r.update(transfer_mean=float(tr.mean()), transfer_sem=float(tr.std() / np.sqrt(len(tr))),
                     transfer_median=float(np.median(tr)))
        return r

    proj = lambda Q, v: (v @ Q) @ Q.T
    x, y = xy_true[fit, 0], xy_true[fit, 1]
    pos_bin = np.clip(x.astype(int), 0, 9) * 10 + np.clip(y.astype(int), 0, 9)
    for g in ([] if args.walk_mode else args.blocks):
        Hf, ht, hd = H_fit[g], H_t[g], H_d[g]          # on CPU: MPS lacks some linalg ops
        mu, d = Hf.mean(0), hd - ht
        Q10 = mean_subspace(Hf, pos_bin, 10)
        Qr2, r2_ridge = ridge_subspace(Hf, torch.tensor(xy_true[fit], dtype=torch.float32), rng)
        gen = torch.Generator().manual_seed(g)
        Qr, _ = torch.linalg.qr(torch.randn(Hf.shape[1], 10, generator=gen))
        conds = {"full": hd, "position PCA 10": ht + proj(Q10, d), "position PCA 2": ht + proj(Q10[:, :2], d),
                 "ridge": ht + proj(Qr2, d), "random 10": ht + proj(Qr, d), "all but position PCA 10": hd - proj(Q10, d)}
        r = {"ridge_r2": r2_ridge, "token_norm_median": float(ht.norm(dim=1).median()),
             "share": {k: float(((proj(Q, d) ** 2).sum(1) / (d ** 2).sum(1)).median())
                       for k, Q in (("position PCA 10", Q10), ("random 10", Qr), ("ridge", Qr2))}}
        for name, h in conds.items():
            r[name] = summarize(single(targets, {g: h})[0])
        for name, Q in (("erase position PCA 10", Q10), ("erase random 10", Qr)):
            r[name] = summarize(single(targets, {g: ht + proj(Q, mu - ht)})[0], erase=True)
        res["blocks"][str(g)] = r
        print(f"block {g} ({time.time() - t0:.0f}s): transfer " + ", ".join(
            f"{k} {v['transfer_mean']:.2f}" for k, v in r.items() if isinstance(v, dict) and "transfer_mean" in v)
              + f"; erase PCA 10 {r['erase position PCA 10']['error_m_median']:.2f} m, random "
              f"{r['erase random 10']['error_m_median']:.2f} m; ridge R2 {r2_ridge:.2f}", flush=True)
        json.dump(res, open(path, "w"), indent=1)
    if not args.walk_mode:
        print(f"patch: blocks {args.blocks} done, {time.time() - t0:.0f}s", flush=True)
        return

    # walk mode: swap the state / pose memory entering a target walk frame for the donor walk's at the donor frame
    order = walk_order(meta)
    step_of = {int(r): s for fi in order.values() for s, r in enumerate(fi)}
    need = {}
    for r in np.r_[targets, donors]:
        need.setdefault(int(meta["walk"][r]), set()).add(step_of[int(r)])
    entering, pred_walk = {}, {}
    ws = sorted(need)
    for b0 in range(0, len(ws), args.batch):
        wb = ws[b0:b0 + args.batch]
        S = expand(S0, len(wb))
        for s in range(len(order[wb[0]])):
            for i, w in enumerate(wb):
                if s in need[w]:
                    entering[int(order[w][s])] = {k: S[k][i:i + 1].cpu() for k in ("state", "mem")}
            cam, S = st.step(st.prep(frames[[order[w][s] for w in wb]]), S)
            p = to_room(cam, T)[0]
            for i, w in enumerate(wb):
                if s in need[w]:
                    pred_walk[int(order[w][s])] = p[i]
    pt = np.array([pred_walk[int(r)] for r in targets])
    pdn = np.array([pred_walk[int(r)] for r in donors])
    res["walk_mode"] = {"clean_error_m": float(np.median(np.hypot(*(pt - xy_true[targets]).T))),
                        "target_donor_distance_m": float(np.median(np.linalg.norm(pdn - pt, axis=1)))}
    for name, keys in (("state and memory", ("state", "mem")), ("state", ("state",)), ("memory", ("mem",))):
        out = []
        for b0 in range(0, len(targets), args.batch):
            tb, db = targets[b0:b0 + args.batch], donors[b0:b0 + args.batch]
            S = expand(S0, len(tb))
            S.update({k: torch.cat([(entering[int(dr)] if k in keys else entering[int(tr)])[k] for tr, dr in zip(tb, db)]).to(st.dev)
                      for k in ("state", "mem")})
            cam, _ = st.step(st.prep(frames[tb]), S, update=False)
            out.append(to_room(cam, T)[0])
        moved = np.concatenate(out) - pt
        g_ = pdn - pt
        tr = (moved * g_).sum(1) / (g_ ** 2).sum(1)
        res["walk_mode"][name] = {"transfer_mean": float(tr.mean()), "transfer_sem": float(tr.std() / np.sqrt(len(tr))),
                                  "transfer_median": float(np.median(tr))}
        print(f"walk mode, swap {name}: transfer {tr.mean():.2f} +/- {tr.std() / np.sqrt(len(tr)):.2f}", flush=True)
    json.dump(res, open(path, "w"), indent=1)
    print(f"patch: walk mode done, {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gate", "acts", "patch"])
    ap.add_argument("--rule", choices=["cut3r", "ttt3r"], default="cut3r")
    ap.add_argument("--repo", help="default ~/wm_models/CUT3R (cut3r) or ~/wm_models/TTT3R (ttt3r)")
    ap.add_argument("--ckpt", default=os.path.join(WM_ROOT, "weights", "cut3r", "cut3r_512_dpt_4_64.pth"))
    ap.add_argument("--data", default="data/walks")
    ap.add_argument("--scene", default="A")
    ap.add_argument("--cond", choices=["walk", "single", "alone", "shuffled"])
    ap.add_argument("--frames", choices=["frames", "probes"], default="frames")
    ap.add_argument("--probes", help="walk only: also read DIR/probes.npy against each walk's final state")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--part", default="1/1", help="acts: I/N, the I-th of N slices of the walks (walk, shuffled) or rows; "
                    "the last part to finish merges them")
    ap.add_argument("--blocks", type=int, nargs="+", default=list(range(1, 13)), help="patch: decoder blocks (1-based)")
    ap.add_argument("--walk-mode", action="store_true", help="patch: only the walk-mode state swap")
    ap.add_argument("--n-fit", type=int, default=2048)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--device")
    args = ap.parse_args()
    repo = args.repo or os.path.join(WM_ROOT, "CUT3R" if args.rule == "cut3r" else "TTT3R")
    st = Stream(repo, args.ckpt, args.rule, device(args.device), record=args.what == "acts")
    print(f"{args.rule}: {repo}, device {st.dev}; unexpected checkpoint keys: {len(st.unexpected)}; stored model: {st.stored}",
          flush=True)
    {"gate": gate, "acts": acts, "patch": patch}[args.what](st, args)


if __name__ == "__main__":
    main()
