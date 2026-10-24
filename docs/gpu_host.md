# Running on a remote GPU host

Every GPU experiment ran on a single RTX PRO 6000 Blackwell (96 GB). Code is edited locally and
synced to a dedicated `mechinterp` Docker container on the host over SSH. The container has its
own volume (`mechinterp-data`) and venv, separate from other projects on the same machine.
`scripts/vast1/` holds the helpers. They are optional: any machine with a CUDA build of torch can
run `python -m spatialcells.<module>` directly from the repository root.

## Setup

Copy `scripts/vast1/host.env.example` to `scripts/vast1/host.env` (gitignored) and set the SSH
target, the base image and its Python, and the container user.

```bash
scripts/vast1/container.sh start      # once: container + volume
scripts/vast1/container.sh env        # venv from requirements.txt
scripts/vast1/sync.sh push            # code -> container (data/ and results/ are not touched)
scripts/vast1/run.sh py -m spatialcells.render_data          # CPU work
scripts/vast1/gpu-run.sh --detach $HOST_LOGS/mechinterp-pilot.log bash scripts/run_pilot.sh
scripts/vast1/sync.sh pull            # results/ -> here
```

`$HOST_LOGS` is an absolute directory on the host. `gpu-run.sh` holds a host-wide GPU lock
(`MI_GPU_LOCK`) for the whole job and refuses to start if a process outside the lock is on the
GPU, so jobs from different projects never overlap.

`requirements.txt` is installed on top of the image's torch (the venv uses
`--system-site-packages`). `vggt_control.py` and the other `vggt_*` modules also need the
[VGGT repository](https://github.com/facebookresearch/vggt) cloned to `/workspace/vggt` in the
container.

## Smaller machines

`scripts/vast1/fetch.sh roomA|roomB|adapter` copies the data a job reads from the container to
this machine. Over a slow link it is faster to render the walks and probes again
(`render_data`, `render_probes`): they are seeded and come out identical to the GPU host's.

- **Apple silicon** (Qwen3-VL-8B needs ~18 GB of unified memory): `scripts/mac/setup.sh` makes
  `.venv-mac` with the GPU host's package versions, and `two_choice.py --device mps` runs the
  behaviour tests there. Its answers are tagged `-mps`. `scripts/mac/run_roomB.sh` checks one
  published room-A result first, then runs experiment 9b.
- **A 12 GB CUDA card on Ubuntu, no container** (`scripts/rtx4070/`, host settings in its
  `host.env`): `setup.sh` (venv, VGGT at the GPU host's commit, VGGT-1B weights), `sync.sh
  push|data|pull`, and `run.sh [--detach LOG] COMMAND`, which holds a GPU lock on that host. It is
  enough for the `vggt_*` modules with smaller batches (`run_vggt_probes.sh`), not for the 8B
  models. Its system disk is small, so the large data live on a second SSD (NTFS, mounted at
  `/mnt/bigd` with `nofail`): `~/habitat-data` and the project's `data/walks*` folders are symlinks
  into `/mnt/bigd/mechinterp/`. If that drive is not mounted, jobs that read them fail with a
  missing-file error. Keep the venv and model weights on the system disk.
