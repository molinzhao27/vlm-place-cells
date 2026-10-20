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
