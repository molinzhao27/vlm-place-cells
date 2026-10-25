# Running on a remote GPU host

The GPU experiments ran on an RTX PRO 6000 Blackwell (96 GB, vast-1), a 12 GB RTX 4070 PC and
Apple silicon (answers from the Mac are tagged `-mps`). Code is edited locally and synced
to a dedicated `mechinterp` Docker container on each container host (vast-1, and vast-3 with an
RTX 5090, 32 GB), which has its own volume (`mechinterp-data`) and venv, separate from other
projects on the same machine. `scripts/vast1/` holds the helpers. They are optional: any machine
with a CUDA build of torch can run `python -m spatialcells.<module>` directly from the repository
root.

## GPU and long jobs: gq

Every GPU job, and every long CPU job, on vast-1, vast-3, the PC or the Mac goes through `gq`, a
queue shared with the other projects on these machines (`~/Documents/ProcessQueues/README.md`;
dashboard at http://127.0.0.1:8790). It is on PATH on every machine, and from the Mac it reaches
the others over SSH. Don't start GPU work any other way, don't take the hosts' lock files
(`~/.codex-gpu-0.lock`, `~/.gpu-4070.lock`, `~/.gpu-0.lock`) yourself, and don't write new waiter
or queue scripts.

```bash
gq submit --on vast1|vast3|pc|mac --project mechinterp --name <short_name> --gpu --est 2h -- <command...>
gq ls                          # every machine
gq log vast1:12 -f             # follow a job's log
gq wait vast1:12               # block until done; exits 0 only if the job succeeded
gq cancel vast1:12 --project mechinterp
```

- Always name the machine, and give `--est` so the queue can show ETAs. Use `--cpu` instead of
  `--gpu` for CPU-only work, and `--after <id>` to start a job only once an earlier one on the same
  machine has succeeded.
- On vast-1 and vast-3 the command runs inside the `mechinterp` container from
  `/workspace/MechInterp`; on the PC and the Mac it runs on the host from the project folder
  (`--cwd` or `--container none` override this). The job doesn't inherit your shell's environment,
  so use explicit interpreters and pass variables with `--env KEY=VALUE`:

  | Machine | Interpreter | Extra settings |
  |---|---|---|
  | vast-1 | `.venv/bin/python` | `--env CC=/opt/worm-build/bin/gcc --env TRITON_CACHE_DIR=/workspace/.cache/triton` for anything that compiles Triton kernels |
  | vast-3 | `.venv/bin/python` | none (gcc is on PATH; the container sets `HF_HOME` and `TRITON_CACHE_DIR`) |
  | PC | `.venv/bin/python` | `--env VGGT_DIR=<VGGT clone>` (`R4_VGGT` in `scripts/rtx4070/host.env`) for the `vggt_*` scripts |
  | Mac | `.venv-mac/bin/python` | pass `--device mps` to the module; gq waits for 25% free memory and keeps the Mac awake |

  For example, Experiment 9b's second LoRA seed on vast-1:
  `gq submit --on vast1 --project mechinterp --name lora_seed2 --gpu --est 1h50m --env CC=/opt/worm-build/bin/gcc --env TRITON_CACHE_DIR=/workspace/.cache/triton -- /bin/bash scripts/run_ft_seed.sh 2`.
- A job's output goes to its log (`gq log`, stored at `~/.gq/logs/mechinterp/` on the job's
  machine), where `RUN_START` and `RUN_END rc=N` lines mark the run.
- Cancel only MechInterp's own jobs. The user sets the projects' priorities; a job may lower its
  own (`--priority N` below the project's) to let others go first, but reordering across projects,
  pausing a machine and `gq prio` are the user's calls.

The older wrappers, `scripts/vast1/gpu-run.sh` (host GPU lock and the shared turn script) and
`scripts/rtx4070/run.sh` (the 4070's lock), are kept so the usage lines in older run scripts still
make sense. Don't start new jobs with them.

## Setup (container hosts)

Copy `scripts/vast1/host.env.example` to `scripts/vast1/host.env` (gitignored) and set the SSH
target, the base image and its Python, and the container user. For another host, copy it to
`host-NAME.env` (also gitignored) and run the helpers with `MI_HOST=NAME`.

```bash
scripts/vast1/container.sh start      # once: container + volume
scripts/vast1/container.sh env        # venv from requirements.txt
scripts/vast1/sync.sh push            # code -> container (data/ and results/ are not touched)
gq submit --on vast1 --project mechinterp --name pilot --gpu --est 1h -- /bin/bash scripts/run_pilot.sh
scripts/vast1/sync.sh pull            # results/ -> here
```

On vast-1, `requirements.txt` is installed on top of the image's torch (the venv uses
`--system-site-packages`). vast-3 has no suitable image, so `MI_HOST=vast3 scripts/vast1/container.sh
image` first builds a plain Python 3.12 one from `scripts/vast1/Dockerfile`, and `env` installs
`requirements-lock.txt`, vast-1's exact package versions with torch 2.14.0+cu132 (the RTX 5090 needs
CUDA 12.8 or later). Its GPU works only through the nvidia runtime (`--runtime=nvidia` with
`NVIDIA_VISIBLE_DEVICES` and `NVIDIA_DRIVER_CAPABILITIES`; `--gpus all` fails there), which
`host.env.example` shows, and it has 30 GB of RAM, so the container is capped at 24 GB. Its `/` is
small: keep data in the Docker volume. `vggt_control.py` and the other `vggt_*` modules also need
the [VGGT repository](https://github.com/facebookresearch/vggt) cloned to `/workspace/vggt` in the
container.

## Smaller machines

`scripts/vast1/fetch.sh roomA|roomB|adapter` copies the data a job reads from the container to
this machine. Over a slow link it is faster to render the walks and probes again
(`render_data`, `render_probes`): they are seeded and come out identical to the GPU host's.

- **Apple silicon** (Qwen3-VL-8B needs ~18 GB of unified memory): `scripts/mac/setup.sh` makes
  `.venv-mac` with the GPU host's package versions, and `two_choice.py --device mps` runs the
  behaviour tests there. Its answers are tagged `-mps`. `scripts/mac/run_roomB.sh` checks one
  published room-A result first, then runs experiment 9b. Run one GPU job at a time (the Mac
  rebooted out of memory with two); gq's `mac` queue does this.
- **A 12 GB CUDA card on Ubuntu, no container** (`scripts/rtx4070/`, host settings in its
  `host.env`): `setup.sh` (venv, VGGT at the GPU host's commit, VGGT-1B weights) and `sync.sh
  push|data|pull`; jobs go through gq's `pc` queue. It is enough for the `vggt_*` modules with
  smaller batches (`run_vggt_probes.sh`), not for the 8B models. Its system disk is small, so the
  large data live on a second SSD (NTFS, mounted at `/mnt/bigd` with `nofail`): `~/habitat-data`
  and the project's `data/walks*` folders are symlinks into `/mnt/bigd/mechinterp/`. If that drive
  is not mounted, jobs that read them fail with a missing-file error. Keep the venv and model
  weights on the system disk.
