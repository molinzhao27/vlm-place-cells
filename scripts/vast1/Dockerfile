# Base image for the MechInterp container on a host without a suitable CUDA torch image (vast-3):
# Python 3.12 with gcc (Triton) and git, plus rsync for sync.sh. torch and every other package go
# into the venv on the volume (container.sh env with MI_REQUIREMENTS=requirements-lock.txt).
#   MI_HOST=vast3 scripts/vast1/container.sh image
FROM python:3.12.14-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends rsync && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
CMD ["sleep", "infinity"]
