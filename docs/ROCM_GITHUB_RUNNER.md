# Self-hosted GitHub Actions Runner for ROCm Docker Builds

The multi-stage Docker build requires ~40 GB of temporary disk space and access to the ROCm compiler toolchain (provided by the `rocm/pytorch` builder image). Standard GitHub-hosted runners have only 14 GB root disk, so a **self-hosted runner** on the Strix Halo machine is required.

## Setup

```bash
# 1. Create a folder for the runner
mkdir -p ~/actions-runner && cd ~/actions-runner

# 2. Download the latest runner
curl -o actions-runner-linux-x64.tar.gz -L \
  https://github.com/actions/runner/releases/latest/download/actions-runner-linux-x64.tar.gz

# 3. Extract
tar xzf actions-runner-linux-x64.tar.gz

# 4. Get a registration token from:
#    GitHub repo → Settings → Actions → Runners → New self-hosted runner
#    Then run the `./config.sh` command shown there, e.g.:
./config.sh --url https://github.com/bjoernellens1/3dgs-mcmc \
  --token <TOKEN> \
  --labels rocm,amd64,linux \
  --name strix-halo-runner

# 5. Install as a service
sudo ./svc.sh install
sudo ./svc.sh start

# 6. Verify
sudo ./svc.sh status
```

## Prerequisites on the runner

- **Docker** with rootless mode or `sudo` access
- **ROCm kernel drivers** installed on the host (`/dev/kfd`, `/dev/dri`)
- At least **100 GB free disk** (the build temporarily uses ~40 GB)
- Git, curl

## Workflow

The CI workflow in `.github/workflows/build-image.yml`:

1. Runs on any self-hosted runner with labels `self-hosted, linux, amd64, rocm`
2. Builds the multi-stage Docker image
3. Pushes to GHCR with tags: `latest` (main), `rocm-port`, `edge` (feature branches), `sha-<short>`
4. Caches Docker layers to GHCR for faster subsequent builds
5. Runs a smoke-test job on `ubuntu-latest` verifying core imports (`torch`, `gsplat`, etc.)

## Tag strategy

| Branch | Tag |
|--------|-----|
| `main` | `latest`, `sha-<commit>` |
| `rocm-port` | `rocm-port`, `sha-<commit>` |
| `feature/*` | `edge`, `sha-<commit>` |
