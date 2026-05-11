# CI: Docker Build Pipeline

The multi-stage Docker build runs automatically on **standard `ubuntu-latest` GitHub runners**. The workflow strips pre-installed bloat (Android SDK, .NET, etc.) before building to free up disk space.

## How it works

1. A `push` or `pull_request` triggers the workflow
2. A cleanup step removes unneeded packages (~30 GB) from the runner
3. Docker builds the multi-stage image using Buildx with registry cache
4. The image is pushed to GHCR with branch-appropriate tags
5. A smoke-test job verifies `torch`, `gsplat`, and key imports

## Tags

| Branch | Tags |
|--------|------|
| `main` | `latest`, `sha-<commit>` |
| `rocm-port` | `rocm-port`, `sha-<commit>` |
| `feature/*` | `edge`, `sha-<commit>` |
| Any PR | `sha-<commit>` (not pushed) |

## Self-hosted runner (optional)

If you want faster builds by caching the `rocm/pytorch` base image locally, set up a self-hosted runner:

```bash
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-linux-x64.tar.gz -L \
  https://github.com/actions/runner/releases/latest/download/actions-runner-linux-x64.tar.gz
tar xzf actions-runner-linux-x64.tar.gz

# Get token from: repo → Settings → Actions → Runners → New self-hosted runner
./config.sh --url https://github.com/bjoernellens1/3dgs-mcmc \
  --token <TOKEN> \
  --labels rocm,amd64,linux \
  --name strix-halo-runner

sudo ./svc.sh install && sudo ./svc.sh start
```

Then change the workflow's `runs-on` to `[self-hosted, linux, amd64, rocm]` and remove the cleanup step.
