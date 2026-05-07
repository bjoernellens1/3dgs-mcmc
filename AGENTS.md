# Agent Notes

- Run runnable verification inside the project containerized setup, not host Python. Prefer `docker compose run --rm train ...` so checks use the ROCm/gsplat runtime this repo targets.
- For training runs, start TensorBoard from inside the container as well and publish it to localhost, e.g. `docker compose run --rm -p 6006:6006 -v "$PWD":/workspace/3dgs-mcmc train tensorboard --logdir /workspace/3dgs-mcmc/output --host 0.0.0.0 --port 6006`.
- Local Mip-NeRF 360 v2 dataset is at `/home/bjoern/Downloads/mipnerf360_v2_dataset`; mount it into the container when running training, for example `-v /home/bjoern/Downloads/mipnerf360_v2_dataset:/data/mipnerf360_v2_dataset`. The default init mode is SFM (`--init_type sfm`); mount the dataset read-write if overriding to `--init_type random`, because scene loading writes `random.ply` into the scene directory.
