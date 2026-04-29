import torch
import math

try:
    from diff_gaussian_rasterization import compute_relocation

    N_max = 51
    binoms = torch.zeros((N_max, N_max), device="cuda", dtype=torch.float32)
    for n in range(N_max):
        for k in range(n + 1):
            binoms[n, k] = math.comb(n, k)

    def compute_relocation_cuda(opacity_old, scale_old, N):
        N = N.clamp(min=1, max=N_max - 1)
        return compute_relocation(opacity_old, scale_old, N, binoms, N_max)
except Exception:
    from utils.rocm_reloc_fallback import compute_relocation_cuda