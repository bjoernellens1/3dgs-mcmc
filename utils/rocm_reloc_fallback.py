import math
import torch

N_MAX = 51


def _precompute_binoms(device):
    """Precompute C(n, j) for n,j in [0, N_MAX). Cached per device."""
    key = str(device)
    if key not in _precompute_binoms._cache:
        table = torch.zeros((N_MAX, N_MAX), device=device, dtype=torch.float32)
        for n in range(N_MAX):
            for j in range(n + 1):
                table[n, j] = math.comb(n, j)
        _precompute_binoms._cache[key] = table
    return _precompute_binoms._cache[key]


_precompute_binoms._cache = {}


def compute_relocation_cuda(opacity_old: torch.Tensor, scale_old: torch.Tensor, N: torch.Tensor):
    """
    PyTorch fallback for diff_gaussian_rasterization.compute_relocation.

    Matches the CUDA kernel in diff-gaussian-rasterization/cuda_rasterizer/utils.cu
    (Equation 9 in "3D Gaussian Splatting as Markov Chain Monte Carlo").

    Uses the hockey-stick identity to collapse the original double loop into a
    single vectorised sum, giving identical results.
    """
    device = opacity_old.device
    dtype = opacity_old.dtype

    orig_opacity_shape = opacity_old.shape
    opacity_old = opacity_old.view(-1)
    scale_old = scale_old.view(-1, 3)
    N = N.view(-1).clamp(min=1, max=N_MAX - 1).long()  # [P]
    P = opacity_old.shape[0]

    # New opacity: alpha_new = 1 - (1 - alpha_old)^(1/N)
    opacity_new = 1.0 - (1.0 - opacity_old) ** (1.0 / N.float())  # [P]

    # Precompute binomial table C(n, j) on the target device
    comb_table = _precompute_binoms(device)  # [N_MAX, N_MAX]

    # Vectorised denom_sum using hockey-stick identity:
    #   denom_sum = sum_{j=1}^{N} C(N, j) * ((-1)^(j-1) / sqrt(j)) * opacity_new^j
    max_n = int(N.max().item())
    if max_n == 0:
        # Should not happen because of clamp(min=1)
        max_n = 1

    j = torch.arange(1, max_n + 1, device=device, dtype=torch.float32)  # [J]

    # Gather C(N_p, j) for each point and each j
    Nj = N.unsqueeze(1)  # [P, 1]
    j_range = j.unsqueeze(0)  # [1, J]
    mask = j_range <= Nj.float()  # [P, J]

    comb_vals = comb_table[N.unsqueeze(1), j.long().unsqueeze(0)]  # [P, J]
    sign = ((-1.0) ** (j - 1)) / torch.sqrt(j)  # [J]
    opacity_pow = opacity_new.unsqueeze(1) ** j.unsqueeze(0)  # [P, J]

    terms = comb_vals * sign.unsqueeze(0) * opacity_pow  # [P, J]
    denom_sum = (terms * mask).sum(dim=1)  # [P]

    coeff = opacity_old / denom_sum.clamp(min=1e-10)
    scale_new = coeff.unsqueeze(-1) * scale_old  # [P, 3]

    return opacity_new.view(orig_opacity_shape), scale_new.view(-1, 3).to(dtype)
