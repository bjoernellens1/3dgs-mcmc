import torch

def distCUDA2(points: torch.Tensor, chunk_size: int = 8192, k: int = 4):
    """
    Approximate PyTorch fallback for simple_knn._C.distCUDA2.
    points: [N, 3] tensor
    returns: squared distance estimate per point
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")

    points = points.contiguous()
    out = torch.empty(points.shape[0], device=points.device, dtype=points.dtype)

    with torch.no_grad():
        for start in range(0, points.shape[0], chunk_size):
            end = min(start + chunk_size, points.shape[0])
            q = points[start:end]
            d = torch.cdist(q.float(), points.float())

            rows = torch.arange(end - start, device=points.device)
            cols = torch.arange(start, end, device=points.device)
            d[rows, cols] = float("inf")

            knn_dists, _ = torch.topk(d, k=k, largest=False)
            out[start:end] = knn_dists.mean(dim=1).to(points.dtype) ** 2

    return out
