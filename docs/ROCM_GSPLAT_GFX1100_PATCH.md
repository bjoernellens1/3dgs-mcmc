# Patching ROCm/gsplat for RDNA3 (gfx1100 / gfx1151)

## Problem Summary

`ROCm/gsplat` release/1.5.3b2 assumes **wavefront size = 64** for all HIP builds (`USE_ROCM`). This is correct for CDNA (Instinct/MI series) but invalid for **RDNA3** consumer APUs such as Strix Halo (`gfx1151`, reported as `gfx1100` via `HSA_OVERRIDE_GFX_VERSION=11.0.0`).

On gfx1100 the hardware wavefront size is **32**. When the compiler instantiates `rocprim::warp_reduce<..., 64>`, `cooperative_groups::tiled_partition<64>`, or `rocprim::warp_reduce<float,64>` on this target, ROCm 7.2 triggers a **static assertion** inside `rocprim/intrinsics/arch.hpp`:

```
rocprim/intrinsics/arch.hpp:260:23: error: static assertion failed due to requirement
'predicate(::rocprim::arch::wavefront::size_from_target())'
```

Additionally, `setup.py` has a **GLM include bug** on the ROCm path: PyTorch's `hipify` copies `gsplat/cuda/csrc/third_party/glm/*.hpp` into `gsplat/hip/...` but skips `*.inl` files and injects CUDA macros that break GLM's `platform.h` checks, causing missing-header errors like:

```
fatal error: 'glm/gtc/type_ptr.hpp' file not found
```

## Files Affected

The wave64 assumptions are concentrated in three rasterization backward kernels and one utility header:

1. `gsplat/cuda/include/Utils.cuh`
2. `gsplat/cuda/csrc/RasterizeToPixels2DGSBwd.cu`
3. `gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu`
4. `gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSBwd.cu`

## Detailed Changes

### 1. Cooperative-group tile size (all 3 `.cu` files)

Replace every occurrence of:

```cpp
cg::thread_block_tile<64> warp = cg::tiled_partition<64>(block);
```

with:

```cpp
cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
```

**Why:** On gfx1100 `tiled_partition<64>` is not a valid cooperative-group size because the hardware wavefront is 32 lanes. ROCm's cooperative-groups implementation delegates to rocPRIM, which enforces this at compile time.

### 2. `rocprim::warp_reduce` template parameter (all 3 `.cu` files + `Utils.cuh`)

Replace explicit `64`-lane instantiations with `32`:

```cpp
// old
rocprim::warp_reduce<int32_t, 64>
rocprim::warp_reduce<float,64>

// new
rocprim::warp_reduce<int32_t, 32>
rocprim::warp_reduce<float,32>
```

In `Utils.cuh` also change the **default template argument** of `rocprim_warpSum` and `rocprim_warpSum_scalar`:

```cpp
// old
template<int LOGICAL_WARP_SIZE = 64>

// new
template<int LOGICAL_WARP_SIZE = 32>
```

**Why:** `rocprim::warp_reduce<T, N>` requires `N` to match the target wavefront size at compile time. Passing `64` on gfx1100 causes the same static assertion.

### 3. `rocprim_warpSum` explicit template arguments (all 3 `.cu` files)

All calls such as:

```cpp
rocprim_warpSum<64>(...)
rocprim_warpSum<CDIM, 64>(...)
rocprim_warpSum<3, 64>(...)
```

must become:

```cpp
rocprim_warpSum<32>(...)
rocprim_warpSum<CDIM, 32>(...)
rocprim_warpSum<3, 32>(...)
```

### 4. Shared-memory scratch size calculations (all 3 `.cu` files)

The number of warps per block is computed as:

```cpp
const uint32_t warps_per_block = (block_size + 63) / 64; // for 64-lane warp
```

Change to:

```cpp
const uint32_t warps_per_block = (block_size + 31) / 32; // for 32-lane warp
```

And the corresponding `sizeof(typename rocprim::warp_reduce<float,64>::storage_type)` must use `32` instead of `64`.

**Why:** If block size is 256, the old formula gives 4 warps of 64 threads. On wave32 there are 8 warps of 32 threads, so the shared-memory scratch array for warp reductions must be sized accordingly.

### 5. Shuffle-based reductions (`Utils.cuh`)

`reduce_max_shuffle` and `manual_warpSum` contain hard-coded loops:

```cpp
for (int offset = 32; offset > 0; offset /= 2) { ... }
```

Replace the initializer with `warpSize / 2` so the loop adapts to the actual hardware wavefront size at runtime:

```cpp
for (int offset = warpSize / 2; offset > 0; offset /= 2) { ... }
```

Also update the 64-lane shuffle mask in `reduce_max_shuffle` to be safe for wave32:

```cpp
// old
const unsigned long long mask = 0xFFFFFFFFFFFFFFFFULL;

// new (runtime adaptive)
const unsigned long long mask = (warpSize == 32)
    ? 0xFFFFFFFFULL
    : 0xFFFFFFFFFFFFFFFFULL;
```

**Why:** On a 32-lane warp, `__shfl_down_sync` with `offset = 32` reads from a non-existent lane. The behaviour is undefined and on some ROCm versions causes wrong reduction results or hangs. Starting at `warpSize/2` (16 for gfx1100) makes the reduction correct for any wavefront size.

### 6. GLM symlink fix (`setup.py`)

`setup.py` must add the GLM submodule path to `include_dirs` on the ROCm branch, and after `CUDAExtension()` returns it should replace the hipified `gsplat/hip/csrc/third_party/glm` directory (which lacks `.inl` files and has corrupted headers) with a **symbolic link** to the original `gsplat/cuda/csrc/third_party/glm`.

The `setup.py` patch looks like:

```python
# inside get_extensions() on the ROCm branch:
glm_path = osp.join(current_dir, "gsplat", "cuda", "csrc", "third_party", "glm")
include_dirs = [
    glm_path,
    osp.join(current_dir, "gsplat", "cuda", "include"),
    ...
]

# after creating the extension:
hip_glm = osp.join(str(current_dir), "gsplat", "hip", "csrc", "third_party", "glm")
cuda_glm = osp.join(str(current_dir), "gsplat", "cuda", "csrc", "third_party", "glm")
if os.path.isdir(hip_glm) and not os.path.islink(hip_glm):
    shutil.rmtree(hip_glm)
    rel = os.path.relpath(cuda_glm, os.path.dirname(hip_glm))
    os.symlink(rel, hip_glm)
```

**Why:** PyTorch's `hipify` tool only copies `.h`/`.hpp`/`.cuh` files; it does **not** copy GLM's `.inl` implementation files. It also inserts `#define __HIP_PLATFORM_AMD__` guards into the headers, which causes GLM's `platform.h` to mis-detect the compiler and error out. Using the unmodified CUDA-side GLM tree (via symlink) works because HIP/clang can compile the GLM headers as-is.

### 7. `import shutil` in `setup.py`

The symlink fix uses `shutil.rmtree`, so `setup.py` needs:

```python
import shutil
```

## Build verification

After applying the patches, build with:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
python setup.py build_ext --inplace
```

All 29 compilation units should complete without the `rocprim` static assertion or GLM header errors. Warnings about unhandled `GLOBAL` enum in `Cameras.cuh` are harmless upstream warnings.

## Runtime considerations

- `HSA_OVERRIDE_GFX_VERSION=11.0.0` is required on Strix Halo (`gfx1151`) because ROCm 7.2 does not officially ship `gfx1151` device code in its bitcode libraries. Overriding to `gfx1100` allows the compiler to use the RDNA3 instruction set it already knows.
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` is recommended to avoid OOM during training.
- `HSA_ENABLE_SDMA=0` is recommended for RDNA3 integrated graphics to avoid DMA engine hangs.

## Summary of patch scripts

Two small Python patch scripts are sufficient to automate everything:

1. `patch_gsplat_setup.py` – adds `import shutil`, fixes `include_dirs`, and creates the GLM symlink.
2. `patch_gsplat_warp32.py` – performs all `64` → `32` replacements in the four source files listed above.

Both are idempotent (string-replacement based) and can be run after cloning the `release/1.5.3b2` branch and updating submodules.
