import os
import re

def patch_file(path, replacements):
    if not os.path.exists(path):
        print(f"Skip: {path} not found")
        return
    with open(path, 'r') as f:
        content = f.read()
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
        else:
            print(f"Warning: pattern not found in {path}: {old!r}")
    with open(path, 'w') as f:
        f.write(content)
    print(f'Patched {path}')

# Patch Utils.cuh
patch_file('gsplat/cuda/include/Utils.cuh', [
    ('template<int LOGICAL_WARP_SIZE = 64>', 'template<int LOGICAL_WARP_SIZE = 32>'),
    ('    const unsigned long long mask = 0xFFFFFFFFFFFFFFFFULL;', '    const unsigned long long mask = (warpSize == 32) ? 0xFFFFFFFFULL : 0xFFFFFFFFFFFFFFFFULL;'),
    ('    for (int offset = 32; offset > 0; offset /= 2) {', '    for (int offset = warpSize / 2; offset > 0; offset /= 2) {'),
    ('    for (int offset = 32 ; offset > 0; offset /= 2) {', '    for (int offset = warpSize / 2 ; offset > 0; offset /= 2) {'),
])

# Patch RasterizeToPixels2DGSBwd.cu
patch_file('gsplat/cuda/csrc/RasterizeToPixels2DGSBwd.cu', [
    ('cg::thread_block_tile<64> warp = cg::tiled_partition<64>(block);', 'cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);'),
    ('rocprim::warp_reduce<float,64>', 'rocprim::warp_reduce<float,32>'),
    ('rocprim_warpSum<CDIM, 64>', 'rocprim_warpSum<CDIM, 32>'),
    ('rocprim_warpSum<3, 64>', 'rocprim_warpSum<3, 32>'),
    ('rocprim_warpSum<64>', 'rocprim_warpSum<32>'),
    ('(block_size + 63) / 64', '(block_size + 31) / 32'),
])

# Patch RasterizeToPixels3DGSBwd.cu
patch_file('gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu', [
    ('cg::thread_block_tile<64> warp = cg::tiled_partition<64>(block);', 'cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);'),
    ('rocprim::warp_reduce<int32_t, 64>', 'rocprim::warp_reduce<int32_t, 32>'),
    ('rocprim::warp_reduce<float,64>', 'rocprim::warp_reduce<float,32>'),
    ('rocprim_warpSum<CDIM, 64>', 'rocprim_warpSum<CDIM, 32>'),
    ('rocprim_warpSum<64>', 'rocprim_warpSum<32>'),
    ('(block_size + 63) / 64', '(block_size + 31) / 32'),
])

# Patch RasterizeToPixelsFromWorld3DGSBwd.cu
patch_file('gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSBwd.cu', [
    ('cg::thread_block_tile<64> warp = cg::tiled_partition<64>(block);', 'cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);'),
    ('rocprim::warp_reduce<float,64>', 'rocprim::warp_reduce<float,32>'),
    ('rocprim_warpSum<CDIM, 64>', 'rocprim_warpSum<CDIM, 32>'),
    ('rocprim_warpSum<64>', 'rocprim_warpSum<32>'),
    ('(block_size + 63) / 64', '(block_size + 31) / 32'),
])

print('Done patching gsplat for wave32 (gfx1100/gfx1151)')
