import os.path as osp
import shutil

with open('setup.py', 'r') as f:
    txt = f.read()

# Patch 0: ensure shutil is imported
if 'import shutil' not in txt:
    txt = txt.replace('import re', 'import re\nimport shutil')

# Patch 0b: respect PYTORCH_ROCM_ARCH env var in get_rocm_arch()
arch_start_idx = txt.find('def get_rocm_arch():')
if arch_start_idx != -1:
    try_idx = txt.find('    try:', arch_start_idx)
    if try_idx != -1:
        txt = txt[:try_idx] + '''    import os
    env_arch = os.environ.get('PYTORCH_ROCM_ARCH', '')
    if env_arch:
        print(f"Using PYTORCH_ROCM_ARCH environment variable: {env_arch}")
        return env_arch
''' + txt[try_idx:]

# Patch 1: add glm_path to ROCm include_dirs
old_inc = '''        include_dirs = [
            osp.join(current_dir, "gsplat", "cuda", "include"),'''
new_inc = '''        glm_path = osp.join(current_dir, "gsplat", "cuda", "csrc", "third_party", "glm")
        include_dirs = [
            glm_path,
            osp.join(current_dir, "gsplat", "cuda", "include"),'''
txt = txt.replace(old_inc, new_inc)

# Patch 2: replace hipified GLM tree with symlink after CUDAExtension creation
old_return = '''        extension = CUDAExtension(
            # Make sure this matches your package structure
            "gsplat.csrc",  # This changes the extension module name to be more standard
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            undef_macros=undef_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        )
        return [extension]'''

new_return = '''        extension = CUDAExtension(
            # Make sure this matches your package structure
            "gsplat.csrc",  # This changes the extension module name to be more standard
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            undef_macros=undef_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        )
        # Fix: hipify copies GLM headers but not .inl files, and modifies them.
        # Replace the hipified GLM tree with a symlink to the original.
        hip_glm = osp.join(str(current_dir), "gsplat", "hip", "csrc", "third_party", "glm")
        cuda_glm = osp.join(str(current_dir), "gsplat", "cuda", "csrc", "third_party", "glm")
        if os.path.isdir(hip_glm) and not os.path.islink(hip_glm):
            shutil.rmtree(hip_glm)
            rel = os.path.relpath(cuda_glm, os.path.dirname(hip_glm))
            os.symlink(rel, hip_glm)
        return [extension]'''

txt = txt.replace(old_return, new_return)

with open('setup.py', 'w') as f:
    f.write(txt)

print('Patched setup.py for GLM symlink fix')
