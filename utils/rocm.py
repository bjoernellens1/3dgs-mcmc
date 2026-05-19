import os
import torch

def is_rocm():
    return torch.version.hip is not None

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def print_device_info():
    print("torch:", torch.__version__)
    print("hip:", torch.version.hip)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print(f"  multi_processor_count: {props.multi_processor_count}")
        print(f"  total_memory: {props.total_memory / 1024**3:.1f} GB")

def setup_rocm_env():
    """Set recommended environment variables for ROCm / Strix Halo."""
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    os.environ.setdefault("HSA_XNACK", "1")
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
