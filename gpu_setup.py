"""
Note, this MUST be the first import in the entry point script (e.g. run.py), before any other imports.
Example:
    import gpu_setup  # noqa: F401 - must be first
    from gpu_setup import device, empty_cache  # for use in the rest of the code
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for Apple MPS; must come before torch import

import torch

assert not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1", \
    "gpu_setup.py must be imported before any other torch imports"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    

def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()