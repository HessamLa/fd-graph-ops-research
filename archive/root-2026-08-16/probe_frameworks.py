#!/usr/bin/env python3
"""Framework probe. Run under EACH interpreter of interest:
     python3 probe_frameworks.py                     (system python)
     source .venv/bin/activate && python probe_frameworks.py   (FDMap venv)
Read-only; prints a compact block to paste back."""
import importlib
import platform
import subprocess
import sys

print("=" * 64)
print("interpreter :", sys.executable)
print("python      :", sys.version.split()[0], "|", platform.platform())

def ver(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "installed (no __version__)")
    except Exception as e:
        return f"not installed ({type(e).__name__})"

for m in ["numpy", "scipy", "jax", "jaxlib", "torch", "triton",
          "torch_scatter", "torch_sparse", "torch_geometric", "dgl",
          "numba", "cupy"]:
    print(f"{m:<16}: {ver(m)}")

try:
    import torch
    print("-" * 64)
    print("torch build :", torch.__version__,
          "| built for cuda:", torch.version.cuda,
          "| cudnn:", torch.backends.cudnn.version())
    print("cuda avail  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  device {i}: {p.name} | cc {p.major}.{p.minor} | "
                  f"{p.total_memory / 2**20:.0f} MiB | SMs {p.multi_processor_count}")
        print("arch list   :", torch.cuda.get_arch_list())
except ImportError:
    pass
except Exception as e:
    print("torch probe :", repr(e))

try:
    import jax
    print("-" * 64)
    print("jax         :", jax.__version__, "| backend:", jax.default_backend(),
          "| devices:", jax.devices())
except ImportError:
    pass
except Exception as e:
    print("jax probe   :", repr(e))

print("-" * 64)
try:
    out = subprocess.run([sys.executable, "-m", "pip", "list"],
                         capture_output=True, text=True, timeout=60).stdout
    keys = ("torch", "jax", "nvidia", "cuda", "cudnn", "triton",
            "numba", "scipy", "numpy", "dgl")
    keep = [l for l in out.splitlines() if any(k in l.lower() for k in keys)]
    print("pip list (relevant):")
    print("\n".join(keep) or "(none)")
except Exception as e:
    print("pip list failed:", repr(e))

try:
    out = subprocess.run([sys.executable, "-m", "pip", "index", "versions", "torch"],
                         capture_output=True, text=True, timeout=60)
    print("pypi torch versions (network, best effort):")
    print((out.stdout or out.stderr).strip() or "(no output)")
except Exception as e:
    print("pip index failed:", repr(e))
print("=" * 64)
