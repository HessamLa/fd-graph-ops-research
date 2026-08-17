# verify_maxwell_triton.py
import torch

cc = torch.cuda.get_device_capability(0)
print(f"GPU: {torch.cuda.get_device_name(0)}, compute capability: {cc}")
assert cc[0] < 7, "This card is actually Volta+; Triton should work."

# --- Attempt a minimal Triton kernel ---
try:
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs,
                 tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
                 mask=mask)

    x = torch.ones(1024, device="cuda")
    out = torch.empty_like(x)
    add_kernel[(4,)](x, x, out, 1024, BLOCK=256)
    print("Triton kernel ran (unexpected on Maxwell!)")
except Exception as e:
    print(f"Triton failed as expected:\n  {type(e).__name__}: {e}")

# --- Attempt a minimal JAX Pallas kernel (Pallas -> Triton on GPU) ---
try:
    import jax, jax.numpy as jnp
    from jax.experimental import pallas as pl

    def kernel(x_ref, o_ref):
        o_ref[...] = x_ref[...] + 1.0

    x = jnp.ones((256,))
    y = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype))(x)
    print("Pallas kernel ran (unexpected on Maxwell!)")
except Exception as e:
    print(f"Pallas failed as expected:\n  {type(e).__name__}: {e}")


import triton
print("triton version:", triton.__version__)

x = torch.ones(1024, device="cuda")
out = torch.empty_like(x)
add_kernel[(4,)](x, x, out, 1024, BLOCK=256)
torch.cuda.synchronize()          # flush async errors
result = out.cpu()                # forces D2H copy, another sync point
if torch.allclose(result, torch.full((1024,), 2.0)):
    print("Triton kernel genuinely ran AND produced correct output")
else:
    print("Kernel 'launched' but output is wrong:", result[:8])
