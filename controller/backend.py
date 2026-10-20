"""Where the JAX parts of the controller run, and in what precision. Set before the first import:

    ADMM_PLATFORM    auto | cpu | gpu      JAX backend; auto takes JAX's default (the GPU when jax[cuda12]
                                           is installed and sees one)
    ADMM_PRECISION   auto | 64 | 32        layer-1 (and batched-loop) floats; auto: float64 on the CPU,
                                           float32 on a GPU

Consumer GPUs (GeForce/RTX) run float64 at 1/32 to 1/64 of their float32 rate, hence the float32
default there. float32 matmuls are pinned to full precision: on Ampere and later GPUs XLA's default
float32 matmul is TF32 (10-bit mantissa), too coarse for the ADMM and the swing MPC, which is float32 on
every backend (the precision its operators were fit at). The pin does not change float64 or CPU results.
Precision is process-wide (jax_enable_x64), so this module is imported before any other JAX use; the
controller modules import it first."""
import os
import jax
import jax.numpy as jnp

_plat = os.environ.get("ADMM_PLATFORM", "auto").strip().lower()
if _plat == "gpu":
    _plat = "cuda"
if _plat != "auto":
    try:
        from jax._src import xla_bridge
        started = xla_bridge.backends_are_initialized()
    except (ImportError, AttributeError):
        started = False
    if started:
        raise RuntimeError("ADMM_PLATFORM is set but JAX already started: import backend before any JAX computation")
    jax.config.update("jax_platforms", _plat)

PLATFORM = jax.default_backend()                               # 'cpu' or 'gpu'
_prec = os.environ.get("ADMM_PRECISION", "auto").strip().lower()
if _prec not in ("auto", "64", "32"):
    raise ValueError(f"ADMM_PRECISION must be auto, 64 or 32, not {_prec!r}")
X64 = PLATFORM == "cpu" if _prec == "auto" else _prec == "64"
jax.config.update("jax_enable_x64", X64)
jax.config.update("jax_default_matmul_precision", "highest")
DTYPE = jnp.float64 if X64 else jnp.float32


def info():
    """One line: backend, devices and precision."""
    devs = ", ".join(f"{d.platform}:{d.device_kind}" for d in jax.devices())
    return f"jax {jax.__version__} | backend {PLATFORM} ({devs}) | {'float64' if X64 else 'float32'}"
