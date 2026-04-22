from __future__ import annotations
import logging
from acme.plugins import register_augmenter

log = logging.getLogger(__name__)

@register_augmenter(phase="cleanup")
def suppress_jax_logging(ctx):
    for name in [
        "root", "jax", "jax._src.dispatch", "jax._src.interpreters.mlir",
        "matplotlib.font_manager", "cobaya",
    ]:
        logging.getLogger(name).disabled = True
    log.info("suppress_jax_logging: JAX/Cobaya loggers suppressed")