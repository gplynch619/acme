from __future__ import annotations
import logging
from pathlib import Path
from OLE.interfaces.cobaya_interface import OLE_callback_function
from acme.plugins import register_augmenter

log = logging.getLogger(__name__)

def _find_ole_theory_key(ctx) -> str:
    theory = ctx.config.get("theory", {})
    ole_entries = [k for k, v in theory.items() if isinstance(v, dict) and "emulate" in v]
    if not ole_entries:
        raise RuntimeError("OLE augmenter: no theory entry with 'emulate' key found")
    if len(ole_entries) > 1:
        raise RuntimeError(f"OLE augmenter: multiple OLE theory entries found: {ole_entries}")
    return ole_entries[0]

@register_augmenter(phase="finalize")
def ole_paths(ctx):
    key = _find_ole_theory_key(ctx)
    if not ctx.output_dir:
        raise RuntimeError("ole_paths: ctx.output_dir is not set — ensure 'output' is defined in cobaya.yaml")
    out = Path(ctx.output_dir).parent
    ctx.config["theory"][key].setdefault("emulator_settings", {}).update({
        "working_directory": str(out / "chains_emulator"),
        "plotting_directory": str(out / "debug_plots"),
    })
    log.info(f"ole_paths: set emulator paths under theory[{key!r}] → {out}")

@register_augmenter(phase="finalize")
def ole_mcmc(ctx):
    key = _find_ole_theory_key(ctx)
    all_sampled = {
        k for k, v in ctx.config.get("params", {}).items()
        if isinstance(v, dict) and "prior" in v
    }
    nuisance = sorted(ctx.nuisance_params & all_sampled)
    cosmo = sorted(all_sampled - ctx.nuisance_params)

    mcmc = ctx.config.setdefault("sampler", {}).setdefault("mcmc", {})
    mcmc["drag"] = False
    mcmc["callback_every"] = 1
    mcmc["callback_function"] = OLE_callback_function
    mcmc["rejection_logging"] = True

    if "camb" in key.lower():
        mcmc.pop("oversample_power", None)
        mcmc["blocking"] = [[1, cosmo], [1, nuisance]]
    else:
        mcmc["oversample_power"] = 0
        mcmc.pop("blocking", None)

    log.info(f"ole_mcmc: {len(cosmo)} cosmo params, {len(nuisance)} nuisance params, theory={key!r}")

@register_augmenter(phase="cleanup")
def ole_dimensionality(ctx):
    key = _find_ole_theory_key(ctx)
    n = sum(
        1 for v in ctx.config.get("params", {}).values()
        if isinstance(v, dict) and "prior" in v
    )
    ctx.config["theory"][key].setdefault("emulator_settings", {})["dimensionality"] = n
    log.info(f"ole_dimensionality: set dimensionality={n} on theory[{key!r}]")