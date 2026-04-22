from __future__ import annotations
import logging
import numpy as np
from acme.plugins import register_augmenter

log = logging.getLogger(__name__)


def _build_candl_like(ctx, like_cls):
    import candl
    import candl.interface

    data_set_file = ctx.kwargs["data_set_file"]
    name = ctx.kwargs.get("name", "candl_like")
    overrides = ctx.kwargs.get("nuisance_params", {})
    like_kwargs = {k: v for k, v in ctx.kwargs.items()
                  if k not in ("data_set_file", "name", "nuisance_params")}

    like = like_cls(data_set_file, **like_kwargs)

    unknown = set(overrides) - set(like.required_nuisance_parameters)
    if unknown:
        raise RuntimeError(
            f"{like_cls.__name__} [{name}]: nuisance_params keys not found "
            f"in candl required_nuisance_parameters: {unknown}"
        )

    cobaya_dict = candl.interface.get_cobaya_info_dict_for_like(like, name=name)
    ctx.config.setdefault("likelihood", {}).update(cobaya_dict)

    params_to_inject = {}
    for par_name in like.required_nuisance_parameters:
        if par_name in overrides:
            params_to_inject[par_name] = overrides[par_name]
            continue
        for prior in like.priors:
            if par_name in prior.par_names:
                idx = prior.par_names.index(par_name)
                loc = float(prior.central_value[idx])
                scale = float(np.sqrt(np.diag(prior.prior_covariance)[idx]))
                params_to_inject[par_name] = {
                    "prior": {"min": loc - 10 * scale, "max": loc + 10 * scale},
                    "ref": loc,
                    "proposal": scale,
                    "latex": par_name,
                }
                break

    ctx.config.setdefault("params", {}).update(params_to_inject)
    ctx.nuisance_params |= set(params_to_inject)
    log.info(f"[{name}] injected {len(params_to_inject)} nuisance params "
             f"({len(overrides)} registry overrides)")


@register_augmenter(phase="likelihood")
def candl_build_likelihoods(ctx):
    _build_candl_like(ctx, candl.Like)


@register_augmenter(phase="likelihood")
def candl_build_lensing(ctx):
    _build_candl_like(ctx, candl.LensLike)


@register_augmenter(phase="finalize")
def planck_act_coupling(ctx):
    likes = set(ctx.config.get("likelihood", {}).keys())
    has_act = any("act" in k.lower() for k in likes)
    has_planck = any("planck" in k.lower() for k in likes)

    if not (has_act and has_planck):
        log.info("planck_act_coupling: ACT+Planck not both present, skipping")
        return

    ctx.config.setdefault("params", {})["A_planck"] = {
        "value": "lambda A_act: A_act",
        "latex": r"A_{\mathrm{Planck}}",
        "derived": True,
    }
    ctx.nuisance_params.discard("A_planck")
    log.info("planck_act_coupling: A_planck set as derived from A_act")
