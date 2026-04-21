from __future__ import annotations
from acme.plugins import register_augmenter

@register_augmenter(phase="likelihood")
def candl_build_likelihoods(ctx):
    return None