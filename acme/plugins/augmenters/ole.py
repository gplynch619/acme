from __future__ import annotations
from acme.plugins import register_augmenter

@register_augmenter(phase="theory")
def ole_paths(ctx):
    return None

@register_augmenter(phase="theory")
def camb_blocking(ctx):
    return None