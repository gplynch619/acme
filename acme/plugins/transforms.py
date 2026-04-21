from __future__ import annotations
from acme.plugins import register_transform

@register_transform
def precision_upgrade_on_small_scale(plan):
    if "small_scale" not in plan.dataset_tags:
        return
    if plan.precision_block is None:
        return
    plan.blocks = [
        plan.precision_block if b == plan.theory_block else b for b in plan.blocks
    ]