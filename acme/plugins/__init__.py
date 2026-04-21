from __future__ import annotations
from typing import Callable

TRANSFORM_REGISTRY: dict[str, Callable] = {}
AUGMENTER_REGISTRY: dict[str, tuple[Callable, str]] = {} #name -> (fn, phase)

def register_transform(fn: Callable) -> Callable:
    TRANSFORM_REGISTRY[fn.__name__] = fn
    return fn

def register_augmenter(phase: str):
    def decorator(fn: Callable) -> Callable:
        AUGMENTER_REGISTRY[fn.__name__] = (fn, phase)
        return fn
    return decorator