from __future__ import annotations
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
import yaml

class AugmenterRef(BaseModel):
    name: str
    phase: Literal["likelihood", "theory", "sampler", "finalize", "cleanup"]
    kwargs: dict = {}
    nuisance_params: dict[str, dict] = {}

class DatasetEntry(BaseModel):
    blocks: list[str]
    tags: list[str] = []
    candl: dict | None = None

class CombinationEntry(BaseModel):
    matches: list[str]
    blocks: list[str]
    tags: list[str] = []
    alias: str | None = None

class ModelEntry(BaseModel):
    param_block: str
    warnings: list[str] = []
    excluded_tags: list[str] = []

class TheoryProfileEntry(BaseModel):
    default_block: str
    precision_block: str | None = None
    augmenters: list[AugmenterRef] = []

class LikelihoodProfileEntry(BaseModel):
    blocks: list[str] = []
    augmenters: list[AugmenterRef] = []

class Registry(BaseModel):
    version: int
    backend_family: str
    datasets: dict[str, DatasetEntry] = {}
    combinations: dict[str, CombinationEntry] = {}
    models: dict[str, ModelEntry] = {}
    theory_profiles: dict[str, TheoryProfileEntry] = {}
    likelihood_profiles: dict[str, LikelihoodProfileEntry] = {}
    transforms: list[str] = []

def load_registry(path: Path) -> Registry:
    path = Path(path).resolve()
    with open(path) as f:
        data = yaml.safe_load(f)
    return Registry.model_validate(data)