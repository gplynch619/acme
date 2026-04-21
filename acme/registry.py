from __future__ import annotations
from typing import Literal
from pydantic import BaseModel
from pathlib import Path
import yaml

class AugmenterRef(BaseModel):
    name: str
    phase: Literal["likelihood", "theory", "sampler", "finalize"]
    kwargs: dict = {}

class TransformRef(BaseModel):
    name: str
    when: dict = {}

class DatasetEntry(BaseModel):
    name: str
    when: dict = {}

class DatasetEntry(BaseModel):
    blocks: list[str]
    tags: list[str] - []
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
    driver_imports: list[str] = []
    augmenters: list[AugmenterRef] = []

class LikelihoodProfileEntry(BaseModel):
    blocks: list[str] = []
    driver_imports: list[str] = []
    augmenters: list[AugmenterRef] = []

class Registry(BaseModel):
    version: int
    backend_family: str
    datasets: dict[str, DatasetEntry] = {}
    combinations: dict[str, CombinationEntry] = {}
    models: dict[str, ModelEntry] = {}
    theory_profiles: dict[str, TheoryProfileEntry] = {}
    likelihood_profiles: dict[str, LikelihoodProfileEntry] = {}
    transforms: list[TransformRef] = []

def load_registry(path: Path) -> Registry:
    path = Path(path).resolve()
    with open(path) as f:
        data = yaml.safe_load(f)
    return Registry.model_validate(data)