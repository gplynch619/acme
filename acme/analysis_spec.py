from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel, model_validator
import yaml

class SlurmDefaults(BaseModel):
    partition: str
    walltime: str
    memory_gb: int
    ntasks_per_node: int

class ConvergenceDefaults(BaseModel):
    metric: str
    target: float

class DataRequest(BaseModel):
    name: str
    datasets: list[str]

class Defaults(BaseModel):
    theory_profile: str
    likelihood_profile: str
    sampler_block: str
    convergence: ConvergenceDefaults
    slurm: SlurmDefaults

class ChainSpec(BaseModel):
    id: str
    model: str
    data: DataRequest
    theory_profile: str | None = None
    likelihood_profile: str | None = None
    sampler_block: str | None = None
    convergence: ConvergenceDefaults | None = None
    slurm: SlurmDefaults | None = None

class AnalysisSpec(BaseModel):
    version: int
    name: str
    backend: str
    registry: Path
    block_library: Path
    output_dir: Path
    defaults: Defaults
    chains: list[ChainSpec]

    @model_validator(mode="before")
    @classmethod
    def resolve_paths(cls, data: dict, info) -> dict:
        spec_dir = (info.context or {}).get("spec_dir")
        if spec_dir:
            spec_dir = Path(spec_dir)
            for field in ("registry", "block_library", "output_dir"):
                if field in data:
                    p = Path(data[field])
                    if not p.is_absolute():
                        data[field] = spec_dir / p
        return data

    @model_validator(mode="after")
    def validate_version_and_backend(self) -> AnalysisSpec:
        if self.version != 1:
            raise ValueError(f"Unsupported version: {self.version}. Expected 1.")
        if self.backend != "cobaya":
            raise ValueError(f"Unsupported backend: '{self.backend}'. Expected 'cobaya'.")
        return self

def load_analysis_spec(path: Path) -> AnalysisSpec:
    path = Path(path).resolve()
    with open(path) as f:
        data = yaml.safe_load(f)
    return AnalysisSpec.model_validate(data, context={"spec_dir": path.parent})