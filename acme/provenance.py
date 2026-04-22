from __future__ import annotations
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel

import acme


class SpecSnapshot(BaseModel):
    name: str
    version: int
    chains_requested: list[str]
    sha256: str


class RegistrySnapshot(BaseModel):
    path: str
    sha256: str


class ResolvedChain(BaseModel):
    id: str
    model: str
    datasets: list[str]
    combo_alias: str | None
    blocks: list[str]
    augmenters: list[dict]
    transforms_applied: list[str]
    model_warnings: list[str]
    sampler_block: str
    slurm: dict
    convergence: dict
    injected_params: dict[str, dict] = {}


class CampaignManifest(BaseModel):
    acme_version: str
    composed_at: str
    spec: SpecSnapshot
    registry: RegistrySnapshot
    chains: list[ResolvedChain]


class ProvenanceRecord(BaseModel):
    acme_version: str
    composed_at: str
    chain_id: str
    spec_name: str
    spec_sha256: str
    registry_sha256: str
    resolved: ResolvedChain


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolved_chain(plan) -> ResolvedChain:
    return ResolvedChain(
        id=plan.id,
        model=plan.model,
        datasets=plan.datasets,
        combo_alias=plan.combo_alias,
        blocks=plan.blocks,
        augmenters=[a.model_dump() for a in plan.augmenters],
        transforms_applied=plan.transforms_applied,
        model_warnings=plan.model_warnings,
        sampler_block=plan.sampler_block,
        slurm=plan.slurm,
        convergence=plan.convergence,
    )


def write_manifest(plans, spec, registry, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    composed_at = _now()
    manifest = CampaignManifest(
        acme_version=acme.__version__,
        composed_at=composed_at,
        spec=SpecSnapshot(
            name=spec.name,
            version=spec.version,
            chains_requested=[c.id for c in spec.chains],
            sha256=_sha256(spec.registry),
        ),
        registry=RegistrySnapshot(
            path=str(registry),
            sha256=_sha256(registry),
        ),
        chains=[_resolved_chain(p) for p in plans],
    )
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))


def write_chain_provenance(plan, spec, registry_path: Path, chain_dir: Path) -> None:
    chain_dir = Path(chain_dir)
    composed_at = _now()
    record = ProvenanceRecord(
        acme_version=acme.__version__,
        composed_at=composed_at,
        chain_id=plan.id,
        spec_name=spec.name,
        spec_sha256=_sha256(spec.registry),
        registry_sha256=_sha256(registry_path),
        resolved=_resolved_chain(plan),
    )
    path = chain_dir / "provenance.json"
    path.write_text(json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True))


def write_analysis_summary(plans, spec, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    lines = [f"# {spec.name}\n"]
    for plan in plans:
        lines.append(f"## {plan.id}")
        lines.append(f"- model: {plan.model}")
        lines.append(f"- datasets: {', '.join(plan.datasets)}")
        if plan.combo_alias:
            lines.append(f"- combination: {plan.combo_alias}")
        lines.append(f"- blocks: {', '.join(plan.blocks)}")
        if plan.transforms_applied:
            lines.append(f"- transforms: {', '.join(plan.transforms_applied)}")
        if plan.augmenters:
            lines.append(f"- augmenters: {', '.join(a.name for a in plan.augmenters)}")
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines))


def write_chain_readme(plan, chain_dir: Path) -> None:
    chain_dir = Path(chain_dir)
    lines = [
        f"# {plan.id}",
        f"model: {plan.model}",
        f"datasets: {', '.join(plan.datasets)}",
        "",
        "## Blocks",
        *[f"- {b}" for b in plan.blocks],
    ]
    if plan.transforms_applied:
        lines += ["", "## Transforms", *[f"- {t}" for t in plan.transforms_applied]]
    if plan.augmenters:
        lines += ["", "## Augmenters", *[f"- {a.name} ({a.phase})" for a in plan.augmenters]]
    (chain_dir / "README.md").write_text("\n".join(lines))
