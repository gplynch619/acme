from __future__ import annotations
import warnings
from typing import Any
from pydantic import BaseModel
from acme.registry import Registry, AugmenterRef

class NormalizedIntent(BaseModel):
    id: str
    model: str
    datasets: list[str]
    theory_profile: str
    likelihood_profile: str
    sampler_block: str
    slurm: dict
    convergence: dict

class BlockPlan(BaseModel):
    id: str
    model: str
    datasets: list[str]
    theory_profile: str
    likelihood_profile: str
    sampler_block: str
    slurm: dict
    convergence: dict
    combo_alias: str | None = None
    dataset_tags: list[str] = []
    theory_block: str | None = None
    precision_block: str | None = None
    blocks: list[str] = []
    augmenters: list[AugmenterRef] = []
    model_warnings: list[str] = []
    transforms_applied: list[str] = []
    nuisance_params: dict[str, dict] = {}

    model_config = {"arbitrary_types_allowed": True}

class CompositionError(Exception):
    def __init__(self, errors:list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))

def normalize(chain, defaults) -> NormalizedIntent:
    return NormalizedIntent(
        id = chain.id,
        model = chain.model,
        datasets = chain.data.datasets,
        theory_profile = chain.theory_profile or defaults.theory_profile,
        likelihood_profile = chain.likelihood_profile or defaults.likelihood_profile,
        sampler_block = chain.sampler_block or defaults.sampler_block,
        slurm = (chain.slurm or defaults.slurm).model_dump(),
        convergence = (chain.convergence or defaults.convergence).model_dump(),
    )

def resolve(intent: NormalizedIntent, registry: Registry) -> BlockPlan:
    model = registry.models[intent.model]
    theory = registry.theory_profiles[intent.theory_profile]
    lik = registry.likelihood_profiles[intent.likelihood_profile]

    blocks: list[str] = []
    augmenters: list[AugmenterRef] = []
    dataset_tags: list[str] = []
    combo_alias: str | None = None
    model_warnings = list(model.warnings)

    #1. param block
    blocks.append(model.param_block)

    #2. theory block + augmenters
    blocks.append(theory.default_block)
    augmenters.extend(theory.augmenters)

    #3. datasets (check for combinations first)
    matched_combo = None
    for combo in registry.combinations.values():
        if set(combo.matches) <= set(intent.datasets):
            if matched_combo is None:
                matched_combo = combo
                combo_alias = combo.alias
            else:
                model_warnings.append(f"Multiple combinations found for datasets {intent.datasets!r} - using first match")

    combo_covered: set[str] = set()
    if matched_combo is not None:
        blocks.extend(matched_combo.blocks)
        dataset_tags.extend(matched_combo.tags)
        combo_covered = set(matched_combo.matches)

    for ds_name in intent.datasets:
        if ds_name in combo_covered:
            continue
        ds = registry.datasets[ds_name]
        blocks.extend(ds.blocks)
        dataset_tags.extend(ds.tags)

    #4. likelihood extra blocks + augmenters
    blocks.extend(lik.blocks)
    augmenters.extend(lik.augmenters)

    #5. sampler block
    blocks.append(intent.sampler_block)

    # make plan
    plan = BlockPlan(
        id = intent.id,
        model=intent.model,
        datasets=intent.datasets,
        theory_profile=intent.theory_profile,
        likelihood_profile=intent.likelihood_profile,
        sampler_block=intent.sampler_block,
        slurm=intent.slurm,
        convergence=intent.convergence,
        combo_alias = combo_alias,
        dataset_tags = list(dict.fromkeys(dataset_tags)),
        theory_block = theory.default_block,
        precision_block = theory.precision_block,
        blocks=blocks,
        augmenters=augmenters,
        model_warnings = model_warnings,
    ) 

    #6. apply transforms
    import acme.plugins.transforms
    from acme.plugins import TRANSFORM_REGISTRY
    for name in registry.transforms:
        fn = TRANSFORM_REGISTRY[name]
        fn(plan)
        plan.transforms_applied.append(name)

    #7. sort augmenters
    phase_order = {"likelihood": 0, "theory": 1, "sampler": 2, "finalize": 3, "cleanup": 4}
    plan.augmenters = sorted(plan.augmenters, key=lambda a: phase_order[a.phase])

    return plan

def validate(plans: list[BlockPlan], library, registry: Registry) -> None:
    from acme.plugins import AUGMENTER_REGISTRY
    errors: list[str] = []

    ids = [p.id for p in plans]
    seen: set[str] = set()
    for chain_id in ids:
        if chain_id in seen:
            errors.append(f"Duplicate chain id: {chain_id!r}")
        seen.add(chain_id)
    
    for plan in plans:
        model = registry.models.get(plan.model)
        if model:
            for tag in model.excluded_tags:
                if tag in plan.dataset_tags:
                    errors.append(
                        f"Chain {plan.id!r}: model {plan.model!r} excludes tag {tag!r} "
                        f"but dataset tags {plan.dataset_tags!r} include it"
                        )

        for block_name in plan.blocks:
            if block_name not in library:
                errors.append(f"Chain {plan.id!r}: block {block_name!r} not found in library")

        for aug in plan.augmenters:
            if aug.name not in AUGMENTER_REGISTRY:
                errors.append(f"Chain {plan.id!r}: augmenter {aug.name!r} not registered")

    for plan in plans:
        # get all static params
        static_params: set[str] = set()
        for block_name in plan.blocks:
            block_yaml = library.get(block_name).raw if block_name in library else {}
            static_params.update(block_yaml.get("params", {}).keys())

        # get injected params and check for collisions
        seen_injected: dict[str, str] = {}  # param_name -> augmenter name
        for aug in plan.augmenters:
            for param_name in aug.nuisance_params:
                if param_name in static_params:
                    errors.append(
                        f"Chain {plan.id!r}: param {param_name!r} is in a static block "
                        f"and also injected by augmenter {aug.name!r}"
                    )
                elif param_name in seen_injected:
                    errors.append(
                        f"Chain {plan.id!r}: param {param_name!r} injected by both "
                        f"{seen_injected[param_name]!r} and {aug.name!r}"
                    )
                else:
                    seen_injected[param_name] = aug.name
        for aug in plan.augmenters:
            plan.nuisance_params.update(aug.nuisance_params)

    for plan in plans:
        for w in plan.model_warnings:
            warnings.warn(f"Chain {plan.id!r}: {w}", stacklevel=2)
    
    if errors:
        raise CompositionError(errors)

class Composer:
    def __init__(self, spec, registry: Registry, library, backend):
        self.spec = spec
        self.registry = registry
        self.library = library
        self.backend = backend

    def compose(self) -> list[BlockPlan]:
        import acme.plugins.transforms
        import acme.plugins.augmenters.ole
        import acme.plugins.augmenters.candl

        intents = [normalize(c, self.spec.defaults) for c in self.spec.chains]
        plans = [resolve(i, self.registry) for i in intents]
        validate(plans, self.library, self.registry)
        return plans