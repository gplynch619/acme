from __future__ import annotations
import json
from string import Template
from acme.composer import BlockPlan, CompositionError
from acme.plugins import AUGMENTER_REGISTRY
import yaml

def _deep_merge(base: dict, override: dict, chain_id: str, block_name: str) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = _deep_merge(result[key], value, chain_id, block_name)
            else:
                raise CompositionError([f"Chain {chain_id!r}: key conflict on {key!r} merging block {block_name!r}"])
        else:
            result[key] = value
    return result

class CobayaBackend:

    name = "cobaya"
    def render_launch(self, plan: BlockPlan, library) -> dict[str, str]:
        config: dict = {}
        for block_name in plan.blocks:
            block = library.get(block_name)
            config = _deep_merge(config, block.raw, plan.id, block_name)

        config.setdefault("output", f"chains/{plan.id}/{plan.id}")

        augmenters_list = []
        for aug in plan.augmenters:
            fn, _ = AUGMENTER_REGISTRY[aug.name]
            augmenters_list.append({
                "name": aug.name,
                "module": fn.__module__,
                "phase": aug.phase,
                "kwargs": aug.kwargs,
                "nuisance_params": aug.nuisance_params,
            })

        return {"launch/cobaya.yaml": yaml.dump(config, default_flow_style=False, sort_keys = True),
                "launch/augmenters.json": json.dumps(augmenters_list, indent=2, sort_keys=True),
        }

    def render_slurm_script(self, plan: BlockPlan) -> str:
        slurm = plan.slurm
        template = Template("""\
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${partition}
#SBATCH --time=${walltime}
#SBATCH --mem=${memory_gb}G
#SBATCH --ntasks-per-node=${ntasks_per_node}
#SBATCH --output=logs/%x_%j.out

mpirun -n ${ntasks_per_node} python -m acme.drivers.cobaya_launch launch
""")
        return template.substitute(
            job_name=plan.id,
            partition=slurm["partition"],
            walltime=slurm["walltime"],
            memory_gb=slurm["memory_gb"],
            ntasks_per_node=slurm["ntasks_per_node"],
        )