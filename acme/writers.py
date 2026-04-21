from __future__ import annotations
from pathlib import Path
from acme.provenance import (
    write_manifest,
    write_chain_provenance,
    write_analysis_summary,
    write_chain_readme,
)


def write_output(plans, spec, registry_path, backend, library, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for plan in plans:
        chain_dir = output_dir / plan.id
        chain_dir.mkdir(parents=True, exist_ok=True)
        launch_dir = chain_dir / "launch"
        launch_dir.mkdir(exist_ok=True)

        files = backend.render_launch(plan, library)
        for rel_path, content in files.items():
            (chain_dir / rel_path).write_text(content)

        (chain_dir / "submit.slurm").write_text(backend.render_slurm_script(plan))

        write_chain_provenance(plan, spec, registry_path, chain_dir)
        write_chain_readme(plan, chain_dir)

    write_manifest(plans, spec, registry_path, output_dir)
    write_analysis_summary(plans, spec, output_dir)
