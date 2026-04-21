from __future__ import annotations
from pathlib import Path
import sys
import click


@click.group()
def main():
    pass


@main.command()
@click.option("--name", required=True, help="Campaign name")
def init(name):
    base = Path(name)
    if base.exists():
        click.echo(f"Directory {name!r} already exists.", err=True)
        sys.exit(1)
    (base / "blocks").mkdir(parents=True)
    (base / f"{name}_spec.yaml").write_text(f"""\
version: 1
name: {name}
backend: cobaya
registry: registry.yaml
block_library: blocks/
output_dir: output/

defaults:
  theory_profile: camb_default
  likelihood_profile: baseline
  sampler_block: cobaya_mcmc_default
  convergence:
    metric: R-1
    target: 0.01
  slurm:
    partition: cosmology
    walltime: "12:00:00"
    memory_gb: 32
    ntasks_per_node: 4

chains: []
""")
    (base / "registry.yaml").write_text("""\
version: 1
backend_family: cobaya
datasets: {}
combinations: {}
models: {}
theory_profiles: {}
likelihood_profiles: {}
transforms: []
""")
    click.echo(f"Initialized campaign scaffold in {name}/")


@main.command()
@click.argument("spec", type=click.Path(exists=True, path_type=Path))
@click.option("--dry-run", is_flag=True)
@click.option("--force", is_flag=True)
@click.option("--output-dir", type=click.Path(path_type=Path))
def compose(spec, dry_run, force, output_dir):
    from acme.analysis_spec import load_analysis_spec
    from acme.registry import load_registry
    from acme.blocks import load_block_library
    from acme.composer import Composer
    from acme.backends.cobaya import CobayaBackend
    from acme.writers import write_output

    analysis_spec = load_analysis_spec(spec)
    registry = load_registry(analysis_spec.registry)
    library = load_block_library(analysis_spec.block_library)
    backend = CobayaBackend()

    out = output_dir or analysis_spec.output_dir

    composer = Composer(analysis_spec, registry, library, backend)
    plans = composer.compose()

    if dry_run:
        for plan in plans:
            click.echo(f"{plan.id}: {plan.blocks}")
        return

    if out.exists() and not force:
        click.echo(f"Output directory {out} exists. Use --force to overwrite.", err=True)
        sys.exit(1)

    write_output(plans, analysis_spec, analysis_spec.registry, backend, library, out)
    click.echo(f"Composed {len(plans)} chain(s) to {out}")


@main.command()
@click.argument("spec", type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", type=click.Path(path_type=Path))
def status(spec, output_dir):
    from acme.analysis_spec import load_analysis_spec

    analysis_spec = load_analysis_spec(spec)
    out = output_dir or analysis_spec.output_dir

    for chain in analysis_spec.chains:
        chain_dir = out / chain.id
        cobaya_yaml = chain_dir / "launch" / "cobaya.yaml"
        provenance = chain_dir / "provenance.json"
        if not chain_dir.exists() or not cobaya_yaml.exists():
            status = "MISSING"
        elif provenance.exists():
            status = "READY"
        else:
            status = "STALE"
        click.echo(f"{chain.id}: {status}")


if __name__ == "__main__":
    main()
