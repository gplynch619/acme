from __future__ import annotations
import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml
import argparse
import logging

from acme.plugins import AUGMENTER_REGISTRY


@dataclass
class AugmenterContext:
    config: dict
    run_kwargs: dict = field(default_factory=dict)
    run_mode: str = "run"
    output_dir: str = ""
    kwargs: dict = field(default_factory=dict)

def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("acme.driver")

    parser = argparse.ArgumentParser(description="Run a composed ACME chain")
    parser.add_argument("launch_dir", help="Path to the chain's launch/ directory")
    parser.add_argument("--dry-run", action="store_true", help="Run augmenters but skip cobaya.run")
    parser.add_argument("--resume", action="store_true", help="Pass resume=True to cobaya.run")

    args = parser.parse_args()
    launch_dir = Path(args.launch_dir)
    config = yaml.safe_load((launch_dir / "cobaya.yaml").read_text())
    augmenters = json.loads((launch_dir / "augmenters.json").read_text())

    # import augmenters modules before importing cobaya
    for entry in augmenters:
        try:
            importlib.import_module(entry["module"])
        except ImportError as e:
            log.error(f"Failed to import augmenter module {entry['module']}: {e}")
            sys.exit(2)

    ctx = AugmenterContext(config=config)

    phase_order = ["likelihood", "theory", "sampler", "finalize"]
    for phase in phase_order:
        for entry in augmenters:
            if entry["phase"] != phase:
                continue
            name = entry["name"]
            if name not in AUGMENTER_REGISTRY:
                log.error(f"Unknown augmenter {name}  - is its module listed in augmenters.json?")
                sys.exit(2)
            fn, _ = AUGMENTER_REGISTRY[name]
            ctx.kwargs = entry.get("kwargs", {})
            try:
                log.info(f"[{phase}] Running augmenter {name} with kwargs {ctx.kwargs}")
                fn(ctx)
            except Exception as e:
                log.exception(f"Failed to run augmenter {name} in phase {phase}:")
                sys.exit(2)

    if args.dry_run:
        print(yaml.dump(ctx.config, default_flow_style=False, sort_keys=True))
        sys.exit(0)

    import cobaya
    try:
        cobaya.run(ctx.config, **ctx.run_kwargs, resume=args.resume)
    except Exception as e:
        log.exception(f"Failed to run cobaya.run: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()