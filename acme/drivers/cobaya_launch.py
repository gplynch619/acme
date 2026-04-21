from __future__ import annotations
import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

@dataclass
class AugmenterContext:
    config: dict
    run_kwargs: dict = field(default_factory=dict)
    run_mode: str = "run"
    output_dir: str = ""
    kwargs: dict = field(default_factory=dict)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["launch"])
    parser.add_argument("--launch-dir", default="launch")
    args = parser.parse_args()

    launch_dir = Path(args.launch_dir)
    config = yaml.safe_load((launch_dir / "cobaya.yaml").read_text())
    augmenters = json.loads((launch_dir / "augmenters.json").read_text())

    # import augmenters modules before importing cobaya
    for entry in augmenters:
        importlib.import_module(entry["module"])

    from acme.plugins import AUGMENTER_REGISTRY

    ctx = AugmenterContext(config=config)

    phase_order = ["likelihood", "theory", "sampler", "finalize"]
    for phase in phase_order:
        for entry in augmenters:
            if entry["phase"] != phase:
                continue
            name = entry["name"]
            fn, _ = AUGMENTER_REGISTRY[name]
            ctx.kwargs = entry.get("kwargs", {})
            fn(ctx)

    import cobaya
    cobaya.run(ctx.config, **ctx.run_kwargs)

if __name__ == "__main__":
    main()