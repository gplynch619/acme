from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel
import yaml

class BlockMeta(BaseModel):
    name: str
    path: Path
    raw: dict

    model_config = {"arbitrary_types_allowed": True}

class BlockLibrary:
    def __init__(self, root: Path, blocks: dict[str, BlockMeta]):
        self._root = root
        self._blocks = blocks

    def get(self, name: str) -> BlockMeta:
        if name not in self._blocks:
            raise KeyError(f"Block not found: {name}")
        return self._blocks[name]

    def names(self) -> list[str]:
        return list(self._blocks.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._blocks

def load_block_library(root: Path) -> BlockLibrary:
    root = Path(root).resolve()
    blocks: dict[str, BlockMeta] = {}
    for path in sorted(root.rglob("*.yaml")):
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        name = path.stem
        if name in blocks:
            raise ValueError(f"Duplicate block name {name!r}: {path} vs {blocks[name].path}")
        blocks[name] = BlockMeta(name=name, path=path, raw=raw)
    return BlockLibrary(root, blocks)

