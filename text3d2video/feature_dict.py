from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class MvFeatureKey:
    module_path: str
    timestep: int
    view_index: int


@dataclass(frozen=True, eq=True)
class Feature3DKey:
    layer: str
    timestep: int
