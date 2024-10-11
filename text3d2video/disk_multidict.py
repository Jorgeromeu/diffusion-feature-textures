from pathlib import Path
from typing import Any, Callable, Dict

import torch


class DiskMultiDict:
    """
    Utility class for storing data on disk with dictionary keys
    """

    def __init__(
        self,
        path: Path,
        save_fun: Callable[[Any], str],
        load_fun: Callable[[str], Any],
        file_extension: str,
        init_empty: bool = False,
    ):
        self.path = path

        self.serialization_fun = save_fun
        self.deserialization_fun = load_fun
        self.file_extension = file_extension

        if not path.exists():
            path.mkdir(parents=True)

        if init_empty:
            self.clear()

    @classmethod
    def dict_to_string_dict(cls, keys: Dict[str, Any]) -> Dict[str, str]:
        return {str(k): str(v) for k, v in keys.items()}

    @classmethod
    def dict_to_str(cls, keys: Dict[str, Any]) -> str:
        str_dict = cls.dict_to_string_dict(keys)
        items = sorted(str_dict.items())
        items_str = [f"{k}={v}" for k, v in items]
        return "&".join(items_str)

    @classmethod
    def str_to_dict(cls, key: str) -> Dict[str, Any]:
        items_str = key.split("&")
        items = [item.split("=") for item in items_str]
        return {k: v for k, v in items}

    def clear(self):
        for file in self.path.iterdir():
            file.unlink()

    def add(self, key: Dict[str, Any], value: Any):
        key_str = self.dict_to_str(key)
        path = self.path / f"{key_str}.{self.file_extension}"
        self.serialization_fun(value, path)

    def get(self, key: Dict[str, Any]) -> Any:
        key_str = self.dict_to_str(key)
        path = self.path / f"{key_str}.{self.file_extension}"
        return self.deserialization_fun(path)

    def keys(self):
        return [self.str_to_dict(f.stem) for f in self.path.iterdir()]

    def values(self):
        return [self.deserialization_fun(f) for f in self.path.iterdir()]

    def items(self):
        return [
            (self.str_to_dict(path.stem), self.deserialization_fun(path))
            for path in self.path.iterdir()
        ]

    def key_values(self, key: str):
        values = set()
        for identifier in self.keys():
            values.add(identifier[key])
        return list(values)

    def __getitem__(self, key: Dict[str, Any]):
        return self.get(key)

    def __setitem__(self, key: Dict[str, Any], value: Any):
        self.add(key, value)

    def __contains__(self, key: Dict[str, Any]):
        return self.dict_to_str(key) in [self.dict_to_str(k) for k in self.keys()]

    def __len__(self):
        return len(self.keys())

    def __repr__(self):
        return f"OnDiskMultiDict(path={self.path}, serialization_fun={self.serialization_fun}, deserialization_fun={self.deserialization_fun})"

    def __delitem__(self, key: Dict[str, Any]):
        pass


class TensorDiskMultiDict(DiskMultiDict):
    """
    Utility class for storing pytorch tensors on disk
    """

    def __init__(self, path: Path, init_empty: bool = False):

        super().__init__(
            path,
            save_fun=lambda t, p: torch.save(t.cpu(), p),
            load_fun=torch.load,
            file_extension="pt",
            init_empty=init_empty,
        )
