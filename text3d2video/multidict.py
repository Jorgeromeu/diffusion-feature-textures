import os
from pathlib import Path
from typing import Any, Callable, Dict


class MultiDict:

    """
    Utility class for dictionary with dict keys
    """

    def __init__(self):
        self._data = dict()

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

    def add(self, key: Dict[str, Any], value: Any):
        self._data[self.dict_to_str(key)] = value

    def get(self, key: Dict[str, Any]) -> Any:
        return self._data.get(self.dict_to_str(key))

    def values(self):
        return list(self._data.values())

    def keys(self):
        return [self.str_to_dict(k) for k in self._data.keys()]

    def items(self):
        return [(self.str_to_dict(k), v) for k, v in self._data.items()]

    def __getitem__(self, key: Dict[str, Any]):
        return self.get(key)

    def __setitem__(self, key: Dict[str, Any], value: Any):
        self.add(key, value)

    def __contains__(self, key: Dict[str, Any]):
        return self.dict_to_str(key) in self._data

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"MultiDict({self._data})"

    def __str__(self):
        return '{' + ','.join([f'{k}: v' for k, v in self._data.items()]) + '}'

    def __delitem__(self, key: Dict[str, Any]):
        del self._data[self.dict_to_str(key)]

    def serialize_multidict(
        self,
        folder: Path,
        extension: str,
        save_fun: Callable
    ):

        for k, v in self.items():
            name = self.dict_to_str(k)
            filename = folder / f'{name}.{extension}'
            save_fun(v, filename)

    @classmethod
    def read_multidict(
        self,
        folder: Path,
        load_fun: Callable
    ):

        d = MultiDict()

        for f in os.listdir(folder):
            no_extension = f.split('.')[0]
            key = self.str_to_dict(no_extension)
            value = load_fun(folder / f)
            d[key] = value

        return d
