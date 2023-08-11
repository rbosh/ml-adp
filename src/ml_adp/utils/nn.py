from typing import Any, Optional, Sequence
from collections import OrderedDict  # To be consistent with Pytorch implementation (even though dicts are ordered now)
import itertools as it

import torch


class ModuleList(torch.nn.Module):
    """Mutable, Fixed-Length List Supporting Pytorch Modules"""
    def __init__(self, *entries) -> None:
        super().__init__()

        self._non_module_entries = OrderedDict(zip(map(str, range(len(entries))), it.repeat(None)))
        self.__setitem__(slice(None, None), entries)

    def __repr__(self) -> str:
        return repr([entry for entry in self])

    def _register_entry(self, idx: int, value: Any) -> None:
        del self._entry_dict_by_idx(idx)[str(idx)]
        if isinstance(value, torch.torch.nn.Module):
            setattr(self, str(idx), value)
        else:
            self._non_module_entries[str(idx)] = value

    def _entry_dict_by_idx(self, idx: int) -> OrderedDict:
        try:
            self._modules[str(idx)]
        except KeyError:
            return self._non_module_entries
        else:
            return self._modules

    def __len__(self) -> int:
        return len(self._non_module_entries) + len(self._modules)

    def __setitem__(self,
                    key: int | slice,
                    value: Optional[Sequence[Optional[Any] | callable]]) -> None:

        idx = list(range(len(self)))[key]
        if isinstance(idx, int):
            self._register_entry(idx, value)
        else:
            if callable(value) or value is None:
                value = it.repeat(value, len(idx))
            for i, entry in zip(idx, value, strict=True):
                self._register_entry(i, entry)

    def __getitem__(self,
                    key: int | slice) -> Optional[Any] | 'ModuleList':
        idx = list(range(len(self)))[key]
        if isinstance(idx, int):
            return self._entry_dict_by_idx(idx)[str(idx)]
        else:
            return self.__class__(*[self._entry_dict_by_idx(idx)[str(idx)] for idx in idx])
