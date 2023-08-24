""" Neural network utilities
"""
from __future__ import annotations

from typing import Any
from collections import OrderedDict  # To be consistent with Pytorch implementation (even though dicts are ordered now)
from collections.abc import Sequence
import itertools as it
from contextlib import contextmanager

import torch
from torch.nn import Module


class ModuleArray(Module, Sequence):
    """Array like container for Pytorch modules"""
    def __init__(self, *entries) -> None:
        super().__init__()

        self._non_module_entries = OrderedDict(zip(map(str, range(len(entries))), it.repeat(None)))
        self.__setitem__(slice(None, None), entries)

    def __repr__(self) -> str:
        return repr([entry for entry in self])

    def _register_entry(self, idx: int, value: Any) -> None:
        del self._entry_dict_by_idx(idx)[str(idx)]
        if isinstance(value, Module):
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
                    value: (None | callable | Sequence[None | callable])) -> None:

        idx = list(range(len(self)))[key]
        if isinstance(idx, int):
            self._register_entry(idx, value)
        else:
            if callable(value) or value is None:
                value = it.repeat(value, len(idx))
            for i, entry in zip(idx, value, strict=True):
                self._register_entry(i, entry)

    def __getitem__(self,
                    key: int | slice) -> (None | callable) | ModuleArray:
        idx = list(range(len(self)))[key]
        if isinstance(idx, int):
            return self._entry_dict_by_idx(idx)[str(idx)]
        else:
            return self.__class__(*[self._entry_dict_by_idx(idx)[str(idx)] for idx in idx])


@contextmanager
def evaluating(model: Module):
    '''Temporarily switch to evaluation mode.
    
    From: https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-
    manager/18998/3
    (MIT Licensed)
    '''
    training = model.training
    try:
        model.eval()
        yield model
    except AttributeError:
        yield model
    finally:
        if training:
            model.train()