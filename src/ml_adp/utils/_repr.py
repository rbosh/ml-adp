from __future__ import annotations
from typing import Any, Optional, TYPE_CHECKING
import shutil

import torch

if TYPE_CHECKING:
    from ml_adp import StateEvolution, CostToGo


PRINT_WIDTH, PRINT_HEIGHT = shutil.get_terminal_size((80, 20))


def table_repr(object: object) -> str:
    if object is None:
        _repr = "None"
    elif object.__class__.__name__ == "function" or object.__class__.__name__ == "method":
        _repr = object.__name__
    elif isinstance(object, torch.nn.Module):
        _repr = repr(object)
    else:
        _repr = str(object)
    
    return _repr


def create_table(module: StateEvolution | CostToGo, width: Optional[int] = None, height: Optional[int] = None) -> str:
        # Prepare ...
        if width is None:
            width = PRINT_WIDTH

        if height is None:
            height = PRINT_HEIGHT

        if height <= 8:
            raise ValueError("Height too small")


        headers = ["state_func"]
        columns = [module.state_functions]

        try:
            columns.append(module.control_functions)
        except AttributeError:
            pass
        else:
            headers.append("control_func")

        try:
            columns.append(module.cost_functions)
        except AttributeError:
            pass
        else:
            headers.append("cost_func")

        index_width = max(len(str(len(module))) + 2, 4)  # Need to be able to fit "(t)" and "time"
        index_column_width = index_width + 2
        content_column_width = (width - index_column_width - len(headers)) // len(headers)
        content_width = content_column_width - 2 # - (2 if optimizer is not None else 0)

        #  Create repr lines ...
        repr_lines = []
        repr_lines.append(f"{module._get_name()}(")
        repr_lines.append(header_row := create_row("time", *headers, index_width=index_width, content_width=content_width))
        repr_lines.append("=" * len(header_row))
        
        time = 0
        initial_row = create_row(f"{time} ", "", *(table_repr(column[time]) for column in columns[1:]), 
                                 index_width=index_width, content_width=content_width)
        repr_lines.append(initial_row)

        upper_slice = [(columns[0][i-1], *(column[i] for column in columns[1:])) for i in range(1, min(height - 8, len(module)))]
        for time, funcs in enumerate(upper_slice, 1):
            row = create_row(f"{time} ", *map(table_repr, funcs), 
                             index_width=index_width, content_width=content_width)
            repr_lines.append(row)

        if (skip := (len(module) - (time + 3))) > 0:  # Count rows to be skipped (skip row and three end rows can stay)
            skip_row = create_row('... ', *(['...'] * len(headers)), index_width=index_width, content_width=content_width)
            repr_lines.append(skip_row)
            time = time + skip  # Move time forward by the number of skipped rows
        
        time = time + 1  # Continue with next time step
        
        lower_slice = [(columns[0][i-1], *(column[i] for column in columns[1:])) for i in range(time, len(module))]
        for time, funcs in enumerate(lower_slice, time):
            row = create_row(f"{time} ", *map(table_repr, funcs), index_width=index_width, content_width=content_width)
            repr_lines.append(row)

        post_problem_row = create_row(f"({len(module)})", table_repr(module.state_functions[-1]), 
                                      *([""] * len(headers[1:])), index_width=index_width, content_width=content_width)
        
        repr_lines.append(post_problem_row)
        repr_lines.append(")")

        return repr_lines


def create_row(index: str, *content: str, index_width: int, content_width: int) -> str:

    row = " " + " | ".join([
        f"{index : >{index_width}}",
        *[f"{shorten_content(content, content_width) : ^{content_width}}" for content in content]
    ])

    return row


def shorten_content(content: str, width: int, placeholder: str = "...") -> str:
    if width < len(placeholder) + 1:
        raise ValueError("Width too small")

    content_lines = content.split("\n")
    if len(content_lines) == 1:
        fini_width = 1
        ini_width = max(width - fini_width - len(placeholder), 0)

        if len(content_lines[0]) > width:
            return content_lines[0][:ini_width] + placeholder + content_lines[0][-fini_width:]
        else:
            return content_lines[0]
        
    else:
        ini_width = max(width - len(placeholder), 1)

        if len(content_lines[0]) + len(placeholder) > width:
            return content_lines[0][:ini_width] + placeholder
        else: # Just return the full line but append placeholder with enough spacing to fill up the whole width:
            return content_lines[0] + f"{placeholder : >{width - len(content_lines[0])}}"


def _training(module: Any) -> Optional[bool]:
    if not isinstance(module, torch.nn.Module):
        return None

    return any([module.training for module in module.modules()])


def _optimizing(module: Any,
                optimizer: Optional[torch.optim.Optimizer] = None) -> Optional[bool]:

    if not isinstance(module, torch.nn.Module) or optimizer is None:
        return None

    module_params = set(module.parameters()) 
    return any([bool(set(param_group['params']) & module_params)
                for param_group in optimizer.param_groups])