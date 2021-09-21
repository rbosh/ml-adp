r"""
Core module providing cost-to-go function implementation :class:`CostToGo`.

More here
"""
from __future__ import annotations

import warnings
import torch
from ml_adp.nn import ModuleList
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from contextlib import ExitStack, contextmanager
from typing import Any, Optional, List, Sequence, MutableSequence, Union, Tuple


class Propagator(torch.nn.Module):
    r"""
    Computes a controlled state evolution.

    Saves a tuple of state functions $(F_i)$ and control functions $(a_i)$
    and, as a callable, implements the map
    $$(x_0, (\xi_0, \dots, \xi_{T+1})) \mapsto x_{T+1}$$
    where
    $$x_{i+1} = F(x_i, a_i(x_i), \xi_i), \quad i=0, \dots, T.$$
    in the sense as specified in `forward`.
    
    For convenience, returns the triple of cleaned-up versions of the complete evolution $(x_i)_{i=0}^{T+1}$,
    of the sequence of controls $(a_i(x_i))_{i=0}^T$ and of the sequence of random effects
    $(\xi_i)_0^{T+1}$ provided by the user.
    """

    def __init__(self,
                 state_functions: Sequence[Optional[Any]],
                 control_functions: Sequence[Optional[Any]]) -> None:
        """
        Construct a :class:`Propagator` object from given state functions and control functions.

        Must provide state and control functions in equal numbers.

        Parameters
        ----------
        state_functions : 
            The container of the state functions.
        control_functions :
            The container of the control functions.
        
        Raises
        ------
        ValueError
            Raised if the lengths of ``state_functions`` and ``control_functions`` differ.
        """
        super(Propagator, self).__init__()
    
        if not len(state_functions) == len(control_functions):
            raise ValueError("Provide equal-length lists as arguments.")
        
        self._state_functions = ModuleList(*state_functions)
        self._control_functions = ModuleList(*control_functions)
        
        #self._state_functions = np.array(state_functions, dtype=object)
        #self._control_functions = np.array(control_functions, dtype=object)
        
        #self._register_modules()
        
    @classmethod
    def from_steps(cls, number_of_steps: int) -> Propagator:
        """
        Construct an empty :class:`Propagator` object of certain length.

        Parameters
        ----------
        number_of_steps
            The length, in terms of a number of steps.

        Returns
        -------
        Propagator
            The :class:`Propagator` of specified length.
        """     
        return Propagator(
            [None] * number_of_steps,
            [None] * number_of_steps
        )
    
    def __repr__(self,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 include_id: bool = False) -> str:
        
        STEP_COL_WIDTH = 5
        COL_WIDTH = 35
        
        desc_width = (COL_WIDTH
                      # Spacing left and right within cell:
                      - 2
                      # opt state indicator counts 2 and wants spacer:
                      - (2 if optimizer is not None else 0))

        repr_lines = []

        #if optimizer is None:
        #    opt_state_ind = ⚙️

        repr_lines.append("Propagator(")
        repr_lines.append("|".join([
            f"{'step' : ^{STEP_COL_WIDTH}}",
            f"{'state_func' : ^{COL_WIDTH}}",
            f"{'control_func' : ^{COL_WIDTH}}"
        ]))
        repr_lines.append("=" * (STEP_COL_WIDTH + 2 * (1 + COL_WIDTH)))

        for step, (state_func, control_func) in enumerate(it.zip_longest(
            [None] + list(self._state_functions),
            self._control_functions
        )):
            opt_state, desc_state = _info(state_func, optimizer=optimizer,
                                          include_id=include_id, width=desc_width)
            opt_control, desc_control = _info(control_func, optimizer=optimizer,
                                              include_id=include_id, width=desc_width)

            cell_state = f"{desc_state : ^{desc_width}}"
            cell_control = f"{desc_control : ^{desc_width}}"

            if optimizer is not None:
                cell_state = opt_state + " " + cell_state
                cell_control = opt_control + " " + cell_control

            if step == 0:
                cell_state = ""
            elif step == len(self): 
                # step = ""
                cell_control = ""

            repr_lines.append(" ".join([
                f"{step : >{STEP_COL_WIDTH}}",
                f"{cell_state : ^{COL_WIDTH}}",
                f"{cell_control : ^{COL_WIDTH}}"
            ]))

        repr_lines.append(")")

        return "\n".join(repr_lines)

    @property
    def state_functions(self) -> MutableSequence[Optional[object]]:
        r"""
        The mutable, zero-based sequence of state functions $(F_i)_{i=1}^{T+1}$.
        
        To manipulate state functions, access the contents of this sequence.
        Immutable as a property in the sense that any attempts to replace the sequence
        by another object will be ignored.  

        Returns
        -------
        MutableSequence[Optional[object]]
            The mutable sequence of state functions.
        """

        return self._state_functions
        

    @state_functions.setter
    def state_functions(self, value: Any) -> None:
        warnings.warn("To change state_functions, access the list contents. Ignored.")
        return

    @property
    def control_functions(self) -> MutableSequence[Optional[object]]:
        r"""
        The mutable, zero-based sequence of control functions $(a_i)_{i=0}^T$.

        To manipulate control functions, access the contents of this sequence.
        Immutable as a property in the sense that any attempts to replace the sequence
        by another object will be ignored.  

        Returns
        -------
        MutableSequence[Optional[object]]
            The mutable sequence of control functions.
        """
        return self._control_functions

    @control_functions.setter
    def control_functions(self, value: Any) -> None:
        warnings.warn("To change control functions, access the list contents. Ignored.")
        return
        
    def __len__(self) -> int:
        r"""
        The length of a Propagator object.

        If the Propagator is $F^a$ with $F=(F_1, \dots, F_{T+1})$ and
        $a = (a_0,\dots, a_T)$, then the length of the Propagator objects 
        is $T$.

        Returns
        -------
        int
            The length of the Propagator.
        """
        assert len(self._state_functions) == len(self._control_functions)
        return len(self._state_functions)
        
    def __getitem__(self, key: Union[int, slice]) -> Propagator:
        r"""
        Get a sub-Propagator.

        If the Propagator object is $F^a$ with $F=(F_1, \dots, F_{T+1})$ and
        $a = (a_0, \dots, a_T)$, then any collection `key` of indices in 
        $\{0, \dots, T\}$ such that 

        Parameters
        ----------
        key :
            Specified the collection of indices.

        Returns
        -------
        Propagator
            The Sub-Propagator

        Raises
        ------
        KeyError
            [description]
        """

        if isinstance(key, int):
            return Propagator([self._state_functions[key]],
                              [self._control_functions[key]])
        elif isinstance(key, slice):
            return Propagator(self._state_functions[key],
                              self._control_functions[key])
        else:
            raise KeyError("Query just like a list.")
    
    def __setitem__(self, key: Union[int, slice], value: Propagator) -> None:
        if isinstance(value, Propagator):
            self._state_functions[key] = value._state_functions
            self._control_functions[key] = value._control_functions
        else:
            raise ValueError("Cannot assign given value.")
        
        self._register_modules()
    
    def __add__(self, other: Propagator) -> Propagator:
        r"""
        Add, i.e. concatenate, two Propagators.

        If the calling Propagator consists of the state functions $(F_i)_{i=1}^{T+1}$
        and the control functions $(a_i)_{i=0}^T$ and the Propagator other consists
        of the state functions $(G_i)_{i=1}^{S+1}$ and the control functions $(b_i)_{i=0}^S$,
        then the return value is the Propagator consisting of the state functions
        $(F_1, \dots, F_{T+1}, G_{1}, \dots, G_{S+1})$ and the control functions 
        $(a_0,\dots, a_T, b_0, \dots, b_{T+1})$.
        In other words, the underlying lists of state and control functions are concatenated.

        Parameters
        ----------
        other : Propagator
            The Propagator object to be appended.

        Returns
        -------
        Propagator
            The resulting Propagator.

        Raises
        ------
        TypeError
            [description]
        """
        if isinstance(other, Propagator):
            return Propagator(
                list(self._state_functions) + list(other._state_functions),
                list(self._control_functions) + list(other._control_functions)
            )
        else:
            raise TypeError("May only add `Propagator`'s")

    def forward(self,
                initial_state: Optional[torch.Tensor] = None,
                random_effects: Optional[Sequence[Optional[torch.Tensor]]] = None) -> List[Optional[torch.Tensor]]:
        r"""
        Compute the controlled state evolution $(s_i)_{i=0}^{T+1}$.

        In more detail, compute $(s_i)_{i=0}^{T+1}$ where
        $$x_{i+1} = F_i(s_i, a_i(s_i), \xi_i), \quad i=0, \dots, T,$$
        where
        
        * $s_0$ is the initial state
        * $F_1, \dots, F_{T+1}$ are the state functions (saved as a list as :attr:`Propagator.state_functions`),
        * $a_0, \dots, a_T$ are the control functions (saved as a list as :attr:`Propagator.control_functions`), and
        * $\xi_0,\dots, \xi_{T+1}$ are the random effects dictating the stochastic behavior at each of the times $0, \dots, T, T+1$.

        Parameters
        ----------
        initial_state : Optional[torch.Tensor], optional
            The initial state $x_0$, by default None.
        random_effects : Optional[Sequence[Optional[torch.Tensor]]], optional
            The sequence of random effects $(\xi_i)_{i=0}^{T+1}$, by default None.

        Returns
        -------
        List[Optional[torch.Tensor]]
            The list containing the state evolution $(s_i)_{i=0}^{T+1}$
        List[Optional[torch.Tensor]]
            The list containing sequence of controls $(a_i(s_i))_{i=0}^T$
        List[Optional[torch.Tensor]]
            The list containing the random effects $(\xi_i)_{i=0}^{T+1}$
        
        
        
        List[Optional[torch.Tensor]]
            [description]
            
        """
        
        if random_effects is None:
            random_effects = [None] * (len(self) + 1)
        
        state = initial_state
        state_batch_size, state_space_size = _get_sizes(state)
        rand_eff = random_effects[0]
        rand_eff_batch_size, rand_eff_space_size = _get_sizes(rand_eff)
        
        states = [None if state is None else state.expand(state_batch_size, state_space_size)]
        rand_effs = [None if rand_eff is None else rand_eff.expand(rand_eff_batch_size, rand_eff_space_size)]
        controls = []
        
        for control_func, state_func, random_effect in it.islice(
            it.zip_longest(self._control_functions,
                           self._state_functions,
                           random_effects[1:]),
            len(self)
        ):
            # TODO Maybe give control_func rand_eff as parameter
            
            control = None if control_func is None else control_func(states[-1])
            control_batch_size, control_space_size = _get_sizes(control)
            controls.append(None if control is None else control.expand(control_batch_size, control_space_size))
            
            rand_eff = random_effect
            rand_eff_batch_size, rand_eff_space_size = _get_sizes(rand_eff)
            rand_effs.append(None if rand_eff is None else rand_eff.expand(rand_eff_batch_size, rand_eff_space_size))
            
            batch_size = max(
                state_batch_size,
                control_batch_size,
                rand_eff_batch_size
            )
            
            args = []
            if state is not None:
                args.append(states[-1].expand(batch_size, state_space_size))
            if control is not None:
                args.append(controls[-1].expand(batch_size, control_space_size))
            if rand_eff is not None:
                args.append(rand_effs[-1].expand(batch_size, rand_eff_space_size))
                
            state = None if state_func is None else state_func(*args)
            state_batch_size, state_space_size = _get_sizes(state)
            states.append(None if state is None else state.expand(state_batch_size, state_space_size))
            
        return states, controls, rand_effs
     

class CostToGo(torch.nn.Module):
    
    r"""
    Sum the costs incurred along a controlled state evolution.

    Saves a propagator object and a list of cost functions $(h_i)$ of equal
    length.
    As a callable, implements the map
    $$(x_0, (\xi_i)_{i=0}^{T+1})) \mapsto \sum_{i=0}^T h_i(x_i, a_i(x_i), \xi_i)$$
    where $a_i$ are the control functions saved by the propagator, $(\xi_i)$ are
    the random effects provided by the user, and $(x_i)_{i=0}^{T+1}$
    is the state evolution as computed by the propagator.
    """
    
    def __init__(self,
                 propagator: Propagator,
                 cost_functions: Sequence[Optional[Any]]) -> None:

        """
        Construct a :class:`CostToGo` object from a given propagator and 
        given cost functions.
        
        Must provide cost functions in a number compatible with ``propagator``.

        Parameters
        ----------
        propagator : 
            The possibly zero-length Propagator
        cost_functions :
            The possibly empty container of cost functions
            
        Raises
        ------
        ValueError
            Raised if the lengths of ``propagator`` and ``control_functions`` differ.
        """
        super(CostToGo, self).__init__()
        
        if not len(propagator) == len(cost_functions):
            raise ValueError("Length mismatch of arguments.")
        
        self.propagator = propagator
        self._cost_functions = ModuleList(*cost_functions)
        #self._cost_functions = np.array(cost_functions, dtype=object)
        
        #self._register_modules()
        
    @classmethod
    def from_steps(cls, number_of_steps: int) -> CostToGo   :
        """
        Construct an empty :class:`CostToGo` object of certain length.

        Parameters
        ----------
        number_of_steps :
            The length, in terms of a number of steps.

        Returns
        -------
        CostToGo
            The :class:`CostToGo` object of specified length.
        """
        cost_functions = [None] * number_of_steps
        propagator = Propagator.from_steps(number_of_steps)

        return CostToGo(propagator, cost_functions)

    def __repr__(self,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 include_id: bool = False) -> str:

        STEP_COL_WIDTH = 5
        COL_WIDTH = 24

        desc_width = (COL_WIDTH
                      # Spacing left and right within cell:
                      - 2
                      # opt state indicator counts 2 and wants spacer:
                      - (2 if optimizer is not None else 0))

        repr_lines = []

        repr_lines.append("CostToGo(")
        repr_lines.append("|".join([
            f"{'step' : ^{STEP_COL_WIDTH}}",
            f"{'state_func' : ^{COL_WIDTH}}",
            f"{'control_func' : ^{COL_WIDTH}}",
            f"{'cost_func' : ^{COL_WIDTH}}"
        ]))

        repr_lines.append("=" * (STEP_COL_WIDTH + 3 * (1 + COL_WIDTH)))

        for step, (state_func, control_func, cost_func) in enumerate(it.zip_longest(
            [None] + list(self.state_functions),
            self.control_functions,
            self.cost_functions
        )):
            opt_state, desc_state = _info(state_func,
                                          optimizer=optimizer,
                                          include_id=include_id,
                                          width=desc_width)
            opt_control, desc_control = _info(control_func,
                                              optimizer=optimizer,
                                              include_id=include_id,
                                              width=desc_width)
            opt_cost, desc_cost = _info(cost_func,
                                        optimizer=optimizer,
                                        include_id=include_id,
                                        width=desc_width)

            cell_state = f"{desc_state : ^{desc_width}}"
            cell_control = f"{desc_control : ^{desc_width}}"
            cell_cost = f"{desc_cost : ^{desc_width}}"

            if optimizer is not None:
                cell_state = opt_state + " " + cell_state
                cell_control = opt_control + " " + cell_control
                cell_cost = opt_cost + " " + cell_cost
            
            if step == 0:
                cell_state = ""
            elif step == len(self):
                #step = ""
                cell_cost = ""
                cell_control = ""

            repr_lines.append(" ".join([
                f"{step : >{STEP_COL_WIDTH}}",
                f"{cell_state : ^{COL_WIDTH}}",
                f"{cell_control : ^{COL_WIDTH}}",
                f"{cell_cost : ^{COL_WIDTH}}"
            ]))

        repr_lines.append(")")

        return "\n".join(repr_lines)

    def descr(self, optimizer=None, include_id=True):
        print(self.__repr__(optimizer=optimizer, include_id=include_id))
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'propagator' and hasattr(self, 'propagator'):
            warnings.warn('Change state and cost functions directly')
            return
        else:
            super().__setattr__(name, value)

    @property
    def state_functions(self) -> MutableSequence[Optional[Any]]:
        r"""
        The mutable, zero-based sequence of state functions $(F_i)_{i=1}^{T+1}$
        as saved by :attr:`CostToGo.propagator`.

        To manipulate state functions, access the contents of this sequence.
        Immutable as a property in the sense that any attempts to replace the sequence
        by another object will be ignored.  

        Returns
        -------
        MutableSequence[Optional[Any]]
            The state functions.
        """
        return self.propagator.state_functions

    @state_functions.setter
    def state_functions(self, value: Any) -> None:
        self.propagator.state_functions = value

    @property
    def control_functions(self) -> MutableSequence[Optional[Any]]:
        r"""
        The mutable, zero-based sequence of control functions 
        $(a_i)_{i=0}^{T}$ as saved by :attr:`CostToGo.propagator`.

        To manipulate control functions, access the contents of this sequence.
        Immutable as a property in the sense that any attempts to replace the sequence
        by another object will be ignored.  

        Returns
        -------
        MutableSequence[Optional[Any]]
            The control functions.
        """
        return self.propagator.control_functions

    @control_functions.setter
    def control_functions(self, value: Any) -> None:
        self.propagator.control_functions = value

    @property
    def cost_functions(self) -> MutableSequence[Optional[Any]]:
        r"""
        The mutable, zero-based sequence of cost functions $(h_i)_{i=0}^T$.

        To manipulate cost functions, access the contents of this sequence.
        Immutable as a property in the sense that any attempts to replace the sequence
        by another object will be ignored.  

        Returns
        -------
        MutableSequence[Optional[object]]
            The mutable sequence of cost functions.
        """
        return self._cost_functions

    @cost_functions.setter
    def cost_functions(self, value: Any) -> None:
        warnings.warn("To change cost functions, access list contents. Ignored.")

    def __getitem__(self, key: Union[int, slice]) -> CostToGo:
        """
        Query a sub :class:`CostToGo` object

        [extended_summary]

        Parameters
        ----------
        key : 
            Contains the indices.

        Returns
        -------
        Propagator
            The sub structure.
        """

        if isinstance(key, int):
            return CostToGo(
                propagator=self.propagator[key],
                cost_functions=[self.cost_functions[key]]
            )
        elif isinstance(key, slice):
            return CostToGo(
                propagator=self.propagator[key],
                cost_functions=self.cost_functions[key]
            )
        else:
            raise KeyError("Query just like a list.")
        
    def __setitem__(self, key: Union[int, slice], value: Propagator) -> None:

        if isinstance(value, CostToGo):
            self.state_functions[key] = value.state_functions
            self.control_functions[key] = value.control_functions
            self.cost_functions[key] = value.cost_functions
        else:
            raise ValueError("Cannot assign given value.")

    def __len__(self) -> int:
        assert len(self.propagator) == len(self.cost_functions)
        return len(self.cost_functions)

    def steps(self) -> int:
        return len(self) - 1

    def __add__(self, other: CostToGo) -> CostToGo:
        r"""
        Add, i.e. concatenate, two CostToGo objects.

        If the calling CostToGo has the cost functions $(h_i)_{i=0}^T$
        and the CostToGo `other` has the cost functions $(k_i)_{i=0}^S$, then
        the resulting CostToGo has the Propagator `self.propagator + other.propagator`
        as returned by `__add__` (link) and has the cost functions
        $(h_0, \dots, h_T, k_0, \dots, k_S)$.
        
        Parameters
        ----------
        other : CostToGo
            The CostToGo to be appended on the right.  

        Returns
        -------
        CostToGo
            The resulting CostToGo object.

        Raises
        ------
        TypeError
            [description]
        """
        if isinstance(other, CostToGo):
            return CostToGo(
                self.propagator + other.propagator,
                list(self.cost_functions) + list(other.cost_functions)
            )
        else:
            raise TypeError("May only add `CostToGo`'s")

    def forward(self,
                initial_state: Optional[torch.Tensor] = None,
                random_effects: Optional[Sequence[Optional[torch.Tensor]]] = None) -> torch.Tensor:
               
        states, controls, random_effects = self.propagator(
            initial_state,
            random_effects
        )
        
        # TODO Adjust sizes of cost_func outputs using _get_sizes
        # Maybe not needed because of broadcasting rules for addition
        
        cost = torch.zeros(size=(1,), device=states[0].device)
        
        for step, cost_func in enumerate(self.cost_functions):
            if cost_func is not None:
                cost = cost + cost_func(
                    states[step],
                    controls[step],
                    random_effects[step]
                ).squeeze()
        
        return cost.unsqueeze(1)

    def plot_state_component_range(
        self,
        *component_ranges: Sequence[torch.Tensor],
        plot_component_index : int = 0,
        random_effects: Optional[List[Optional[torch.Tensor]]] = None,
        versus: Optional[CostToGo] = None,
        plot_size=(8, 5)
    ) -> plt.Axes:

        component_ranges = list(component_ranges)

        # Make range of scalars
        for component_range in component_ranges:
            if isinstance(component_range, torch.Tensor):
                component_range.squeeze()  # Does this even do anything? TODO No!
            else:
                for component in component_range:
                    component.squeeze()

        # `component_range` will have plot component removed:
        plot_component_range = component_ranges.pop(plot_component_index)
        length_plot_range = len(plot_component_range)

        states_range = map(
            torch.Tensor.cpu,
            plot_component_range.squeeze()
        )
        states_range = np.array(list(map(
            torch.Tensor.numpy,
            states_range
        )))

        if isinstance(versus, CostToGo):
            versus = [versus]
        if versus is None:
            versus = []
        cost_to_gos = [self] + versus

        number_plots = int(np.prod(list(map(len, component_ranges))))
        fig, axs = plt.subplots(nrows=number_plots, squeeze=False)
        fig_size = (plot_size[0],) + (number_plots * plot_size[1],)
        fig.set_size_inches(fig_size)

        with torch.no_grad(), ExitStack() as stack:
            cost_to_gos = [stack.enter_context(_evaluating(cost_to_go))
                           for cost_to_go in cost_to_gos]

            for i, fixed_comps in enumerate(it.product(*component_ranges)):
                fixed_states = list(fixed_comps)
                # The following inserts the plot components between the other components at the right index:
                states = it.starmap(
                    _list_insert,
                    zip(
                        it.repeat(fixed_states),
                        it.repeat(plot_component_index),
                        plot_component_range
                    )
                )
                states = list(map(
                    torch.stack,
                    states
                ))

                for number, cost_to_go in enumerate(cost_to_gos):
                    costs = map(
                        cost_to_go,
                        states,
                        it.repeat(random_effects, length_plot_range)
                    )
                    costs = map(torch.mean, costs)
                    costs = map(torch.Tensor.cpu, costs)
                    costs = map(torch.Tensor.numpy, costs)
                    costs = np.array(list(costs))
                    axs[i, 0].plot(
                        states_range,
                        costs,
                        # TODO Put `info` method here
                        label=f"CostToGo({id(cost_to_go)}) [{number}]"
                    )

                axs[i, 0].set_xlabel('state')
                axs[i, 0].set_ylabel('cost')
                title = list(map(torch.Tensor.item, fixed_comps))
                title.insert(plot_component_index, "\U000000B7")  # \cdot
                axs[i, 0].set_title(str(title))
                axs[i, 0].legend()

        return axs


"""
Repr Helpers

"""

def _training(module: Any) -> Optional[bool]:
    if not isinstance(module, torch.nn.Module):
        return None

    return any([module.training for module in module.modules()])


def _optimizing(module: Any,
                optimizer: Optional[torch.optim.Optimizer] = None):

    if not isinstance(module, torch.nn.Module) or optimizer is None:
        return None

    module_params = set(module.parameters()) 
    return any([bool(set(param_group['params']) & module_params)
                for param_group in optimizer.param_groups])


def _name(module, width=24, include_id=False, **kwargs):

    TRAIN_STATUS = getattr(kwargs,
                           'train_status',
                           {True: "train", False: "eval", None: "-"})

    if module is None:
        return "None"

    if module.__class__.__name__ == "function":
        name = module.__name__
    else:
        name = type(module).__name__

    ID_LEN = getattr(kwargs,
                     'id_width',
                     6)
    id_ph = ".."
    id_len_short = ID_LEN - len(id_ph)

    details_len = 2 + max(map(len, TRAIN_STATUS.values()))

    name_width = width - (details_len+ID_LEN+1 if include_id else details_len)

    placeholder = "..."
    fini_width = 2
    ini_width = max(name_width - fini_width - len(placeholder), 0)

    name = (name[:ini_width] + placeholder + name[-fini_width:]
            if len(name) > name_width
            else name)

    m_id = str(id(module))
    m_id = f'{(id_ph + m_id[-id_len_short:]) if len(m_id)>ID_LEN else m_id}'

    name += ("("
             + TRAIN_STATUS[_training(module)]
             + (";" + m_id if include_id else "")
             + ")")

    return name


def _info(module,
          optimizer: torch.optim.Optimizer = None,
          include_id=False,
          width=24,
          **kwargs) -> Tuple[Any, Union[str, Any]]:

    OPT_STATUS = getattr(kwargs,
                         'opt_status_indicators',
                         {True: "X", False: "-", None: " "})
                         # {True: "🔥", False: "🧊", None: "➖"})

    opt_status = OPT_STATUS[_optimizing(module, optimizer=optimizer)]
    name = _name(module, width=width, include_id=include_id)

    return opt_status, name


"""
Helper Functions

TODO Export into another helper module someday maybe

"""

 
def _get_sizes(inputs: torch.Tensor):

    if inputs is None:
        return 0, 0
    
    try:
        inputs_space_size = inputs.size(1)
    except IndexError:
        # This means no two axes were given
        # => Assume batch_axis not present
        inputs_batch_size = 1
        try:
            inputs_space_size = inputs.size(0)
        except IndexError:
            # i.e. scalar Tensor given
            inputs_space_size = 1
    else:
        inputs_batch_size = inputs.size(0)

    return inputs_batch_size, inputs_space_size


@contextmanager
def _evaluating(model: torch.nn.Module):
    '''
    Temporarily switch to evaluation mode.
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

def _list_insert(list1: list, position: int, element: object) -> list:
    out = list1.copy()
    out.insert(position, element)
    return out
