r"""
Core module providing cost-to-go function implementation :class:`CostToGo`.
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

        Saves state functions $(F_0, \dots, F_T)$ and control functions $(a_0,\dots, A_T)$
        and, as a callable, implements, essentially, the map
        $$(s_0, (\xi_0, \dots, \xi_{T+1})) \mapsto s_{T+1}$$
        where
        $$s_{i+1} = F_i(s_i, a_i(s_i), \xi_i), \quad a_t = a_t(s_t),\quad i=0, \dots, T.$$
        For more detail, see :func:`ml_adp.cost.Propagator.forward`.
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
            [None] * (number_of_steps + 1),
            [None] * (number_of_steps + 1)
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
                step = f" {step} "
                cell_state = ""
            elif step == len(self):
                step = f"({step})"
                cell_cost = ""
                cell_control = ""
            else:
                step = f" {step} "

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
        The mutable, zero-based sequence of state functions $(F_i)_{i=0}^T$
        
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
        The length of ``self`` as a propagator

        If ``self`` is $F^A$ with the state functions $F=(F_0,\dots, F_T)$ and the control functions
        $A=(A_0,\dots, A_T)$, then the length of ``self`` is considered to be $T+1$.

        Returns
        -------
        int
            The length, $T+1$
        """
        assert len(self._state_functions) == len(self._control_functions)
        return len(self._state_functions)
        
    def steps(self) -> int:
        r"""
        The number of steps of ``self`` as a propagator

        If ``self`` is $F^A$ with the state functions $F=(F_0,\dots, F_T)$ and the control functions
        $A=(A_0,\dots, A_T)$, then the number of steps of ``self`` is considered to be $T$.

        Returns
        -------
        int
            The number of steps, $T$
        """
        return len(self) - 1
    
    
    def __getitem__(self, key: Union[int, slice]) -> Propagator:
        r"""
            Return a :class:`Propagator` substructure of ``self`` given by a subrange of times
        
            If ``self`` is a $T$-step :class:`Propagator` $h^{F, A}$ and ``key`` specifies the subset $I=\{i_0,\dots, i_k\}$ of $\{0, \dots, T\}$,
            then return the :class:`Propagator` $G^B $given by the list of state functions $G=(F_{i_0},\dots, F_{i_k})$ and
            the list of control functions $(B_{i_0}, \dots, B_{i_k})$.

            Parameters
            ----------
            key : Union[int, slice]
                Specify the subrange; ``int``'s are singleton ranges

            Returns
            -------
            Propagator
                The substructure :class:`Propagator` $G^B$

            Raises
            ------
            KeyError
                If ``key`` does not specify a valid subrange
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

        If ``self`` is $T$-step :class:`Propagator` $F^A$ and ``other`` is $S$-step :class:`Propagator`
        $k^{G, B}$, then return the concatenated $(T+S)$-step :class:`Propagator` $H^C$
        given by the list of state functions $l=(F_0, \dots, F_T, G_0, \dots, G_S)$ and
        the list of control functions $C=(A_0,\dots, A_T, B_0,\dots, B_S)$.

        Parameters
        ----------
        other : Propagator
            The :class:`Propagator` to be appended on the right

        Returns
        -------
        Propagator
            The concatenation of ``self`` and `other`

        Raises
        ------
        TypeError
            Raised, if ``other`` is not a :class:`Propagator`
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
        Compute controlled state evolution.

        In more detail, compute
        $$\left((s_t)_{t=0}^{T+1}, (a_t)_{t=0}^T, (\xi_t)_{t=0}^{T+1}\right).$$

        Parameters
        ----------
        initial_state : Optional[torch.Tensor], optional
            The initial state $s_0$, by default None.
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
        """
        
        if random_effects is None:
            random_effects = [None] * (len(self) + 1)
            
        rand_effs = []
        states = []
        
        state_args = []
        control_args = []
        
        rand_eff = random_effects[0]
        if rand_eff is not None:
            if rand_eff.dim() <= 1:
                rand_eff = rand_eff.expand(1, *(rand_eff.size() or [1]))
            control_args.append(rand_eff)
        rand_effs.append(rand_eff)
        
        state = initial_state
        if state is not None:
            if state.dim() <= 1:
                state = state.expand(1, *(state.size() or [1]))
            state_args.append(state)
            control_args.insert(0, state)
        states.append(state)
        
        controls = []

        for control_func, state_func, random_effect in it.islice(
            it.zip_longest(self._control_functions,
                           self._state_functions,
                           random_effects[1:]),
            len(self)
        ):
            control = None if control_func is None else control_func(*control_args)
            control_args = []
            if control is not None:
                if control.dim() <= 1:
                    control = control.expand(1, *(control.size() or [1]))
                state_args.append(control)
            controls.append(control)
            
            rand_eff = random_effect
            if rand_eff is not None:
                if rand_eff.dim() <= 1:
                    rand_eff = rand_eff.expand(1, *(rand_eff.size() or [1]))
                state_args.append(rand_eff)
                control_args.append(rand_eff)
            rand_effs.append(rand_eff)
                 
            state = None if state_func is None else state_func(*state_args)
            state_args = []
            if state is not None:
                if state.dim() <= 1:
                    state = state.expand(1, *(state.size() or [1]))
                state_args.append(state)
                control_args.insert(0, state)
            states.append(state)
            
        return states, controls, rand_effs
     

class CostToGo(torch.nn.Module):
    
    r"""
    Sum the costs incurred along a controlled state evolution.

    Saves a :class:`Propagator` $F^A$ and a list of cost functions $(h_i)$ of equal
    length.
    As a callable, implements the map
    $$h^{F, A}\colon (s_0, (\xi_i)_{i=0}^{T+1})) \mapsto \sum_{i=0}^T h_i(s_i, A_i(s_i), \xi_i)$$
    where $A_i$ are the control functions saved by the propagator, $(\xi_i)$ are
    the random effects provided by the user, and $(s_i)_{i=0}^{T+1}$
    is the state evolution as computed by the propagator.
    """
    
    def __init__(self,
                 propagator: Propagator,
                 cost_functions: Sequence[Optional[Any]]) -> None:

        """
            Construct a :class:`CostToGo` object from a given :class:`Propagator` ``propagator`` and 
            a seqeuence of cost functions.
            
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
                Raised if the lengths of ``propagator`` and ``cost_functions`` differ.
        """
        super(CostToGo, self).__init__()
        
        if not len(propagator) == len(cost_functions):
            raise ValueError("Length mismatch.")
        
        self.propagator = propagator
        self._cost_functions = ModuleList(*cost_functions)
        self._dummy_zero = torch.nn.Parameter(torch.tensor([0.]))
        
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
        cost_functions = [None] * (number_of_steps + 1)
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
                step = f" {step} "
                cell_state = ""
            elif step == len(self):
                step = f"({step})"
                cell_cost = ""
                cell_control = ""
            else:
                step = f" {step} "
            


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
        $(A_i)_{i=0}^{T}$ as saved by :attr:`CostToGo.propagator`.

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
        r"""
            Return a :class:`CostToGo` substructure of ``self`` given by a subrange of times

            If ``self`` is a $T$-step :class:`CostToGo` $h^{F, A}$ and ``key`` specifies the subset $I=\{i_0,\dots, i_k\}$ of $\{0, \dots, T\}$,
            then return the :class:`CostToGo` $k^{G, B}$ given by the list of state functions $G=(F_{i_0},\dots, F_{i_k})$,
            the list of control functions $(B_{i_0}, \dots, B_{i_k})$ and the list of cost functions $k=(h_{i_0},\dots, h_{i_k})$.

            Parameters
            ----------
            key : Union[int, slice]
                Specify the subrange, `int`'s are singleton ranges

            Returns
            -------
            CostToGo
                The substructure :class:`CostToGo` $k^{G, B}$

            Raises
            ------
            KeyError
                If ``key`` does not specify a valid subrange
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
        r"""
        The length of ``self`` as a cost-to-go

        If ``self`` is $h^{F, A}$ with the state functions $F=(F_0,\dots, F_T)$, the control functions
        $A=(A_0,\dots, A_T)$ and the cost functions $h=(h_0,\dots, h_T)$ then the length of ``self`` is considered to be $T+1$.

        Returns
        -------
        int
            The length, $T+1$
        """
        assert len(self.propagator) == len(self.cost_functions)
        return len(self.cost_functions)

    def steps(self) -> int:
        r"""
        The number of steps of ``self`` as a cost-to-go

        If ``self`` is $h^{F,A}$ with the state functions $F=(F_0,\dots, F_T)$, the control functions
        $A=(A_0,\dots, A_T)$ and the cost functions $h=(h_0,\dots, h_T)$, then the number of steps of ``self`` is considered to be $T$.

        Returns
        -------
        int
            The number of steps, $T$
        """
        return len(self) - 1

    def __add__(self, other: CostToGo) -> CostToGo:
        r"""
            Add, i.e. concatenate, two CostToGo objects.
            
            If ``self`` is $T$-step :class:`CostToGo` $h^{F, A}$ and ``other`` is 
            $S$-step :class:`CostToGo` $k^{G, B}$, then return the concatenated $(T+S)$-step :class:`CostToGo`
            $l^{H, C}$ given by the list of state functions $l=(F_0, \dots, F_T, G_0, \dots, G_S)$,
            the list of control functions $C=(A_0,\dots, A_T, B_0,\dots, B_S)$ and the list of cost functions
            $l=(h_0,\dots, h_T, k_0,\dots, k_S)$.
            
            Parameters
            ----------
            other : CostToGo
                The :class:`CostToGo` to be appended on the right  

            Returns
            -------
            CostToGo
                The concatenation of ``self`` and `other`

            Raises
            ------
            TypeError
                Raised, if ``other`` is not a :class:`CostToGo`
        """
        if isinstance(other, CostToGo):
            return CostToGo(
                self.propagator + other.propagator,
                list(self.cost_functions) + list(other.cost_functions)
            )
        else:
            raise TypeError("Can only add `CostToGo`'s")

    def forward(self,
                initial_state: Optional[torch.Tensor] = None,
                random_effects: Optional[Sequence[Optional[torch.Tensor]]] = None) -> torch.Tensor:
               
        states, controls, random_effects = self.propagator(
            initial_state,
            random_effects
        )
        
        cost = torch.zeros(size=(1,1), device=next(self.parameters()).device)
        
        for step, cost_func in enumerate(self.cost_functions):
            if cost_func is not None:
                cost_args = []
                if states[step] is not None:
                    cost_args.append(states[step])
                if controls[step] is not None:
                    cost_args.append(controls[step])
                if random_effects[step] is not None:
                    cost_args.append(random_effects[step])
                cost_step = cost_func(*cost_args)
                if cost_step.dim() <= 1:
                    cost_step = cost_step.expand(1, *[cost_step.size() or [1]])
                cost = cost + cost_step 
        
        return cost

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
