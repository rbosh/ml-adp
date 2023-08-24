""" Implementations of state evolutions and cost accumulations
"""
from __future__ import annotations

import torch
from torch import Tensor

from ml_adp.utils.nn import ModuleList
from ml_adp.utils._repr import create_table

import itertools as it
from collections.abc import Sequence, Iterable, Callable
from typing import Any, Optional


Data = dict[str, float | Tensor]


class StateEvolution(torch.nn.Module):
    r"""Compute (controlled) state evolutions

        Saves state functions 
        $$\mathrm{state\_func}_0(s_0 [, a_0], \xi_1), \quad \dots, \quad \mathrm{state\_func}_T(s_T [, a_T], \xi_{T+1})$$
        and, optionally, control functions 
        $$\mathrm{control\_func}_0(s_0),\dots, \mathrm{control\_func}_T(s_T).$$
        As a callable, implements the map
        $$\mathrm{state\_evo} \colon (s_0, (\xi_t)_{t=1}^{T+1}) \mapsto s_{T+1}$$
        where
        $$s_{t+1} = \mathrm{state\_func}_t(s_t [, a_t], \xi_{t+1}), \quad t = 0, \dots, T$$
        and, if applicable, 
        $$a_t = \mathrm{control\_func}_t(s_t), \quad t=0, \dots, T.$$
        For more detail, see :meth:`evolve`.
    """
    def __init__(self, state_functions: Sequence[Optional[Callable]],
                 control_functions: Optional[Sequence[Optional[Callable]]] = None) -> None:
        """ Construct :class:`StateEvolution` object from sequences of state functions and, optionally, control functions

            If providing both state and control functions, must provide them in equal numbers.

            Parameters
            ----------
            state_functions
                The sequence of state functions
            control_functions
                The sequence of the control functions (optional)
            
            Raises
            ------
            ValueError
                Raised if the lengths of ``state_functions`` and ``control_functions`` differ.
        """
        super().__init__()
        
        if not (controlled := (control_functions is not None)):
            control_functions = it.repeat(None, len(state_functions))
        
        assert len(state_functions) == len(control_functions)

        self._state_functions = ModuleList(*state_functions)
        self._control_functions = ModuleList(*control_functions)
        self._controlled = controlled

    @property
    def controlled(self) -> bool:
        """ Indicates whether :class:`StateEvolution` is controlled

            To activate :attr:`control_functions` as an attribute, set to ``True`` (to deactivate, set to ``False``).
        """
        return self._controlled
    
    @controlled.setter
    def controlled(self, value: bool) -> None:
        if isinstance(value, bool):
            self._controlled = value
        else:
            raise TypeError("Must be bool.")
        
    @property
    def state_functions(self) -> Sequence[Optional[Callable]]:
        r"""The sequence of state functions
        
            Contains the state functions 
            $$\mathrm{state\_func}_0(s_0 [, a_0], \xi_1), \dots, \mathrm{state\_func}_T(s_T [, a_T], \xi_{T+1}).$$
            To manipulate state functions, access the entries of this property
            (any attempts to replace this property with another object result in an error).  
        """

        return self._state_functions
    
    @state_functions.setter
    def state_functions(self, value: Any) -> None:
        raise AttributeError("To change state functions, modify list entries directly.")
    
    @property
    def control_functions(self) -> Sequence[Optional[Callable]]:
        r"""The sequence of control functions (if any)

            If ``self`` is controlled (i.e. :attr:`controlled` is ``True``), then this property is accessible as an attribute and containes the control functions
            $$\mathrm{control\_func}_0(s_0),\dots, \mathrm{control\_func}_T(s_T).$$
            In this case, to manipulate control functions, access the entries of this property
            (any attempts to replace this sequence by another object result in an error).

            If ``self`` is not controlled (i.e. :attr:`controlled` is ``False``), then accessing this attribute results in an error.
        """
        if self.controlled:
            return self._control_functions
        else:
            raise AttributeError("Not controlled.")
        
    @control_functions.setter
    def control_functions(self, value: Any) -> None:
        if self.controlled:
            raise AttributeError("To change control functions, modify list entries directly.")
        else:
            raise AttributeError("Not controlled.")

    def __len__(self) -> int:
        r"""Return the length of ``self``

            The length of a :class:`StateEvolution` is equal to the number of states it produces.
            It is equal to the number of functions it has (and of control functions, if it has any).

            Returns
            -------
            int
                The length of ``self``
        """
        assert (length := len(self._state_functions)) == len(self._control_functions)
        return length
    
    def steps(self) -> int:
        r"""Return the number of steps of ``self``

            The number of steps of a :class:`StateEvolution` is the number of steps of the state evolution it implements.
            It is equal to its length (see :meth:`__len__`) minus one.

            Returns
            -------
            int
                The number of steps of ``self``
        """
        return len(self) - 1
    
    @classmethod
    def from_steps(cls, steps: int, controlled: bool = True) -> StateEvolution:
        r"""Construct empty :class:`StateEvolution` with a certain number of steps

            A :class:`StateEvolution` is empty if all its state functions (and control functions, if applicable) are ``None``.

            Parameters
            ----------
            steps
                The number of steps (see :func:`steps`).
            controlled
                Indicates whether the constructed :class:`StateEvolution` is controlled, by default ``True``.

            Returns
            -------
            StateEvolution
                The empty :class:`StateEvolution` of specified number of steps.
        """

        state_functions = [None] * (steps + 1)
        if controlled:
            control_functions = [None] * (steps + 1)
        else:
            control_functions = None

        return StateEvolution(state_functions, control_functions)
    
    def as_table(self, width: Optional[int] = None, height: Optional[int] = None) -> str:
        """ Return a string representation of ``self`` as a table

            Parameters
            ----------
            width
                The width of the table (optional).
            height
                The height of the table (optional).

            Returns
            -------
            str
                The string representation of ``self`` as a table.
        
        """
        return "\n".join(create_table(self, width=width, height=height))
    
    def __repr__(self) -> str:
        return self.as_table()

    def __getitem__(self, key: int | slice) -> StateEvolution:

        r"""Construct a :class:`StateEvolution` from a subrange of the state functions (and the control functions) of ``self``
        
            Slices, using ``key``, into :attr:`state_functions` (and :attr:`control_functions`, if applicable) at the same time and combine as a new :class:`StateEvolution`.

            Parameters
            ----------
            key
                Specifies the subrange of state functions (and control functions) to take (:class:`int`'s indicate singleton indices)

            Returns
            -------
            StateEvolution
                The resulting :class:`StateEvolution`
        """
        if isinstance(key, int):
            return StateEvolution(
                self._state_functions[key],
                [self._control_functions[key]] if self.controlled else None
            )
        elif isinstance(key, slice):
            return StateEvolution(
                self._state_functions[key],
                self._control_functions[key] if self.controlled else None
            )
        else:
            raise KeyError("Query using int's or slices.")

    def __add__(self, other: StateEvolution) -> StateEvolution:
        """ Compose :class:`StateEvolution`'s

            Creates a new :class:`StateEvolution` by concatenating the state functions (and control functions) of ``self`` and ``other``.
            Functionally, the resulting :class:`StateEvolution` implements the composition of the state evolutions implemented by ``self`` and ``other``.

            Parameters
            ----------
            other
                The :class:`StateEvolution` to be appended on the right

            Returns
            -------
            StateEvolution
                The composition of ``self`` and ``other``
        """
        if isinstance(other, StateEvolution):
            return StateEvolution(
                list(self._state_functions) + list(other._state_functions),
                list(self._control_functions) + list(other._control_functions) if self.controlled or other.controlled else None
            )
        else:
            raise TypeError("May only add `StateEvolution`'s")

    def evolve(self,
               initial_state: Optional[Data] = None,
               random_effects: Optional[Iterable[Optional[Data]]] = None) -> list[Data]:
        r"""Compute the state evolution in terms of an initial state and a sequence of random effects

            Return all states (merged with controls, if applicable) 
            $$(s_0[, a_0]), \dots, (s_{T + 1} [, a_{T+1}])$$
            of the (controlled) state evolution described by
            $$s_{t+1} = \mathrm{state\_func}_t(s_t [, a_t], \xi_{t+1}), \quad t=0, \dots, T$$
            and, if applicable,
            $$\quad a_t = \mathrm{control\_func}(s_t), \quad t=0, \dots, T.$$

            Parameters
            ----------
            initial_state
                The initial state $s_0$, optional.
            random_effects
                The sequence of random effects $(\xi_t)_{t=1}^{T+1}$, optional.
        """
        
        if random_effects is None:
            random_effects = [None] * len(self)
            
        states = []
        
        # Prep initial state s_0
        if (state := initial_state) is None:
            state = {}
        
        for state_func, control_func, random_effect in it.islice(
            it.zip_longest(self._state_functions,  # zip_longest pads random_effects with None's on the right
                           self._control_functions if self.controlled else it.repeat(None),
                           random_effects),
            len(self)
        ):
            # Produce time-t control $a_t=A_t(s_t)$
            control = {} if control_func is None else control_func(**state)
            state = state | control  # Merge control onto state
    
            states.append(state)
            
            # Next time step t+1:
            # Prep time-(t+1) random effect xi_{t+1}
            if random_effect is None:
                random_effect = {}
                
            # Produce time-(t+1) state s_{t+1} = F_t(s_t, a_t, xi_{t+1}) 
            state = {} if state_func is None else state_func(**state, **random_effect)

        states.append(state)  # Append post-problem scope state    
        
        return states
     
    def forward(self,
                initial_state: Optional[Data] = None,
                random_effects: Optional[Iterable[Optional[Data]]] = None) -> Data:
        r"""Compute and return final state of state evolution

            More precisely, implements
            $$\mathrm{state\_evo} \colon (s_0,(\xi_t)_{t=1}^{T+1})\mapsto s_{T+1}.$$
            For more detail, see :meth:`evolve`.
        """
        states = self.evolve(initial_state, random_effects)
        return states[-1]
                  

class CostAccumulation(torch.nn.Module):
    r"""Accumulate costs along (controlled) state evolutions
    
        Saves a :class:`StateEvolution` and a sequence of cost functions
        $$\mathrm{cost\_func}_0(s_0 [, a_0]), \quad \dots, \quad \mathrm{cost\_func}_T(s_T [, a_T])$$
        of equal lengths.
        As a callable, implements the map
        $$\mathrm{cost\_acc} \colon (s_0, (\xi_t)_{t=1}^{T+1})) \mapsto \sum_{t=0}^T \mathrm{cost\_func}_t(s_t [, a_t])$$
        where $(s_t [,a_t])_{t=0}^T$ is the state evolution (including controls, if applicable) computed by the :class:`StateEvolution`.
    """
    def __init__(self,
                 state_evolution: StateEvolution,
                 cost_functions: Sequence[Optional[Callable]]) -> None:
        r"""Construct :class:`CostAccumulation` from given :class:`StateEvolution` and sequence of cost functions
        
        Parameters
        ----------
        state_evolution
            The underlying :class:`StateEvolution`
        cost_functions
            The sequence of cost functions

        Raises
        ------
        ValueError
            Raised if the lengths of ``state_evolution`` and ``cost_functions`` differ.
        """
        
        super().__init__()

        assert len(state_evolution) == len(cost_functions)
        
        self._state_evolution = state_evolution
        self._cost_functions = ModuleList(*cost_functions)

    @property
    def state_evolution(self) -> StateEvolution:
        """ The underlying :class:`StateEvolution`"""
        return self._state_evolution
    
    @state_evolution.setter
    def state_evolution(self, value: Any) -> None:
        raise AttributeError("To change state evolution, modify state (and control) functions directly.")

    @property
    def controlled(self) -> bool:
        """ Indicates whether the underlying :class:`StateEvolution` is controlled
        
            To activate :attr:`control_functions` as an attribute, set to ``True`` (to deactivate, set to ``False``).
        """
        return self.state_evolution.controlled
    
    @controlled.setter
    def controlled(self, value: bool) -> None:
        self.state_evolution.controlled = value

    @property
    def state_functions(self) -> Sequence[Optional[Callable]]:
        r"""The sequence of state functions of the underlying :class:`StateEvolution`

        Contains the state functions
        $$\mathrm{state\_func}_0(s_0 [, a_0], \xi_1), \dots, \mathrm{state\_func}_T(s_T [, a_T], \xi_{T+1}).$$
        To manipulate state functions, access the entries of this property
        (any attempts to replace this property with another object result in an error;
        see :attr:`StateEvolution.state_functions`).
        """
        return self.state_evolution.state_functions

    @state_functions.setter
    def state_functions(self, value: Any) -> None:
        self.state_evolution.state_functions = value

    @property
    def control_functions(self) -> Sequence[Optional[Callable]]:
        r"""The sequence of control functions of the underlying :class:`StateEvolution`

        Contains the control functions
        $$\mathrm{control\_func}_0(s_0),\dots, \mathrm{control\_func}_T(s_T).$$
        To manipulate control functions, access the entries of this property
        (any attempts to replace this property with another object result in an error; 
        see :attr:`StateEvolution.control_functions`).
        """
        return self.state_evolution.control_functions

    @control_functions.setter
    def control_functions(self, value: Any) -> None:
        self.state_evolution.control_functions = value

    @property
    def cost_functions(self) -> Sequence[Optional[Callable]]:
        r"""The sequence of cost functions

            Contains the cost functions
            $$\mathrm{cost\_func}_0(s_0 [, a_0]), \dots, \mathrm{cost\_func}_T(s_T [, a_T]).$$
            To manipulate cost functions, access the entries of this property
            (any attempts to replace this property with another object result in an error).
        """
        return self._cost_functions

    @cost_functions.setter
    def cost_functions(self, value: Any) -> None:
        raise AttributeError("To change cost functions, modify list entries directly.")
    
    def __len__(self) -> int:
        """ Return the length of ``self``

            The length of a :class:`CostAccumulation` is equal to the number of time points at which costs are accumulated.
            It is equal to the length of the underlying :class:`StateEvolution` (see :meth:`StateEvolution.__len__`) and the number of its cost functions (see :meth:`CostAccumulation.cost_functions`).
        """
        assert (length := len(self.state_evolution)) == len(self.cost_functions)
        return length
    
    def steps(self) -> int:
        """ Return the number of steps of ``self``
        
            The number of steps of a :class:`CostAccumulation` is the number of steps between the time points at which costs are accumulated.
            It is equal to its length minus one (see :meth:`__len__`)  and to the number of steps of the underlying :class:`StateEvolution` (see :meth:`StateEvolution.steps`).
        """
        return len(self) - 1
    
    @classmethod
    def from_steps(cls, steps: int, controlled: bool = True) -> CostAccumulation:
        """ Construct empty :class:`CostAccumulation` with a certain number of steps

        A :class:`CostAccumulation` is empty if all its state functions and cost functions (and control functions, if applicable) are ``None``.
        
        Parameters
        ----------
        steps
            The number of steps (see :func:`steps`).
        controlled
            Indicates whether the constructed :class:`CostAccumulation` is controlled, by default ``True``.

        Returns
        -------
        CostAccumulation
            The empty :class:`CostAccumulation` of specified number of steps.
        """
        state_evolution = StateEvolution.from_steps(steps, controlled=controlled)
        cost_functions = [None] * (steps + 1)

        return CostAccumulation(state_evolution, cost_functions)
    
    def as_table(self, width: Optional[int] = None, height: Optional[int] = None) -> str:
        """ Return a string representation of ``self`` as a table
        
            Parameters
            ----------
            width
                The width of the table (optional).
            height
                The height of the table (optional).
            
            Returns
            -------
            str
                The string representation of ``self`` as a table.
        """
        return "\n".join(create_table(self, width=width, height=height))
    
    def __repr__(self) -> str:
        return self.as_table()

    def __getitem__(self, key: int | slice) -> CostAccumulation:
        """ Construct a :class:`CostAccumulation` from a subrange of the state evolution and the cost functions of ``self``

            Slices, using ``key``, into the underlying :attr:`state_evolution` and :attr:`cost_functions` at the same time and combines as a new :class:`CostAccumulation`.

            Parameters
            ----------
            key
                Specifies the subrange of the state evolution and cost functions to take (``int``'s indicate singleton indices)

            Returns
            -------
            CostAccumulation
                The resulting :class:`CostAccumulation`
        """

        if isinstance(key, int):
            return CostAccumulation(
                self.state_evolution[key],
                [self.cost_functions[key]]
            )
        elif isinstance(key, slice):
            return CostAccumulation(
                self.state_evolution[key],
                self.cost_functions[key]
            )
        else:
            raise KeyError("Query using int's or slices.")
        
    def __add__(self, other: CostAccumulation) -> CostAccumulation:
        """ Compose :class:`CostAccumulation`'s
        
            Creates a new :class:`CostAccumulation` by concatenating the :class:`StateEvolution`'s and cost functions of ``self`` and ``other``.
            Functionally, the resulting :class:`CostAccumulation` implements the composition of the cost accumulations implemented by ``self`` and ``other``.

            Parameters
            ----------
            other
                The :class:`CostAccumulation` to be appended on the right

            Returns
            -------
            CostAccumulation
                The composition of ``self`` and ``other``
        """
        if isinstance(other, CostAccumulation):
            return CostAccumulation(
                self.state_evolution + other.state_evolution,
                list(self.cost_functions) + list(other.cost_functions)
            )
        else:
            raise TypeError("Can only add `CostAccumulation`'s")

    def forward(self,
                initial_state: Optional[Data] = None,
                random_effects: Optional[Iterable[Optional[Data]]] = None) -> Data:
        r"""Accumulate total cost incurred along underlying state evolution

            More precisely, implements
            $$\mathrm{cost\_acc} \colon (s_0,(\xi_t)_{t=1}^{T+1})\mapsto \sum_{t=0}^T \mathrm{cost\_func}_t(s_t [, a_t])$$
            where $(s_t [,a_t])_{t=0}^T$ is the state evolution (with controls, if applicable) implemented by the underlying :class:`StateEvolution` (see :meth:`StateEvolution.evolve`).
        """
        states = self.state_evolution.evolve(initial_state, random_effects)

        cost = 0.
        for step, cost_func in enumerate(self.cost_functions):
            if cost_func is not None:
                cost += cost_func(**states[step])

        return cost
