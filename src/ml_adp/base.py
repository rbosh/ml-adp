r"""
Base module providing cost-to-go function implementation :class:`CostToGo`.
"""
from __future__ import annotations

import warnings
import torch
from torch import Tensor

from ml_adp.utils.nn import ModuleList
from ml_adp.utils._repr import create_table

import itertools as it
from typing import Any, Optional, Sequence, MutableSequence, Union, Tuple


Sample = dict[str, float | Tensor]


class Propagator(torch.nn.Module):
    r"""Compute Controlled State Evolutions

        Saves state functions $(F_0, \dots, F_T)$ and control functions $(A_0,\dots, A_T)$
        and, as a callable, implements the map
        $$F^A\colon (s_0, (\xi_t)_{t=1}^{T+1}) \mapsto s_{T+1}$$
        where
        $$s_{t+1} = F_t(s_t, a_t, \xi_{t+1}), \quad a_t = A_t(s_t),\quad t=0, \dots, T.$$
    """

    def __init__(self,
                 state_functions: Sequence[Optional[Any]],
                 control_functions: Sequence[Optional[Any]]) -> None:
        r"""Construct :class:`Propagator` Object From Given State Functions and Control Functions

            Must provide state functions and control functions in equal numbers.

            Parameters
            ----------
            state_functions
                The container of the state functions.
            control_functions
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
    def from_steps(cls, steps: int) -> Propagator:
        """ Construct Empty :class:`Propagator` From Number of Time Steps

            Parameters
            ----------
            number_of_steps
                The number of time steps (see :func:`steps`).

            Returns
            -------
            Propagator
                The empty :class:`Propagator` of specified number of steps.
        """     
        return Propagator(
            [None] * (steps + 1),
            [None] * (steps + 1)
        )
    
    def as_table(self, width: Optional[int] = None, height: Optional[int] = None, optimizer: Optional[torch.optim.Optimizer] = None) -> str:
        return "\n".join(create_table(self, width=width, height=height, optimizer=optimizer))

    def __repr__(self) -> str:
        return self.as_table()

    @property
    def state_functions(self) -> MutableSequence[Optional[object]]:
        r"""The Mutable, Zero-based Sequence of State Functions $(F_0,\dots, F_T)$
        
            To manipulate state functions, access the contents of this sequence.
            Immutable as a property in the sense that any attempts to replace the sequence
            by another object will be ignored.  

            Returns
            -------
            MutableSequence[Optional[object]]
                The sequence of state functions $F_0,\dots, F_T$.
        """

        return self._state_functions
        

    @state_functions.setter
    def state_functions(self, value: Any) -> None:
        warnings.warn("To change state_functions, access the list contents. Ignored.")
        return

    @property
    def control_functions(self) -> MutableSequence[Optional[object]]:
        r"""The Mutable, Zero-Based Sequence of Control Functions $(A_0,\dots, A_T)$

            To manipulate control functions, access the contents of this sequence.
            Immutable as a property in the sense that any attempts to replace the sequence
            by another object will be ignored.  

            Returns
            -------
            MutableSequence[Optional[object]]
                The sequence of control functions $A_0,\dots, A_T$.
        """
        return self._control_functions

    @control_functions.setter
    def control_functions(self, value: Any) -> None:
        warnings.warn("To change control functions, access the list contents. Ignored.")
        return
        
    def __len__(self) -> int:
        r"""Return the Length of :class:`Propagator`'s

            If ``self`` has the state functions $F=(F_0,\dots, F_T)$ and the control functions
            $A=(A_0,\dots, A_T)$, then the *length* of ``self`` is considered to be $T+1$.

            Returns
            -------
            int
                The length $T+1$
        """
        assert len(self._state_functions) == len(self._control_functions)
        return len(self._state_functions)
        
    def steps(self) -> int:
        r"""Return the Number of Steps of :class:`Propagator`'s

            If ``self`` is $F^A$ with the state functions $F=(F_0,\dots, F_T)$ and the control functions
            $A=(A_0,\dots, A_T)$, then return the number of steps between the time points $0,\dots, T$ 
            (which is $T$).
    
            Returns
            -------
            int
                The number of steps $T$
        """
        return len(self) - 1
    
    
    def __getitem__(self, key: Union[int, slice]) -> Propagator:
        r"""Return Sub-:class:`Propagator`'s

            If ``self`` has the state functions $F=(F_0,\dots, F_T) and the control functions
            $A=(A_0,\dots, A_T)$ and ``key`` specifies the subset $I=\{t_0,\dots, t_k\}$ of $\{0, \dots, T\}$,
            then return the :class:`Propagator` given by the state functions $(F_{t_0},\dots, F_{t_k})$, and the control functions $(A_{i_0}, \dots, A_{i_k})$.

            Parameters
            ----------
            key
                Specifies the subset $I$ of indices (``int``'s indicate singletons)

            Returns
            -------
            Propagator
                The sub-:class:`Propagator`

            Raises
            ------
            KeyError
                Raised if ``key`` does not specify a valid subrange
        """

        if isinstance(key, int):
            return Propagator([self._state_functions[key]],
                              [self._control_functions[key]])
        elif isinstance(key, slice):
            return Propagator(self._state_functions[key],
                              self._control_functions[key])
        else:
            raise KeyError("Query using int's or slices.")
    
    def __setitem__(self, key: Union[int, slice], value: Propagator) -> None:
        if isinstance(value, Propagator):
            self._state_functions[key] = value._state_functions
            self._control_functions[key] = value._control_functions
        else:
            raise ValueError("Cannot assign given value.")
        
        self._register_modules()
    
    def __add__(self, other: Propagator) -> Propagator:
        r"""Concatenate :class:`Propagator` Objects
            
            If ``self`` has the state functions $F=(F_0,\dots, F_T)$ and the control functions
            $A=(A_0,\dots, A_T)$ and ``other`` is as well a :class:`Propagator` with the state functions $G=(G_0,\dots, G_S)$ and the control functions
            $B=(B_0,\dots, B_S)$, then return the concatenated $(T+S+1)$-step :class:`Propagator` given by the state functions $(F_0, \dots, F_T, G_0, \dots, G_S)$ and the control functions $(A_0,\dots, A_T, B_0,\dots, B_S)$.
            
            Parameters
            ----------
            other
                The :class:`Propagator` to be appended on the right  

            Returns
            -------
            Propagator
                The concatenation of ``self`` and ``other``

            Raises
            ------
            TypeError
                Raised if ``other`` is not a :class:`Propagator`
        """
        if isinstance(other, Propagator):
            return Propagator(
                list(self._state_functions) + list(other._state_functions),
                list(self._control_functions) + list(other._control_functions)
            )
        else:
            raise TypeError("May only add `Propagator`'s")

    def propagate(self,
                  initial_state: Optional[Sample] = None,
                  random_effects: Optional[Sequence[Optional[Sample]]] = None) -> tuple[list[Sample], list[Sample]]:
        r"""Compute Controlled State Evolution and Corresponding Sequence of Controls

            More precisely, implements
            $$(s_0,(\xi_t)_{t=1}^{T+1})\mapsto\left((s_t)_{t=0}^{T+1}, (a_t)_{t=0}^T, (\xi_t)_{t=1}^{T+1}\right).$$
            where
            $$s_{t+1} = F_t(s_t,a_t, \xi_{t+1}), \quad a_t = A_t(s_t, \xi_t),\quad t=0,\dots, T$$
            and $(F_0,\dots, F_T)$ are the state functions and $(A_0,\dots, A_T)$ are the control functions
            saved in :attr:`Propagator.state_functions` and :attr:`Propagator.control_functions`, respectively.

            Parameters
            ----------
            initial_state
                The initial state $s_0$, by default None.
            random_effects
                The sequence of random effects $(\xi_i)_{i=0}^{T+1}$, by default None.

            Returns
            -------
            List[Optional[torch.Tensor]]
                The list containing the state evolution $(s_i)_{i=0}^{T+1}$
            List[Optional[torch.Tensor]]
                The list containing sequence of controls $(a_i(s_i))_{i=0}^T$
            List[Optional[torch.Tensor]]
                The list containing the random effects $(\xi_i)_{i=1}^{T+1}$
        """
        
        if random_effects is None:
            random_effects = [None] * len(self)
            
        states = []
        controls = []
        
        # Prep initial state s_0
        if (state := initial_state) is None:
            state = {}
        states.append(state)
        
        for control_func, state_func, random_effect in it.islice(
            it.zip_longest(self._control_functions,
                           self._state_functions,
                           random_effects),
            len(self)
        ):
            # Produce time-t control $a_t=A_t(s_t)$
            controls.append(control := {} if control_func is None else control_func(**state))
            
            # Next time step t+1:
            # Prep time-(t+1) random effect xi_{t+1}
            if random_effect is None:
                random_effect = {}
                
            # Produce time-(t+1) state s_{t+1} = F_t(s_t, a_t, xi_{t+1}) 
            states.append(state := {} if state_func is None else state_func(**state, **control, **random_effect))
            
        return states, controls
     
    def forward(self,
                initial_state: Optional[Sample] = None,
                random_effects: Optional[Sequence[Optional[Sample]]] = None) -> Sample:

        state, _ = self.propagate(initial_state, random_effects)
        
        return state[-1]


class CostToGo(torch.nn.Module):
    
    r"""Compute Total Cost Incurred Along Controlled State Evolutions

        Saves a :class:`Propagator` and a list of cost functions $(K_0, \dots, K_T)$ of equal
        lengths.
        As a callable, implements the map
        $$K^{F, A}\colon (s_0, (\xi_t)_{t=1}^{T+1})) \mapsto \sum_{t=0}^T K_t(s_t, A_t(s_t))$$
        where $(A_0,\dots, A_T)$ are the control functions saved by the :class:`Popagator`
        and $(s_t)_{t=0}^T$ is the state evolution as computed by the :class:`Propagator`.
    """
    
    def __init__(self,
                 propagator: Propagator,
                 cost_functions: Sequence[Optional[Any]]) -> None:

        r"""Construct a :class:`CostToGo` object from :class:`Propagator` and list of cost functions
            
            Must provide list ``cost_functions`` of cost functions compatible in length with ``propagator``.

            Parameters
            ----------
            propagator
                The possibly zero-length Propagator
            cost_functions
                The possibly empty sequence of cost functions
                
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
        
    @classmethod
    def from_steps(cls, steps: int) -> CostToGo   :
        r"""Construct an Empty :class:`CostToGo` From Number of Time Steps

            Parameters
            ----------
            number_of_steps :
                The number of time steps (see :func:`steps`).

            Returns
            -------
            CostToGo
                The empty :class:`CostToGo` of specified number of steps.
        """
        cost_functions = [None] * (steps + 1)
        propagator = Propagator.from_steps(steps)

        return CostToGo(propagator, cost_functions)

    def as_table(self, width: Optional[int] = None, height: Optional[int] = None, optimizer: Optional[torch.optim.Optimizer] = None) -> str:
        return "\n".join(create_table(self, width=width, height=height, optimizer=optimizer))
    
    def __repr__(self) -> str:
        return self.as_table()
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'propagator' and hasattr(self, 'propagator'):
            warnings.warn('Change state and cost functions directly')
            return
        else:
            super().__setattr__(name, value)

    @property
    def state_functions(self) -> MutableSequence[Optional[Any]]:
        r"""The Mutable, Zero-Based Sequence of State Functions $(F_0,\dots, F_T)$
            Saved by :attr:`CostToGo.propagator`

            To manipulate state functions, access the contents of this sequence.
            Immutable as a property in the sense that any attempts to replace the sequence
            by another object will be ignored.  

            Returns
            -------
            MutableSequence[Optional[Any]]
                The sequence of state functions $F_0,\dots, F_T$.
        """
        return self.propagator.state_functions

    @state_functions.setter
    def state_functions(self, value: Any) -> None:
        self.propagator.state_functions = value

    @property
    def control_functions(self) -> MutableSequence[Optional[Any]]:
        r"""The Mutable, Zero-Based Sequence of Control Functions 
            $(A_0,\dots, A_T)$ Saved by :attr:`CostToGo.propagator`

            To manipulate control functions, access the contents of this sequence.
            Immutable as a property in the sense that any attempts to replace the sequence
            by another object will be ignored.  

            Returns
            -------
            MutableSequence[Optional[Any]]
                The sequence of control functions $A_0,\dots, A_T$.
        """
        return self.propagator.control_functions

    @control_functions.setter
    def control_functions(self, value: Any) -> None:
        self.propagator.control_functions = value

    @property
    def cost_functions(self) -> MutableSequence[Optional[Any]]:
        r"""The Mutable, Zero-Based Sequence of Cost Functions $(K_0,\dots, K_T)$

            To manipulate cost functions, access the contents of this sequence.
            Immutable as a property in the sense that any attempts to replace the sequence
            by another object will be ignored.  

            Returns
            -------
            MutableSequence[Optional[object]]
                The sequence of cost functions $K_0,\dots, K_T$.
        """
        return self._cost_functions

    @cost_functions.setter
    def cost_functions(self, value: Any) -> None:
        warnings.warn("To change cost functions, access list contents. Ignored.")

    def __getitem__(self, key: Union[int, slice]) -> CostToGo:
        r"""Return Sub-:class:`CostToGo`'s

            If ``self`` has the state functions $F=(F_0,\dots, F_T)$, the control functions
            $A=(A_0,\dots, A_T)$ and the cost functions $k=(K_0,\dots, K_T)$ and ``key`` specifies 
            the subset $I=\{t_0,\dots, t_k\}$ of $\{0, \dots, T\}$,
            then return the :class:`CostToGo` given by the state functions $(F_{t_0},\dots, F_{t_k})$,
            the control functions $(A_{t_0}, \dots, A_{t_k})$ and the cost functions $(K_{t_0},\dots, K_{t_k})$.

            Parameters
            ----------
            key
                Specifies the subset $I$ of indices (``int``'s indicate singletons)

            Returns
            -------
            CostToGo
                The sub-:class:`CostToGo`

            Raises
            ------
            KeyError
                Raised if ``key`` does not specify a valid subrange
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
            raise KeyError("Query using int's or slices.")
        
    def __setitem__(self, key: Union[int, slice], value: CostToGo) -> None:

        if isinstance(value, CostToGo):
            self.state_functions[key] = value.state_functions
            self.control_functions[key] = value.control_functions
            self.cost_functions[key] = value.cost_functions
        else:
            raise ValueError("Cannot assign given value.")

    def __len__(self) -> int:
        r"""Return the Length of :class:`CostToGo`'s

            If ``self`` has the state functions $F=(F_0,\dots, F_T)$, the control functions
            $A=(A_0,\dots, A_T)$ and the cost functions $k=(K_0,\dots, K_T)$, then the length of ``self`` is $T+1$.

            Returns
            -------
            int
                The length $T+1$
        """
        assert len(self.propagator) == len(self.cost_functions)
        return len(self.cost_functions)

    def steps(self) -> int:
        r"""Return the Number of Steps of :class:`CostToGo`'s

            If ``self`` has the state functions $F=(F_0,\dots, F_T)$, the control functions
            $A=(A_0,\dots, A_T)$ and the cost functions $k=(K_0,\dots, K_T)$, then return the
            number of steps between the time points $0,\dots, T$ (which is $T$).


            Returns
            -------
            int
                The number of steps $T$
        """
        return len(self) - 1

    def __add__(self, other: CostToGo) -> CostToGo:
        r"""Concatenate :class:`CostToGo`'s
            
            If ``self`` has the state functions $F=(F_0,\dots, F_T)$, the control functions
            $A=(A_0,\dots, A_T)$ and the cost functions $k=(K_0,\dots, K_T)$ and ``other`` is as well a :class:`CostToGo` with the state functions $G=(G_0,\dots, G_S)$, the control functions
            $B=(B_0,\dots, B_S)$ and the cost functions $L=(L_0,\dots,L_S)$, then return the concatenated $(T+S+1)$-step :class:`CostToGo` given by the state functions $(F_0, \dots, F_T, G_0, \dots, G_S)$, the control functions $(A_0,\dots, A_T, B_0,\dots, B_S)$ and the  cost functions $(K_0,\dots, K_T, L_0,\dots, L_S)$.
            
            Parameters
            ----------
            other
                The :class:`CostToGo` to be appended on the right  

            Returns
            -------
            CostToGo
                The concatenation of ``self`` and `other`

            Raises
            ------
            TypeError
                Raised if ``other`` is not a :class:`CostToGo`
        """
        if isinstance(other, CostToGo):
            return CostToGo(
                self.propagator + other.propagator,
                list(self.cost_functions) + list(other.cost_functions)
            )
        else:
            raise TypeError("Can only add `CostToGo`'s")

    def forward(self,
                initial_state: Optional[Sample] = None,
                random_effects: Optional[Sequence[Optional[Sample]]] = None) -> float | Tensor:
               
        states, controls = self.propagator.propagate(initial_state, random_effects)
        
        cost = 0.
        
        for step, cost_func in enumerate(self.cost_functions):
            if cost_func is not None:
                cost += cost_func(**states[step], **controls[step])
        
        return cost

