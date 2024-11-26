import importlib
import inspect
from abc import ABC, abstractmethod
from inspect import Parameter
from itertools import chain
from typing import Iterable, Callable, List

import pandas as pd
from more_itertools.recipes import unique_everseen
from pandas import DatetimeIndex

ONE_D_COL = 0
ONE_D_COLS = (ONE_D_COL,)
EPSILON = pd.Timedelta('1ns')


def as_valid_result(value):
    if isinstance(value, pd.DataFrame):
        return value
    elif isinstance(value, pd.Series):
        return pd.DataFrame({ONE_D_COL: value})
    else:
        raise ValueError(f"Results must be dataframes, got {type(value)}")


def to_series(value):
    if isinstance(value, pd.DataFrame):
        if value.shape[-1] != 1:
            raise ValueError("Can't convert multidimensional df to a series")
        return value.iloc[:, 0]
    else:
        return value


def get_slice(start_dt, end_dt):
    slice_start = start_dt + EPSILON if start_dt is not None else None
    return slice(slice_start, end_dt)


def get_data_for_node(node, start_dt, end_dt, working_data, type_error=True):
    if isinstance(node, Node):
        return working_data[node][get_slice(start_dt, end_dt)]
    elif type_error:
        raise TypeError(f"{node} not a node")
    else:
        return node


class Node(ABC):

    def __init__(self, name: str, parents: Iterable["Node"], columns: Iterable[str] = ONE_D_COLS):
        self._name = name
        self.parents = list(parents)
        self.columns = list(columns)

    def __repr__(self):
        return f'{self._name}[{", ".join(str(c) for c in self.columns)}]'

    @abstractmethod
    def get_required_evaluations(self, start_dt, end_dt):
        pass

    @abstractmethod
    def evaluate(self, start_dt, end_dt, working_data) -> pd.DataFrame:
        """Evaluates the node over the given date range given working data."""

    def graph_start_dt(self) -> pd.Timestamp:
        return min(n.graph_start_dt() for n in self.parents)

    def calc(self):
        """Convinient way to evaluate up to the present day"""
        from tsgraph.evaluation import evaluate_nodes
        result = evaluate_nodes([self], start_dt=None, end_dt=pd.Timestamp.now())
        return result[self]


class OutputNode(Node):

    def __init__(self, input_node: Node):
        super().__init__('output_node', [input_node], columns=input_node.columns)
        self.data = []
        self._input_node = input_node
        self._current_dt = None

    def get_required_evaluations(self, start_dt, end_dt):
        if self._current_dt is None or end_dt > self._current_dt:
            return [(self._input_node, (self._current_dt, end_dt))]
        else:
            return []

    def evaluate(self, start_dt, end_dt, working_data) -> pd.DataFrame:
        exact_range = False
        if self._current_dt is None or end_dt > self._current_dt:
            result = get_data_for_node(self._input_node, self._current_dt, end_dt, working_data)
            self.data.append(result)
            exact_range = start_dt == self._current_dt
            self._current_dt = end_dt
        return result if exact_range else self.as_df()[get_slice(start_dt, end_dt)]

    def as_df(self):
        if len(self.data) != 1:
            self.data = [pd.concat(self.data)]
        return self.data[0]


def output_node(node) -> OutputNode:
    return OutputNode(node)


class FunctionRegistry():
    def __init__(self):
        self.registry = {}
        self.reverse_lookup = {}

    def register(self, func):
        key = func.__module__, func.__name__
        if key in self.registry:
            return
        if func in self.reverse_lookup:
            raise ValueError("Same function can't be registered under different names")
        if func.__name__ == '<lambda>':
            raise ValueError("Cannot register lambdas.")
        if func.__module__ == '__main__':
            raise ValueError("Cannot register functions defined in __main__ script.")
        self.registry[key] = func
        self.reverse_lookup[func] = key

    def get_key(self, func):
        return self.reverse_lookup[func]

    def get_func(self, key):
        if key in self.registry:
            return self.registry[key]
        # If it's not in there, we must try to import the module, which should make it become registered.
        importlib.import_module(key[0])
        if key not in self.registry:
            raise ValueError(f"Function {key[1]} not found by importing {key[0]}")
        return self.registry[key]


_FUNCTION_REGISTRY = FunctionRegistry()


class FuncNode(Node):

    def __init__(self, func: Callable, *args, columns: Iterable = ONE_D_COLS, **kwargs):
        super().__init__(name=func.__name__, parents=self.get_parents(args, kwargs), columns=columns)
        _FUNCTION_REGISTRY.register(func)
        self._func = func
        self._args = args
        self._kwargs = kwargs
        sig = inspect.signature(func)
        self._is_stateful = 'state' in sig.parameters
        if self._is_stateful:
            if 'state' in self._kwargs:
                del self._kwargs['state']
            self._initial_state = kwargs.get('state')
            if self._initial_state is None and sig.parameters['state'].default != Parameter.empty:
                self._initial_state = sig.parameters['state'].default
        else:
            self._initial_state = None
        self._state = self._initial_state
        if 'columns' in sig.parameters:
            self._kwargs['columns'] = columns
        self._current_dt = None

    def __getstate__(self):
        state = self.__dict__.copy()
        func = state.pop('_func')
        state['_func_key'] = _FUNCTION_REGISTRY.get_key(func)
        return state

    def __setstate__(self, state):
        key = state['_func_key']
        func = _FUNCTION_REGISTRY.get_func(key)
        self.__dict__.update(state)
        del self.__dict__['_func_key']
        self._func = func

    @staticmethod
    def get_parents(args, kwargs):
        parents = [v for v in chain(args, kwargs.values()) if isinstance(v, Node)]
        if len(parents) == 0:
            raise ValueError("Cannot have no node inputs")
        return parents

    def get_required_evaluations(self, start_dt, end_dt):
        # First what evaluation do we require.
        if self._is_stateful and (self._current_dt is None or start_dt is None or start_dt != self._current_dt):
            self_eval = (None, end_dt)
        else:
            self_eval = (start_dt, end_dt)
        # Next, we require the parent evaluations over the same range.
        return [(p, self_eval) for p in self.parents]

    def evaluate(self, start_dt, end_dt, working_data) -> pd.DataFrame:
        eval_start_dt = start_dt
        if self._is_stateful and start_dt != self._current_dt:
            self.reset()
            eval_start_dt = None

        args = [get_data_for_node(v, eval_start_dt, end_dt, working_data, type_error=False) for v in self._args]
        kwargs = {k: get_data_for_node(v, eval_start_dt, end_dt, working_data, type_error=False)
                  for k, v in self._kwargs.items()}
        if self._is_stateful:
            kwargs['state'] = self._state
            result, state = self._func(*args, **kwargs)
            self._state = state
        else:
            result = self._func(*args, **kwargs)
        self._current_dt = end_dt
        return as_valid_result(result.loc[get_slice(start_dt, end_dt)])

    def reset(self):
        self._current_dt = None
        self._state = self._initial_state


class NodeDecorator:
    def __init__(self, func: Callable[..., pd.DataFrame]):
        self.func = func

    def __call__(self, *args, columns=None, **kwargs) -> Node:
        if columns is None:
            parents = FuncNode.get_parents(args, kwargs)
            columns = self.infer_columns(parents)
        return FuncNode(self.func, *args, columns=columns, **kwargs)

    @abstractmethod
    def infer_columns(self, parents: List[Node]):
        """The strategy this decorator uses to infer the columns from the arguments given to the function"""


class HomogeneousNodeDecorator(NodeDecorator):
    """
    Decorator that requires all input nodes to have the same columns.
    """

    def infer_columns(self, parents: List[Node]):
        all_column_specs = list(unique_everseen(n.columns for n in parents))
        if len(all_column_specs) > 1:
            raise ValueError(f"Different columns found in input nodes - {all_column_specs}")
        return all_column_specs[0]


class ScalarNodeDecorator(NodeDecorator):
    """
    A decorator that produces nodes that are 1d with the standard 1d columns.
    """

    def infer_columns(self, parents: List[Node]):
        return ONE_D_COLS


node = HomogeneousNodeDecorator
scalar_node = ScalarNodeDecorator


class DataFrameNode(Node):

    def __init__(self, df: pd.DataFrame):
        super().__init__('df_node', [], columns=df.columns)
        if df.empty:
            raise ValueError("Cannot use empty dataframe for a node")
        self._df = df

    def get_required_evaluations(self, start_dt, end_dt):
        return []

    def evaluate(self, start_dt, end_dt, working_data) -> pd.DataFrame:
        slice_start = start_dt + EPSILON if start_dt is not None else self.graph_start_dt()
        return self._df[slice_start:end_dt]

    def graph_start_dt(self) -> pd.Timestamp:
        return self._df.index[0]


def df_node(df):
    return DataFrameNode(as_valid_result(df))


def empty_df(columns):
    return pd.DataFrame([], index=DatetimeIndex([]), columns=columns)
