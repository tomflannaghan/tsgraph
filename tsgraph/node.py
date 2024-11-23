import inspect
from abc import ABC, abstractmethod
from inspect import Parameter
from itertools import chain
from typing import Iterable, Callable

import pandas as pd

from tsgraph.curve import Curve, empty_df

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


class Node(ABC):

    def __init__(self, parents: Iterable["Node"], columns: Iterable[str] = ONE_D_COLS):
        self._parents = list(parents)
        self.columns = list(columns)
        self.current_dt = None
        self.prev_dt = None
        self.prev_result = None

    def advance_child(self, node, end_dt) -> pd.DataFrame:
        if not isinstance(node, Node):
            return node
        if node.prev_result is not None and node.prev_dt == self.current_dt and node.current_dt == end_dt:
            return node.prev_result
        if node.current_dt is not None and node.current_dt != self.current_dt:
            node.reset_all()
            node.advance(self.current_dt)  # Advance to our current time.
        return node.advance(end_dt)

    def advance(self, end_dt) -> pd.DataFrame:
        if end_dt == self.current_dt:
            return empty_df(columns=self.columns)
        result = self.evaluate(self.current_dt, end_dt)
        result = as_valid_result(result)
        if list(result.columns) != self.columns:
            raise ValueError(f"Incorrect columns returned: {list(result.columns)} != {self.columns}")
        self.prev_dt = self.current_dt
        self.current_dt = end_dt
        self.prev_result = result
        return result

    @abstractmethod
    def evaluate(self, start_dt, end_dt) -> pd.DataFrame:
        """Evaluates the node over the given date range."""

    def reset(self):
        """Resets the node"""
        self.current_dt = None
        self.prev_dt = None
        self.prev_result = None

    def hard_reset(self):
        self.reset()

    def reset_all(self):
        for n in self._parents:
            n.reset_all()
        self.reset()

    def hard_reset_all(self):
        for n in self._parents:
            n.hard_reset_all()
        self.hard_reset()

    def graph_start_dt(self) -> pd.Timestamp:
        return min(n.graph_start_dt() for n in self._parents)

    def calc(self):
        """Convinient way to evaluate up to the present day"""
        self.reset()
        return self.advance(pd.Timestamp.now())


class OutputNode(Node):

    def __init__(self, input_node: Node):
        super().__init__([input_node], columns=input_node.columns)
        self.data = Curve(columns=input_node.columns)
        self._input_node = input_node

    def __repr__(self):
        return f'output_node({id(self)})'

    def evaluate(self, start_dt, end_dt) -> pd.DataFrame:
        if self.current_dt is None or end_dt > self.current_dt:
            result = self.advance_child(self._input_node, end_dt)
            self.data.append(result)
        if start_dt is None:
            start_dt = self.graph_start_dt()
        slice_start = start_dt + EPSILON if start_dt is not None else self.graph_start_dt()
        return self.data[slice_start:end_dt].as_df()

    def hard_reset(self):
        super().hard_reset()
        self.data = Curve(columns=self.columns)

    def reset(self):
        pass

    def reset_all(self):
        pass


def output_node(node) -> OutputNode:
    return OutputNode(node)


class FuncNode(Node):

    def __init__(self, func: Callable, *args, columns: Iterable = ONE_D_COLS, **kwargs):
        super().__init__(parents=self.get_parents(args, kwargs), columns=columns)
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

    def __repr__(self):
        return f'{self._func.__name__}({id(self)})'

    @staticmethod
    def get_parents(args, kwargs):
        parents = [v for v in chain(args, kwargs.values()) if isinstance(v, Node)]
        if len(parents) == 0:
            raise ValueError("Cannot have no node inputs")
        return parents

    def evaluate(self, start_dt, end_dt) -> pd.DataFrame:
        if start_dt != self.current_dt:
            self.reset_all()
            return self.advance(end_dt)[start_dt:]

        args = [self.advance_child(v, end_dt) for v in self._args]
        kwargs = {k: self.advance_child(v, end_dt) for k, v in self._kwargs.items()}
        if self._is_stateful:
            kwargs['state'] = self._state
            result, state = self._func(*args, **kwargs)
            self._state = state
        else:
            result = self._func(*args, **kwargs)
        return result

    def reset(self):
        super().reset()
        self._state = self._initial_state


def node(func: Callable[..., pd.DataFrame]) -> Callable[..., FuncNode]:
    """
    A decorator for constructing function nodes. It assumes that all node inputs have either
    the same set of columns or are 1d default columns. It then assumes the output column set matches.
    """

    def wrapped(*args, **kwargs):
        parents = FuncNode.get_parents(args, kwargs)
        columns = ONE_D_COLS
        for n in parents:
            if n.columns != ONE_D_COLS:
                if columns == ONE_D_COLS:
                    columns = n.columns
                elif columns != n.columns:
                    raise ValueError("Different columns found in input nodes. Can't infer column type.")

        return FuncNode(func, *args, columns=columns, **kwargs)

    return wrapped


def scalar_node(func: Callable[..., pd.DataFrame]) -> Callable[..., FuncNode]:
    """
    A decorator for constructing function nodes that return 1 dimensional results.
    """

    def wrapped(*args, **kwargs):
        return FuncNode(func, *args, columns=ONE_D_COLS, **kwargs)

    return wrapped



class DataFrameNode(Node):

    def __init__(self, df: pd.DataFrame):
        super().__init__([], columns=df.columns)
        if df.empty:
            raise ValueError("Cannot use empty dataframe for a node")
        self._df = df

    def __repr__(self):
        return f'df_node({id(self)})'

    def evaluate(self, start_dt, end_dt) -> pd.DataFrame:
        slice_start = start_dt + EPSILON if start_dt is not None else self.graph_start_dt()
        return self._df[slice_start:end_dt]

    def graph_start_dt(self) -> pd.Timestamp:
        return self._df.index[0]


def df_node(df):
    return DataFrameNode(as_valid_result(df))
