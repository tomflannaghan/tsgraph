from typing import List, Iterable

import pandas as pd


class Curve:
    def __init__(self, dfs: Iterable[pd.DataFrame] = (), columns: List[str] = None):
        self._dfs = []
        self.columns = None
        self.extend(dfs)
        if columns is None:
            raise ValueError("Cannot construct a curve with no dataframes or columns specified.")

    def append(self, df):
        df_columns = list(df.columns)
        if self.columns is None:
            self.columns = df_columns
        if df_columns != self.columns:
            raise ValueError(f"Columns don't match, require {self.columns}")
        if df.empty:
            return
        if len(self._dfs) > 0:
            if self._dfs[-1].index[-1] >= df.index[0]:
                raise ValueError("Non-ordered dataframes.")
        self._dfs.append(df)

    def extend(self, dfs: Iterable[pd.DataFrame] | "Curve"):
        if isinstance(dfs, Curve):
            dfs = dfs._dfs
        for df in dfs:
            self.append(df)

    def as_df(self) -> pd.DataFrame:
        return pd.concat(self._dfs)

    def get_end_dt(self) -> pd.Timestamp | None:
        if len(self._dfs) == 0:
            return None
        else:
            return self._dfs[-1].index[-1]

    def __getitem__(self, item) -> "Curve":
        if not isinstance(item, slice):
            raise("Requires a slice object")
        return Curve(dfs=[df.loc[item] for df in self._dfs], columns=self.columns)
