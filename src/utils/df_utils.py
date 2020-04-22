from typing import Iterable

from pandas import DataFrame
from tabulate import tabulate


def format_df_columns(df: DataFrame, column_names: Iterable[str], fmt: str):
    for c in column_names:
        format_df_column(df, c, fmt)


def format_df_column(df: DataFrame, column_name: str, fmt: str):
    df[column_name] = df[column_name].map(fmt.format)


def df_to_latex_table(df: DataFrame) -> str:
    return tabulate(df, tablefmt="latex_raw", headers="keys").replace("% ", "\\%").replace("_", "\\_")
