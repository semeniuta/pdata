import pandas as pd
import numpy as np


def pd_iterate_over_rows(df):
    """
    Iterate over rows of a dataframe
    where is row is a Series object
    """

    for i in range(len(df)):
        row = df_new.iloc[i]
        yield row


def pd_isin_multiple_cols(df1, df2):
    """
    For all columns in df2,
    select a subset of df1 with the matched values
    in the columns with the same names
    """

    mask = np.ones(len(df1), dtype=bool)

    for colname in df2.columns:

        col_match = df1[colname].isin(df2[colname])
        mask = mask & col_match

    return mask


def pd_content_in_columns_is_identical(df1, df2, colnames):
    """
    Check whether data in the selected 
    identically-named columns in two dataframes
    is identical. 
    """

    common_colnames = set(df1.columns).intersection(df2.columns)
    assert set(colnames).issubset(common_colnames)

    return np.all(df1[colnames] == df2[colnames])
