import pandas as pd
import numpy as np


def pd_isin_multiple_cols(df1, df2):
    """
    For all columns in df2 
    and columns of df1 having the same names,
    select a subset of df1 consisting of rows
    that exist in df2.
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

    Will only work for the dataframes with a common index. 
    Otherwise, a ValueError is raised:
    "Can only compare identically-labeled DataFrame objects". 
    """

    common_colnames = set(df1.columns).intersection(df2.columns)
    assert set(colnames).issubset(common_colnames)

    return np.all(df1[colnames] == df2[colnames])


def pd_col_is_identical_no_index(df1, df2, colname):
    """
    Check whether an identically-named column in 
    two dataframes has the same data. 
    The function disregards dataframe index. 
    """

    c1 = np.array(df1[colname])
    c2 = np.array(df2[colname])

    return np.all(c1 == c2)


def pd_find_switchpoints(target_series):
    """
    Given a Pandas Series (or a 1D NumPy array),
    return indices using (0 ... n-1) indexing
    that correspond to cases when a value 
    in the original Series changes.

    As such, index i in the output NumPy array
    shows that target_series[i - 1] != target_series[i].

    The output array will always contain 0 as the first 
    element. 
    """

    uniq, indices = np.unique(target_series, return_inverse=True)

    diff = np.diff(indices)

    switchpoints = np.argwhere(diff != 0).reshape(-1)

    return np.concatenate((np.zeros(1, dtype=int), switchpoints + 1))


def pd_get_cols_subset_by_template(df, template, rng):
    """
    Select certain columns from the supplied dataframe
    corresponding to a template like "X_{}"
    and a range values (e.g. ingegers like range(1, 4)) 
    inserted into the template. 
    """

    colnames = [template.format(el) for el in rng]
    return df[colnames]


def pd_cols_with_identical_values(df):
    """
    Given a dataframe, return a dictionary mapping
    column names to a single value for columns
    containing an identical value.
    """

    n = len(df)

    res = dict()

    for colname in df.columns:
    
        col = df[colname]
        first_val = col.iloc[0]
        all_same = (np.sum(col == first_val) == n)
        
        if all_same:
            res[colname] = first_val

    return res


def pd_rolling_mean(series, window_size):
    """
    Compute rolling mean on a Series 
    with the given window_size
    and return only non-None rows,
    i.e. starning from row number window_size - 1
    and until the end. 
    """
    
    rolling_mean = series.rolling(window_size).mean()
    
    return rolling_mean[window_size - 1:]