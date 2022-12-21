import pandas as pd
dt = pd.read_csv('E:\\machine learning\\airlines.csv')
print(dt)
dm = pd.get_dummies(dt)
print(dm)
"""
pd.get_dummies(
    data,
    prefix=None,
    prefix_sep='_',
    dummy_na: 'bool' = False,
    columns=None,
    sparse: 'bool' = False,
    drop_first: 'bool' = False,
    dtype: 'Dtype | None' = None,
) -> 'DataFrame'
Docstring:
Convert categorical variable into dummy/indicator variables.

Parameters
----------
data : array-like, Series, or DataFrame
    Data of which to get dummy indicators.
prefix : str, list of str, or dict of str, default None
    String to append DataFrame column names.
    Pass a list with length equal to the number of columns
    when calling get_dummies on a DataFrame. Alternatively, `prefix`
    can be a dictionary mapping column names to prefixes.
prefix_sep : str, default '_'
    If appending prefix, separator/delimiter to use. Or pass a
    list or dictionary as with `prefix`.
dummy_na : bool, default False
    Add a column to indicate NaNs, if False NaNs are ignored.
columns : list-like, default None
    Column names in the DataFrame to be encoded.
    If `columns` is None then all the columns with
    `object`, `string`, or `category` dtype will be converted.
sparse : bool, default False
    Whether the dummy-encoded columns should be backed by
    a :class:`SparseArray` (True) or a regular NumPy array (False).
drop_first : bool, default False
    Whether to get k-1 dummies out of k categorical levels by removing the
    first level.
dtype : dtype, default np.uint8
    Data type for new columns. Only a single dtype is allowed.

Returns
-------
DataFrame
    Dummy-coded data.

See Also
--------
Series.str.get_dummies : Convert Series to dummy codes.
:func:`~pandas.from_dummies` : Convert dummy codes to categorical ``DataFrame``.

Notes
-----
Reference :ref:`the user guide <reshaping.dummies>` for more examples.

Examples
--------
>>> s = pd.Series(list('abca'))

>>> pd.get_dummies(s)
   a  b  c
0  1  0  0
1  0  1  0
2  0  0  1
3  1  0  0

>>> s1 = ['a', 'b', np.nan]

>>> pd.get_dummies(s1)
   a  b
0  1  0
1  0  1
2  0  0

>>> pd.get_dummies(s1, dummy_na=True)
   a  b  NaN
0  1  0    0
1  0  1    0
2  0  0    1

>>> df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
...                    'C': [1, 2, 3]})

>>> pd.get_dummies(df, prefix=['col1', 'col2'])
   C  col1_a  col1_b  col2_a  col2_b  col2_c
0  1       1       0       0       1       0
1  2       0       1       1       0       0
2  3       1       0       0       0       1

>>> pd.get_dummies(pd.Series(list('abcaa')))
   a  b  c
0  1  0  0
1  0  1  0
2  0  0  1
3  1  0  0
4  1  0  0

>>> pd.get_dummies(pd.Series(list('abcaa')), drop_first=True)
   b  c
0  0  0
1  1  0
2  0  1
3  0  0
4  0  0

>>> pd.get_dummies(pd.Series(list('abc')), dtype=float)
     a    b    c
0  1.0  0.0  0.0
1  0.0  1.0  0.0
2  0.0  0.0  1.0
File:      c:\users\semil\appdata\local\programs\python\python310\lib\site-packages\pandas\core\reshape\encoding.py
Type:      function
"""