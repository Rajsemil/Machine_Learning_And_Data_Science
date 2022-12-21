import pandas as pd
from sklearn.preprocessing import OneHotEncoder
dt = pd.read_csv('E:\\machine learning\\airlines.csv')
print(dt)
oh_enc = OneHotEncoder(sparse = True)
print(oh_enc)
oh_enc_arr = oh_enc.fit_transform(dt[['Time.Year', 'Time.Label']])
print(oh_enc_arr)
"""
OneHotEncoder(
    *,
    categories='auto',
    drop=None,
    sparse=True,
    dtype=<class 'numpy.float64'>,
    handle_unknown='error',
    min_frequency=None,
    max_categories=None,
)
Docstring:     
Encode categorical features as a one-hot numeric array.

The input to this transformer should be an array-like of integers or
strings, denoting the values taken on by categorical (discrete) features.
The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
encoding scheme. This creates a binary column for each category and
returns a sparse matrix or dense array (depending on the ``sparse``
parameter)

By default, the encoder derives the categories based on the unique values
in each feature. Alternatively, you can also specify the `categories`
manually.

This encoding is needed for feeding categorical data to many scikit-learn
estimators, notably linear models and SVMs with the standard kernels.

Note: a one-hot encoding of y labels should use a LabelBinarizer
instead.

Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

Parameters
----------
categories : 'auto' or a list of array-like, default='auto'
    Categories (unique values) per feature:

    - 'auto' : Determine categories automatically from the training data.
    - list : ``categories[i]`` holds the categories expected in the ith
      column. The passed categories should not mix strings and numeric
      values within a single feature, and should be sorted in case of
      numeric values.

    The used categories can be found in the ``categories_`` attribute.

    .. versionadded:: 0.20

drop : {'first', 'if_binary'} or an array-like of shape (n_features,),             default=None
    Specifies a methodology to use to drop one of the categories per
    feature. This is useful in situations where perfectly collinear
    features cause problems, such as when feeding the resulting data
    into an unregularized linear regression model.

    However, dropping one category breaks the symmetry of the original
    representation and can therefore induce a bias in downstream models,
    for instance for penalized linear classification or regression models.

    - None : retain all features (the default).
    - 'first' : drop the first category in each feature. If only one
      category is present, the feature will be dropped entirely.
    - 'if_binary' : drop the first category in each feature with two
      categories. Features with 1 or more than 2 categories are
      left intact.
    - array : ``drop[i]`` is the category in feature ``X[:, i]`` that
      should be dropped.

    .. versionadded:: 0.21
       The parameter `drop` was added in 0.21.

    .. versionchanged:: 0.23
       The option `drop='if_binary'` was added in 0.23.

    .. versionchanged:: 1.1
        Support for dropping infrequent categories.

sparse : bool, default=True
    Will return sparse matrix if set True else will return an array.

dtype : number type, default=float
    Desired dtype of output.

handle_unknown : {'error', 'ignore', 'infrequent_if_exist'},                      default='error'
    Specifies the way unknown categories are handled during :meth:`transform`.

    - 'error' : Raise an error if an unknown category is present during transform.
    - 'ignore' : When an unknown category is encountered during
      transform, the resulting one-hot encoded columns for this feature
      will be all zeros. In the inverse transform, an unknown category
      will be denoted as None.
    - 'infrequent_if_exist' : When an unknown category is encountered
      during transform, the resulting one-hot encoded columns for this
      feature will map to the infrequent category if it exists. The
      infrequent category will be mapped to the last position in the
      encoding. During inverse transform, an unknown category will be
      mapped to the category denoted `'infrequent'` if it exists. If the
      `'infrequent'` category does not exist, then :meth:`transform` and
      :meth:`inverse_transform` will handle an unknown category as with
      `handle_unknown='ignore'`. Infrequent categories exist based on
      `min_frequency` and `max_categories`. Read more in the
      :ref:`User Guide <one_hot_encoder_infrequent_categories>`.

    .. versionchanged:: 1.1
        `'infrequent_if_exist'` was added to automatically handle unknown
        categories and infrequent categories.

min_frequency : int or float, default=None
    Specifies the minimum frequency below which a category will be
    considered infrequent.

    - If `int`, categories with a smaller cardinality will be considered
      infrequent.

    - If `float`, categories with a smaller cardinality than
      `min_frequency * n_samples`  will be considered infrequent.

    .. versionadded:: 1.1
        Read more in the :ref:`User Guide <one_hot_encoder_infrequent_categories>`.

max_categories : int, default=None
    Specifies an upper limit to the number of output features for each input
    feature when considering infrequent categories. If there are infrequent
    categories, `max_categories` includes the category representing the
    infrequent categories along with the frequent categories. If `None`,
    there is no limit to the number of output features.

    .. versionadded:: 1.1
        Read more in the :ref:`User Guide <one_hot_encoder_infrequent_categories>`.

Attributes
----------
categories_ : list of arrays
    The categories of each feature determined during fitting
    (in order of the features in X and corresponding with the output
    of ``transform``). This includes the category specified in ``drop``
    (if any).

drop_idx_ : array of shape (n_features,)
    - ``drop_idx_[i]`` is the index in ``categories_[i]`` of the category
      to be dropped for each feature.
    - ``drop_idx_[i] = None`` if no category is to be dropped from the
      feature with index ``i``, e.g. when `drop='if_binary'` and the
      feature isn't binary.
    - ``drop_idx_ = None`` if all the transformed features will be
      retained.

    If infrequent categories are enabled by setting `min_frequency` or
    `max_categories` to a non-default value and `drop_idx[i]` corresponds
    to a infrequent category, then the entire infrequent category is
    dropped.

    .. versionchanged:: 0.23
       Added the possibility to contain `None` values.

infrequent_categories_ : list of ndarray
    Defined only if infrequent categories are enabled by setting
    `min_frequency` or `max_categories` to a non-default value.
    `infrequent_categories_[i]` are the infrequent categories for feature
    `i`. If the feature `i` has no infrequent categories
    `infrequent_categories_[i]` is None.

    .. versionadded:: 1.1

n_features_in_ : int
    Number of features seen during :term:`fit`.

    .. versionadded:: 1.0

feature_names_in_ : ndarray of shape (`n_features_in_`,)
    Names of features seen during :term:`fit`. Defined only when `X`
    has feature names that are all strings.

    .. versionadded:: 1.0

See Also
--------
OrdinalEncoder : Performs an ordinal (integer)
  encoding of the categorical features.
sklearn.feature_extraction.DictVectorizer : Performs a one-hot encoding of
  dictionary items (also handles string-valued features).
sklearn.feature_extraction.FeatureHasher : Performs an approximate one-hot
  encoding of dictionary items or strings.
LabelBinarizer : Binarizes labels in a one-vs-all
  fashion.
MultiLabelBinarizer : Transforms between iterable of
  iterables and a multilabel format, e.g. a (samples x classes) binary
  matrix indicating the presence of a class label.

Examples
--------
Given a dataset with two features, we let the encoder find the unique
values per feature and transform the data to a binary one-hot encoding.

>>> from sklearn.preprocessing import OneHotEncoder

One can discard categories not seen during `fit`:

>>> enc = OneHotEncoder(handle_unknown='ignore')
>>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
>>> enc.fit(X)
OneHotEncoder(handle_unknown='ignore')
>>> enc.categories_
[array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
>>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
array([[1., 0., 1., 0., 0.],
       [0., 1., 0., 0., 0.]])
>>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
array([['Male', 1],
       [None, 2]], dtype=object)
>>> enc.get_feature_names_out(['gender', 'group'])
array(['gender_Female', 'gender_Male', 'group_1', 'group_2', 'group_3'], ...)

One can always drop the first column for each feature:

>>> drop_enc = OneHotEncoder(drop='first').fit(X)
>>> drop_enc.categories_
[array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
>>> drop_enc.transform([['Female', 1], ['Male', 2]]).toarray()
array([[0., 0., 0.],
       [1., 1., 0.]])

Or drop a column for feature only having 2 categories:

>>> drop_binary_enc = OneHotEncoder(drop='if_binary').fit(X)
>>> drop_binary_enc.transform([['Female', 1], ['Male', 2]]).toarray()
array([[0., 1., 0., 0.],
       [1., 0., 1., 0.]])

Infrequent categories are enabled by setting `max_categories` or `min_frequency`.

>>> import numpy as np
>>> X = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object).T
>>> ohe = OneHotEncoder(max_categories=3, sparse=False).fit(X)
>>> ohe.infrequent_categories_
[array(['a', 'd'], dtype=object)]
>>> ohe.transform([["a"], ["b"]])
array([[0., 0., 1.],
       [1., 0., 0.]])
File:           c:\users\semil\appdata\local\programs\python\python310\lib\site-packages\sklearn\preprocessing\_encoders.py
Type:           type
Subclasses:     
"""