# Custom Estimators

## Overview
1. [_Object-Oriented Programming (OOP)_](./object-oriented-programming.md)
2. [_Inheritance (OOP)_](./inheritance.md)
3. [_Estimators_](./estimators.md)
4. [_Transformers_](./transformers)
5. **Custom Estimators**
6. Pipeline
7. Common Scikit-learn modules

---

In the last two parts of this series we took a look at some of the _estimators_ that `Scikit-learn` provides out of the box.

> **Note:** Transformers are estimators as well.

However, `Scikit-learn` allows us to define our custom _estimators_ by inheriting from some base classes in the `sklearn.base` module.

One thing to keep in mind is that every _estimator_, regardless of its purpose and functionality inherits from the `sklearn.base.BaseEstimator` class.

Creating an _estimator_ is as easy as:
```python
>>> from sklearn.base import BaseEstimator
>>> class MyEstimator(BaseEstimator):
...     pass
```

Congratulations, you have just created your own custom _estimator_. However, at this stage it does nothing.

In fact, the only methods defined in `MyEstimator` are:
- `__getstate__(self)`
- `__setstate__(self, state)`
- `get_params(self, deep=True)`
- `set_params(self, **params)`

These are inherited from `sklearn.base.BaseEstimator` and are used internally by `Scikit-learn`, so we rarely interact directly with these methods.

In order to create a useful _estimator_, we must first recognize its purpose to identify the category it fits in.  
Some categories include:
- Transformation
- Regression
- Classification
- Clustering

The `sklearn.base` module exposes other base classes in addition to `BaseEstimator` for specific types of _estimators_.
The `TransformerMixin` class for _transformers_, `RegressorMixin` for _regressors_, `ClassifierMixin` for _classifiers_, e.t.c

For example, `sklearn.base.TransformerMixin` can be inherited in tandem with `sklearn.base.BaseEstimator` to define a custom `Transformer` _estimator_. This is how all built-in `Scikit-learn` _estimators_ operate.

```python
>>> from sklearn.base import BaseEstimator, TransformerMixin
>>> class MyTransformer(BaseEstimator, TransformerMixin):
...     pass
```

> **Note:**  
> The `sklearn.base.BaseEstimator` must be inherited as well.  
> This is how `Scikit-learn` knows that a class is indeed an _estimator_.


Each `*Mixin` class requires a different set of methods to be defined in their subclasses to function appropriately.

As an example, the `TransformerMixin` class requires its subclasses to define the following methods:
- `.fit(self, X, y=None)`
- `.transform(self, X, y=None)`

`TransfomerMixin` subclasses inherit the `.fit_transform(self, X, y=None)`, which executes the `.fit` and `.transform` methods in sequence.

```python
>>> import numpy as np
>>> import pandas as pd
>>> from sklearn.datasets import load_iris
>>> 
>>> seed = 0
>>> np.random.seed(seed)
>>> 
>>> X, y = load_iris(return_X_y=True)
>>> 
>>> 
>>> class MyTransformer(BaseEstimator, TransformerMixin):
...     def fit(self, X, y=None):
...         '''
...         1. learns from the data (fits the data)
...         2. returns self
...         '''
...         return self
...     def transform(self, X, y=None):
...         '''
...         return a transformation of the input X data.
...         '''
...         return X
>>> 
>>> my_transformer = MyTransformer()
>>> X_transformed = my_transformer.fit_transform(X)
```

Other `*Mixin` classes include:
- `RegressorMixin` Regression estimators
- `ClassifierMixin` Classification estimators
- `ClusterMixin` Clustering estimators
- `OutlierMixin` Outlier detection estimators

### MyLabelEncoder

In the last tutorial, we used `sklearn.preprocessing.LabelEncoder` to encode our iris `target` column to numeric values.

Let's implement our own custom LabelEncoder.

But first, we'll load the data.

```python
>>> import numpy as np
>>> import pandas as pd
>>> from sklearn.datasets import load_iris
>>> 
>>> seed = 0
>>> np.random.seed(seed)
>>> 
>>> iris = load_iris()
>>> 
>>> labels = iris['target_names'][iris['target']] 
>>> columns = iris['feature_names'] + ['target']
>>> values = np.c_[iris['data'], labels]
>>> df = pd.DataFrame(values, columns=columns)
>>> df.sample(5, random_state=seed)
```
|     |   sepal length (cm) |   sepal width (cm) |   petal length (cm) |   petal width (cm) | target     |
|----:|--------------------:|-------------------:|--------------------:|-------------------:|:-----------|
| 114 |                 5.8 |                2.8 |                 5.1 |                2.4 | virginica  |
|  62 |                 6   |                2.2 |                 4   |                1   | versicolor |
|  33 |                 5.5 |                4.2 |                 1.4 |                0.2 | setosa     |
| 107 |                 7.3 |                2.9 |                 6.3 |                1.8 | virginica  |
|   7 |                 5   |                3.4 |                 1.5 |                0.2 | setosa     |

Our first step now is to identify the type of _estimator_ we are creating, and use the appropriate `*Mixin` class.

**Hint**: Transformer

```python
>>> from sklearn.base import BaseEstimator, TransformerMixin
>>> class MyLabelEncoder(BaseEstimator, TransformerMixin):
...     pass
```

Next, we define the required methods for `TransformerMixin`.

`TransformerMixin` requires both `.fit(self, X, y=None)` and `.transform(self, X, y=None)`, where `.fit` returns the current _estimator_ object, and `.transform` returns the transformed data.

```python
>>> class MyLabelEncoder(BaseEstimator, TransformerMixin):
...     def fit(self, X, y=None):
...         ''' Not yet implemented '''
...         return self
... 
...     def transform(self, X, y=None):
...         ''' Not yet implemented '''
...         pass
```

In the _transformation_ phase, we want to swap the target names with their corresponding integer pair:

| target names | integer |
|:------------:|:-----------:|
|setosa|0|
|versicolor|1|
|virginica|2|

However, we need to compute the corresponding integer pairs in the `.fit` method when we first receive the training data. The first flower name will be encoded as 0, the second as 1, and third as 2.

```python
>>> class MyLabelEncoder(BaseEstimator, TransformerMixin):
...     def fit(self, X, y=None):
...         # Get the unique values from array X
...         unique_values = np.unique(X) # ['setosa', 'versicolor', 'virginica']
... 
...         # create a dictionary that maps unique_values to integers
...         # mapping = { 'setosa': 0, 'versicolor': 1, 'virginica': 2 }
...         mapping = dict()
...         for integer, value in enumerate(unique_values):
...             mapping[value] = integer
... 
...         # save the mapping on the current object to be used in the .transform method
...         self.mapping = mapping
... 
...         # Scikit-learn expects the .fit method to return the current object
...         return self
... 
...     def transform(self, X, y=None):
...         mapping = self.mapping
...         
...         # Swap each occurrence of the unique values with their integer pairs
...         transformed_X = []
...         for iris_name in X:
...             integer = mapping[iris_name] # get the flower's corresponding integer
...             transformed_X.append(integer)
...         
...         # return the transformed data as a numpy array
...         return np.array(transformed_X)
```

Let's compare our `MyLabelEncoder` with `sklearn.preprocessing.LabelEncoder`.

```python
>>> from sklearn.preprocessing import LabelEncoder
>>> label_encoder = LabelEncoder()
>>> my_label_encoder = MyLabelEncoder()
>>> values = df['target'].values
>>> 
>>> sklearn_encoding = label_encoder.fit_transform(df['target'])
>>> custom_encoding = my_label_encoder.fit_transform(df['target']) # .fit_transform inherited from TransformerMixin
>>> 
>>> np.all(sklearn_encoding == custom_encoding) # all elements in both encoded arrays are equal
True
```

Our custom `MyLabelEncoder` produces the same result as `sklearn.preprocessing.LabelEncoder`.

> **Note:**  
> The `.fit` method is for extracting information from the data passed in (usually the training set) on how to transform subsequent data. Therefore, it should only be used once, and on the training set only.
> 
> Similarly, the `.fit_transform` method should be used only once on the training set because it calls `.fit` internally. For, every subsequent attempt to transform the data, use `.transform`.


### MyRandomClassifier
As a bonus, let's create a custom classification _estimator_ that **randomly** classifies instances in a dataset.

```python
>>> # 1. Identify estimator type
>>> # classification estimators inherit from ClassifierMixin
>>> from sklearn.base import BaseEstimator, ClassifierMixin
>>>
>>> # 2. Defined required methods
>>> # ClassifierMixin require .fit and .predict from its subclasses
>>> class MyRandomClassifier(BaseEstimator, ClassifierMixin):
...     def fit(self, X, y=None):
...         ''' Not yet implemented '''
...         return self
...     def predict(self, X):
...         ''' Not yet implemented '''
...         pass
```

The `ClassifierMixin` requires a `.predict` method defined in its `Subclasses`, and provides us with a `.score` method.

Since we are **randomly** classifying instances of the data, we might want to allow users of our `MyRandomClassifier` _estimator_ to provide a `random_state` value for reproducing the results of the model.

The `random_state` is not a parameter that the _estimator_ learns, rather it is explicitly provided by users of our _estimator_.

Parameters that are not computed during the `.fit` method (learned from the data) are called _hyperparameters_ and are accepted in the `__init__(...)` method.

```python
>>> class MyRandomClassifier(BaseEstimator, ClassifierMixin):
...    def __init__(self, random_state=None):
...        self.random_state = random_state
... 
...    def fit(self, X, y):
...        ''' Extract labels from the training set, in the `y` parameter.
...        '''
...        self.labels = np.unique(y)
...        return self # 👈 required
... 
...    def predict(self, X):
...        ''' Randomly classifies the rows in the data.
...        In order to reproduce the random results via random_state,
...        we use a np.random.Generator object which implements the .choice method similar to np.random.choice
...        '''
...        generator = np.random.default_rng(self.random_state) # gets np.random.Generator object
...        labels = self.labels
...        predictions = generator.choice(labels, size=len(X)) # generate random predictions
...        return predictions
>>> 
>>> clf = MyRandomClassifier(random_state=seed)
>>> clf.fit(X, y)
MyRandomClassifier(random_state=0)
>>> clf.predict(X)
array([2, 1, 1, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 0, 2,
       2, 0, 1, 2, 1, 0, 2, 2, 2, 0, 0, 2, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0,
       0, 2, 1, 1, 0, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 0,
       1, 2, 2, 1, 1, 0, 1, 1, 2, 2, 0, 2, 1, 1, 2, 1, 0, 0, 2, 1, 1, 1,
       2, 1, 0, 2, 0, 0, 2, 1, 0, 0, 1, 2, 1, 2, 0, 0, 2, 2, 0, 0, 2, 1,
       1, 0, 2, 1, 2, 2, 2, 0, 2, 0, 1, 1, 2, 0, 2, 0, 1, 1, 2, 0, 2, 2,
       2, 0, 2, 2, 0, 1, 1, 0, 1, 1, 2, 2, 1, 1, 1, 2, 0, 1])
>>> # Accuracy Score
>>> clf.score(X, y) # implemented by ClassifierMixin
0.38666666666666666
```

### Conclusion
> **Further Reading:**  
> - [Random Generator](https://numpy.org/doc/stable/reference/random/generator.html?highlight=generator#random-generator)
> - [NumPy Random Seed, Explained](https://www.sharpsightlabs.com/blog/numpy-random-seed/)

Understanding how `Scikit-learn` _estimators_ work under the hood will help you write cleaner code that interacts nicely with the `Scikit-learn` API. Especially _pipelines_, which we will be taking a look at next.

---
| [Prev - Transformers](./transformers.md) | Next |
|:-------------------------------------|--------------------:|
