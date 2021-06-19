# Estimators

## Overview
1. [_Object-Oriented Programming (OOP)_](./object-oriented-programming.md)
2. [_Inheritance (OOP)_](./inheritance.md)
3. **Estimators**
4. Transformers
5. Writing Custom Estimators & Transformers
6. Pipeline
7. Common Scikit-learn modules

**Prerequisite:**
- Basic understanding of `numpy`

---

`Scikit-learn` is an open source machine learning library that supports supervised and unsupervised learning.  
It also provides various tools for model fitting, data preprocessing, model selection and evaluation, and many other utilities.

In this tutorial, we will be using a very popular _classification dataset_ used in machine-learning: the iris dataset.  
`Scikit-learn` provides a `load_iris` _function_ to retrieve this _dataset_ from the `sklearn.datasets` module.
```python
>>> from sklearn.datasets import load_iris
>>> X, y = load_iris(return_X_y=True)
>>> X.shape
(150, 4)
>>> y.shape
(150,)
```

According to the `Scikit-learn` documentation, it provides _dozens of built-in machine learning algorithms and models_.  
These models (aka `estimators`) are implemented as _classes_ using the _OOP_ paradigm, and provide common _methods_ for processing data.

The [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
from the `sklearn.ensemble` module is one such `estimator` for _classification_ problems.
```python
>>> from sklearn.ensemble import RandomForestClassifier
```

We can instantiate the `RandomForestClassifier`.

Each _estimator_ receives different arguments during instantiation depending on the algorithm,
so it is handy to have access to the online documentation for these models.

Fortunately, most _estimator classes_ in `Scikit-learn` provide sensible default arguments,
so we can start using the models without worrying too much about the arguments to pass in.
```python
>>> # Instantiate RandomForestClassifier estimator
>>> estimator = RandomForestClassifier(random_state=0)
```

> **Note:**  
Some `Scikit-learn` _estimators_ accept an optional `random_state` argument during instantiation.
It is recommended to set this argument to a constant `int` throughout your program.
This is to ensure a consistent result when you run your program multiple times.

Every _estimator_ in `Scikit-learn` implements a [`fit`](https://scikit-learn.org/stable/glossary.html#term-fit) method that accepts training data for learning.
```python
>>> # X is the features of your training set
>>> # y is the label for your training set
>>> clf.fit(X, y)
```

Since it is best practice to set aside some of the data for evaluating the ability of a model, we will split the data (`X, y`) to training and test sets.

`Scikit-learn` provides utilities for working with your data.
The one we are going to use to split our data is the [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) _function_ in the `sklearn.model_selection` module.
```python
>>> from sklearn.model_selection import train_test_split
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
>>> X_train.shape, X_test.shape
((112, 4), (38, 4))
>>> y_train.shape, y_test.shape
((112,), (38,))
```

Now we can call the RandomForestClassifier.fit method on X_train and y_train.
```python
>>> clf = RandomForestClassifier(random_state=0)
>>> clf.fit(X_train, y_train)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)
```

After training the _classifier_ on the training set, we can now make predictions with the `predict` method.  
As an example, let's make a prediction on the first 5 elements of the test set, and compare with the actual results.
```python
>>> y_pred = clf.predict(X_test[0:5])
>>> y_pred # predictions
array([1, 2, 1, 0, 1])
>>> y_test[0:5] # true result
array([1, 2, 1, 0, 1])
```

> **Note:**  
`Scikit-learn` uses `numpy` _arrays_ in the background for working with data.
Therefore it is advised to be familiar with the basic of `numpy` before starting out with `Scikit-learn`


Our `RandomForestClassifier` _estimator_ predicted the first 5 elements correctly.

Considering that we are not interested in the _predictions_ themselves, we just want to know how well our model performed.  
The `RandomForestClassifier` provides a `score` _method_, to determine how accurate our model is on a test set.
```python
>>> clf.score(X_test, y_test)
0.9736842105263158
```

It appears that our model predicted `97%` of our test set correctly.

I would encourage you to take a look at the [`Scikit-learn`](https://scikit-learn.org/stable/) documentation to get familiar with several models and functions the package provides.

In the next tutorial, we will be taking a look at `Transformers` and how they are used for preprocessing data before training.

---
| [Prev - Inheritance (OOP)](./inheritance.md) | [Next - Transformers](./transformers.md) |
|:---------------------------------------------|-----------------------------------------:|
