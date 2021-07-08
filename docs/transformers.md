# Transformers

## Overview
1. [_Object-Oriented Programming (OOP)_](./object-oriented-programming.md)
2. [_Inheritance (OOP)_](./inheritance.md)
3. [_Estimators_](./estimators.md)
4. **Transformers**
5. Writing Custom Estimators & Transformers
6. Pipeline
7. Common Scikit-learn modules

**Prerequisite:**
- Basic understanding of `numpy`

---

Last time we were introduced to `Scikit-learn` _estimators_. \
These are classes that implement machine-learning algorithms that are trained on our data
using the `.fit` method. \
We also saw how to make predictions using the `.predict` method.

We trained the `RandomForestClassifier` _estimator_ on the `iris` dataset for classifying the type of flower, and we got a `97%` accuracy.

```python
>>> from sklearn.datasets import load_iris
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.model_selection import train_test_split
>>>
>>> # Load data
>>> X, y = load_iris(return_X_y=True)
>>> X.shape # 150 flowers, 4 features
(150, 4)
>>> y.shape
(150,)
>>>
>>> # Instantiate model
>>> clf = RandomForestClassifier(random_state=0)
>>>
>>> # Train model
>>> clf.fit(X_train, y_train)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)
>>>
>>> # Evaluate model accuracy
>>> clf.score(X_test, y_test)
0.9736842105263158
```

In the example above, the `iris` dataset was already prepared to be trained on using the `RandomForestClassifier.fit` method.

However, most data in the real-world are not so clean. \
In fact, datasets may contain missing information, outliers, and other issues.

On the other hand, most _estimators_ in `Scikit-learn` expect to receive data in a particular format to perform any kind of procedure. \
Almost, if not all, _estimators_ in `Scikit-learn` expect the data to be numeric, and they cannot function with data that contains missing values.

Take a look at the following dataset:

|     |   sepal length (cm) |   sepal width (cm) |   petal length |   petal width | target     |
|----:|--------------------:|-------------------:|---------------:|--------------:|:-----------|
| 114 |                 5.8 |                2.8 |          0.051 |         0.024 | virginica  |
|  62 |                 6   |                2.2 |          0.04  |         0.01  | versicolor |
|  33 |                 5.5 |                4.2 |          0.014 |         0.002 | setosa     |
| 107 |                 7.3 |                2.9 |          0.063 |         0.018 | virginica  |
|   7 |                 5   |                3.4 |          0.015 |         0.002 | setosa     |

The `target` column here would be an issue for most `Scikit-learn` _estimators_ because of its datatype is not numeric. \
Luckily, `Scikit-learn` provides a some classes that implement certain procedures to _transform_ your data into a more compatible format for your model,
one of which is the `LabelEncoder` class from the `sklearn.preprocessing` module.

```python
>>> from sklearn.preprocessing import LabelEncoder
>>> label_encoder = LabelEncoder()
```

All _transformers_ in `Scikit-learn` implement the following methods:
- `.fit`  
extracts essential information from the provided data for transforming the subsequent data.

- `.transform`  
returns a transformation of the data.

You may recall that the `.fit` method was present in the `RandomForestClassifier`. Transformers are essentially _estimators_. We established last time that _estimators_ train on provided data to make predictions, _transformers_ also train on provided data, but rather than returning a prediction, they return a transformation of the input data. In fact, making predictions is essentially the same concept: you receive input data and return output data that is based on a transformation of the input data.

```python
>>> target = df['target']
>>> label_encoder.fit(target)
>>> result = label_encoder.transform(target)
>>> df['target'] = result
>>> df
|     |   sepal length (cm) |   sepal width (cm) |   petal length |   petal width |   target |
|----:|--------------------:|-------------------:|---------------:|--------------:|---------:|
| 114 |                 5.8 |                2.8 |          0.051 |         0.024 |        2 |
|  62 |                 6   |                2.2 |          0.04  |         0.01  |        1 |
|  33 |                 5.5 |                4.2 |          0.014 |         0.002 |        0 |
| 107 |                 7.3 |                2.9 |          0.063 |         0.018 |        2 |
|   7 |                 5   |                3.4 |          0.015 |         0.002 |        0 |
```

|     |   sepal length (cm) |   sepal width (cm) |   petal length |   petal width |   target |
|----:|--------------------:|-------------------:|---------------:|--------------:|---------:|
| 114 |                 5.8 |                2.8 |          0.051 |         0.024 |        2 |
|  62 |                 6   |                2.2 |          0.04  |         0.01  |        1 |
|  33 |                 5.5 |                4.2 |          0.014 |         0.002 |        0 |
| 107 |                 7.3 |                2.9 |          0.063 |         0.018 |        2 |
|   7 |                 5   |                3.4 |          0.015 |         0.002 |        0 |

> **Note:**  
`LabelEncoder` should only be used for the column being predicted, and not on feature columns.  
If you are looking to encode feature columns, consider other forms of encoding classes such as `sklearn.preprocessing.OneHotEncoder` or `sklearn.preprocessing.OrdinalEncoder`.  
Alternatively, consider the `category_encoders` [package](https://contrib.scikit-learn.org/category_encoders/)

**What Happened?** \
In simplest terms, all occurrences of `setosa`, `versicolor`, and `virginica` were replaced with `0`, `1`, and `2` respectively.

| target names | encoded int |
|:------------:|:-----------:|
|setosa|0|
|versicolor|1|
|virginica|2|

During the `.fit` phase, the `label_encoder` extracted the unique values within the `target` array (`setosa`, `versicolor`, and `virginica`)
and stored them internally along with their corresponding mapping. Since `setosa` appeared first, it was paired with integer `0`, `versicolor` with `1` and `virginica` with `2`.

In the `.transform` phase, the `label_encoder` returned a new array with each occurrence of `setosa`, `versicolor`, and `virginica` swapped with their paired integer.

---
| [Prev - Estimators](./estimators.md) | [Next](./custom_transformer_classes.md) |
|:-------------------------------------|--------------------:|
