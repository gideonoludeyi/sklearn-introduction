# Transformers

## Overview
1. [_Object-Oriented Programming (OOP)_](./object-oriented-programming.md)
2. [_Inheritance (OOP)_](./inheritance.md)
3. [_Estimators_](./estimators.md)
4. **Transformers**
5. [Custom Estimators](./custom_estimators.md)
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

```python
>>> import numpy as np
>>> import pandas as pd
>>> from sklearn.datasets import load_iris
>>>
>>> def make_dirty(iris_data):
...     """ a utility function to dirty up the iris dataset
...     to make it unpresentable to most Scikit-learn estimators.
...     Changes:
...     1. Transformed iris target values into their respective text names
...         ('setosa', 'versicolor', 'virginica')
...     2. Scaled dimensions of 'petal length' and 'petal width'
...     """
... 
...     features = iris_data['data']
...     target = iris_data['target']
...     columns = iris_data['feature_names'] + ['target']
... 
...     df = pd.DataFrame(np.c_[features, target], columns=columns)
...     
...     # change target from number to text
...     df['target'] = iris_data['target_names'][target]
... 
...     # make petal features with dimensions in meters
...     df[['petal length', 'petal width']] = df[['petal length (cm)', 'petal width (cm)']] * .01
... 
...     return df[['sepal length (cm)', 'sepal width (cm)', 'petal length', 'petal width', 'target']]
>>> 
>>> iris = load_iris()
>>> df = make_dirty(iris)
>>> df.sample(5, random_state=0)
```
|     |   sepal length (cm) |   sepal width (cm) |   petal length |   petal width | target     |
|----:|--------------------:|-------------------:|---------------:|--------------:|:-----------|
| 114 |                 5.8 |                2.8 |          0.051 |         0.024 | virginica  |
|  62 |                 6   |                2.2 |          0.04  |         0.01  | versicolor |
|  33 |                 5.5 |                4.2 |          0.014 |         0.002 | setosa     |
| 107 |                 7.3 |                2.9 |          0.063 |         0.018 | virginica  |
|   7 |                 5   |                3.4 |          0.015 |         0.002 | setosa     |

The `target` column is an issue for most `Scikit-learn` _estimators_ because its datatype is not numeric. \
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

You may recall that the `.fit` method was present in the `RandomForestClassifier`. \
Transformers are essentially _estimators_. \
We established last time that _estimators_ train on provided data to make predictions. \
Well, _transformers_ also train on provided data, but rather than output a prediction, they return a transformation of the input data. \
In fact, making predictions is the same concept: you receive input data, apply one or more transformations on the data, and return the result.

```python
>>> target = df['target']
>>> # .fit phase
>>> label_encoder.fit(target)
>>> # .transform phase
>>> result = label_encoder.transform(target)
>>> df['target'] = result
>>> df.sample(5, random_state=0)
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
|   setosa     |      0      |
|  versicolor  |      1      |
|  virginica   |      2      |

During the `.fit` phase, the `label_encoder` extracted the unique values within the `target` array (`setosa`, `versicolor`, and `virginica`)
and stored them internally along with their corresponding mapping. Since `setosa` appeared first, it was paired with integer `0`, `versicolor` with `1` and `virginica` with `2`.

In the `.transform` phase, the `label_encoder` returned a new array with each occurrence of `setosa`, `versicolor`, and `virginica` swapped with their paired integer.

Another, more subtle issue, with this data is the inconsistent scale of the features.

|       |   sepal length (cm) |   sepal width (cm) |   petal length |   petal width |
|:------|--------------------:|-------------------:|---------------:|--------------:|
| mean  |            5.84333  |           3.05733  |       0.03758  |    0.0119933  |
| std   |            0.828066 |           0.435866 |       0.017653 |    0.00762238 |

The `mean` and `std` of the `sepal` and `petal` features vary by a great deal.

Depending on the algorithm you're working with, the difference in scale between features can negatively affect the performance of the model.

The example below trains a _Support Vector Machine Classifier_ on the current dataset and evaluates it accuracy.


```python
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.svm import SVC # Support Vector Classifier
>>> X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length', 'petal width']]
>>> y = df['target']
>>> X.shape
(150, 4)
>>> y.shape
(150,)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
>>>
>>> clf = SVC(random_state=0)
>>> clf.fit(X_train, y_train)
SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=0, shrinking=True, tol=0.001,
    verbose=False)
>>> clf.score(X_test, y_test)
0.7631578947368421
```

Apply scaling on each feature in the dataset using the `StandardScaler` transformer from the `sklearn.preprocessing` module before fitting the model yields a better accuracy.

```python
>>> from sklearn.preprocessing import StandardScaler
>>> scaler = StandardScaler()
>>> scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
>>> X_train_transformed = scaler.transform(X_train)
>>> # X_train_transformed = scaler.fit_transform(X_train) # composes the previous two steps
>>>
>>> clf = SVC(random_state=0)
>>> clf.fit(X_train_transformed, y_train)
SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=0, shrinking=True, tol=0.001,
    verbose=False)
>>> X_test_transformed = scaler.transform(X_test)
>>> clf.score(X_test_transformed, y_test)
0.9736842105263158
```
As you can see, the accuracy went up from `76%` all the way to `97%`. \
The boost in performance is as a result of a similar scale between all features.

|      |   sepal length (cm) |   sepal width (cm) |   petal length |   petal width |
|:-----|--------------------:|-------------------:|---------------:|--------------:|
| mean |          -0.0498882 |          0.0127753 |     -0.0214369 |    -0.0306981 |
| std  |           0.954636  |          1.00373   |      0.984748  |     0.979828  |

> **Note:** \
_Support Vector Machine_ is used in this tutorial to illustrate the significance of scaling for certain machine-learning algorithms, because _Random Forest_ models are not affected by scale.

Hopefully, this example sheds light on the effectiveness of _transformers_ for manipulating your data. \
For further reading, you can refer to the `Scikit-learn` documentation on [Dataset transformations](https://scikit-learn.org/stable/data_transforms.html)

In the next tutorial we will be creating our own estimators and transformers using what we have learned so far in this series.

---

| [Prev - Estimators](./estimators.md) | [Next - Custom Estimators](./custom_estimators.md) |
|:-------------------------------------|--------------------:|
