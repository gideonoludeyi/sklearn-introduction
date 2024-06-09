# Custom Estimators

## Overview
1. [_Object-Oriented Programming (OOP)_](./object-oriented-programming.md)
2. [_Inheritance (OOP)_](./inheritance.md)
3. [_Estimators_](./estimators.md)
4. [_Transformers_](./transformers.md)
5. [_Custom Estimators_](./custom_estimators.md)
6. **Pipeline**
7. Common Scikit-learn modules

---

In the [Custom Estimators](./custom_estimators.md) article, we walked through defining our own estimators for processing data or generating predictions. We made a `MyLabelEncoder` transformer that encodes target labels as unique integers, similar to scikit-learn's `LabelEncoder` transformer. We also implemented our own `MyRandomClassifier` classifier that predicts a target label at random. Scikit-learn allows us to create other types of estimators for regression, outlier detection, clustering, and more.

By itself, a classifier or regressor is usually not enough to solve machine learning tasks. Real-world data usually require several steps to prepare the data before a classification or regression model can be trained or used to make predictions. Scikit-learn's pipeline system provides a convenient way to compose all these steps into a single end-to-end model for solving machine learning tasks.

We will attempt to build a classification model for the iris dataset to show the usefulness of the pipeline system. The training and test data can be obtained via the following:
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

seed = 0
np.random.seed(seed)

iris = load_iris()

labels = iris['target_names'][iris['target']]
columns = iris['feature_names'] + ['target']
values = np.c_[iris['data'], labels]
df = pd.DataFrame(values, columns=columns)
train_df, test_df = train_test_split(df, random_state=seed)
print(train_df.sample(5, random_state=seed))
```

|     | sepal length (cm) | sepal width (cm) | petal length (cm) | petal width (cm) | target     |
| --: | ----------------: | ---------------: | ----------------: | ---------------: | :--------- |
|  64 |               5.6 |              2.9 |               3.6 |              1.3 | versicolor |
| 128 |               6.4 |              2.8 |               5.6 |              2.1 | virginica  |
| 119 |               6.0 |              2.2 |               5.0 |              1.5 | virginica  |
| 105 |               7.6 |              3.0 |               6.6 |              2.1 | virginica  |
|  25 |               5.0 |              3.0 |               1.6 |              0.2 | setosa     |

Without the pipeline system, the code to train and test the model might look like the following:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# transformers
my_label_encoder = MyLabelEncoder() # our custom label encoder
standard_scaler = StandardScaler() # a feature transformer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean') # another feature transformer

clf = MyRandomClassifier(random_state=seed) # our custom classifier

X_train = train_df.drop('target', axis='columns')
y_train = train_df['target']

y_train = my_label_encoder.fit_transform(y_train)
X_train = standard_scaler.fit_transform(X_train)
X_train = imp_mean.fit_transform(X_train)
clf.fit(X_train, y_train)

# repeat for test data
X_test = test_df.drop('target', axis='columns')
y_test = test_df['target']

y_test = my_label_encoder.transform(y_test)
X_test = standard_scaler.transform(X_test)
X_test = imp_mean.transform(X_test)

score = clf.score(X_test, y_test)

print(score)
```

A pattern starts to emerge in the code where the output of each transformer becomes the input for the next transformer. Instead of doing this manually, we can generalize the implementation to an arbitrary number of transformations before reaching the classifier:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

my_label_encoder = MyLabelEncoder() # our custom label encoder
feature_transformers = [ # define all transformations as a sequence
	StandardScaler(),
	SimpleImputer(missing_values=np.nan, strategy='mean')
]
clf = MyRandomClassifier(random_state=seed) # our custom classifier

X_train = train_df.drop('target', axis='columns')
y_train = train_df['target']

y_train = my_label_encoder.fit_transform(y_train)

for transformer in feature_transformers:
	X_train = transformer.fit_transform(X_train)
clf.fit(X_train, y_train)

# repeat for test data
X_test = test_df.drop('target', axis='columns')
y_test = test_df['target']

y_test = my_label_encoder.transform(y_test)

for transformer in feature_transformers:
	X_test = transformer.transform(X_test)
score = clf.score(X_test, y_test)

print(score)

# ...
# repeat for predictions at a later time
X = ...
for transformer in feature_transformers:
	X = transformer.transform(X)
predictions = clf.predict(X)

```

What our new implementation does is define a sequence of transformations to apply on the features `X` before it reaches the classifier. This makes it easier to easily add or remove transformations by updating the `feature_transformers` list. For example, we could add a `SelectKBest(k=3)` transformer at the beginning of the `feature_transformers` to use only the top 3 features that are useful in predicting the target. In other words, we can incorporate feature selection into our model with a single change in the code. This ability to easily compose transformations is effectively what scikit-learn's [`Pipeline`](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html) class does for us.

> [!NOTE]
> You may notice that the label encoder is not included in the list of transformers. Unlike other transformers, the label encoder transforms the target labels `y` instead of the features `X`, therefore it is applied outside of the pipeline.

The new implementation using `Pipeline` looks like the following:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

my_label_encoder = MyLabelEncoder() # our custom label encoder

estimators = [
	('standard_scaler', StandardScaler()),
	('mean_imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
	('random_clf', MyRandomClassifier(random_state=seed)) # our custom classifier
]
pipeline = Pipeline(steps=estimators)

X_train = train_df.drop('target', axis='columns')
y_train = train_df['target']

y_train = my_label_encoder.fit_transform(y_train)
pipeline.fit(X_train, y_train)

# repeat for test data
X_test = test_df.drop('target', axis='columns')
y_test = test_df['target']

y_test = my_label_encoder.transform(y_test)
score = pipeline.score(X_test, y_test)

print(score)
```

A `Pipeline` is instantiated with a list of estimators that describe the steps in the pipeline. It expects all but the last element in the list to be transformers, but the last element can be any type of estimator.

The pipeline instance behaves the same as any other estimator: it defines a `.fit` method for training, as well as other estimator-related methods (eg. `.predict`). Under the hood, the pipeline exposes all the methods of the final estimator in the steps sequence. This means our pipeline will have a `.score` and `.predict` method because the final estimator `MyRandomClassifier` is a classifier. When any of these estimator-related methods are called on the pipeline:
1.  it calls `.transform` on all the transformers except the last estimator, passing the result of the previous transformation as input to the next transformer in the sequence, and
2. it then calls same method on the last estimator using the final transformation as the input.

The difference between the pipeline's `.fit` method and the other estimator-related methods is that the pipeline calls `.fit_transform` on each of the transformers instead of `.transform`.

With scikit-learn's pipeline system, we can compose many estimators and treat them as a single estimator, which makes it easy to use because we only need to `.fit` once and `.predict` once for each prediction.

---
| [Prev - Custom Estimators](./custom_estimators.md) |
|:-------------------------------------|
