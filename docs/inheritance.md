# Inheritance (OOP)

## Overview
1. [_Object-Oriented Programming (OOP)_](./object-oriented-programming.md)
2. **Inheritance (OOP)**
3. Estimators
4. Transformers
5. Writing Custom Estimators & Transformers
6. Pipeline
7. Common Scikit-learn modules

---

**What is Inheritance?**  
Inheritance is an _OOP_ concept in which a _class_ inherits all the _properties_ and _methods_ of another _class_.

The _class_ that is _inherited_ is commonly referred to as the `Parent Class` or the `Superclass`, while the _class_ that _inherits_ the `Superclass` is known as the `Subclass`.

You can think of the `Subclass` as a extension of the `Superclass` because it is extending the functionality of the `Superclass`. Everything that the `Superclass` can do, the `Subclass` can do the same (and more).

**Example:**  
Extending the example from **_Object-Oriented Programming (OOP)_**, suppose that within the _Social Media_ platform, we want to model `Users` with additional abilities. For example, a `Business` account that should provide the same functionality as a `User`, plus the following:
1. Advertise products
2. Sell products
3. Has a Business name
4. Provides a description of the type of business
5. Provides a physical address of the business

We can account for this feature via _Inheritance_, whereby the `Business` _class_ _extend_ (_inherit_) the `User` _class_.

```python
def create_post(title, text):
    """a utility function for creating a post
    """
    return {
        'title': title,
        'text': text,
        'likes': []
    }
```
```python
class User:
    def __init__(self, username, joined_date):
        self.username = username
        self.joined_date = joined_date
        self.friends = []
        self.posts = []

    def add_friend(self, new_friend):
        username = self.username
        friend_username = new_friend.username

        self.friends.append(friend_username)
        new_friend.friends.append(username)

    def publish_post(self, post):
        self.posts.append(post)

    def like_post(self, post):
        username = self.username
        post['likes'].append(username)
```

The syntax for _Python_ Inheritance is the following:
```python
class Subclass(Superclass1, Superclass2, ..., SuperclassN):
    pass
```

> **Note:**  
In _Python_, `Subclasses` can inherit multiple `Superclasses`, however, this is not usually the case in other programming languages. 
For example, in Java, a `Subclass` can inherit only one `Superclass`.

```python
class Business(User):
    """An extension of the User class that can do the following:
    1. Advertise products
    2. Sell products
    3. Has a Business name
    4. Provides a description of the type of business
    5. Provides a physical address of the business
    """
```

By default, the `Business` _class_ inherits the full functionality of the `User` class. This means that the `__init__` method in the `User` _class_ will be executed when instantiating a `User` _object_.

```python
>>> # the Business class calls the User __init__ method by default
>>> Business()
TypeError: __init__() missing 2 required positional arguments: 'username' and 'joined_date'
>>>
>>> business = Business('My Business', '2019-01-01')
>>> business.username, business.joined_date
('My Business', '2019-01-01')
```

We want to run our own `Business.__init__` method to _initialize_ properties exclusive to `Business` _objects_, but don't want to repeat the code in `User.__init__`.
To accomplish this, we need to _override_ the `User.__init__` method.

**Method Override:**
When a _method_ defined in a `Subclass` has the same name as another method in the `Superclass`, 
the `Subclass` method takes precedence and _overrides_ the `Superclass` method.

```python
class Business(User):
    """An extension of the User class that can do the following:
    1. Advertise products
    2. Sell products
    3. Has a Business name
    4. Provides a description of the type of business
    5. Provides a physical address of the business
    """
    def __init__(self, business_name, product, price, description, joined_date, address):
        """this __init__ method overrides the User.__init__ method,
        but we still have access to User.__init__ via `super()`.
        """
        # Superclass = super() # calling super gives a reference of the Superclass
        # Superclass.__init__(username, joined_date)
        # Usually shortened to:
        super().__init__(business_name, joined_date)

        self.product = product
        self.price = price
        self.description = description
        self.address = address
```
```python
>>> business = Business('My Business', 
...                     product='Watch',
...                     price=12.99,
...                     description='The next generation digital Watch shop', 
...                     joined_date='2019-09-01', 
...                     address='XXX Clockwork Lane')
>>>
>>> # We even have access to `username` defined in `User`
>>> business.username
'My Business'
>>>
>>> # Business has the same functionalities as the User class
>>> post = create_post('My Business Post 1', '...')
>>> business.publish_post(post) # publish_post is defined for User but works with Business objects
>>> business.posts
[
    {
        'likes': [],
        'text': '...',
        'title': 'My Business Post 1'
    }
]
```

Below is the complete implementation for the `Business` _class_
```python
class Business(User):
    """An extension of the User class that can do the following:
    1. Advertise products
    2. Sell products
    3. Has a Business name
    4. Provides a description of the type of business
    5. Provides a physical address of the business
    """
    def __init__(self, business_name, product, price, description, joined_date, address):
        super().__init__(business_name, joined_date)

        self.product = product
        self.price = price
        self.description = description
        self.address = address

    def advertise_product(self):
        title = f'{self.product} on Sale'
        text = f'Buy {self.product} now for ${self.price}'
        
        post = create_post(title, text)

        self.publish_post(post) # we can call methods on the current object, even if it was defined in the Superclass

    def sell_product(self, customer):
        print(f'{customer.username} bought {self.product} for ${self.price}')
```

```python
>>> business = Business('My Business', 
...                     product='Watch',
...                     price=12.99,
...                     description='The next generation digital Watch shop', 
...                     joined_date='2019-09-01', 
...                     address='XXX Clockwork Lane')
>>> user = User('johndoe', '2015-04-20')
>>>
>>> business.advertise_product()
>>> business.posts
[
    {
        'likes': [],
        'text': 'Buy Watch now for $12.99',
        'title': 'Watch on Sale'
    }
]
>>>
>>> business.sell_product(user)
'johndoe bought Watch for $12.99'
```

A `Business` user can do everything that a basic `User` can do, plus some extra privileges such as:
- Advertise products
- Sell products

And this is all powered by inheritance.

Any number of _classes_ can extend a single `Superclass`, and it leads to less code repetition (notice we did not re-implement the `publish_post` in `Business` _class_ even though we used it in the `advertise_product` method).

Next time, we'll finally be into the `Scikit-learn` library, starting with the `Estimator` API for working with machine-learning models.

---
> **Quick Tip:**  
The `isinstance` _function_ determines if an _object_ is an _instance_ of a _class_ either directly or via inheritance:  
`isinstance(object, class)`
```python
>>> isinstance(user, User) # <- user is an instance of User
True
>>> isinstance(business, Business) # <- business is an instance of Business
True
>>> isinstance(business, User) # <- business is an instance of User (via Inheritance)
True
>>> isinstance(user, Business) # <- user is NOT an instance of Business
False
```

---
| [Prev - Object-Oriented Programming (OOP)](./object-oriented-programming.md "Object-Oriented Programming (OOP)")  | [Next - Estimators]   |
|:----------------------------------------------------------------------------------------------------------------- |-----------------------------------------------------:|
