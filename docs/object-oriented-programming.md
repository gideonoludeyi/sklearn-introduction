# Object-Oriented Programming (OOP)

## Overview
1. **Object-Oriented Programming (OOP)**
2. Inheritance (OOP)
3. Estimators
4. Transformers
5. Custom Estimators
6. Pipeline
7. Common Scikit-learn modules

---

**What is Object-Oriented Programming?**  
A style of programming that emphasizes the use of _objects_ to represent and process data in a program.

**Basic Concepts**  
- Object (Instance)
- Class
- Property (Attribute, Field, Feature)
- Method

We will use an example to motivate the use of the **OOP** paradigm.

**Example:**  
Suppose we are required to write a program to simulate the interactions between _users_ on a _Social Media_ platform.

For the basic requirements, we need to be able to:
1. Represent each _user_ data (`username`, `birthdate`, `friends`, `posts`).
2. Add a new `friend`
3. Publish a `post`
4. Like a `post`

### [Basic Solution](#basic-solution)
A basic solution would be to store each _user's_ data as a `dict`, and use _functions_ to manipulate the data.

```python
{
    'username': 'john_doe',
    'joined_date': 'YYYY-MM-DD',
    'friends': [...], # list of usernames
    'posts': [
        {
            'title': 'Post 1',
            'text': 'A new post',
            'likes': [...] # list of usernames
        },
        ... # other posts 
    ],
}
```

**Requirement 1:** _Represent user data_  
```python
def create_user(username, joined_date):
    user = {
        'username': username,
        'joined_date': joined_date,
        'friends': [], # no friends for new user
        'posts': [], # no posts for new user
    }

    return user
```
```python
>>> johndoe = create_user('johndoe', '2015-04-20') # creating a new user dictionary
>>> johndoe
{
    'friends': [],
    'joined_date': '2018-04-20',
    'posts': [],
    'username': 'johndoe'
}
>>> mikesmith = create_user('mikesmith', '2020-10-31') # creating a new user dictionary
>>> mikesmith
{
    'friends': [],
    'joined_date': '2020-10-31',
    'posts': [],
    'username': 'mikesmith'
}
```

**Requirement 2:** _Add friend_
```python
def add_friend(user1, user2):
    username1 = user1['username']
    username2 = user2['username']

    user1['friends'].append(username2)
    user2['friends'].append(username1)
```
```python
>>> add_friend(johndoe, mikesmith) # adding friends
>>> johndoe['friends']
['mikesmith']
>>> mikesmith['friends']
['johndoe']
```

**Requirement 3:** _Publish post_
```python
def publish_post(user, post):
    user['posts'].append(post)
```
```python
>>> post = {
...     'title': 'Post 1',
...     'text': 'A new post',
...     'likes': []
... }
>>> publish_post(johndoe, post) # publish a new post
>>> johndoe['posts']
[
    {
        'likes': [],
        'text': 'A new post',
        'title': 'Post 1'
    }
]
```

**Requirement 4:** _Like post_
```python
def like_post(user, post):
    username = user['username']

    post['likes'].append(username)
```
```python
>>> like_post(mikesmith, post) # liking a post
>>> johndoe['posts']
[
    {
        'likes': ['mikesmith'],
        'text': 'A new post',
        'title': 'Post 1'
    }
]
```

### [OOP Solution](#oop-solution)
The [_Basic Solution_](#basic-solution) already solves the _requirements_ for the program.

In fact, what we have done is _conceptually_ inline with the _OOP_ paradigm.

> **Recall:**  
_Object-Oriented Programming_ is a style of programming that emphasizes the use of _objects_ to represent and process data in a program.

**What is an Object?**  
In simplest terms, an _object_ is a `data type` or `data structure`.

`string, integer, boolean, list` are all _objects_.

Even the `dict` we've been using the represent the _user_ data is an _object_.

There are two type of features that make _objects_ powerful:
- **Properties** - _variables_ that belong to an _object_. These variables that are accessible only through the _object_.
- **Methods** - _functions_ that are bound to, and interact specifically with the _object_.

Conceptually, the _key-value_ pairs in the _user_ `dict` are like an _object's properties_.  
And the _functions_ we've defined to manipulate the _user_ `dict` are the _methods_ associated with that specific type of _data structure_.

However, in order to create an _object_, we need to define its structure. This is accomplished with a _class_.

**What is a Class?**  
A _class_ is a _blueprint_ of an _object's_ structure.
Just as a house requires a _blueprint_ that defines its structure, an _object_ requires a _class_ in order to be constructed.

Specifically, a _class_ defines the **Properties** and **Methods** that its _objects_ possess.

To define a _class_, we use a special keyword called `class`:
```python
>>> class User:
...     pass
```

And we create an _object_ (aka _instance_) of the _class_ by calling it like a _function_. This is called _instantiation_.
```python
>>> user = User()
```

> **Note:**  
`pass` is a special keyword in python for avoiding a common error that occurs when there is no code within an indentation block.
```python
SyntaxError: unexpected EOF while parsing
```

Right now the `user` _object_ does not have any _properties defined_.

We can assign _properties_ to an _object_ using the `<object>.<property>` syntax to access its _properties_:
```python
>>> user.username = 'johndoe'
>>> user.username
'johndoe'
```

> **Note:**  
Attempting to access a property that does not exist on an _object_ will result in an error.
```python
>>> user.name
AttributeError: 'User' object has no attribute 'name'
```

We can even combine the process of _instantiating_ the _object_ and _initializing_ its _properties_ into a single _function_ `create_user_object`.

This ensures that every `User` _object_ we create has the expected _properties_ defined when we use the `create_user_object` _function_.
```python
def create_user_object(username, joined_date):
    user = User()

    user.username = username
    user.joined_date = joined_date
    user.friends = [] # no friends for new user
    user.posts = [] # no post for new user

    return user
```
```python
>>> johndoe = create_user_object('johndoe', '2018-04-20')
>>> johndoe.username, johndoe.joined_date
('johndoe', '2018-04-20')
>>> mikesmith = create_user_object('mikesmith', '2020-10-31')
>>> mikesmith.username, mikesmith.joined_date
('mikesmith', '2020-10-31')
```

Now that we've looked at _properties_, let's move on to _methods_.

**Methods** are _functions_ that are bound to a particular _class_ and are used by _objects_ of that _class_.

We define a _method_ on a _class_ like this:
```python
class MyClass:
    def my_method(self, ...):
        pass
```

And call it like this:
```python
>>> my_object = MyClass()
>>> my_object.my_method(...)
```

It is similar to a _function_ definition except for 2 notable differences:
1. The definition exists within the indentation block of the _class_ definition.
    - This means that most of the code relating to the _class_ are bundled up in the _class_ definition.
    - It's now clear the _method_ is meant only for _objects_ of that _class_.

2. The first argument is always the current _object_ calling the _method_. By convention, the name of that first argument is called `self`.
    - We don't have to explicitly pass in the _object_ for its _methods_ to gain access to it.

```python
class User:
    # Requirement 2: Add friend
    def add_friend(self, new_friend):
        username = self.username
        friend_username = new_friend.username

        self.friends.append(friend_username)
        new_friend.friends.append(username)

    # Requirement 3: Publish post
    def publish_post(self, post):
        self.posts.append(post)

    # Requirement 4: Like post
    def like_post(self, post):
        username = self.username
        post['likes'].append(username) # append is a method on the `list` object
        
# Requirement 1: Represent user data
def create_user_object(username, birthdate):
    user = User()

    user.username = username
    user.birthdate = birthdate
    user.friends = [] # no friends for new user
    user.posts = [] # no post for new user

    return user
```
```python
>>> # Requirement 1: Represent user data
>>> johndoe = create_user_object('johndoe', '2015-04-20')
>>> mikesmith = create_user_object('mikesmith', '2020-10-31')
>>>
>>> # Requirement 2: Add friend
>>> johndoe.add_friend(mikesmith) # instead of add_friend(johndoe, mikesmith)
>>> johndoe.friends
['mikesmith']
>>> mikesmith.friends
['johndoe']
>>>
>>> # Requirement 3: Publish post
>>> post = {
...     'title': 'Post 1',
...     'text': 'A new post',
...     'likes': []
... }
>>> johndoe.publish_post(post) # instead of publish_post(johndoe, post)
>>> johndoe.posts
[
    {
        'likes': [],
        'text': 'A new post',
        'title': 'Post 1'
    }
]
>>>
>>> # Requirement 4: Like post
>>> mikesmith.like_post(post) # instead of like_post(mikesmith, post)
>>> post
{
    'likes': ['mikesmith'],
    'text': 'A new post',
    'title': 'Post 1'
}
```

Now that we have moved most of the code into the `User` _class_, it is now easy to see which _methods_ an _object_ can use.  
Depending on the text editor or IDE you're using, you could even get auto-complete features.

It would be nice if the process for _instantiating_ and _initializing_ a `User` was also bundled together in the _class_ definition with the rest of the code.  
If only there was a way to automatically _initialize_ the properties of an _object_ at the same time when we _instantiate_ it via `MyClass()`.

Fortunately, _Python_ provides a solution for this.  
There are a set of special _method_ definitions that _Python_ watches out for in a _class_.  
If present, these special _methods_ enhance the functionalities of the _classes_ that define them and their _objects_.

These _methods_ are commonly referred to as [_dunder (double-underscore) methods_](https://docs.python.org/3/reference/datamodel.html#special-method-names),
due to their naming convention (`def __methodname__(self, ...)`).

One of those special methods is the `__init__` method.  
Whenever we construct an _object_ (i.e: calling `MyClass(...)`),
_Python_ automatically calls the `__init__` method in the background after the _object_ has been _instantiated_.
```python
>>> # When we do this 👇
>>> my_object = MyClass(...)

>>> # Python does this 👇 for us if __init__ is defined
>>> my_object = MyClass()
>>> my_object.__init__(...)
```

We can now move the _initialization_ process for `User` _objects_ from `create_user_object` to the `__init__` method.
```python
class User:
    # Requirement 1: Represent user data
    def __init__(self, username, joined_date):
        # user = User() # we don't need this

        self.username = username
        self.joined_date = joined_date
        self.friends = [] # no friends for new user
        self.posts = [] # no post for new user

        # return user # we don't need this

    # Requirement 2: Add friend
    def add_friend(self, new_friend):
        username = self.username
        friend_username = new_friend.username

        self.friends.append(friend_username)
        new_friend.friends.append(username)

    # Requirement 3: Publish post
    def publish_post(self, post):
        self.posts.append(post)

    # Requirement 4: Like post
    def like_post(self, post):
        username = self.username
        post['likes'].append(username) # Note: .append is a method for `list` objects
```

Now every code related to `User` is defined in the _class_ definition.  
Both the _properties_ and _methods_ are visible in the same location.
```python
>>> johndoe = User('johndoe', '2018-04-20')
>>> mikesmith = User('mikesmith', '2020-10-31')
```

### Conclusion

#### Naming Convention
The built-in _classes_ in _Python_ are typically in lower-case because they are used frequently and recognized by most _Python_ developers as _classes_.
However, when defining custom _classes_, it is standard to use `PascalCase` casing for custom class names
to ensure that readers of the code recognize they are _classes_ at a glance.

#### Documentation
When defining a _class_ it is advised to provide documentation using a multi-line string,
and to provide documentation for its `methods` in the same manner.

```python 
>>> class MyClass:
...     """Description and purpose of the class goes here.
...     Also describe the properties that belong to this class.
...     The rest of the class definition goes below
...     """
...     
...     def my_method(self, x, y):
...         """Description about my_method preferably talking about
...         the purpose of the method and what its arguments are for
...         """
...         pass
```

> **Quick Tip:**  
if you call the built-in `help` _function_ on a _class_ or _object_, it outputs the documentation for that class.

```python
>>> help(MyClass)
Help on class MyClass in module __main__:

class MyClass(builtins.object)
 |  Description and purpose of the class goes here.
 |  Also describe the properties that belong to this class.
 |  The rest of the class definition goes below
 |  
 |  Methods defined here:
 |  
 |  my_method(self, x, y)
 |      Description about my_method preferably talking about
 |      the purpose of the method and its arguments are for
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
```

Hopefully through these examples ([_Basic Solution_](#basic-solution) and [_OOP Solution_](#oop-solution)), you now realize how powerful the **Object-Oriented Programming** paradigm is.

This style of programming will come up frequently as you advance in your programming journey.

As an exercise, you can try implementing a `Post` _class_ with what we've learned so far and integrate it with the current code.

---
| [Prev - Overview](./index.md) | [Next - Inheritance (OOP)](./inheritance.md) |
|:------------------------------|---------------------------------------------:|
