#### Python中的私有变量

在Python把一个类里面的变量定义为私有变量的方法是通过**变量命名**来定义的，可以把一个属性，方法的名字前面至少加两个下划线，后面至多一个下划线命名来实现。

```python
class A:
  def __init(self):
    self.__v = 0
    
  def print_v(self):
    print(self.__v)
    
  def __private_fn(self):
    print(self.__v)
    
if __name__ == "__main__":
  o = A()     // correct
  o.print_v() // correct
  o.__private_fn() // error
```

私有变量的作用就是通过对变量和函数的命名，告诉这段代码的使用者，这个变量或者这个函数名字不应该被使用。还有一个非常重要的作用就是，就是在继承的时候避免父类的变量被覆盖。

```python
class A:
  valid_kwds = ['a']
  def __init__(self, **kwargs):
    for key, val in kwargs.items():
      if key in self.valid_kwds:
        print(key, val)
        
class B(A):
	valid_kwds = ['b']
 	def __init__(self, **kwargs):
    left_kwargs = {}
    for key, val in kwargs.items():
      if key in self.valid_kwds:
        print(key, val)
      else:
        left_kwargs[key] = val
    super().__init__(**left_kwargs)
 
if __name__ == "__main__":
  o = B(a=2, b=2) // only b and 3 print
  // should change valid_kwds to __valid_kwds in Class A and B definition
  // 在继承的过程中，class定义的私有属性不会被覆盖
```

Python中定义的私有属性，在编译过后的字节码，是将私有变量编译为\_classname\_\_properties_name（name mangling），也就是私有变量是在编译期实现的，也就是我们显示定义的私有变量才能执行name mangling，而使用setattr这种方式是不会定义出私有变量。