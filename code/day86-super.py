class ParentClass:
    def __init__(self,name):
        self.name = name
        
class BaseClass(ParentClass):      # inheritance 
    def __init__(self,name,greet):
        super().__init__(name)     # call name method from parent class
        self.greet = greet

Parent = ParentClass("This is from parent class")
base = BaseClass(name="From parent class", greet="Base class method")

print(base.greet)
print(base.name)
