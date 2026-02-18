import math

class Tensor:
    def __init__(self,value):
        self.value = value
        self.grad = None
    
    def __add__(self,tensor_to_add):
        return Tensor(self.value + tensor_to_add.value)
    
    def __mult__(self,tensor_to_mult):
        return Tensor(self.value*tensor_to_mult.value)
    

if __name__ == "__main__":
    a = Tensor(3.0)
    b = Tensor(4.0)
    c = a+b
    print(c.value)
    d = a.__mult__(b)
    print(d.value)