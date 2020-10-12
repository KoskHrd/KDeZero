import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 1. get the function
            x, y = f.input, f.output    # 2. get the IO of the function
            x.grad = f.backward(y.grad) # 3. call the backward method
            if x.creator is not None:
                funcs.append(x.creator) # add the former function to the list

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)    # store the creator in the output variables
        self.input = input
        self.output = output    # also store outputs
        return output

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def square(x):
    return Square()(x)  # write in one line

def exp(x):
    return Exp()(x)

def main1():
    x = Variable(np.array(0.5))
    a = square(x)
    b = exp(a)
    y = square(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)

def main2():
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    print(x.grad)

def main3():
    x = Variable(np.array(1.0)) # OK
    x = Variable(None)  # OK
    x = Variable(1.0)   # NG : raise an error!

if __name__ == "__main__":
    # main1()
    # main2()
    main3()
    pass