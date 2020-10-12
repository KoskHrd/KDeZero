import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator    # 1. get the function
        if f is not None:
            x = f.input # 2. get the input of the function
            x.grad = f.backward(self.grad)  # 3. call the backward method of the function
            x.backward()    # 4. call the backward method of the former variable (recursive)

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
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

def main2():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # backward
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)


class Variable_without_Backward:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

class Function_without_Backward:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable_without_Backward(y)
        output.set_creator(self)    # store the creator in the output variables
        self.input = input
        self.output = output    # also store outputs
        return output

class Square_without_Backward(Function_without_Backward):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp_without_Backward(Function_without_Backward):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def main1():
    A = Square_without_Backward()
    B = Exp_without_Backward()
    C = Square_without_Backward()

    x = Variable_without_Backward(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # follow the nodes of the calculation graph in the opposite direction
    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x

    y.grad = np.array(1.0)

    C = y.creator   # 1. get the function
    b = C.input # 2. get the input of the function
    b.grad = C.backward(y.grad) # 3. call the backward method of the function
    B = b.creator   # 1. get the function
    a = B.input # 2. get the input of the function
    a.grad = B.backward(b.grad) # 3. call the backward method of the function
    A = a.creator   # 1. get the function
    x = A.input # 2. get the input of the function
    x.grad = A.backward(a.grad) # 3. call the backward method of the function
    print(x.grad)


if __name__ == "__main__":
    main1()
    main2()
    pass