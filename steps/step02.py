from step01 import Variable

class Function:
    def __call__(self, input):
        x = input.data  # take out data
        y = self.forward(x)  # actual calculation in 'forward'
        output = Variable(y)    # return as 'Variable'
        return output

    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

def main():
    import numpy as np
    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    print(type(y))
    print(y.data)

if __name__ == "__main__":
    main()
    pass


# from step01 import Variable

# class Function:
#     def __call__(self,input):
#         x = input.data  # take out data
#         y = x ** 2  # actual calculation
#         output = Variable(y)    # return as 'Variable'
#         return output

# import numpy as np
# x = Variable(np.array(10))
# f = Function()
# y = f(x)

# print(type(y))
# print(y.data)