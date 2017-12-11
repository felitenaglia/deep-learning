import theano
a = theano.tensor.vector() # declare variable
b = theano.tensor.vector()
out =  a ** 2 + b ** 2 + 2 * a * b               # build symbolic expression
f = theano.function([a, b], out)   # compile function
print(f([0, 1, 2], [2, 3, 4]))
