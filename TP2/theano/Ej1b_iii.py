import numpy
import theano
import theano.tensor as T
from math import sqrt
rng = numpy.random

N = 400                                   # training sample size
feats = 784                               # number of input variables
hidden_layer = 100                        # Número de capas ocultas


# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 10000

# Declare Theano symbolic variables
x = T.dmatrix("x")
y = T.dvector("y")

# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
w0 = theano.shared(rng.randn(feats, hidden_layer), name="w0")
w1 = theano.shared(rng.randn(hidden_layer) * sqrt(2.0/hidden_layer), name="w1")

# initialize the bias term
b0 = theano.shared(0., name="b0")
b1 = theano.shared(0., name="b1")

print("Initial model:")
print(w0.get_value())
print(b0.get_value())
print(w1.get_value())
print(b1.get_value())

# Construct Theano expression graph
primer_capa = T.dot(x,w0)+b0
segunda_capa = T.nnet.relu(T.dot(primer_capa, w1) + b1)
p_1 = 1 / (1 + T.exp(-segunda_capa))  
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w0 ** 2).sum() + 0.01 * (w1**2).sum() 	# The cost to minimize
gw0, gb0, gw1, gb1 = T.grad(cost, [w0, b0, w1, b1])             # Compute the gradient of the cost


# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w0, w0 - 0.1 * gw0), (b0, b0 - 0.1 * gb0), (w1, w1 - 0.1*gw1), (b1,b1-0.1*gb1)))
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print("Final model:")
print(w0.get_value())
print(b0.get_value())
print(w1.get_value())
print(b1.get_value())

print("target values for D:")
print(D[1])
print("prediction on D:")
pred = predict(D[0])
print(pred)
print("accuracy:")
print(sum(pred==D[1])/N)
