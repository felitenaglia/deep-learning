import numpy
import theano
import theano.tensor as T
import math
rng = numpy.random

N = 400                                   # training sample size
feats = 784                               # number of input variables

tamanio_batch = 10

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
w = theano.shared(rng.randn(feats), name="w")

# initialize the bias term
b = theano.shared(0., name="b")

print("Initial model:")
print(w.get_value())
print(b.get_value())

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

seed_state = numpy.random.get_state()
numpy.random.shuffle(D[0]) # Mezclo la primer componente del par
numpy.random.set_state(seed_state) # Vuelvo la semilla como estaba antes, para que sea la misma permutacion
numpy.random.shuffle(D[1]) 

# Divido en batches de igual tamaño salvo, quizás, el último
NBatches = math.ceil(N/tamanio_batch)
batch = [0]*NBatches
for i in range(NBatches):
    base = i*tamanio_batch
    limite = (i+1)*tamanio_batch
    batch[i] = (D[0][base:limite],D[1][base:limite])

# Train
for i in range(training_steps):
    for j in batch :
        pred, err = train(j[0], j[1])

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))
