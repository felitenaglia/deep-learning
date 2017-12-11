import numpy
import theano
import theano.tensor as T
import TP1
rng = numpy.random

def escala_gris(img):
    return numpy.dot(img,[0.3333,0.3333,0.3333]) # Media de los 3 canales

def procesar_imagen(img):
    img = TP1.normalize(img)
    img = TP1.resize_image(img,28)
    img = escala_gris(img)
    return img.flatten()    


aviones = TP1.get_images_clase("airplanes")
aviones_cl = [0] * len(aviones)
motos = TP1.get_images_clase("Motorbikes")
motos_cl = [1] * len(motos)

numpy.random.shuffle(aviones)
numpy.random.shuffle(motos)

NTest = 200; # Por clase

test = aviones[-NTest:] + motos[-NTest:]
test_cl = aviones_cl[-NTest:] + motos_cl[-NTest:]

aviones = aviones[:-NTest]
motos = motos[:-NTest]
aviones_cl = aviones_cl[:-NTest]
motos_cl = motos_cl[:-NTest]

merged = list(map(procesar_imagen,aviones+motos))
test = list(map(procesar_imagen,test))

D = (merged, aviones_cl + motos_cl)
Tst = (test, test_cl)
N = len(D)


training_steps = 10000

feats = 784                               # number of input variables

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

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))

print("predcition on T:")
pred = predict(Tst[0])
print(pred)
print("Correcto (%): ")
print(sum(pred == Tst[1])*100/len(Tst[1]))

