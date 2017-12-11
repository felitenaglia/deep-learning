from keras.preprocessing.image import ImageDataGenerator

import h5py

imgDataGen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=True,		#Media 0
    featurewise_std_normalization=False,
    samplewise_std_normalization=True,	#Varianza 1
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1/255)			#Las lleva a [0,1]

# Lee las imagenes del directorio Caltech101, determinando las clases según los subdirectorios presentes.
# Existen parametros opcionales, como el tamaño del minibatch; si queremos restringir a determinadas clases;
# si queremos transformar las imagenes a escala de gris, entre otros.
# Tambien reescala a 28x28
dataset = imgDataGen.flow_from_directory('./Caltech101',target_size=(28,28))

# 'dataset' luego se podria iterar, devolviendo un minibatch de 32 (el valor por defecto) en cada iteracion.
# Es un iterador infinito, por lo que repetira muestras a partir de alguna iteración.
# Usualmente se utiliza con model.fit_generator(...) para entrenar el modelo, indicando alli
# limite de muestras por epoca


#Ejercicio 2 c 
#Volcado del dataset a un archivo hdf5:
#Se guardará una muy pequeña parte del conjunto de imagenes ya que es solamente a modo de ejemplo

primer_batch = dataset.next()

archivo = h5py.File('caltech.h5','w')

#Se crean dos 'datasets' en el archivo, uno es la entrada y el otro la clase correspondiente
archivo.create_dataset("dset_caltech101",data=primer_batch[0])
archivo.create_dataset("clases_caltech101",data=primer_batch[1])

archivo.close()

#Ejemplo de como leer
archivo = h5py.File('caltech.h5','r+')

imgs = archivo['/dset_caltech101']
clases = archivo['/clases_caltech101']

archivo.close()


