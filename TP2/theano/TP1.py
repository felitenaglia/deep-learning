import os, subprocess
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def get_images_clase(clase):
    inputdir = "./Caltech101/"
    extension = '.jpg'
    r = []
    for fileName in os.listdir(os.path.join(inputdir, clase)):
        if fileName.endswith(extension):
            r.append(misc.imread(os.path.join(os.path.join(inputdir, clase), fileName)))
    return r


def to_float(face):
    face = np.float32(face)
    face /= 255
    return face    

def mean_by_channel(face):
    r = []
    for i in range(face.shape[2]):
        r.append(face[...,i].mean())
    return np.array(r)

def zero_mean(face):
    face -= mean_by_channel(face)
    return face

def var_by_channel(face):
    r = []
    for i in range(face.shape[2]):
        r.append(face[...,i].var())
    return np.array(r)

def one_variance(face):
    variance = var_by_channel(face)
    face /= list(map(sqrt,variance)  )  
    return face

def show(face):
    m = face.min()
    M = face.max()
    face = (face-m)/(M-m)
    plt.imshow(face) 
    plt.axis('off')
    plt.show()

def add_depth(face):
    if len(face.shape) < 3:
        x, y = face.shape
        ret = np.empty((x, y, 3), dtype=np.float32)
        ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  face
        return ret
    else:
        return face

def make_squared(face):
    lx, ly, _ = face.shape
    tam = max(lx, ly)
    color = mean_by_channel(face)
    if ly<tam:
        face = np.append([[color]*((tam-ly)//2)]*lx, face, axis=1)
        face = np.append(face, [[color]*((tam-ly+1)//2)]*lx, axis=1)
    elif lx<tam:
        face = np.append([[color]*ly]*((tam-lx)//2), face, axis=0)
        face = np.append(face, [[color]*ly]*((tam-lx+1)//2), axis=0)
    return face

# Elimina la cuarta capa para imágenes png que contengan transparencia.
def drop_transparency(face):
    face = np.array([ face[...,0], face[...,1], face[...,2] ])
    face = np.rollaxis(face, 0, 3)
    return face

def resize_image(face,tam):
    return to_float(misc.imresize(face, (tam, tam))) 


def normalize(face):
    face = add_depth(face)
    face = make_squared(to_float(face))
    face = zero_mean(face)
    face = one_variance(face)
    return face

