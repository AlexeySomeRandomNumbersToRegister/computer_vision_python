from CNN.forward import *
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import gzip

def extract_data(filename, num_images, IMAGE_WIDTH):

    print('Извлечение', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        return data

def extract_labels(filename, num_images):

    print('Извлечение', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def initializeFilter(size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01

# Возвращает индексы элемента с наибольшим значением в массиве, игнорируя значения NaN
def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs    

def predict(image, f1, f2, w3, w4, b1, b2, b3, b4, conv_s = 1, pool_f = 2, pool_s = 2):

    conv1 = convolution(image, f1, b1, conv_s) # первая свертка
    conv1[conv1<=0] = 0 # ReLU
    
    conv2 = convolution(conv1, f2, b2, conv_s) # вторая свертка
    conv2[conv2<=0] = 0 # ReLU
    
    pooled = maxpool(conv2, pool_f, pool_s) # maxpooling
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten
    
    z = w3.dot(fc) + b3 # Первый полносвязный слой
    z[z<=0] = 0 # ReLU 
    
    out = w4.dot(z) + b4 # Второй полносвязный слой
    probs = softmax(out) # прогнозирование вероятности классов с помощью функции активации softmax
    
    return np.argmax(probs), np.max(probs)
    
def draw_bbox(ax, image_data, pred, actual):
    # Отображаем изображение
    ax.imshow(image_data, cmap='gray')
    
    # Находим координаты объекта (пикселей, значения которых больше 0)
    object_coords = np.argwhere(image_data > 0)
    
    if object_coords.size > 0:
        ymin, xmin = np.min(object_coords, axis=0)
        ymax, xmax = np.max(object_coords, axis=0)

        x_buffer = 1
        y_buffer = 1
        xmin -= x_buffer
        ymin -= y_buffer
        xmax += x_buffer
        ymax += y_buffer
        
        # Создаем прямоугольник bbox
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red')
        ax.add_patch(rect)  # Добавляем прямоугольник на изображение
    
    ax.set_title(f"Predicted class: {pred}, Actual class: {actual}")
    # ax.set_title(f"Predicted class: {pred}, (prob: {prob:.2f}) Actual class: {actual}")
    ax.axis('off')