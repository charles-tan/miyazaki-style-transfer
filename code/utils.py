import tensorflow as tf

import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

def tensor_to_image(tensor, file_name="", show=False):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    tensor = tensor[0]
    if show:
        PIL.Image.fromarray(tensor).show()
    if file_name != "":
        t = tf.image.encode_jpeg(tensor, quality=100, format='rgb')
        tf.io.write_file(file_name, t)

def load_img(path_to_img):
    max_dim = 512

    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def show_plot(vals, title=""):
    x = np.arange(len(vals))
    y = np.asarray(vals)
    plt.plot(x,y)
    plt.title(title)
    plt.show()
