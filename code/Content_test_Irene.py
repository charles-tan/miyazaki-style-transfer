import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

# Utils
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

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    #return PIL.Image.fromarray(tensor)
    
    if len(tensor.shape) > 3:
        tensor = tf.squeeze(tensor, axis=0)

    plt.imshow(tensor)
    plt.show()

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)



# Set up the content model
def vgg_model(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)

    return model

class ContentModel(tf.keras.models.Model):
    def __init__(self, content_image, content_layers):
        super(ContentModel, self).__init__()
        self.content_image = content_image
        self.content_layers = content_layers
        self.num_content_layers = len(self.content_layers)
        self.vgg = vgg_model(content_layers)
        self.vgg.trainable = False
        self.content_targets = self.call(content_image)
        style_outputs = self.vgg(content_image*255)
        for name, output in zip(self.content_layers, style_outputs):
            print(name)
            print("  shape: ", output.numpy().shape)
            print("  min: ", output.numpy().min())
            print("  max: ", output.numpy().max())
            print("  mean: ", output.numpy().mean())
            print()

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        content_outputs = self.vgg(preprocessed_input)
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, [content_outputs])}

        return content_dict

    def content_loss_calc(self, outputs):
        content_outputs = outputs
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-self.content_targets[name])**2) 
                                for name in content_outputs.keys()])
        content_loss /= self.num_content_layers
        return content_loss

if __name__ == '__main__':
    content_layers = ['block5_conv2'] 
    content_extractor = ContentModel(content_layers)
    num_content_layers = len(content_layers)
    content_image = load_img('../Content_images/college_hill.png')
    content_image = tf.image.resize(content_image, [200,200])
    style_image = load_img('../style_images/kokurikozaka049.jpg')
    style_image = tf.image.resize(style_image, [200,200])

    # # Testing
    # results = content_extractor(tf.constant(content_image))

    # print("Contents:")
    # for name, output in sorted(results['content'].items()):
    #   print("  ", name)
    #   print("    shape: ", output.numpy().shape)
    #   print("    min: ", output.numpy().min())
    #   print("    max: ", output.numpy().max())
    #   print("    mean: ", output.numpy().mean())

    # Gradient Descent
    content_targets = content_extractor(content_image)['content']
    # image = tf.Variable(style_image)
    image = tf.Variable(tf.random.uniform(content_image.shape, minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None))
    optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    style_weight=1e-2
    content_weight=1e4

    @tf.function()
    def train_step(image, style_loss):
        w
        sign(clip_0_1(image))

    for i in range(1000): 
        train_step(image, 0)
    tensor_to_image(image)
