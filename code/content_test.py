import tensorflow as tf
# import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools


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


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    #return PIL.Image.fromarray(tensor)
    plt.imshow(tensor)
    plt.show()


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    for layer in vgg.layers:
        print(layer.name)

    outputs = [vgg.get_layer("block5_conv2").output]
    # outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model




class ContentModel(tf.keras.models.Model):
    def __init__(self, content_layers):
        super(ContentModel, self).__init__()
        self.content_layers = content_layers
        self.vgg = vgg_layers(self.content_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)

        # vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        # vgg.trainable = False
        # outputs = [vgg.get_layer(name).output for name in self.content_layers]
        # model = tf.keras.Model([vgg.input], outputs)

        out = self.vgg(preprocessed_input)

        # content_dict = {content_name: value
        #                 for content_name, value
        #                 in zip(self.content_layers, out)}
        content_dict = {'block5_conv2': out}
        return content_dict

def content_loss_calc(content_outputs, content_targets):
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    # content_loss *= content_weight / num_content_layers
    return content_loss




# @tf.function()
def train_step(content_model, style_model, image, content_weight, style_weight, content_targets, style_targets):
    optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
  
    with tf.GradientTape() as tape:
        content_outputs = content_model(image)
        # style_outputs = style_model(image)
        content_loss = content_loss_calc(content_outputs, content_targets)

        ## set arbiturary style loss
        # style_loss = style_loss_calc(style_outputs, style_targets) ### style_loss_calc wait to be implemented
        style_loss = 0

        # if loss inside models
        # content_loss = content_model.content_loss_calc(content_outputs, content_targets)
        # style_loss = style_model.style_loss_calc(style_outputs, style_targets) ### style_loss_calc wait to be implemented

        loss = content_weight*content_loss + style_weight*style_loss

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


def main():
    content_image = tf.image.resize(load_img("content_images/10000_suns.jpeg"), [200,200])
    content_image_1 = tf.image.resize(load_img("style_images/nighttime.jpeg"), [200,200])
    #content_image_1 = load_img("style_images/nighttime.jpeg").resize((200,200))
    content_layers = ['block5_conv2']
    # style_layers = ...
    content_model = ContentModel(content_layers)
    # style_model = StyleModel(style_layers)
    style_model = None
    content_targets = content_model(content_image_1)
    # style_targets = style_model(style_images)
    style_targets = None
    input = tf.Variable(content_image)
    train_step(content_model, style_model, input, 0.0001, 0.02, content_targets, style_targets)
    train_step(content_model, style_model, input, 0.0001, 0.02, content_targets, style_targets)
    train_step(content_model, style_model, input, 0.0001, 0.02, content_targets, style_targets)
    tensor_to_image(input)

if __name__ == "__main__":
    main()