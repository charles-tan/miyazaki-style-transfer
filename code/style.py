import os
import tensorflow as tf
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import IPython.display as display

import matplotlib as mpl

import time
import utils

def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

class StyleModel(tf.keras.models.Model):
    def __init__(self, style_image, style_layers):
        super(StyleModel, self).__init__()
        self.style_image = style_image
        self.style_layers = style_layers
        self.num_style_layers = len(self.style_layers)
        self.vgg = vgg_layers(self.style_layers)
        self.vgg.trainable = False
        self.style_targets = self.call(tf.constant(style_image))
        style_outputs = self.vgg(style_image*255)
        for name, output in zip(self.style_layers, style_outputs):
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
        outputs = self.vgg(preprocessed_input)
        style_outputs = (outputs[:self.num_style_layers])

        style_outputs = [self.gram_matrix(style_output)
                        for style_output in style_outputs]

        style_dict = {style_name: value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}

        return style_dict

    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)

    def style_loss_calc(self, outputs):
        style_outputs = outputs
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-self.style_targets[name])**2) 
                            for name in style_outputs.keys()])
        style_loss /= self.num_style_layers
        
        return style_loss
