import os
import tensorflow as tf
from classifier_model import classifier_layers
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


class StyleModel(tf.keras.models.Model):
    def __init__(self, style_image, style_layers):
        super(StyleModel, self).__init__()
        self.style_image = style_image
        self.style_layers = style_layers
        self.num_style_layers = len(self.style_layers)
        self.classifier = classifier_layers(self.style_layers)
        self.classifier.trainable = False
        self.style_targets = self.call(tf.constant(style_image))
        style_outputs = self.classifier(style_image*255)

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0

        # vgg16
        preprocessed_input = tf.keras.applications.vgg16.preprocess_input(inputs)

        # vgg 19
        # preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)

        # resnet50 / resnet101
        # preprocessed_input = tf.keras.applications.resnet50.preprocess_input(inputs)

        outputs = self.classifier(preprocessed_input)
        style_outputs = (outputs[:self.num_style_layers])

        style_outputs = [self.gram_matrix(style_output)
                        for style_output in style_outputs]

        style_dict = {style_name: value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}

        return style_dict

    # Gram matrix function found here: https://www.geeksforgeeks.org/neural-style-transfer-with-tensorflow/
    def gram_matrix(self, A):
        channels = int(A.shape[-1])
        a = tf.reshape(A, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        r = gram / tf.cast(n, tf.float32)
        return r

    def style_loss_calc(self, outputs):
        style_outputs = outputs
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-self.style_targets[name])**2) 
                            for name in style_outputs.keys()])
        style_loss /= self.num_style_layers
        
        return style_loss
