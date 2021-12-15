import os
import tensorflow as tf
from classifier_model import classifier_layers
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'



class ContentModel(tf.keras.models.Model):
    def __init__(self, content_image, content_layers):
        super(ContentModel, self).__init__()
        self.content_image = content_image
        self.content_layers = content_layers
        self.num_content_layers = len(self.content_layers)
        self.classifier = classifier_layers(self.content_layers)
        self.classifier.trainable = False
        self.content_targets = self.call(content_image)

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        # vgg16
        preprocessed_input = tf.keras.applications.vgg16.preprocess_input(inputs)

        # vgg 19
        # preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)

        # resnet50 / resnet101
        # preprocessed_input = tf.keras.applications.resnet50.preprocess_input(inputs)

        content_outputs = self.classifier(preprocessed_input)
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, [content_outputs])}

        return content_dict

    def content_loss_calc(self, outputs):
        content_outputs = outputs
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-self.content_targets[name])**2) 
                                for name in content_outputs.keys()])
        content_loss /= self.num_content_layers
        return content_loss
