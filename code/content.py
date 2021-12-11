import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


# Set up the content model
def vgg_model(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load pretrained VGG, trained on imagenet data
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
