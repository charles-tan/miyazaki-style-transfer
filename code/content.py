### JUST A SKETCH ###
# many unimplemented functions
# assume style model implemented

import tensorflow as tf


class ContentModel(tf.keras.models.Model):
  def __init__(self, content_layers):
    super(ContentModel, self).__init__()
    self.content_layers = content_layers

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in self.content_layers]

    model = tf.keras.Model([vgg.input], outputs)

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, model)}

    return {'content': content_dict}

def content_loss_calc(content_outputs, content_targets):
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    # content_loss *= content_weight / num_content_layers
    return content_loss




@tf.function()
def train_step(content_model, style_model, image, content_weight, style_weight):
  
  with tf.GradientTape() as tape:
    content_outputs = content_model(image)
    style_outputs = style_model(image)
    content_loss = content_loss_calc(content_outputs, content_targets)
    style_loss = style_loss_calc(style_outputs, style_targets) ### style_loss_calc wait to be implemented
    # if loss inside models
    # content_targets = content_model.content_model(content_images)
    # style_targets = style_model.style_model(style_images)
    loss = content_weight*content_loss + style_weight*style_loss

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

def main():
    content_layers = ...
    style_layers = ...
    content_model = ContentModel(content_layers)
    style_layers = StyleModel(style_layers)
    content_targets = content_model(content_images)
    style_targets = style_model(style_images)