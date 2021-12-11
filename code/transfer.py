from style import StyleModel
from content import ContentModel
import utils

import tensorflow as tf

import IPython.display as display

@tf.function()
def train_step(image, content_model, style_model):
    total_variation_weight=30
    with tf.GradientTape() as tape:
        content_outputs = content_model(image)
        style_outputs = style_model(image)
        style_loss = style_model.style_loss_calc(style_outputs)
        content_loss = content_model.content_loss_calc(content_outputs)
        loss = content_weight * content_loss + style_weight * style_loss
        loss += total_variation_weight*tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(utils.clip_0_1(image))

if __name__ == '__main__':
    style_path = 'style_images/kokurikozaka049.jpg'
    content_path = 'content_images/college_hill.png'
    style_image = utils.load_img(style_path)
    content_image = utils.load_img(content_path)

    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    style_model = StyleModel(style_image, style_layers)
    content_model = ContentModel(content_image, content_layers)

    image = tf.Variable(content_image)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    style_weight=1e-2
    content_weight=1e4

    num_style_layers = 5
    num_content_layers = len(content_layers)

    import time
    start = time.time()

    epochs = 4
    steps_per_epoch = 100

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image, content_model, style_model)
            print(".", end='', flush=True)
        display.clear_output(wait=True)
        utils.tensor_to_image(image).show()
        print("Train step: {}".format(step))

    end = time.time()
    print("Total time: {:.1f}".format(end-start))

    fin_image = utils.tensor_to_image(image)
    fin_image.show()