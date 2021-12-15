from style import StyleModel
from content import ContentModel
import utils

import re
import tensorflow as tf

import IPython.display as display

@tf.function()
def train_step(image, content_model, style_model, style_weight=1e-2, content_weight=1e4, variation_weight=30):
    with tf.GradientTape() as tape:
        content_outputs = content_model(image)
        style_outputs = style_model(image)
        style_loss = style_model.style_loss_calc(style_outputs)
        content_loss = content_model.content_loss_calc(content_outputs)
        variation_loss = tf.image.total_variation(image)
        loss =  content_weight*content_loss + style_weight*style_loss + variation_weight*variation_loss

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))
    return (content_loss, style_loss, variation_loss, loss)


if __name__ == '__main__':
    style_path = 'style_images/kokurikozaka049.jpg'
    content_path = 'content_images/college_hill.png'
    style_path_name = re.split('\.|/', style_path)[-2]
    content_path_name = re.split('\.|/', content_path)[-2]
    style_image = utils.load_img(style_path)
    content_image = utils.load_img(content_path)
    save_dir = 'results/'

    # vgg19
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    # resnet50
    # content_layers = ['conv5_block3_3_conv']
    # style_layers = ['conv1_conv', 
    #                 'conv2_block1_1_conv',
    #                 'conv3_block1_1_conv',
    #                 'conv4_block1_1_conv',
    #                 'conv5_block1_1_conv']

    style_model = StyleModel(style_image, style_layers)
    content_model = ContentModel(content_image, content_layers)

    image = tf.Variable(content_image)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    num_style_layers = len(style_layers)
    num_content_layers = len(content_layers)

    import time
    
    start = time.time()

    epochs = 5

    steps_per_epoch = 200

    content_loss_list = []
    style_loss_list = []
    variation_loss_list = []
    loss_list = []
    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            content_loss, style_loss, variation_loss, loss = train_step(image, content_model, style_model, style_weight=1e-2, content_weight=1e4, variation_weight=100)
            content_loss_list.append(content_loss)
            style_loss_list.append(style_loss)
            variation_loss_list.append(variation_loss)
            loss_list.append(loss)
            print(".", end='', flush=True)
        display.clear_output(wait=True)
        utils.tensor_to_image(image, file_name=(save_dir + '/' + style_path_name + '_' + content_path_name + '_' + str(n) + ".jpg"))
        utils.tensor_to_image(image, show=True)
        print("Train step: {}".format(step))
        print("content_loss", content_loss, "style_loss", style_loss, "variation_loss", variation_loss, "total_loss", loss)

    end = time.time()
    print("Total time: {:.1f}".format(end-start))
    
    import numpy as np
    utils.show_plot(np.log(content_loss_list), "log content loss")
    utils.show_plot(np.log(style_loss_list), "log style loss")
    utils.show_plot(np.log(variation_loss_list), "log variation loss")
    utils.show_plot(np.log(loss_list), "log total weighted loss")

    # for n in range(1):
    #     for m in range(steps_per_epoch):
    #         step += 1
    #         content_loss, style_loss, variation_loss, loss = train_step(image, content_model, style_model, style_weight=0, content_weight=1e4)
    #         content_loss_list.append(content_loss)
    #         style_loss_list.append(style_loss)
    #         variation_loss_list.append(variation_loss)
    #         loss_list.append(loss)
    #         print(".", end='', flush=True)
    #     display.clear_output(wait=True)
    #     # utils.tensor_to_image(image, file_name=(save_dir + '/' + style_path_name + '_' + content_path_name + '_' + str(n) + ".jpg"))
    #     utils.tensor_to_image(image, show=True)
    #     print("Train step: {}".format(step))
    #     print("content_loss", content_loss, "style_loss", style_loss, "variation_loss", variation_loss, "total_loss", loss)

    
    # end = time.time()
    # print("Total time: {:.1f}".format(end-start))