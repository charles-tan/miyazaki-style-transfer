import tensorflow as tf

def classifier_layers(layer_names):
    # vgg16
    # classifier = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')

    # vgg19
    # classifier = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')

    # resnet50
    classifier = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')

    # resnet 101
    # classifier = tf.keras.applications.resnet.ResNet101(include_top=False, weights='imagenet')
    
    classifier.trainable = False
    outputs = [classifier.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([classifier.input], outputs)
    return model