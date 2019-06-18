import keras
from keras.models import Sequential, Model
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Activation, Flatten, Dense
from keras.layers import SpatialDropout3D, Add
from keras.engine import Input
from keras.layers.convolutional import AveragePooling3D
from Models.coord import CoordinateChannel3D

def add_layer(x, num_filters, d_rate, batch_norm):
    
    x = Conv3D(num_filters, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    
    if d_rate > 0:
        x = SpatialDropout3D(d_rate)(x)

    x = Conv3D(num_filters, kernel_size=(3, 3, 3), padding='same')(x)
    
    if batch_norm:
        x = BatchNormalization(axis=-1)(x)
    
    x = Activation('relu')(x)
    return MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='valid')(x)

def CNN_3D(input_shape, sf=4, n_layers=6, d_rate=.3, batch_norm=True, regression=False, coord_conv=False):
    
    filters = [sf * (2 ** n) for n in range(n_layers)]
    
    ip = Input(shape = input_shape)
    
    if coord_conv:
        x = CoordinateChannel3D()(ip)
        x = add_layer(x, filters[0], d_rate, batch_norm)
    else:
        x = add_layer(ip, filters[0], d_rate, batch_norm)
        
    for l in range(1, len(filters)):
        x = add_layer(x, filters[l], d_rate, batch_norm)
    
    x = Flatten()(x)
    x = Dense(1)(x)
    
    if regression:
        x = Activation('linear')(x)
    else:
        x = Activation('sigmoid')(x)
        
    model = Model(ip, x)
    return model

def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """

    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)
    
def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2

def Unet_Inspired(input_shape, depth=6, n_base_filters=16):
    
    inputs = Input(input_shape)
    current_layer = inputs
    
    level_output_layers = list()
    level_filters = list()
    
    
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=0.3)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer
        
    
    avgpool = AveragePooling3D(pool_size = 4, strides=(1, 1, 1))(current_layer)
    flatten = Flatten()(avgpool)
    dense = Dense(units=1, kernel_initializer="he_normal", activation="sigmoid")(flatten)

    activation_block = Activation('sigmoid')(dense)

    model = Model(inputs=inputs, outputs=activation_block)
    
    return model


