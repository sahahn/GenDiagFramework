import keras
from keras.models import Sequential, Model
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Activation, Flatten, Dense


def CNN_3D(input_shape):
    filters = [4, 8, 16, 32, 64]
    model = Sequential()

    model.add(Conv3D(filters=filters[0], kernel_size=(3, 3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv3D(filters=filters[0], kernel_size=(3, 3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='valid'))
    
    model.add(Conv3D(filters=filters[1], kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(filters=filters[1], kernel_size=(3, 3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='valid'))
              
    model.add(Conv3D(filters=filters[2], kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(filters=filters[2], kernel_size=(3, 3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='valid'))
    
    model.add(Conv3D(filters=filters[3], kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(filters=filters[3], kernel_size=(3, 3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='valid'))
    
    model.add(Conv3D(filters=filters[4], kernel_size=(3, 3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(filters=filters[4], kernel_size=(3, 3, 3), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2, padding='valid'))
              
    model.add(Flatten())
    model.add(Dense(1))
    
    #model.summary()
    return model

