
import keras
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Deconvolution3D, LeakyReLU
from keras.layers import Reshape, Dense, Flatten, Activation, BatchNormalization

def encoder_model_200(input_shape=(160,192,160,1), bottleneck=200):
    
    enc = Sequential()
    
    enc.add(Conv3D(64, 5, padding='same', strides=2, input_shape=input_shape))
    enc.add(LeakyReLU(alpha=.2))
    enc.add(Conv3D(128, 5, padding='same', strides=2))
    enc.add(LeakyReLU(alpha=.2))
    enc.add(Conv3D(128, 5, padding='same', strides=2))
    enc.add(LeakyReLU(alpha=.2))
    enc.add(Conv3D(256, 5, padding='same', strides=2))
    enc.add(LeakyReLU(alpha=.2))
    enc.add(Conv3D(256, 5, padding='same', strides=2))
    enc.add(LeakyReLU(alpha=.2))
    
    enc.add(Flatten())
    enc.add(Dense(bottleneck, activation = 'sigmoid'))
    
    enc.add(Dense(256*5*6*5))
    enc.add(Reshape((5,6,5,256)))

    enc.add(Deconvolution3D(filters=256, kernel_size=4, padding='same',
                            strides=2, activation='relu'))
    enc.add(Deconvolution3D(filters=256, kernel_size=4, padding='same',
                            strides=2, activation='relu'))
    enc.add(Deconvolution3D(filters=128, kernel_size=4, padding='same',
                            strides=2, activation='relu'))
    enc.add(Deconvolution3D(filters=64, kernel_size=4, padding='same',
                            strides=2, activation='relu'))
    enc.add(Deconvolution3D(filters=1, kernel_size=4, padding='same',
                            strides=2, activation='relu'))    
    enc.summary()
    return enc


def brain_enc_model(input_shape=(160,192,160,1)):
    
    enc=Sequential()
    enc.add(Conv3D(8, (3,3,3), padding='same', activation='relu', input_shape=input_shape))
    enc.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same'))
    
    enc.add(Conv3D(8, (3,3,3), padding='same', activation='relu'))
    enc.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same'))
    
    enc.add(Conv3D(8, (3,3,3), padding='same', activation='relu'))
    enc.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same'))
    
    enc.add(Deconvolution3D(filters=8, kernel_size=4, padding='same', strides=2, activation='relu'))
    enc.add(Deconvolution3D(filters=8, kernel_size=4, padding='same', strides=2, activation='relu'))
    enc.add(Deconvolution3D(filters=8, kernel_size=4, padding='same', strides=2, activation='relu'))
    
    enc.summary()
    return enc


def CNN_3D_AE(input_shape=(160,192,160,1), bottleneck=200):
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
    model.add(Dense(bottleneck, activation = 'sigmoid'))
    model.add(Dense(64*5*6*5))
    model.add(Reshape((5,6,5,64)))
    

    model.add(Deconvolution3D(filters=64, kernel_size=3, padding='same',
                            strides=2, activation='relu'))
    model.add(Deconvolution3D(filters=32, kernel_size=3, padding='same',
                            strides=2, activation='relu'))
    model.add(Deconvolution3D(filters=16, kernel_size=3, padding='same',
                            strides=2, activation='relu'))
    model.add(Deconvolution3D(filters=8, kernel_size=3, padding='same',
                            strides=2, activation='relu'))
    model.add(Deconvolution3D(filters=1, kernel_size=3, padding='same',
                            strides=2, activation='relu'))    
    
    model.summary()
    return model

CNN_3D_AE()