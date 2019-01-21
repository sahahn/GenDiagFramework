
import keras
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Deconvolution3D, LeakyReLU
from keras.layers import Reshape, Dense, Flatten

def encoder_model_200(input_shape=(160,192,160,1)):
    
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
    enc.add(Dense(200, activation = 'sigmoid'))
    
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