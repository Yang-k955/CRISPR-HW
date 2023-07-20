
import os
import keras
from keras.layers import Conv2D,MaxPool1D, MaxPooling1D, AveragePooling1D, MaxPool2D, GlobalAveragePooling1D, GlobalMaxPool1D, Input, Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, BatchNormalization, concatenate, Activation, Multiply, Permute, Reshape, Lambda, Add,multiply,Flatten
from keras.models import Model
from keras.initializers import RandomUniform,glorot_normal,VarianceScaling

VOCAB_SIZE = 25
EMBED_SIZE = 100
MAXLEN = 24

def attention(x, g, TIME_STEPS):
    """
    inputs.shape = (batch_size, time_steps, input_dim)
    """
    input_dim = int(x.shape[2])
    x1 = K.permute_dimensions(x, (0, 2, 1))
    g1 = K.permute_dimensions(g, (0, 2, 1))

    x2 = Reshape((input_dim, TIME_STEPS))(x1)
    g2 = Reshape((input_dim, TIME_STEPS))(g1)

    x3 = Dense(TIME_STEPS, kernel_initializer=RandomUniform(seed=2023))(x2)
    g3 = Dense(TIME_STEPS, kernel_initializer=RandomUniform(seed=2023))(g2)
    x4 =  keras.layers.add([x3, g3])
    a = Dense(TIME_STEPS, activation="softmax", use_bias=False)(x4)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = multiply([x, a_probs])
    return output_attention_mul

def residual_block(x, num_filters):
    res = x
    x = Conv1D(num_filters, kernel_size=5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(num_filters, kernel_size=5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([res, x])
    x = Activation('relu')(x)
    return x


def CRISPR_HW():
    input = Input(shape=(24,))
    embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)
    conv1 = Conv1D(70, 4, activation="relu", name="conv1")(embedded)
    batchnor1 = BatchNormalization()(conv1)
    conv2 = Conv1D(40, 6, activation="relu", name="conv2")(batchnor1)
    batchnor2 = BatchNormalization()(conv2)

    #---------Resnet---------#
    c1 = residual_block(batchnor2, 40)
    c11 = BatchNormalization()(c1)

    # ---------LSTM---------#
    c22 = Bidirectional(LSTM(20, return_sequences=True, activation='relu'))(batchnor2)
    c22 = BatchNormalization()(c22)

    # ---------Attention---------#
    c31 = Conv1D(40, 9, activation='relu', name="c31")(embedded)
    batchnor3 = BatchNormalization()(c31)
    c32 = Lambda(lambda x: attention(x[0], x[1], 16))([batchnor2,batchnor3])

    merged = concatenate([c11, c22, c32])
    flat = Flatten()(merged)

    dense1 = Dense(300, activation="relu", name="dense1")(flat)
    drop1 = Dropout(0.2)(dense1)

    dense2 = Dense(150, activation="relu", name="dense2")(drop1)
    drop2 = Dropout(0.2)(dense2)

    output = Dense(2, activation="softmax", name="dense3")(drop2)
    model = Model(inputs=[input], outputs=[output])
    return model
