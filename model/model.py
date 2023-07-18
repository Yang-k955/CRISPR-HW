
import os

import keras
from keras.layers import Conv2D,MaxPool1D, MaxPooling1D, AveragePooling1D, MaxPool2D, GlobalAveragePooling1D, GlobalMaxPool1D, Input, Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, BatchNormalization, concatenate, Activation, Multiply, Permute, Reshape, Lambda, Add,multiply,Flatten
from keras.models import Model
from keras.initializers import RandomUniform,glorot_normal,VarianceScaling
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Attention

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

def CRISPR_HW_noatt():
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
    #c31 = Conv1D(40, 9, activation='relu', name="c31")(embedded)
    #batchnor3 = BatchNormalization()(c31)
    #c32 = Lambda(lambda x: attention(x[0], x[1], 16))([batchnor2,batchnor3])

    merged = concatenate([c11, c22])
    flat = Flatten()(merged)

    dense1 = Dense(300, activation="relu", name="dense1")(flat)
    drop1 = Dropout(0.2)(dense1)

    dense2 = Dense(150, activation="relu", name="dense2")(drop1)
    drop2 = Dropout(0.2)(dense2)

    output = Dense(2, activation="softmax", name="dense3")(drop2)
    model = Model(inputs=[input], outputs=[output])

    return model

def CRISPR_HW_noresnet():
    input = Input(shape=(24,))
    embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)
    conv1 = Conv1D(70, 4, activation="relu", name="conv1")(embedded)
    batchnor1 = BatchNormalization()(conv1)
    conv2 = Conv1D(40, 6, activation="relu", name="conv2")(batchnor1)
    batchnor2 = BatchNormalization()(conv2)

    #---------Resnet---------#
    #c1 = residual_block(batchnor2, 40)
    #c11 = BatchNormalization()(c1)

    # ---------LSTM---------#
    c22 = Bidirectional(LSTM(20, return_sequences=True, activation='relu'))(batchnor2)
    c22 = BatchNormalization()(c22)

    # ---------Attention---------#
    c31 = Conv1D(40, 9, activation='relu', name="c31")(embedded)
    batchnor3 = BatchNormalization()(c31)
    c32 = Lambda(lambda x: attention(x[0], x[1], 16))([batchnor2,batchnor3])


    merged = concatenate([c22, c32])
    flat = Flatten()(merged)

    dense1 = Dense(300, activation="relu", name="dense1")(flat)
    drop1 = Dropout(0.4)(dense1)

    dense2 = Dense(80, activation="relu", name="dense2")(drop1)
    drop2 = Dropout(0.4)(dense2)

    output = Dense(2, activation="softmax", name="dense3")(drop2)
    model = Model(inputs=[input], outputs=[output])

    return model

def CRISPR_HW_noblstm():
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
    #batchnor2 = BatchNormalization()(batchnor2)
    #c22 = Bidirectional(LSTM(20, return_sequences=True, activation='relu'))(batchnor2)

    # ---------Attention---------#
    c31 = Conv1D(40, 9, activation='relu', name="c31")(embedded)
    batchnor3 = BatchNormalization()(c31)
    c32 = Lambda(lambda x: attention(x[0], x[1], 16))([batchnor2,batchnor3])

    merged = concatenate([c11, c32])
    flat = Flatten()(merged)

    dense1 = Dense(300, activation="relu", name="dense1")(flat)
    drop1 = Dropout(0.2)(dense1)

    dense2 = Dense(150, activation="relu", name="dense2")(drop1)
    drop2 = Dropout(0.2)(dense2)

    output = Dense(2, activation="softmax", name="dense3")(drop2)
    model = Model(inputs=[input], outputs=[output])

    return model

def CRISPR_HW_Linear():
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
    c22 = Bidirectional(LSTM(20, return_sequences=True, activation='relu'))(c11)

    # ---------Attention---------#
    mp = MaxPooling1D(pool_size=2, strides=None, padding='valid')(c22)
    ap = AveragePooling1D(pool_size=2, strides=None, padding='valid')(c22)

    c32 = Lambda(lambda x: attention(x[0], x[1], 8))([mp,ap])

    flat = Flatten()(c32)

    dense1 = Dense(300, activation="relu", name="dense1")(flat)
    drop1 = Dropout(0.2)(dense1)

    dense2 = Dense(150, activation="relu", name="dense2")(drop1)
    drop2 = Dropout(0.2)(dense2)

    output = Dense(2, activation="softmax", name="dense3")(drop2)
    model = Model(inputs=[input], outputs=[output])
    return model

def CRISPR_HW_nodense():
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

    output = Dense(2, activation="softmax", name="dense3")(flat)
    model = Model(inputs=[input], outputs=[output])
    return model

def crispr_HW_onehot():
    # 有问题
    input = Input(shape=(1, 24, 7))
    # 定义模型输入
    embedded = Conv2D(30, (1,10) ,activation='relu')(input)
    embedded = BatchNormalization()(embedded)
    embedded = Reshape((15, 30))(embedded)

    batchnor1 = Conv1D(70, 5, activation="relu", name="conv1")(embedded)
    batchnor1 = BatchNormalization()(batchnor1)

    conv2 = Conv1D(40, 6, activation="relu", name="conv2")(batchnor1)
    batchnor2 = BatchNormalization()(conv2)

    #---------Resnet---------#
    c1 = residual_block(batchnor2, 40)
    c11 = BatchNormalization()(c1)

    # ---------LSTM---------#
    c22 = Bidirectional(LSTM(20, return_sequences=True, activation='relu'))(batchnor2)
    c22 = BatchNormalization()(c22)

    # ---------Attention---------#
    c31 = Conv1D(40, 10, activation='relu', name="c31")(embedded)
    batchnor3 = BatchNormalization()(c31)
    c32 = Lambda(lambda x: attention(x[0], x[1], 6))([batchnor2,batchnor3])

    merged = concatenate([c11, c22, c32])
    flat = Flatten()(merged)

    dense1 = Dense(300, activation="relu", name="dense1")(flat)
    drop1 = Dropout(0.2)(dense1)

    dense2 = Dense(100, activation="relu", name="dense2")(drop1)
    drop2 = Dropout(0.2)(dense2)

    output = Dense(2, activation="softmax", name="dense3")(drop2)
    model = Model(inputs=[input], outputs=[output])
    return model

def CnnCrispr():
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    input = Input(shape=(23,))
    embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,trainable=True)(input)
    bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
    bi_lstm_relu = Activation('relu')(bi_lstm)

    conv1 = Conv1D(10, (5))(bi_lstm_relu)
    conv1_relu = Activation('relu')(conv1)
    conv1_batch = BatchNormalization()(conv1_relu)

    conv2 = Conv1D(20, (5))(conv1_batch)
    conv2_relu = Activation('relu')(conv2)
    conv2_batch = BatchNormalization()(conv2_relu)

    conv3 = Conv1D(40, (5))(conv2_batch)
    conv3_relu = Activation('relu')(conv3)
    conv3_batch = BatchNormalization()(conv3_relu)

    conv4 = Conv1D(80, (5))(conv3_batch)
    conv4_relu = Activation('relu')(conv4)
    conv4_batch = BatchNormalization()(conv4_relu)

    conv5 = Conv1D(100, (5))(conv4_batch)
    conv5_relu = Activation('relu')(conv5)
    conv5_batch = BatchNormalization()(conv5_relu)

    flat = Flatten()(conv5_batch)
    drop = Dropout(0.3)(flat)
    dense = Dense(20)(drop)
    dense_relu = Activation('relu')(dense)
    prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
    model = Model(inputs=[input], outputs=[prediction])
    return model

def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=True,
              name=None, trainable=True):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      kernel_initializer=glorot_normal(seed=2023),
                      name=name, trainable=trainable)(x)

    # x = layers.BatchNormalization(axis=-1,scale=True)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def CRISPR_Net():
    inputs = Input(shape=(1, 24, 7), name='main_input')
    branch_0 = Conv2D(10,(1,1),strides=1,padding='same',use_bias=True,
                      kernel_initializer=glorot_normal(seed=2023),name=None, trainable=True)(inputs)
    branch_1 = Conv2D(10, (1, 2), strides=1, padding='same', use_bias=True,
                      kernel_initializer=glorot_normal(seed=2023), name=None, trainable=True)(inputs)
    branch_2 = Conv2D(10, (1, 3), strides=1, padding='same', use_bias=True,
                      kernel_initializer=glorot_normal(seed=2023), name=None, trainable=True)(inputs)
    branch_3 = Conv2D(10, (1, 4), strides=1, padding='same', use_bias=True,
                      kernel_initializer=glorot_normal(seed=2023), name=None, trainable=True)(inputs)

    branches = [inputs, branch_0, branch_1, branch_2, branch_3]
    mixed = concatenate(branches, axis=-1)
    mixed = Reshape((24, 47))(mixed)
    blstm_out = Bidirectional(
        LSTM(15, kernel_initializer=glorot_normal(seed=2023), return_sequences=True, input_shape=(24, 47),
             name="LSTM_out"))(mixed)
    # inputs_rs = Reshape((24, 7))(inputs)
    # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
    blstm_out = Flatten()(blstm_out)
    x = Dense(80, kernel_initializer=glorot_normal(seed=2023), activation='relu')(blstm_out)
    x = Dense(20, kernel_initializer=glorot_normal(seed=2023), activation='relu')(x)
    x = Dropout(0.35, seed=2023)(x)
    prediction = Dense(2, kernel_initializer=glorot_normal(seed=2023), activation='softmax', name='main_output')(x)
    model = Model(inputs=[inputs], outputs=[prediction])
    return model

def CRISPR_IP():
    inputs = Input(shape=(1, 24, 7), name='main_input')
    conv_1_output = Conv2D(60, (1, 7), padding='valid')(inputs)
    conv_1_output_reshape = Reshape((18, 60))(conv_1_output)
    #conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
    conv_1_output_reshape2 = Permute((2, 1))(conv_1_output_reshape)
    conv_1_output_reshape_average = AveragePooling1D(pool_size=2, strides=None, padding='valid')(conv_1_output_reshape2)
    conv_1_output_reshape_max = MaxPool1D(pool_size=2, strides=None, padding='valid')(conv_1_output_reshape2)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))\
        (concatenate([conv_1_output_reshape_average, conv_1_output_reshape_max],axis=-1))
    attention_1_output = Lambda(lambda x: attention(x[0], x[1], 30))([bidirectional_1_output, bidirectional_1_output])
    average_1_output = AveragePooling1D(pool_size=2, strides=None, padding='valid')(attention_1_output)
    max_1_output = MaxPool1D(pool_size=2, strides=None, padding='valid')(attention_1_output)

    concat_output = concatenate([average_1_output, max_1_output],axis=-1)

    flatten_output = Flatten()(concat_output)
    linear_1_output = BatchNormalization()(
        Dense(200, activation='relu')(flatten_output))
    linear_2_output = Dense(100, activation='relu')(linear_1_output)
    linear_2_output_dropout = Dropout(0.9)(linear_2_output)
    output = Dense(2, activation='softmax')(linear_2_output_dropout)
    model = Model(inputs=[inputs], outputs=[output])
    return model

def crispr_offt():
    VOCAB_SIZE = 16
    EMBED_SIZE = 90
    MAXLEN = 23
    input = Input(shape=(23,))
    embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

    conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
    batchnor1 = BatchNormalization()(conv1)

    conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
    batchnor2 = BatchNormalization()(conv2)

    conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
    batchnor3 = BatchNormalization()(conv3)

    conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
    x = Lambda(lambda x: attention(x[0], x[1], 11))([conv11, batchnor3])

    flat = Flatten()(x)
    dense1 = Dense(40, activation="relu", name="dense1")(flat)
    drop1 = Dropout(0.4)(dense1)

    dense2 = Dense(20, activation="relu", name="dense2")(drop1)
    drop2 = Dropout(0.4)(dense2)

    output = Dense(2, activation="softmax", name="dense3")(drop2)
    model = Model(inputs=[input], outputs=[output])
    return model

def cnn_std():
    input = Input(shape=(1, 23, 4))
    conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
    conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
    conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
    conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

    conv_output = concatenate([conv_1, conv_2, conv_3, conv_4])

    bn_output = BatchNormalization()(conv_output)

    pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

    flatten_output = Flatten()(pooling_output)

    x = Dense(100, activation='relu')(flatten_output)
    x = Dense(23, activation='relu')(x)
    x = Dropout(rate=0.15)(x)

    output = Dense(2, activation="softmax")(x)

    model = Model(inputs=[input], outputs=[output])
    return model

if __name__ == "__main__":
    model = crispr_HW_onehot()
    model.summary()
