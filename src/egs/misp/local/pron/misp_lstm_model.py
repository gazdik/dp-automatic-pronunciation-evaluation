#!/usr/bin/python3

# Copyright 2018 Peter Gazdik

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

import math
import keras
import keras.backend as K
from keras.models import Model
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Input
from keras.layers import LSTM
from keras.callbacks import LearningRateScheduler
from model_interface import ModelInterface


class MispLSTMModel(ModelInterface):
    lr_0 = 0.001
    drop = 0.5
    epochs_drop = 1

    @classmethod
    def build(cls, input_dim, out_dim, hidden_act='relu', out_act='softmax'):
        input_shape = (None, input_dim)

        x = Input(shape=input_shape)
        y = LSTM(256, recurrent_dropout=0.2)(x)
        y = Dense(out_dim, activation=out_act)(y)

        m = Model(inputs=x, outputs=y)

        opt = keras.optimizers.Adam(lr=MispLSTMModel.lr_0)
        # opt = keras.optimizers.SGD(lr=MispModel.lr_0, decay=0, momentum=0.5,
        #                            nesterov=True)
        m.compile(opt, loss=cls.loss, metrics=[cls.accuracy])

        m.summary()

        return cls(m)

    @classmethod
    def load(cls, model_fname):
        m = keras.models.load_model(
            model_fname,
            custom_objects={
                'loss': cls.loss,
                'accuracy': cls.accuracy
            }
        )

        return cls(m)

    def train_decay(self, train_generator, validation_generator, out_fname,
                    epochs=1, monitor='val_loss', mode='min', min_delta=0,
                    patience=10):

        lrs = LearningRateScheduler(MispLSTMModel.step_decay, verbose=1)
        self.train(train_generator, validation_generator, out_fname,
                   epochs=epochs, monitor=monitor, mode=mode,
                   min_delta=min_delta, patience=patience, callbacks=[lrs])


    @staticmethod
    def step_decay(epoch):
        lr = MispLSTMModel.lr_0 \
             * math.pow(MispLSTMModel.drop,
                        math.floor((1 + epoch) / MispLSTMModel.epochs_drop))

        return lr

    @staticmethod
    def loss(y_true, y_pred):
        y_true_s, y_pred_s = MispLSTMModel.mask_invalid(y_true, y_pred)
        return K.binary_crossentropy(y_true_s, y_pred_s)


    @staticmethod
    def accuracy(y_true, y_pred):
        y_true_s, y_pred_s = MispLSTMModel.mask_invalid(y_true, y_pred)
        return K.mean(K.equal(y_true_s, K.round(y_pred_s)), axis=-1)

    @staticmethod
    def mask_invalid(y_true, y_pred):
        mask = K.tf.not_equal(y_true, -1)
        y_true_s = K.tf.boolean_mask(y_true, mask)
        y_pred_s = K.tf.boolean_mask(y_pred, mask)
        return y_true_s, y_pred_s
