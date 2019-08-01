#!/usr/bin/python3

# Copyright 2019 Peter Gazdik

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
import keras.layers
from keras.models import Model
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Input
from keras.callbacks import LearningRateScheduler
from model_interface import ModelInterface


class PdfModel(ModelInterface):
    lr_0 = 0.1
    drop = 0.5
    epochs_drop = 2

    @classmethod
    def build(cls, input_dim, out_dim, hidden_act='relu', out_act='softmax',
              units_per_layer=512):
        input_shape = (input_dim,)

        x = Input(shape=input_shape)
        y = Dense(units_per_layer, activation=hidden_act)(x)
        y = Dropout(0.1)(y)
        y = Dense(units_per_layer, activation=hidden_act)(y)
        y = Dropout(0.1)(y)
        y = Dense(units_per_layer, activation=hidden_act)(y)
        y = Dropout(0.1)(y)
        y = Dense(out_dim, activation=out_act)(y)

        m = Model(inputs=x, outputs=y)

        opt = keras.optimizers.SGD(lr=PdfModel.lr_0,
                                   decay=0, momentum=0.5, nesterov=True)
        m.compile(opt, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

        m.summary()

        return cls(m)

    @classmethod
    def transfer_and_build(cls, base_model_fname, out_dim, out_act='softmax'):
        base_model = keras.models.load_model(base_model_fname)
        base_model.layers.pop()

        y = base_model.layers[-1].output
        y = Dense(out_dim, activation=out_act, name='pdf_output')(y)
        m = Model(inputs=base_model.input, outputs=y)

        opt = keras.optimizers.SGD(lr=PdfModel.lr_0,
                                   decay=0, momentum=0.5, nesterov=True)
        m.compile(opt, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

        m.summary()

        return cls(m)

    def train_decay(self, train_generator, validation_generator, out_fname,
                    epochs=1, monitor='val_loss', mode='min', min_delta=0,
                    patience=10):

        lrs = LearningRateScheduler(PdfModel.step_decay, verbose=1)
        self.train(train_generator, validation_generator, out_fname,
                   epochs=epochs, monitor=monitor, mode=mode,
                   min_delta=min_delta, patience=patience, callbacks=[lrs])

    @staticmethod
    def step_decay(epoch):
        lr = PdfModel.lr_0 \
             * math.pow(PdfModel.drop,
                        math.floor((1 + epoch) / PdfModel.epochs_drop))

        return lr

    @classmethod
    def load(cls, model_fname):
        m = keras.models.load_model(model_fname)

        return cls(m)





