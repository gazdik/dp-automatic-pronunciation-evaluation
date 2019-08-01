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
from keras.models import Model
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Input
from keras.callbacks import LearningRateScheduler
from model_interface import ModelInterface
from pfeats_model import PFeatsModel


class PdfPfeatsModel(ModelInterface):
    lr_0 = 0.001
    drop = 0.5
    epochs_drop = 2

    @classmethod
    def build(cls, input_dim, out_dim, hidden_act='relu', out_act=('softmax', 'sigmoid')):
        input_shape = (input_dim,)

        # Shared layers
        x = Input(shape=input_shape)
        y = Dense(512, activation=hidden_act)(x)
        y = Dropout(0.1)(y)
        y = Dense(512, activation=hidden_act)(y)
        y = Dropout(0.1)(y)
        y = Dense(512, activation=hidden_act)(y)
        y = Dropout(0.1)(y)

        y_pdf = PdfPfeatsModel.build_pdf_branch(
            y, out_dim[0], hidden_act, out_act[0]
        )
        y_pfeats = PdfPfeatsModel.build_pfeats_branch(
            y, out_dim[1], hidden_act, out_act[1]
        )

        # Create model
        m = Model(inputs=x, outputs=[y_pdf, y_pfeats])

        # Compile model
        losses = {
            'pdf_output': 'sparse_categorical_crossentropy',
            'pfeats_output': cls.pfeats_loss
        }
        loss_weights = {'pdf_output': 1.0, 'pfeats_output': 1.0}
        metrics = {
            'pdf_output': ['accuracy'],
            'pfeats_output': [cls.pfeats_accuracy]
        }

        opt = keras.optimizers.Adam(lr=PdfPfeatsModel.lr_0)
        m.compile(opt, loss=losses , loss_weights=loss_weights,
                  metrics=metrics)

        return cls(m)

    @staticmethod
    def pfeats_loss(y_true, y_pred):
        return PFeatsModel.loss(y_true, y_pred)

    @staticmethod
    def pfeats_accuracy(y_true, y_pred):
        return PFeatsModel.accuracy(y_true, y_pred)

    @classmethod
    def load(cls, model_fname):
        m = keras.models.load_model(
            model_fname,
            custom_objects={
                'pfeats_loss': cls.pfeats_loss,
                'pfeats_accuracy': cls.pfeats_accuracy
            }
        )

        return cls(m)

    def train_decay(self, train_generator, validation_generator, out_fname,
                    epochs=1, monitor='val_loss', mode='min', min_delta=0,
                    patience=10):

        lrs = LearningRateScheduler(PdfPfeatsModel.step_decay, verbose=1)
        self.train(train_generator, validation_generator, out_fname,
                   epochs=epochs, monitor=monitor, mode=mode,
                   min_delta=min_delta, patience=patience, callbacks=[lrs])

    @staticmethod
    def step_decay(epoch):
        lr = PdfPfeatsModel.lr_0 \
             * math.pow(PdfPfeatsModel.drop,
                        math.floor((1 + epoch) / PdfPfeatsModel.epochs_drop))

        return lr

    @staticmethod
    def build_pdf_branch(inputs, out_dim, hidden_act, out_act):
        x = Dense(out_dim, activation=out_act, name='pdf_output')(inputs)
        return x

    @staticmethod
    def build_pfeats_branch(inputs, out_dim, hidden_act, out_act):
        x = Dense(out_dim, activation=out_act, name='pfeats_output')(inputs)
        return x

